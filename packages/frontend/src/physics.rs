//! The **Box3D** world — a scene-derived rigid-body sim, stepped in-line by the
//! game loop.
//!
//! The exported scene's `Collider` nodes become the world: the table + walls are
//! static bodies (box hulls), the ball is a dynamic sphere. The ball drops in and
//! bounces; WASD/arrow keys roll it, Space makes it hop, clicks drop more.
//!
//! Box3D is C (vendored at `vendor/box3d`), compiled into this same wasm module
//! by `box3d-sys` — no bridge, no copies. The simulation is **single-threaded**,
//! so the world runs Box3D with `workerCount = 1`: its internal parallel-for
//! takes the inline serial path and no task scheduler is touched.
//!
//! The sim runs at a **fixed timestep** ([`SIM_HZ`]) decoupled from the wall
//! clock: the game loop's accumulator ([`crate::game`]) calls [`World::step`]
//! exactly as many fixed steps as elapsed real time bought (capped, to avoid a
//! spiral of death). Each step publishes the ball's pose into a plain prev/curr
//! double-buffer; the render side interpolates prev→curr by a single alpha for
//! smooth motion regardless of the fixed rate. Because physics and render share
//! one thread and one clock, that single-alpha interpolation is exact — no jitter
//! buffer needed.
//!
//! Input is read straight from the shared [`Input`] block each step (main writes
//! it from the DOM handlers). Audio cues — the ball's roll + its wall/floor/ball
//! impacts — are pushed into a `cues` buffer the game loop drains into the audio
//! player; they're classified from Box3D's begin-touch contact events plus the
//! world's hit events (which carry the approach speed).

use std::collections::HashMap;
use std::ffi::c_void;

use box3d_sys as b3;
use wasm_bindgen::prelude::*;

use crate::protocol::{
    AudioMsg, ColliderInit, ColliderShapeMsg, Input, HELD_BACK, HELD_FORWARD, HELD_LEFT,
    HELD_RIGHT, MAX_BALLS, ROLE_BALL, ROLE_FLOOR, ROLE_WALL, SIM_HZ,
};

/// A published body pose: `(position, quaternion)` as plain arrays. The render
/// side lerps position and slerps rotation between the prev/curr pair.
pub type Pose = ([f32; 3], [f32; 4]);

/// Fixed simulation step, derived from the shared [`SIM_HZ`] (the one rate knob).
const FIXED_DT_SECS: f32 = (1.0 / SIM_HZ) as f32;
/// Box3D's internal solver sub-steps per `b3World_Step` (its *Soft Step* solver
/// iterates inside one step). Upstream advises 4 at a 60 Hz step; our outer rate
/// is already 240 Hz, so 1 sub-step lands on the same 240 Hz solver cadence —
/// raise this only if contact quality demands it (cost is linear).
const SUB_STEPS: i32 = 1;
const GRAVITY: f32 = -9.81;
/// How far above its authored position the ball starts, so it drops in.
const DROP_HEIGHT: f32 = 2.0;
/// Horizontal rolling **acceleration** (m/s²) applied while a movement key is
/// held. Expressed per *second*, not per tick, so the feel is identical at any
/// [`SIM_HZ`]: each step adds `MOVE_ACCEL * FIXED_DT_SECS` to the velocity. The
/// impulse is mass-scaled at apply time so the feel is independent of ball size.
const MOVE_ACCEL: f32 = 27.0;
/// Cap on horizontal (xz) rolling speed. The per-tick kick is applied
/// continuously while a key is held; with the ball's light damping that would
/// otherwise accumulate into a runaway velocity, so we clamp the horizontal
/// speed each tick (vertical/jump motion is left untouched).
const MAX_SPEED: f32 = 6.0;
/// Launch velocity for a jump (also mass-scaled into an impulse).
const JUMP_DV: f32 = 3.2;
/// Cap on the horizontal speed a touch fling can set (m/s). Deliberately above
/// [`MAX_SPEED`] so a hard flick reads as a real throw, but bounded so the
/// ball can't be launched arbitrarily fast over the rails.
const FLING_MAX: f32 = 9.0;
/// Restitution/friction for the static table + walls.
const STATIC_RESTITUTION: f32 = 0.5;
const STATIC_FRICTION: f32 = 0.9;
/// Restitution/friction/damping for the dynamic ball.
const BALL_RESTITUTION: f32 = 0.6;
const BALL_FRICTION: f32 = 0.9;
const BALL_LINEAR_DAMPING: f32 = 0.1;
const BALL_ANGULAR_DAMPING: f32 = 0.4;
/// Target rate (Hz) for the continuous roll cue. The audio side glides between
/// values, so ~20 Hz stays smooth; kept as a rate so the cue traffic doesn't
/// balloon with [`SIM_HZ`].
const ROLL_CUE_HZ: f64 = 20.0;
/// Emit the roll cue every Nth step — derived so it fires ~[`ROLL_CUE_HZ`] at any
/// sim rate (3 at 60 Hz, 12 at 240 Hz). Impacts are emitted immediately.
const ROLL_EVERY: u32 = (SIM_HZ / ROLL_CUE_HZ + 0.5) as u32;
/// Below this normalized roll speed the ball is treated as effectively still.
const ROLL_FLOOR: f32 = 0.04;
/// Vertical speed (m/s) that maps a landing to full intensity.
const LAND_FULL_SPEED: f32 = 6.5;
/// Minimum approach speed (m/s) for a contact to count as a real impact — below
/// this it's resting/grinding contact, not a hit (avoids machine-gun retriggers
/// when a ball is shoved steadily against a wall or settling on the floor).
/// Enforced by Box3D itself: this is the world's `hitEventThreshold`.
const HIT_MIN_SPEED: f32 = 0.8;
/// Minimum *time* between successive impacts of the same kind — a debounce so
/// shoving the ball against a rail doesn't machine-gun the knock.
const IMPACT_COOLDOWN_SECS: f64 = 0.13;
/// The same debounce in steps, derived from [`SIM_HZ`] (8 at 60 Hz, 31 at 240 Hz).
const IMPACT_COOLDOWN: u32 = (IMPACT_COOLDOWN_SECS * SIM_HZ + 0.5) as u32;
/// Below this height the ball has escaped the table; we drop it back to spawn
/// (the walls are low rails, so a determined shove can pop the ball over them).
const FALL_LIMIT: f32 = -2.0;

const fn v3(x: f32, y: f32, z: f32) -> b3::b3Vec3 {
    b3::b3Vec3 { x, y, z }
}

const QUAT_IDENTITY: b3::b3Quat = b3::b3Quat {
    v: v3(0.0, 0.0, 0.0),
    s: 1.0,
};

/// Box3D assert hook: surface the condition in the console before the trap
/// (without this an assert is an opaque `RuntimeError: unreachable`).
/// Returning nonzero keeps the debugger break (loud failure).
unsafe extern "C" fn box3d_assert(
    condition: *const core::ffi::c_char,
    file: *const core::ffi::c_char,
    line: i32,
) -> i32 {
    let cstr = |p: *const core::ffi::c_char| {
        if p.is_null() {
            "?".into()
        } else {
            core::ffi::CStr::from_ptr(p).to_string_lossy()
        }
    };
    tracing::error!("BOX3D ASSERT: {} ({}:{line})", cstr(condition), cstr(file));
    1
}

/// Route Box3D's printf output (its default assert/warning formatting) and
/// asserts to the console. Idempotent; called once at world creation.
fn install_box3d_hooks() {
    b3::wasm_shim::set_shim_log(|msg| tracing::warn!("box3d: {msg}"));
    unsafe { b3::b3SetAssertFcn(box3d_assert) };
}

/// The live Box3D world plus everything the game loop needs to drive it: the
/// player ball, the click-dropped balls, and a prev/curr pose double-buffer per
/// body for the render side to interpolate.
pub struct World {
    world: b3::b3WorldId,
    ball_body: b3::b3BodyId,
    ball_shape: b3::b3ShapeId,
    ball_mass: f32,
    ball_radius: f32,
    /// The player ball's authored spawn (fall-through respawn target).
    spawn: b3::b3Vec3,
    /// Ground plane the ball settles on (`spawn.y`), used to place dropped balls.
    base_y: f32,

    /// Click-dropped balls: body id + drop point (its own fall-through target).
    drop_bodies: Vec<b3::b3BodyId>,
    drop_spawns: Vec<b3::b3Vec3>,

    /// Prev/curr player pose (interpolated by the render alpha).
    player_prev: Pose,
    player_curr: Pose,
    /// Prev/curr pose per dropped ball (parallel to `drop_bodies`).
    ball_prev: Vec<Pose>,
    ball_curr: Vec<Pose>,

    /// Shapes the player ball is currently touching (`other shape index1` →
    /// role), maintained from begin/end contact events. Membership ⇒ grounded.
    contacts: HashMap<i32, u8>,

    /// Input edge-detect state (jump / fling counters last seen).
    last_jump_seq: u32,
    last_fling_seq: u32,

    tick: u32,
    last_wall_tick: u32,
    last_land_tick: u32,
    last_clack_tick: u32,
}

impl World {
    /// Build the world from the scene's collider list: the table + walls become
    /// static bodies, the one dynamic collider becomes the player ball (dropped
    /// from [`DROP_HEIGHT`] above its authored spawn).
    pub fn new(colliders: &[ColliderInit], spawn: [f32; 3]) -> Result<World, JsValue> {
        install_box3d_hooks();
        tracing::info!(
            "physics: building Box3D world — {SIM_HZ} Hz (dt {:.2}ms, {SUB_STEPS} solver \
             sub-steps, roll every {ROLL_EVERY}, impact cooldown {IMPACT_COOLDOWN})",
            1000.0 / SIM_HZ
        );

        // Single-threaded: `workerCount = 1` takes Box3D's inline serial task
        // path — its internal pthread scheduler (which doesn't exist on wasm) is
        // never touched, and no enqueue/finish callbacks are wired up.
        let world = unsafe {
            let mut def = b3::b3DefaultWorldDef();
            def.gravity = v3(0.0, GRAVITY, 0.0);
            def.hitEventThreshold = HIT_MIN_SPEED; // impacts are voiced from hit events
            def.workerCount = 1;
            b3::b3CreateWorld(&def)
        };
        if !unsafe { b3::b3World_IsValid(world) } {
            return Err(JsValue::from_str("physics: b3CreateWorld failed"));
        }

        // Static geometry: the table + walls, one static body per collider placed
        // at its world pose. Each shape carries its gameplay role in userData so
        // contact events can be classified.
        for c in colliders.iter().filter(|c| !c.dynamic) {
            unsafe {
                let mut body_def = b3::b3DefaultBodyDef();
                body_def.position = v3(c.translation[0], c.translation[1], c.translation[2]);
                body_def.rotation = b3::b3Quat {
                    v: v3(c.rotation[0], c.rotation[1], c.rotation[2]),
                    s: c.rotation[3],
                };
                let body = b3::b3CreateBody(world, &body_def);

                let mut shape_def = b3::b3DefaultShapeDef();
                shape_def.baseMaterial.friction = STATIC_FRICTION;
                shape_def.baseMaterial.restitution = STATIC_RESTITUTION;
                shape_def.userData = c.role as usize as *mut c_void;
                // Contact events are OR'd across a pair's shapes (contact.c) — the
                // player ball's own enable already covers ball-vs-static. Enabling
                // here too would flood the arrays with dropped-ball contacts.
                shape_def.enableContactEvents = false;
                create_shape(body, &shape_def, &c.shape);
            }
        }

        // The dynamic ball, dropped from above its authored spawn.
        let ball_def = colliders
            .iter()
            .find(|c| c.dynamic)
            .ok_or_else(|| JsValue::from_str("physics: no dynamic collider (ball) in scene"))?;
        let spawn_v = v3(spawn[0], spawn[1] + DROP_HEIGHT, spawn[2]);
        let (ball_body, ball_shape) = unsafe {
            let mut body_def = b3::b3DefaultBodyDef();
            body_def.r#type = b3::b3_dynamicBody;
            body_def.position = spawn_v;
            body_def.linearDamping = BALL_LINEAR_DAMPING;
            body_def.angularDamping = BALL_ANGULAR_DAMPING;
            body_def.enableSleep = false;
            // Continuous collision so the ball can't tunnel through the thin rails
            // at speed (Box3D bullets do CCD against static + non-bullet bodies).
            body_def.isBullet = true;
            let body = b3::b3CreateBody(world, &body_def);

            let mut shape_def = b3::b3DefaultShapeDef();
            shape_def.density = 1.0;
            shape_def.baseMaterial.friction = BALL_FRICTION;
            shape_def.baseMaterial.restitution = BALL_RESTITUTION;
            shape_def.userData = ball_def.role as usize as *mut c_void;
            // Contact events maintain the grounded/contacts set (roll cue); hit
            // events voice the impacts (they carry position + approach speed).
            shape_def.enableContactEvents = true;
            shape_def.enableHitEvents = true;
            let shape = create_shape(body, &shape_def, &ball_def.shape);
            (body, shape)
        };
        let ball_mass = unsafe { b3::b3Body_GetMass(ball_body) };
        if ball_mass <= 0.0 || !ball_mass.is_finite() {
            return Err(JsValue::from_str("physics: ball has no mass"));
        }
        let ball_radius = match ball_def.shape {
            ColliderShapeMsg::Ball { radius } => radius,
            _ => 0.5,
        };

        let seed: Pose = ([spawn_v.x, spawn_v.y, spawn_v.z], [0.0, 0.0, 0.0, 1.0]);
        Ok(World {
            world,
            ball_body,
            ball_shape,
            ball_mass,
            ball_radius,
            spawn: spawn_v,
            base_y: spawn[1],
            drop_bodies: Vec::new(),
            drop_spawns: Vec::new(),
            player_prev: seed,
            player_curr: seed,
            ball_prev: Vec::new(),
            ball_curr: Vec::new(),
            contacts: HashMap::new(),
            last_jump_seq: 0,
            last_fling_seq: 0,
            tick: 0,
            last_wall_tick: 0,
            last_land_tick: 0,
            last_clack_tick: 0,
        })
    }

    /// Drop a new silver ball at table position `(x, z)`. Returns `false` (and
    /// does nothing) once [`MAX_BALLS`] have been dropped.
    pub fn drop_ball(&mut self, x: f32, z: f32) -> bool {
        if self.drop_bodies.len() >= MAX_BALLS {
            tracing::warn!("ball drop ignored — MAX_BALLS ({MAX_BALLS}) reached");
            return false;
        }
        let spawn_at = v3(x, DROP_HEIGHT + self.base_y, z);
        unsafe {
            let mut body_def = b3::b3DefaultBodyDef();
            body_def.r#type = b3::b3_dynamicBody;
            body_def.position = spawn_at;
            body_def.linearDamping = BALL_LINEAR_DAMPING;
            body_def.angularDamping = BALL_ANGULAR_DAMPING;
            // Cheaper than the player: sleep allowed (a settled pile costs
            // ~nothing), no bullet CCD.
            let body = b3::b3CreateBody(self.world, &body_def);
            let mut shape_def = b3::b3DefaultShapeDef();
            shape_def.density = 1.0;
            shape_def.baseMaterial.friction = BALL_FRICTION;
            shape_def.baseMaterial.restitution = BALL_RESTITUTION;
            shape_def.userData = ROLE_BALL as usize as *mut c_void;
            shape_def.enableHitEvents = true; // its drop + collisions make sound too
            b3::b3CreateSphereShape(
                body,
                &shape_def,
                &b3::b3Sphere {
                    center: v3(0.0, 0.0, 0.0),
                    radius: self.ball_radius,
                },
            );
            self.drop_bodies.push(body);
        }
        self.drop_spawns.push(spawn_at);
        let seed: Pose = ([spawn_at.x, spawn_at.y, spawn_at.z], [0.0, 0.0, 0.0, 1.0]);
        self.ball_prev.push(seed);
        self.ball_curr.push(seed);
        true
    }

    /// Advance the world one fixed step: apply input, step Box3D, classify
    /// contacts/hits into `cues`, and roll the prev/curr pose buffers forward.
    /// `camera_yaw` rotates the roll/fling input so W/A/S/D stay view-relative.
    pub fn step(&mut self, input: &Input, camera_yaw: f32, cues: &mut Vec<AudioMsg>) {
        self.tick = self.tick.wrapping_add(1);

        // ── Apply input before stepping (polled from the shared block) ──────
        unsafe {
            let held = input.held();
            let mut dir = v3(0.0, 0.0, 0.0);
            if held & HELD_FORWARD != 0 {
                dir.z -= 1.0;
            }
            if held & HELD_BACK != 0 {
                dir.z += 1.0;
            }
            if held & HELD_LEFT != 0 {
                dir.x -= 1.0;
            }
            if held & HELD_RIGHT != 0 {
                dir.x += 1.0;
            }
            let dir_sq = dir.x * dir.x + dir.z * dir.z;
            if dir_sq > 0.0 {
                // `dir` is in the CAMERA frame (W = away from the camera). Rotate
                // it into world space by the camera yaw so W/A/S/D stay
                // view-relative at any orbit angle. Basis from `OrbitCamera::eye()`
                // — the eye sits at (sinθ, cosθ)·r, so camera-forward is (−sinθ,
                // −cosθ) and camera-right is (cosθ, −sinθ); at θ = 0 this is the
                // identity (forward = −Z, right = +X).
                let (s, c) = camera_yaw.sin_cos();
                let wx = dir.x * c + dir.z * s;
                let wz = -dir.x * s + dir.z * c;
                let k = MOVE_ACCEL * FIXED_DT_SECS * self.ball_mass / dir_sq.sqrt();
                b3::b3Body_ApplyLinearImpulseToCenter(
                    self.ball_body,
                    v3(wx * k, 0.0, wz * k),
                    true,
                );
                // Clamp horizontal speed so the held impulse can't run away,
                // leaving the vertical component (gravity / jumps) intact.
                let v = b3::b3Body_GetLinearVelocity(self.ball_body);
                let horiz_sq = v.x * v.x + v.z * v.z;
                if horiz_sq > MAX_SPEED * MAX_SPEED {
                    let scale = MAX_SPEED / horiz_sq.sqrt();
                    b3::b3Body_SetLinearVelocity(self.ball_body, v3(v.x * scale, v.y, v.z * scale));
                }
            }
            // Jump is edge-triggered: act when main's counter has advanced.
            let seq = input.jump_seq();
            if seq != self.last_jump_seq {
                self.last_jump_seq = seq;
                b3::b3Body_ApplyLinearImpulseToCenter(
                    self.ball_body,
                    v3(0.0, JUMP_DV * self.ball_mass, 0.0),
                    true,
                );
            }
            // Touch fling — edge-triggered like the jump. The swipe velocity
            // arrives in the camera frame; rotate by the same yaw, then SET the
            // horizontal velocity (a throw reads as "the ball goes where I
            // flicked", not a nudge on top of existing motion). Vertical velocity
            // is left alone (gravity / jumps).
            if let Some((fx, fz)) = input.poll_fling(&mut self.last_fling_seq) {
                let (s, c) = camera_yaw.sin_cos();
                let mut wx = fx * c + fz * s;
                let mut wz = -fx * s + fz * c;
                let speed_sq = wx * wx + wz * wz;
                if speed_sq > FLING_MAX * FLING_MAX {
                    let k = FLING_MAX / speed_sq.sqrt();
                    wx *= k;
                    wz *= k;
                }
                let v = b3::b3Body_GetLinearVelocity(self.ball_body);
                b3::b3Body_SetLinearVelocity(self.ball_body, v3(wx, v.y, wz));
            }
        }

        unsafe { b3::b3World_Step(self.world, FIXED_DT_SECS, SUB_STEPS) };

        // ── Player ball: fall-through safety net, then publish pose ─────────
        let mut teleported = false;
        if unsafe { b3::b3Body_GetPosition(self.ball_body) }.y < FALL_LIMIT {
            unsafe {
                b3::b3Body_SetTransform(self.ball_body, self.spawn, QUAT_IDENTITY);
                b3::b3Body_SetLinearVelocity(self.ball_body, v3(0.0, 0.0, 0.0));
                b3::b3Body_SetAngularVelocity(self.ball_body, v3(0.0, 0.0, 0.0));
            }
            self.contacts.clear();
            teleported = true;
        }
        let (pos, quat) = body_pose(self.ball_body);

        // ── Contact + hit events → grounded set + impact sounds ─────────────
        unsafe {
            let events = b3::b3World_GetContactEvents(self.world);
            for i in 0..events.beginCount as usize {
                let ev = &*events.beginEvents.add(i);
                let other = if ev.shapeIdA == self.ball_shape {
                    ev.shapeIdB
                } else if ev.shapeIdB == self.ball_shape {
                    ev.shapeIdA
                } else {
                    continue;
                };
                self.contacts
                    .insert(other.index1, b3::b3Shape_GetUserData(other) as usize as u8);
            }
            for i in 0..events.endCount as usize {
                let ev = &*events.endEvents.add(i);
                let other = if ev.shapeIdA == self.ball_shape {
                    ev.shapeIdB
                } else if ev.shapeIdB == self.ball_shape {
                    ev.shapeIdA
                } else {
                    continue;
                };
                self.contacts.remove(&other.index1);
            }

            for i in 0..events.hitCount as usize {
                let ev = &*events.hitEvents.add(i);
                let role_a = b3::b3Shape_GetUserData(ev.shapeIdA) as usize as u8;
                let role_b = b3::b3Shape_GetUserData(ev.shapeIdB) as usize as u8;
                let (x, y, z) = (ev.point.x, ev.point.y, ev.point.z);
                // Voice by the "surface" hit: a rail knock, a floor thud, or
                // (ball-on-ball) the steel clack.
                if role_a == ROLE_WALL || role_b == ROLE_WALL {
                    if self.tick.wrapping_sub(self.last_wall_tick) >= IMPACT_COOLDOWN {
                        self.last_wall_tick = self.tick;
                        let intensity = (ev.approachSpeed / MAX_SPEED).clamp(0.12, 1.0);
                        cues.push(AudioMsg::WallHit { x, y, z, intensity });
                    }
                } else if role_a == ROLE_FLOOR || role_b == ROLE_FLOOR {
                    if self.tick.wrapping_sub(self.last_land_tick) >= IMPACT_COOLDOWN {
                        self.last_land_tick = self.tick;
                        let intensity = (ev.approachSpeed / LAND_FULL_SPEED).clamp(0.12, 1.0);
                        cues.push(AudioMsg::Land { x, y, z, intensity });
                    }
                } else if self.tick.wrapping_sub(self.last_clack_tick) >= IMPACT_COOLDOWN {
                    // ball-on-ball → the steel-sphere clack (its own debounce
                    // window, so a rail knock and a clack in the same instant both
                    // sound — they're distinct events).
                    self.last_clack_tick = self.tick;
                    let intensity = (ev.approachSpeed / MAX_SPEED).clamp(0.12, 1.0);
                    cues.push(AudioMsg::BallClack { x, y, z, intensity });
                }
            }
        }
        let grounded = self.contacts.values().any(|&role| role != ROLE_WALL);

        // ── Continuous roll cue (throttled) ─────────────────────────────────
        if self.tick.is_multiple_of(ROLL_EVERY) {
            let v = unsafe { b3::b3Body_GetLinearVelocity(self.ball_body) };
            let horiz = (v.x * v.x + v.z * v.z).sqrt();
            let speed = if grounded {
                (horiz / MAX_SPEED).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let speed = if speed < ROLL_FLOOR { 0.0 } else { speed };
            cues.push(AudioMsg::Roll {
                speed,
                x: pos[0],
                y: pos[1],
                z: pos[2],
            });
        }

        // Roll the player's prev/curr buffer (snap on teleport so the render side
        // never interpolates across the respawn).
        self.player_prev = if teleported {
            (pos, quat)
        } else {
            self.player_curr
        };
        self.player_curr = (pos, quat);

        // ── Dropped balls: same fall-through net, publish each pose ──────────
        for (i, body) in self.drop_bodies.iter().enumerate() {
            let ball_teleported = unsafe { b3::b3Body_GetPosition(*body) }.y < FALL_LIMIT;
            if ball_teleported {
                unsafe {
                    b3::b3Body_SetTransform(*body, self.drop_spawns[i], QUAT_IDENTITY);
                    b3::b3Body_SetLinearVelocity(*body, v3(0.0, 0.0, 0.0));
                    b3::b3Body_SetAngularVelocity(*body, v3(0.0, 0.0, 0.0));
                }
            }
            let pose = body_pose(*body);
            self.ball_prev[i] = if ball_teleported {
                pose
            } else {
                self.ball_curr[i]
            };
            self.ball_curr[i] = pose;
        }
    }

    /// The player ball's `(prev, curr)` poses for interpolation.
    pub fn player_poses(&self) -> (Pose, Pose) {
        (self.player_prev, self.player_curr)
    }

    /// How many balls have been dropped.
    pub fn ball_count(&self) -> usize {
        self.drop_bodies.len()
    }

    /// A dropped ball's `(prev, curr)` poses for interpolation.
    pub fn ball_poses(&self, index: usize) -> (Pose, Pose) {
        (self.ball_prev[index], self.ball_curr[index])
    }
}

/// Read a body's current pose as `(position, quaternion)` arrays.
fn body_pose(body: b3::b3BodyId) -> Pose {
    unsafe {
        let p = b3::b3Body_GetPosition(body);
        let q = b3::b3Body_GetRotation(body);
        ([p.x, p.y, p.z], [q.v.x, q.v.y, q.v.z, q.s])
    }
}

/// Create the Box3D shape for a scene collider on `body` (which carries the
/// world pose; the shape is in the body's local frame). Capsule/cylinder/cone
/// are Y-axis-aligned, matching the scene's local frames.
///
/// SAFETY: `body` must be a live body id and `def` a cookie-valid shape def.
unsafe fn create_shape(
    body: b3::b3BodyId,
    def: &b3::b3ShapeDef,
    shape: &ColliderShapeMsg,
) -> b3::b3ShapeId {
    match *shape {
        ColliderShapeMsg::Cuboid {
            half_extents: [x, y, z],
        } => {
            let hull = b3::b3MakeBoxHull(x, y, z);
            b3::b3CreateHullShape(body, def, &hull.base)
        }
        ColliderShapeMsg::Ball { radius } => b3::b3CreateSphereShape(
            body,
            def,
            &b3::b3Sphere {
                center: v3(0.0, 0.0, 0.0),
                radius,
            },
        ),
        ColliderShapeMsg::Capsule {
            half_height,
            radius,
        } => b3::b3CreateCapsuleShape(
            body,
            def,
            &b3::b3Capsule {
                center1: v3(0.0, -half_height, 0.0),
                center2: v3(0.0, half_height, 0.0),
                radius,
            },
        ),
        // The hull builders return heap hulls; the world interns a copy into its
        // hull database on shape creation, so destroying ours right after is safe
        // (upstream samples do exactly this).
        ColliderShapeMsg::Cylinder {
            half_height,
            radius,
        } => {
            // Spans y ∈ [yOffset, yOffset + height] → center with -half_height.
            let hull = b3::b3CreateCylinder(2.0 * half_height, radius, -half_height, 16);
            let shape = b3::b3CreateHullShape(body, def, hull);
            b3::b3DestroyHull(hull);
            shape
        }
        ColliderShapeMsg::Cone {
            half_height,
            radius,
        } => {
            // b3CreateCone builds a truncated cone spanning y ∈ [0, height] (no
            // offset — base sits at the node origin rather than centered) and
            // asserts both radii > 0, so the apex is 5% of the base: acceptable
            // approximations; the shipped scene has no cones.
            let hull = b3::b3CreateCone(2.0 * half_height, radius, radius * 0.05, 16);
            let shape = b3::b3CreateHullShape(body, def, hull);
            b3::b3DestroyHull(hull);
            shape
        }
    }
}
