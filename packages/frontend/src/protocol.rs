//! The shared types + constants the game's pieces pass around, and the
//! `postMessage` protocol between the two threads.
//!
//! This is the **single-threaded** template: the *simulation* is single-threaded
//! (one `requestAnimationFrame` loop, no parallelism, no shared-memory
//! coordination), but that loop lives on a **worker** so it never fights the
//! main thread for a frame. The split is deliberately minimal:
//!
//! - **Main thread** owns the DOM, input, and WebAudio.
//! - **Worker thread** ([`crate::game`]) owns the renderer AND the physics
//!   world — one loop, one memory, no sharing.
//!
//! They talk only by `postMessage` (serialized with `serde` +
//! `serde_wasm_bindgen`), and every message here is a **cold path** — one per
//! input event / gesture / settings change / audio cue, never per frame. There
//! is no shared `WebAssembly.Memory`, so no atomics, no futexes, and no
//! COOP/COEP: the worker instantiates its own memory from the same module.
//!
//! ```text
//!   main ──(OffscreenCanvas transferred, at spawn)──▶ worker
//!   main ──(InputMsg / CameraMsg / DropMsg / ResizeMsg / QualityMsg)──▶ worker
//!   worker ──(RenderMsg: progress / ready / gpu-info / error)──▶ main
//!   worker ──(AudioMsg: roll / wall-hit / land / clack)──▶ main ──▶ WebAudio
//!   worker ──(StatsMsg, ~1 Hz)──▶ main ──▶ stats panel
//! ```
//!
//! Every crossing is a `postMessage` on a cold path (input events, layout
//! changes, audio cues, ~1 Hz stats) — never per-frame render data. The two
//! threads share nothing else.

use std::cell::Cell;

use serde::{Deserialize, Serialize};

/// **The simulation's fixed step rate, in hertz — the single knob that sets the
/// physics/render timestep.** The fixed step is `1 / SIM_HZ` seconds; the sim
/// advances this many physics steps per second of real time. Both the physics
/// step and the render interpolation derive from *this* constant, so they can
/// never disagree.
///
/// ## Why a *fixed* step — and why retuning it is safe
///
/// A rigid-body solver is only stable and deterministic at a **constant** `dt`;
/// variable-`dt` stepping makes restitution / friction / penetration drift frame
/// to frame. So the world always advances in `1/SIM_HZ` chunks and the game loop
/// simply runs *as many* of them as real time has bought since the last frame
/// (the accumulator in [`crate::game`]). Raising `SIM_HZ` means **more sub-steps
/// per frame**, not a faster or slower game.
///
/// ## Why raising it helps — and what it costs
///
/// * **Latency** — rendering interpolates between the last two published poses, so
///   it shows the world up to one step in the past. Smaller step ⇒ less of it: the
///   interpolation floor is `~1000/SIM_HZ` ms (≈16.7 at 60, ≈4.2 at 240).
/// * **Collision accuracy** — the ball moves a shorter distance per step, so less
///   penetration and less reliance on CCD.
/// * **Cost** — CPU scales linearly (2× the rate ⇒ 2× the `step()` calls). Each
///   step here is ~0.2 ms, so even 240 Hz is a few % of one core; a heavy sim is
///   where the trade-off would start to bite.
///
/// ## What makes this a *one-number* change
///
/// Nothing else is written in "ticks". Every force / threshold in
/// [`crate::physics`] is either a per-*second* physical quantity (velocities,
/// accelerations, gravity, damping — all `dt`-invariant) or is *derived* from
/// `SIM_HZ` (the roll-cue divider, the impact cooldown, the catch-up cap).
/// Change this number and the feel is identical — only the step granularity
/// (hence latency + cost) moves. **If you add a new per-step force, express it
/// per second and multiply by the step `dt`** — don't bake the rate into a
/// constant, or you'll reintroduce the very coupling this removes.
pub const SIM_HZ: f64 = 240.0;

/// The most balls a session can drop (clicks past this are ignored). Bounded
/// by the renderer's spatial index more than by Box3D — see BOX3D.md's rstar
/// note: ≥~300 concurrently-moving bodies can trip an upstream rstar panic.
pub const MAX_BALLS: usize = 200;

/// Held-movement bits packed into [`Input`]'s held bitset.
pub const HELD_FORWARD: u32 = 1 << 0;
pub const HELD_BACK: u32 = 1 << 1;
pub const HELD_LEFT: u32 = 1 << 2;
pub const HELD_RIGHT: u32 = 1 << 3;

/// The player's live input, held **on the worker** and polled by
/// [`crate::physics`] once per fixed step. The DOM handlers on the main thread
/// capture keystrokes and post [`InputMsg`]s; the worker's message handler
/// applies them here, so both the writer (message handler) and the reader
/// (physics step) are on the same (worker) thread — plain [`Cell`]s, no
/// synchronization needed.
///
/// The discrete actions (jump, fling) are edge-detected by a monotonic counter
/// the reader diffs against its last-seen value, so one press = exactly one
/// action even though physics *polls* rather than receiving an event.
#[derive(Default)]
pub struct Input {
    /// Bitset of currently-held movement keys (`HELD_*`).
    held: Cell<u32>,
    /// Bumped once per discrete jump press (edge-detected by physics).
    jump_seq: Cell<u32>,
    /// Bumped once per touch-swipe fling gesture (edge-detected by physics).
    fling_seq: Cell<u32>,
    /// Swipe release velocity (m/s) in the CAMERA frame — x right, −z away from
    /// the camera; physics rotates it by the camera yaw like the held keys.
    fling_x: Cell<f32>,
    fling_z: Cell<f32>,
}

impl Input {
    pub fn new() -> Self {
        Self::default()
    }

    /// Main side: set/clear one held-key bit.
    pub fn set_held(&self, mask: u32, down: bool) {
        let cur = self.held.get();
        self.held.set(if down { cur | mask } else { cur & !mask });
    }

    /// Physics side: the current held-key bitset.
    pub fn held(&self) -> u32 {
        self.held.get()
    }

    /// Main side: register a discrete jump press.
    pub fn bump_jump(&self) {
        self.jump_seq.set(self.jump_seq.get().wrapping_add(1));
    }

    /// Physics side: the jump counter (diff against your last-seen to edge-detect).
    pub fn jump_seq(&self) -> u32 {
        self.jump_seq.get()
    }

    /// Main side: publish a touch-swipe fling with the given CAMERA-frame
    /// velocity (m/s — x right, −z away from the camera).
    pub fn bump_fling(&self, vx: f32, vz: f32) {
        self.fling_x.set(vx);
        self.fling_z.set(vz);
        self.fling_seq.set(self.fling_seq.get().wrapping_add(1));
    }

    /// Physics side: the fling counter's current value (seed your last-seen).
    pub fn fling_seq(&self) -> u32 {
        self.fling_seq.get()
    }

    /// Physics side: consume a pending fling. Returns the camera-frame swipe
    /// velocity when `last_seq` is behind (and advances it to current).
    pub fn poll_fling(&self, last_seq: &mut u32) -> Option<(f32, f32)> {
        let seq = self.fling_seq.get();
        if seq == *last_seq {
            return None;
        }
        *last_seq = seq;
        Some((self.fling_x.get(), self.fling_z.get()))
    }
}

/// One scene-derived collider, ready to hand to the physics engine (Box3D). The
/// shape is in local space; `translation`/`rotation` place it in the world. Scale
/// is intentionally NOT carried — the collider has no scale (its placement is a
/// rotation + translation isometry); the fit lives entirely in the shape extents.
#[derive(Debug, Clone)]
pub struct ColliderInit {
    pub shape: ColliderShapeMsg,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    /// `true` for the one dynamic body (the ball); `false` for static geometry.
    pub dynamic: bool,
    /// Gameplay role, used to classify collision sounds — see `ROLE_*`.
    pub role: u8,
}

/// Collider shapes the physics side knows how to build (mirrors
/// `awsm_renderer_scene::ColliderShape`; ellipsoid is dropped — unused here).
#[derive(Debug, Clone)]
pub enum ColliderShapeMsg {
    Cuboid { half_extents: [f32; 3] },
    Ball { radius: f32 },
    Capsule { half_height: f32, radius: f32 },
    Cylinder { half_height: f32, radius: f32 },
    Cone { half_height: f32, radius: f32 },
}

/// Collider role tags, shared between the scene-derived collider builder and the
/// physics-side collision classifier.
pub const ROLE_FLOOR: u8 = 0; // the tabletop — ball landings thud here
pub const ROLE_WALL: u8 = 1; // the rails — ball knocks against these
pub const ROLE_BALL: u8 = 2; // the dynamic ball itself

/// Gameplay-driven audio cues (worker → main). Physics owns the ball's motion +
/// contacts, so it decides when sounds fire and how loud; the worker posts each
/// cue to main, which feeds it to the WebAudio player via
/// [`crate::audio::AudioController::on_audio`] (WebAudio is main-thread-only).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "audio", rename_all = "kebab-case")]
pub enum AudioMsg {
    /// Continuous rolling state: normalized speed (0..1) + ball world position.
    Roll { speed: f32, x: f32, y: f32, z: f32 },
    /// The ball struck a wall — `intensity` (0..1) from the impact speed.
    WallHit {
        x: f32,
        y: f32,
        z: f32,
        intensity: f32,
    },
    /// The ball landed on the table — `intensity` (0..1) from the drop speed.
    Land {
        x: f32,
        y: f32,
        z: f32,
        intensity: f32,
    },
    /// Two balls collided — `intensity` (0..1) from the approach speed. Voiced
    /// by the steel-sphere clack (`sfx_ball_clack`, a modal-synthesis worklet
    /// whose `intensity` param this drives); falls back to the wall knock when
    /// the loaded audio export predates the clack.
    BallClack {
        x: f32,
        y: f32,
        z: f32,
        intensity: f32,
    },
}

/// Radians of camera yaw per CSS pixel of horizontal drag. Lives here because
/// TWO integrators must agree on it without exchanging the angle: the worker's
/// `OrbitCamera` (the visual + physics W/A/S/D basis) and the main thread's
/// mirrored yaw (fed to the audio listener so the stereo image orbits with the
/// view). Both apply `yaw -= dx * CAMERA_ORBIT_SENSITIVITY` to the same drag
/// deltas from the same start, so they stay in lockstep — main updates audio
/// immediately while the worker updates the visual at its next frame.
pub const CAMERA_ORBIT_SENSITIVITY: f32 = 0.005;

// ── The postMessage protocol ─────────────────────────────────────────────────
// Each enum is `serde`-tagged with a distinct discriminator so a receiver can
// try-deserialize each variant it expects and ignore the rest. All cold paths.

/// Main → worker: a player-input event. The main thread captures the DOM event
/// and posts this; the worker applies it to its local [`Input`] block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "input", rename_all = "kebab-case")]
pub enum InputMsg {
    /// A held movement key went down/up (`HELD_*` mask).
    Held { mask: u32, down: bool },
    /// A discrete jump press (edge-triggered).
    Jump,
    /// A touch-swipe fling with a CAMERA-frame velocity (m/s).
    Fling { vx: f32, vz: f32 },
}

/// Main → worker: a camera gesture. The worker owns the `OrbitCamera`; main just
/// forwards the pointer deltas (and separately mirrors the yaw for audio).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cam", rename_all = "kebab-case")]
pub enum CameraMsg {
    /// A right-drag delta (CSS pixels): yaw from `dx`, pitch from `dy`.
    Orbit { dx: f32, dy: f32 },
    /// A wheel delta — dollies the camera in/out.
    Zoom { dy: f32 },
}

/// Main → worker: the user clicked the canvas at the given NDC point (x right,
/// y up, both −1..1). The worker (which owns the camera) unprojects it onto the
/// tabletop and drops a ball there.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "drop", rename_all = "kebab-case")]
pub enum DropMsg {
    Ball { ndc_x: f32, ndc_y: f32 },
}

/// Main → worker: the canvas backing size changed (main's `ResizeObserver` — only
/// main sees layout). The worker owns the transferred `OffscreenCanvas`, so it
/// applies the new **device-pixel** size there.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "resize", rename_all = "kebab-case")]
pub enum ResizeMsg {
    Canvas { width: u32, height: u32 },
}

/// Main → worker: a runtime anti-aliasing change (a Settings toggle). The worker
/// recompiles the affected renderer pipelines (`set_anti_aliasing`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "quality", rename_all = "kebab-case")]
pub enum QualityMsg {
    AntiAlias { msaa: bool, smaa: bool },
}

/// Worker → main: load progress + lifecycle + the GPU facts main needs to seed
/// the resolution scale and cap the backing store.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "msg", rename_all = "kebab-case")]
pub enum RenderMsg {
    /// A human-readable load-progress line for the loading screen.
    Progress { message: String },
    /// The first few frames have presented — the scene is on screen.
    Ready,
    /// GPU capability facts, posted once at startup, before the scene load.
    GpuInfo {
        is_fallback: bool,
        max_texture_dim: u32,
    },
    /// Something failed in the worker.
    Error { message: String },
}

/// Worker → main: sampled engine stats (~1 Hz) for the top-right panel. The
/// worker owns the per-frame data, so it computes the smoothed rates and posts
/// finished numbers; main only formats them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "stats", rename_all = "kebab-case")]
pub struct StatsMsg {
    /// Presented frames per second (vsync-capped).
    pub fps: f32,
    /// Avg render CPU ms per frame (workload metric, monitor-independent).
    pub frame_ms: f32,
    /// Avg physics CPU ms per fixed step.
    pub step_ms: f32,
    /// Fixed steps per second (locked to `SIM_HZ` when healthy).
    pub sps: f32,
    /// Balls dropped so far.
    pub balls: u32,
}
