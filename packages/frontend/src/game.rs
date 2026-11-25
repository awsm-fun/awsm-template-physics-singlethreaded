//! The game worker: one `requestAnimationFrame` loop that owns the renderer AND
//! the physics world, on a worker thread.
//!
//! Everything the game does — physics, rendering, camera, drop unprojection —
//! runs here, so the main thread stays free for DOM, input, and audio. The two
//! threads share nothing; input arrives as [`InputMsg`]/[`CameraMsg`]/… and
//! audio cues + the Stats panel's reports go back. Each frame:
//!
//! 0. take the camera pose main posted for its previous frame (exactly one
//!    main frame of orbit/zoom per drawn frame — see [`CameraQueue`]),
//! 1. drain click-drops (unproject onto the tabletop, spawn a ball),
//! 2. step the fixed-timestep [`World`] as many times as elapsed real time
//!    bought (the accumulator),
//! 3. advance the scene's clips + texture flows (`update_animations`) and its
//!    particles ([`ParticleDriver`](awsm_renderer_scene_loader::ParticleDriver))
//!    by the frame's time,
//! 4. **interpolate** each body's prev→curr pose by one alpha and write the
//!    resulting matrix into the renderer's shared-arena transform slot (after
//!    the clips, so physics wins on the bodies it drives),
//! 5. update the orbit camera + render,
//! 6. post the frame's audio cues (and, ~1 Hz, the Stats panel's report) to
//!    main.
//!
//! The Settings and the Stats panel are awsm-renderer-kit's: main posts the
//! kit's `anti-alias` / `resize` / `stats-view` / `stats-page` here, and this
//! worker applies the settings and runs the stats collector (see `stats`),
//! which times the frame's phases and the steps' phases only while their
//! groups are open.
//!
//! ## Pacing (why this shape, for smoothest motion)
//!
//! Physics and render share one thread, so the pacing is the classic
//! fixed-timestep accumulator + **render-time interpolation** (Gaffer's "Fix
//! Your Timestep"): accumulate real elapsed time, step a constant `dt` as many
//! times as it bought, then render the pose *interpolated between the last two
//! steps* by the leftover fraction. This is smoother than the two alternatives:
//! stepping a variable `dt` to "now" makes Box3D's solver non-deterministic, and
//! stepping fixed `dt` but rendering the raw final state lets the sub-step
//! remainder beat against the display clock (visible micro-stutter). Showing the
//! world ~one step in the past costs `~1000/SIM_HZ` ms of latency — ≈4 ms at
//! 240 Hz, imperceptible. ("Physics running as much as it can between frames"
//! only helps when render samples the sim across a *separate* thread through a
//! jitter buffer; with physics and render on one thread, the accumulator's
//! `while` loop already does exactly that within the frame.)

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use awsm_renderer::buffer::shared_arena::foreign_write;
use glam::{Mat4, Quat, Vec3};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;
use web_sys::OffscreenCanvas;

use awsm_renderer::transforms::TransformKey;
use awsm_renderer_settings::{
    apply_anti_aliasing, post_to_page, scene_anti_aliasing, AaMode, SettingsToPage,
    SettingsToWorker,
};
use awsm_renderer_stats::{
    post_report, stats_logging, FrameInput, RendererTimings, StatsCollector, StatsToWorker,
};

use crate::physics::{Pose, World};
use crate::protocol::{
    AudioMsg, CameraMsg, CameraPose, ColliderInit, ColliderShapeMsg, DropMsg, Input, InputMsg,
    RenderMsg, ROLE_FLOOR, SIM_HZ,
};
use crate::stats::{self, Phase, PhysicsStats, NOT_RENDERER};

/// Player input, held on the worker and fed by [`InputMsg`]s.
type SharedInput = Rc<Input>;
/// The one orbit camera (render + physics W/A/S/D basis), driven by [`CameraMsg`].
type SharedCamera = Rc<RefCell<OrbitCamera>>;
/// The camera poses main has posted, newest last (see [`CameraQueue`]).
type SharedCameraQueue = Rc<RefCell<CameraQueue>>;
/// Pending click-drops as NDC points, drained + unprojected by the loop.
type DropQueue = Rc<RefCell<Vec<(f32, f32)>>>;
/// A pending Settings anti-aliasing change, applied off the render path.
type PendingAa = Rc<Cell<Option<AaMode>>>;
/// The Stats panel's numbers (awsm-renderer-stats): the render loop feeds
/// it, main's `stats-view` / `stats-page` reach it through the message
/// handler.
type SharedStats = Rc<RefCell<StatsCollector>>;
/// The self-referencing rAF callback cell (so the loop can reschedule itself).
type RafCell = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// Base-color factor for the PLAYER ball (multiplies the albedo texture) — a
/// red tint so yours reads instantly against the silver click-dropped balls.
const PLAYER_TINT: [f32; 4] = [1.0, 0.25, 0.2, 1.0];

/// Fixed step in ms (drives the accumulator) — mirrors [`SIM_HZ`].
const FIXED_DT_MS: f64 = 1000.0 / SIM_HZ;
/// Clamp a single elapsed gap (e.g. a backgrounded tab) so we never try to
/// simulate minutes of backlog at once — the most real time one frame absorbs.
const MAX_FRAME_MS: f64 = 100.0;
/// Cap on fixed steps per frame — bounds catch-up so a hitch can't trigger a
/// spiral of death. Derived to cover a full [`MAX_FRAME_MS`] gap at the current
/// rate (≈6 at 60 Hz, ≈24 at 240 Hz), so it scales with [`SIM_HZ`].
const MAX_SUBSTEPS: u32 = (MAX_FRAME_MS / 1000.0 * SIM_HZ + 0.999) as u32;
/// Smoothing weight of each frame gap in the display-period estimate used to
/// pick the camera pose (see [`CameraQueue`]); only needs to be roughly right.
const PERIOD_EMA_ALPHA: f64 = 0.1;
/// The awsm-renderer release the vendored Basis transcoder files came with,
/// appended to their URLs as `?v=…`: those files keep their names from one
/// release to the next, so a browser that cached an older transcoder (wasm
/// included: the worker carries the query over to it) would otherwise pair
/// it with the new JS. Bump it together with awsm-renderer in Cargo.toml.
const BASIS_VERSION: &str = "0.46.0";

/// A drawn ball that moves farther than this in one frame (m) teleported
/// (physics' fall-through respawn): under TAA its motion history is reset,
/// so the respawned ball does not reproject the history of where it fell
/// from. A ball never moves this far in a frame on its own (its speed is
/// capped well below it).
const TELEPORT_JUMP_M: f32 = 1.5;

/// Lens + clip planes for the one camera this app has. Shared by the render
/// path (`set_camera`) and the click-unprojection so the two can't drift apart
/// — a mismatch here silently drops balls in the wrong place.
const CAMERA_FOV_Y_DEG: f32 = 55.0;
const CAMERA_NEAR: f32 = 0.1;
const CAMERA_FAR: f32 = 400.0;

/// A minimal orbit camera (after the renderer's `model-tests` `OrbitCamera`):
/// it circles a fixed `look_at` point at spherical `(yaw, pitch, radius)` (no
/// pan — the table can never leave the frame). The pose comes from main (see
/// [`CameraPose`] and [`CameraQueue`]); physics rotates W/A/S/D by its `yaw`.
pub struct OrbitCamera {
    pub look_at: Vec3,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl OrbitCamera {
    pub fn new(look_at: Vec3, pose: CameraPose) -> Self {
        Self {
            look_at,
            radius: pose.radius,
            yaw: pose.yaw,
            pitch: pose.pitch,
        }
    }

    /// Take main's pose (already clamped there).
    pub fn set_pose(&mut self, pose: CameraPose) {
        self.yaw = pose.yaw;
        self.pitch = pose.pitch;
        self.radius = pose.radius;
    }

    /// Camera world position from spherical coords (yaw 0 = looking from +Z).
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.look_at
            + Vec3::new(
                self.radius * cp * sy,
                self.radius * sp,
                self.radius * cp * cy,
            )
    }
}

/// The last few camera poses main posted, each stamped with the time of the
/// main frame it belongs to (absolute ms, see [`CameraMsg::Pose`]).
///
/// Each frame the worker draws the newest pose stamped at least half a
/// display period before its own frame time. Main and this worker share the
/// browser's frame clock, so that is main's PREVIOUS frame: posted a whole
/// frame ago and certainly here, however this frame's message races the
/// worker's callback. The newest (this frame's) pose is never used yet, which
/// is what makes it exactly one frame of input per drawn frame.
#[derive(Default)]
pub struct CameraQueue {
    poses: std::collections::VecDeque<(f64, CameraPose)>,
}

impl CameraQueue {
    /// Poses kept; only the newest two or three can ever be chosen.
    const KEEP: usize = 4;

    fn push(&mut self, t: f64, pose: CameraPose) {
        if self.poses.len() == Self::KEEP {
            self.poses.pop_front();
        }
        self.poses.push_back((t, pose));
    }

    /// The newest pose stamped no later than `cutoff` (absolute ms), if any.
    fn at(&self, cutoff: f64) -> Option<CameraPose> {
        self.poses
            .iter()
            .rev()
            .find(|(t, _)| *t <= cutoff)
            .map(|(_, pose)| *pose)
    }
}

/// Worker entry (called by the bootstrap after init): unpack the transferred
/// `OffscreenCanvas` + page origin + desired AA, build the WebGPU device, and
/// kick off the async load + render loop.
pub fn start(payload: JsValue) -> Result<(), JsValue> {
    use awsm_renderer::core::renderer::{AwsmRendererWebGpuBuilder, DeviceRequestLimits};
    use awsm_renderer::web_global::navigator_gpu;
    use awsm_renderer_scene_loader::basis::{configure, BasisWorkerConfig};

    let canvas: OffscreenCanvas =
        js_sys::Reflect::get(&payload, &JsValue::from_str("canvas"))?.unchecked_into();
    // The worker has a `blob:` base URL, so relative fetches can't resolve —
    // main passes the page origin so we can build the absolute scene.toml URL.
    let origin = js_sys::Reflect::get(&payload, &JsValue::from_str("origin"))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    // Provide the Basis codec URLs (the crate hardcodes none). We run the player
    // load path ON THIS render worker, whose `blob:` base can't resolve a
    // relative URL — so the URLs must be ABSOLUTE. `app_base` is the directory
    // the app is served from (where Trunk copy-file serves workers/basis-worker.js
    // + vendor/basis/…). It keeps the deploy PATH, not just the origin, so this
    // works under a subpath deploy (e.g. /experiments/<slug>/); a bare origin
    // would 404 there. It is also NOT `origin` above (in dev that's the separate
    // live-media server, which does not serve the codec).
    {
        let app_base = js_sys::Reflect::get(&payload, &JsValue::from_str("app_base"))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_default();
        let base = app_base.trim_end_matches('/');
        configure(BasisWorkerConfig::player(
            format!("{base}/workers/basis-worker.js?v={BASIS_VERSION}"),
            format!("{base}/vendor/basis/basis_transcoder.js?v={BASIS_VERSION}"),
        ));
    }
    // The STARTUP anti-aliasing mode (main's Settings: the stored choice,
    // `?aa=`, or the device default), as its id, so the renderer BUILDS with
    // it — compiling exactly the variants this session needs.
    let aa_id = js_sys::Reflect::get(&payload, &JsValue::from_str("aa"))
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    let desired_aa = AaMode::from_id(&aa_id)
        .ok_or_else(|| JsValue::from_str(&format!("game worker: unknown aa mode {aa_id:?}")))?;

    let gpu = match navigator_gpu() {
        Some(gpu) => gpu,
        None => {
            let msg = "no navigator.gpu — this browser has no WebGPU";
            tracing::error!("{msg}");
            post_to_main(&RenderMsg::Error {
                message: msg.to_string(),
            });
            return Ok(());
        }
    };
    let gpu_builder = AwsmRendererWebGpuBuilder::new_with_offscreen_canvas(gpu, canvas.clone())
        .with_device_request_limits(DeviceRequestLimits::max_all());

    wasm_bindgen_futures::spawn_local(async move {
        if let Err(err) = run(gpu_builder, canvas, origin, desired_aa).await {
            tracing::error!("game worker: {err:?}");
            post_to_main(&RenderMsg::Error {
                message: format!("{err:?}"),
            });
        }
    });
    Ok(())
}

async fn run(
    gpu_builder: awsm_renderer::core::renderer::AwsmRendererWebGpuBuilder,
    canvas: OffscreenCanvas,
    origin: String,
    desired_aa: AaMode,
) -> Result<(), JsValue> {
    use awsm_renderer::camera::CameraParams;
    use awsm_renderer::AwsmRendererBuilder;

    // ── Local state, created BEFORE the slow load so the message handler is
    // live immediately (messages received with no listener are dropped, not
    // queued — main starts posting resize/input as soon as it has spawned us) ─
    let input: SharedInput = Rc::new(Input::new());
    // Initial framing: looking just above the table, from main's start pose.
    #[allow(clippy::arc_with_non_send_sync)]
    let camera: SharedCamera = Rc::new(RefCell::new(OrbitCamera::new(
        Vec3::new(0.0, 0.4, 0.0),
        CameraPose::START,
    )));
    #[allow(clippy::arc_with_non_send_sync)]
    let camera_queue: SharedCameraQueue = Rc::new(RefCell::new(CameraQueue::default()));
    #[allow(clippy::arc_with_non_send_sync)]
    let drops: DropQueue = Rc::new(RefCell::new(Vec::new()));
    // Runtime Settings AA changes only — the STARTUP config goes straight into
    // the renderer build below (`with_anti_aliasing`), so boot compiles exactly
    // the variants the session needs and never recompiles. (This used to seed a
    // boot recompile whenever the stored pref differed from the build default —
    // a long, silent, avoidable stall on every load for e.g. every touch
    // device, whose default is MSAA off.)
    let pending_aa: PendingAa = Rc::new(Cell::new(None));
    let stats: SharedStats = Rc::new(RefCell::new(StatsCollector::new(stats::config())));
    install_message_handler(
        canvas.clone(),
        input.clone(),
        camera_queue.clone(),
        drops.clone(),
        pending_aa.clone(),
        stats.clone(),
    );
    // The panel's "starting…" until the scene is on screen.
    post_report(stats.borrow().placeholder());

    // The builder reports its coarse boot phases through this handler. Since the
    // deferred-boot renderer, `build()` compiles NO pipelines (it reserves them,
    // compiled by the labeled `ensure_config_pipelines` step below), so only
    // `Init` — device + core GPU resources — takes visible time here.
    let mut renderer = AwsmRendererBuilder::new(gpu_builder)
        .with_anti_aliasing(desired_aa.to_anti_aliasing())
        // GPU timing for the Stats panel's GPU group (the CPU tier is `Off`
        // except while the renderer group is open: the stats collector's
        // `begin_renderer_frame` switches it): `Frame` tier = timestamp
        // queries on the renderer's passes, folded into one whole-frame span
        // (cheap; silently off without `timestamp-query`).
        .with_logging(stats_logging())
        .with_phase_handler(|phase| {
            use awsm_renderer::RendererLoadingPhase as P;
            if let P::Init = phase {
                post_progress("renderer init: GPU device + core resources…");
            }
        })
        .build()
        .await
        .map_err(|e| JsValue::from_str(&format!("build renderer: {e}")))?;
    post_progress(&format!(
        "WebGPU device + renderer ready (anti-aliasing: {})",
        AaMode::from_anti_aliasing(&renderer.anti_aliasing).label()
    ));

    // Shared-arena mode: every scene node gets a stable arena slot we can write
    // an absolute world matrix into each frame. (Not a threading feature — just
    // the renderer's raw-matrix write path; here it's all one thread anyway.)
    renderer.transforms.enable_shared_arena();

    // ── Warm-up + scene fetch, CONCURRENTLY ─────────────────────────────────
    // `build()` compiled NO pipeline — it only reserved them. We know our config
    // here (AA is baked into the builder), so warm exactly that set now via
    // `ensure_config_pipelines` (only what's needed — not an eager compile of
    // everything). That warm-up is GPU/driver work; the `scene.toml` fetch is
    // network work; the two share no data, so we run them under one `join!` and
    // let the executor interleave them — the compile hides the fetch latency
    // instead of following it. First visit compiles; warm visits are a no-op.
    let bundle_base = format!("{}/bundle", origin.trim_end_matches('/'));
    let scene_url = format!("{bundle_base}/scene.toml");
    post_progress("compiling core render pipelines + fetching scene… (first visit can take a while — cached after)");
    let (compiled, scene) =
        futures::join!(renderer.ensure_config_pipelines(), fetch_scene(&scene_url));
    let compiled =
        compiled.map_err(|e| JsValue::from_str(&format!("ensure_config_pipelines: {e}")))?;
    let mut scene =
        scene.map_err(|e| JsValue::from_str(&format!("load scene {scene_url}: {e}")))?;
    // The loader applies the scene's authored anti-aliasing. This app chooses
    // AA itself (the Settings prefs the renderer was built with), so write that
    // choice into the scene: the loader then keeps it and compiles nothing
    // twice.
    scene.post_process.anti_aliasing = scene_anti_aliasing(&renderer.anti_aliasing);
    tracing::info!(
        "game worker: loaded scene {scene_url} ({} nodes)",
        scene.nodes.len()
    );
    post_progress(&format!(
        "core pipelines ready ({compiled}) · scene parsed ({} nodes)",
        scene.nodes.len()
    ));

    // The renderer's shared-arena mode is FLAT — each slot is an absolute world
    // matrix, no parent→child propagation — so physics must drive the node that
    // carries the MESH, not the parent "Ball" group. Find that mesh node.
    let ball_group = scene
        .nodes
        .iter()
        .find(|n| n.name == "Ball")
        .ok_or_else(|| JsValue::from_str("scene has no node named 'Ball'"))?;
    let ball_node_id = find_mesh_node(ball_group)
        .ok_or_else(|| JsValue::from_str("'Ball' has no renderable mesh node to drive"))?;

    // Pull the physics world straight out of the scene's collider nodes.
    let (colliders, spawn, ball_visual_scale) = derive_physics(&scene, ball_node_id);
    tracing::info!(
        "game worker: derived {} colliders (ball scale {ball_visual_scale}, spawn {spawn:?})",
        colliders.len()
    );
    let ball_radius = colliders
        .iter()
        .find(|c| c.dynamic)
        .and_then(|c| match c.shape {
            ColliderShapeMsg::Ball { radius } => Some(radius),
            _ => None,
        })
        .unwrap_or(0.5);
    let floor = floor_box(&colliders);

    // Build the Box3D world now (before the slow GPU commit) so a bad scene
    // fails fast.
    let mut world = World::new(&colliders, spawn)?;

    // Materialize the scene (assets fetched from our origin next to scene.toml).
    let assets = awsm_renderer_scene_loader::assets::HttpAssets::new(bundle_base.clone());
    // `LoadPhase::label()` is the loader's granular, human-readable progress line
    // ("Fetching textures 3/9…", "Compiling pipelines (4)…"), deduped — several
    // phases re-report the same counts back-to-back, which reads as spam.
    let mut last_phase_line = String::new();
    let loaded = awsm_renderer_scene_loader::load_scene_for_player(
        &mut renderer,
        &scene,
        &assets,
        |phase| {
            let line = phase.label();
            // "0/0" phases are commit bookkeeping over an empty registry — noise.
            if line != last_phase_line && !line.contains("0/0") {
                post_progress(&line);
                last_phase_line = line;
            }
        },
    )
    .await
    .map_err(|e| JsValue::from_str(&format!("load_scene_for_player: {e}")))?;
    // The loader builds the scene's particle emitters; the host ticks them
    // each frame (as it advances the clips), in the render loop below.
    let mut particles = awsm_renderer_scene_loader::ParticleDriver::new(&loaded);

    // Relay GPU-commit progress, deduped (the callback fires per resolution).
    // Normally a cheap no-op — the loader already ran the real commit.
    let mut last_commit_line = String::new();
    renderer
        .commit_load(|s| {
            let Some(line) = s.phase_label() else {
                return;
            };
            if line != last_commit_line && !line.contains("0/0") {
                post_progress(&line);
                last_commit_line = line;
            }
        })
        .await
        .map_err(|e| JsValue::from_str(&format!("commit_load: {e}")))?;
    renderer.update_transforms();
    post_progress("gpu commit complete");

    // ── The PLAYER ball is always distinct ───────────────────────────────────
    // Clicking mints duplicates of this mesh, and a duplicate copies the source
    // mesh's material key — so swap in the player's look up front, leaving the
    // original material as the duplicates' source of truth (`ball_material`).
    let ball_mesh = loaded
        .nodes
        .get(&ball_node_id)
        .and_then(|h| h.meshes.first().copied())
        .ok_or_else(|| JsValue::from_str("ball node has no mesh"))?;
    let ball_material = renderer
        .meshes
        .get(ball_mesh)
        .map_err(|e| JsValue::from_str(&format!("ball mesh lookup: {e}")))?
        .material_key;
    // Preferred: the scene ships the player look as a MATERIAL VARIANT on the
    // ball mesh (the editor's "Material variants" — e.g. the red billiard skin).
    // Fallback: tint the base material's color factor red.
    let variant_material = loaded
        .node_material_variants
        .get(&ball_node_id)
        .and_then(|vs| vs.iter().find(|v| v.name == "Ball_Player_Red"))
        .map(|v| v.key);
    let player_key = match variant_material {
        Some(key) => {
            tracing::info!("player ball: using the authored 'Ball_Player_Red' variant");
            key
        }
        None => {
            tracing::info!("player ball: no 'Ball_Player_Red' variant — tinting the base material");
            let mut tinted = renderer
                .materials
                .get(ball_material)
                .map_err(|e| JsValue::from_str(&format!("ball material lookup: {e}")))?
                .clone();
            if let awsm_renderer::materials::Material::Pbr(pbr) = &mut tinted {
                pbr.base_color_factor = PLAYER_TINT;
            }
            renderer.materials.insert(
                tinted,
                &renderer.textures,
                &renderer.dynamic_materials,
                &renderer.extras_pool,
            )
        }
    };
    renderer
        .set_mesh_material(ball_mesh, player_key)
        .map_err(|e| JsValue::from_str(&format!("set player material: {e}")))?;

    // ── Arena binding for the player ball's transform slot ──────────────────
    let ball_tk = loaded
        .nodes
        .get(&ball_node_id)
        .map(|h| h.transform)
        .ok_or_else(|| JsValue::from_str("ball node produced no transform"))?;
    let dirty_words_addr = renderer
        .transforms
        .arena_dirty_words_addr()
        .ok_or_else(|| JsValue::from_str("shared arena not enabled"))?;
    let binding = renderer
        .transforms
        .arena_slot_binding(ball_tk)
        .ok_or_else(|| JsValue::from_str("ball slot has no arena binding"))?;

    tracing::info!("game worker: scene ready, starting render loop");
    post_progress("starting render loop…");

    // ── The render loop ─────────────────────────────────────────────────────
    let perf = worker_performance()?;

    #[allow(clippy::arc_with_non_send_sync)]
    let cell = Rc::new(RefCell::new(Some(renderer)));
    let reconfiguring = Rc::new(Cell::new(false));
    #[allow(clippy::arc_with_non_send_sync)]
    let raf: RafCell = Rc::new(RefCell::new(None));
    let raf_init = raf.clone();
    let raf_run = raf.clone();

    // Per-frame carried state (owned by the FnMut closure).
    let mut accumulator: f64 = 0.0;
    let mut last_vsync: Option<f64> = None;
    // For choosing the camera pose (see `CameraQueue`): this worker's time
    // origin, to put its frame times on main's absolute clock, and a smoothed
    // display period.
    let time_origin = perf.time_origin();
    let mut period_ms: f64 = 1000.0 / 60.0;
    let mut frame_count: u32 = 0;
    let mut cues: Vec<AudioMsg> = Vec::new();
    // One arena binding per minted (dropped-ball) visual duplicate, with its
    // transform (for a teleport's motion-history reset).
    let mut ball_bindings: Vec<(
        awsm_renderer::buffer::shared_arena::SlotBinding,
        TransformKey,
    )> = Vec::new();
    // Where each ball was drawn last frame (the player first, then the
    // dropped balls), to spot a teleport.
    let mut drawn_at: Vec<Vec3> = Vec::new();
    // The Stats panel's physics group (the collector itself is `stats`).
    let mut physics_stats = PhysicsStats::new(perf.clone());

    *raf_init.borrow_mut() = Some(Closure::new(move |vsync_ms: f64| {
        // A pending Settings anti-aliasing change: apply it off the render path
        // (async recompile). Take the renderer OUT of the cell for the awaits;
        // `reconfiguring` keeps frames off it until it's put back.
        if !reconfiguring.get() {
            if let Some(mode) = pending_aa.take() {
                reconfiguring.set(true);
                let cell = cell.clone();
                let done = reconfiguring.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let Some(mut r) = cell.borrow_mut().take() else {
                        done.set(false);
                        return;
                    };
                    // awsm-renderer-settings posts main the `aa-compile-*`
                    // messages that raise and lower the "compiling pipelines"
                    // overlay (the first switch in each direction really
                    // compiles; later ones hit the variant cache, and
                    // switching between TAA presets compiles nothing and shows
                    // no overlay). `aa-compile-done` is always posted, even on
                    // failure.
                    let applied = apply_anti_aliasing(&mut r, mode, |event| {
                        post_to_page(&SettingsToPage::from(event));
                    })
                    .await;
                    match applied {
                        Ok(()) => tracing::info!(
                            "game worker: anti-aliasing applied: {}",
                            AaMode::from_anti_aliasing(&r.anti_aliasing).label()
                        ),
                        Err(message) => {
                            tracing::error!("game worker: {message}");
                            post_to_main(&RenderMsg::Error { message });
                        }
                    }
                    *cell.borrow_mut() = Some(r);
                    done.set(false);
                });
            }
        }
        if reconfiguring.get() {
            reschedule(&raf_run);
            return;
        }
        let mut cell_ref = cell.borrow_mut();
        let Some(r) = cell_ref.as_mut() else {
            reschedule(&raf_run);
            return;
        };
        let work_t0 = perf.now();
        let mut stats_ref = stats.borrow_mut();
        let stats = &mut *stats_ref;
        // Phase timing for the Stats panel, only while it shows the
        // renderer's breakdown (awsm-renderer-stats).
        stats.begin_renderer_frame(r, work_t0);
        let timing = stats.renderer_detail();
        let lap = |stats: &mut StatsCollector, phase: usize| {
            if timing {
                stats.lap(phase, perf.now());
            }
        };

        // 0. ── Camera: main's pose from its previous frame (see CameraQueue) ─
        if let Some(prev) = last_vsync {
            let gap = vsync_ms - prev;
            if gap > 0.0 && gap < MAX_FRAME_MS {
                period_ms += (gap - period_ms) * PERIOD_EMA_ALPHA;
            }
        }
        let cutoff = time_origin + vsync_ms - 0.5 * period_ms;
        if let Some(pose) = camera_queue.borrow().at(cutoff) {
            camera.borrow_mut().set_pose(pose);
        }

        // 1. ── Drain click-drops: unproject onto the tabletop, spawn a ball ──
        if let Some((floor_he, floor_tr)) = floor {
            let pending: Vec<(f32, f32)> = drops.borrow_mut().drain(..).collect();
            for (ndc_x, ndc_y) in pending {
                if let Some((x, z)) = unproject_to_table(
                    &camera.borrow(),
                    &canvas,
                    floor_he,
                    floor_tr,
                    ball_radius,
                    ndc_x,
                    ndc_y,
                ) {
                    world.drop_ball(x, z);
                }
            }
        }

        // 2. ── Fixed-timestep physics catch-up ──────────────────────────────
        let frame_ms = match last_vsync {
            Some(prev) => (vsync_ms - prev).clamp(0.0, MAX_FRAME_MS),
            None => 0.0,
        };
        last_vsync = Some(vsync_ms);
        accumulator += frame_ms;
        let yaw = camera.borrow().yaw;
        let mut substeps: u32 = 0;
        lap(stats, Phase::ReadApp as usize);
        let step_t0 = perf.now();
        let mut step_timer = physics_stats.step_timer(stats);
        while accumulator >= FIXED_DT_MS && substeps < MAX_SUBSTEPS {
            world.step(&input, yaw, &mut cues, step_timer.as_deref_mut());
            accumulator -= FIXED_DT_MS;
            substeps += 1;
        }
        if substeps == MAX_SUBSTEPS {
            accumulator = 0.0; // hit the cap — drop the backlog (avoid a spiral)
        }
        // The steps are physics time, not the renderer's: no phase gets it.
        lap(stats, NOT_RENDERER);
        let step_ms = perf.now() - step_t0;

        // 3. ── Audio cues → main ─────────────────────────────────────────────
        for cue in cues.drain(..) {
            post_audio(&cue);
        }
        lap(stats, Phase::SoundCues as usize);

        // 4. ── Scene clips, texture flows, particles ────────────────────────
        // Advanced by the same frame time as physics. Before the body writes
        // below, so physics has the last word on the slots it drives. (Don't
        // animate a body or its parents, though: `update_transforms` rewrites
        // the slots of a subtree a clip moved, over the physics pose.)
        let _ = r.update_animations(frame_ms);
        particles.tick(r, (frame_ms / 1000.0) as f32);
        lap(stats, Phase::Animate as usize);

        // 5. ── Interpolate poses → write transforms ─────────────────────────
        // After the catch-up loop `accumulator` is the leftover fraction of a
        // step; alpha blends prev→curr (classic fixed-timestep interpolation).
        let alpha = (accumulator / FIXED_DT_MS) as f32;
        // A teleport (see `TELEPORT_JUMP_M`): under TAA the ball's motion
        // history goes, in the frame the jump is drawn.
        let teleported = |drawn_at: &mut Vec<Vec3>, index: usize, pos: Vec3| {
            if drawn_at.len() <= index {
                drawn_at.push(pos);
                return false;
            }
            let jumped = drawn_at[index].distance(pos) > TELEPORT_JUMP_M;
            drawn_at[index] = pos;
            jumped
        };
        let (pp, pc) = world.player_poses();
        let pose = pose_lerp(pp, pc, alpha);
        if teleported(&mut drawn_at, 0, pose.0) {
            r.reset_motion_history(ball_tk);
        }
        write_slot(binding, dirty_words_addr, ball_visual_scale, pose);

        // Dropped balls: mint the silver visual duplicate for any ball we haven't
        // seen yet, then interpolate every slot with the same alpha.
        let ball_count = world.ball_count();
        while ball_bindings.len() < ball_count {
            let (_, curr) = world.ball_poses(ball_bindings.len());
            let tk = r.transforms.insert(
                awsm_renderer::transforms::Transform {
                    translation: Vec3::from_array(curr.0),
                    rotation: Quat::from_array(curr.1),
                    scale: Vec3::splat(ball_visual_scale),
                },
                None,
            );
            match r.duplicate_mesh_with_transform(ball_mesh, tk) {
                Ok(dup) => {
                    // The duplicate copies the source's CURRENT material (the
                    // player's red tint). Silver it back.
                    if let Err(err) = r.set_mesh_material(dup, ball_material) {
                        tracing::warn!("dropped ball material reset failed: {err}");
                    }
                    match r.transforms.arena_slot_binding(tk) {
                        Some(b) => ball_bindings.push((b, tk)),
                        None => {
                            tracing::error!("dropped ball has no arena slot");
                            break;
                        }
                    }
                }
                Err(err) => {
                    tracing::error!("dropped ball duplicate failed: {err}");
                    break;
                }
            }
        }
        for (i, (b, tk)) in ball_bindings.iter().enumerate() {
            let (bp, bc) = world.ball_poses(i);
            let pose = pose_lerp(bp, bc, alpha);
            if teleported(&mut drawn_at, i + 1, pose.0) {
                r.reset_motion_history(*tk);
            }
            write_slot(*b, dirty_words_addr, ball_visual_scale, pose);
        }
        lap(stats, Phase::ReadApp as usize);

        // 6. ── Camera + render ──────────────────────────────────────────────
        let cam = camera.borrow();
        // ONE camera API: a view matrix plus projection params. The renderer
        // supplies the depth convention (reverse-Z by default) and the live
        // surface aspect itself, so neither can drift from what it actually
        // renders with — hand-rolling a projection here would invert every
        // depth test. Focus on the orbit pivot so DoF tracks the dolly;
        // aperture stays at the f/5.6 default.
        let view = Mat4::look_at_rh(cam.eye(), cam.look_at, Vec3::Y);
        let mut params =
            CameraParams::perspective(CAMERA_FOV_Y_DEG.to_radians(), CAMERA_NEAR, CAMERA_FAR);
        params.focus_distance = cam.radius;
        let _ = r.set_camera(view, params);
        drop(cam);
        lap(stats, Phase::SetCamera as usize);
        r.update_transforms();
        lap(stats, Phase::UpdateTransforms as usize);
        if let Err(err) = r.render(None) {
            tracing::warn!("game worker: render error: {err}");
        }
        lap(stats, Phase::Render as usize);

        // 7. ── Stats + ready signal ─────────────────────────────────────────
        // The renderer's time is the frame minus the physics steps; the
        // physics group carries the whole worker's lane of the frame budget.
        let worker_ms = perf.now() - work_t0;
        let input = FrameInput::new(vsync_ms, worker_ms - step_ms);
        let window = stats.end_frame(input, &RendererTimings);
        physics_stats.frame(substeps, step_ms, worker_ms);
        if let Some(window) = window {
            let section = physics_stats.section(stats, &window, ball_count);
            post_report(stats.report(&window, vec![section]));
        }
        frame_count = frame_count.wrapping_add(1);
        if frame_count == 3 {
            post_to_main(&RenderMsg::Ready);
            // The scene is on screen: the stats warm up from here.
            post_report(stats.mark_ready());
            physics_stats.ready();
        }
        drop(stats_ref);
        drop(cell_ref);
        reschedule(&raf_run);
    }));
    if let Some(cb) = raf_init.borrow().as_ref() {
        awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref())?;
    }
    // Keep the RAF closure alive for the session.
    std::mem::forget(raf);
    Ok(())
}

/// Install this worker's `onmessage`, replacing the bootstrap's init handler
/// (which has done its job). Applies main's messages — the kit's Settings and
/// stats ones, then input/camera/drop — to the shared local state the render
/// loop reads.
fn install_message_handler(
    canvas: OffscreenCanvas,
    input: SharedInput,
    camera_queue: SharedCameraQueue,
    drops: DropQueue,
    pending_aa: PendingAa,
    stats: SharedStats,
) {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    let cb = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
        let data = e.data();
        // The kit's messages first: Settings' `resize` and `anti-alias`, the
        // stats panel's `stats-view` and `stats-page`.
        match SettingsToWorker::from_js(&data) {
            // Canvas resize (the Settings panel follows the canvas's layout
            // size and the resolution slider): the worker owns the
            // OffscreenCanvas, so it applies the new device-pixel size.
            Some(Ok(SettingsToWorker::Resize { width, height })) => {
                if width > 0 && height > 0 {
                    canvas.set_width(width);
                    canvas.set_height(height);
                }
                return;
            }
            // An anti-aliasing change: the render loop applies it (a newer
            // one replaces one not applied yet).
            Some(Ok(SettingsToWorker::AntiAlias { mode })) => {
                pending_aa.set(Some(mode));
                return;
            }
            Some(Err(err)) => {
                post_to_main(&RenderMsg::Error {
                    message: format!("a settings message did not parse: {err}"),
                });
                return;
            }
            None => {}
        }
        match StatsToWorker::from_js(&data) {
            Some(Ok(message)) => {
                // The render loop holds the collector only during a frame,
                // and a message cannot arrive then (one thread).
                stats.borrow_mut().handle(message);
                return;
            }
            Some(Err(err)) => {
                post_to_main(&RenderMsg::Error {
                    message: format!("a stats message did not parse: {err}"),
                });
                return;
            }
            None => {}
        }
        if let Ok(msg) = serde_wasm_bindgen::from_value::<InputMsg>(data.clone()) {
            match msg {
                InputMsg::Held { mask, down } => input.set_held(mask, down),
                InputMsg::Jump => input.bump_jump(),
                InputMsg::Fling { vx, vz } => input.bump_fling(vx, vz),
            }
            return;
        }
        if let Ok(CameraMsg::Pose { t, pose }) =
            serde_wasm_bindgen::from_value::<CameraMsg>(data.clone())
        {
            camera_queue.borrow_mut().push(t, pose);
            return;
        }
        if let Ok(DropMsg::Ball { ndc_x, ndc_y }) = serde_wasm_bindgen::from_value::<DropMsg>(data)
        {
            drops.borrow_mut().push((ndc_x, ndc_y));
        }
    });
    scope.set_onmessage(Some(cb.as_ref().unchecked_ref()));
    cb.forget();
}

/// Reschedule the render loop for the next frame.
fn reschedule(raf: &RafCell) {
    if let Some(cb) = raf.borrow().as_ref() {
        let _ = awsm_renderer::web_global::request_animation_frame(cb.as_ref().unchecked_ref());
    }
}

/// Lerp/slerp a `(prev, curr)` pose pair by `alpha`.
fn pose_lerp(prev: Pose, curr: Pose, alpha: f32) -> (Vec3, Quat) {
    (
        Vec3::from_array(prev.0).lerp(Vec3::from_array(curr.0), alpha),
        Quat::from_array(prev.1).slerp(Quat::from_array(curr.1), alpha),
    )
}

/// Write an interpolated pose into a shared-arena transform slot as an absolute
/// world matrix (the ball mesh's authored scale baked in — the slot holds the
/// absolute world matrix, and this is that slot's sole writer).
fn write_slot(
    binding: awsm_renderer::buffer::shared_arena::SlotBinding,
    dirty_words_addr: usize,
    scale: f32,
    (pos, rot): (Vec3, Quat),
) {
    let mat = Mat4::from_scale_rotation_translation(Vec3::splat(scale), rot, pos);
    let cols = mat.to_cols_array();
    // SAFETY: `binding` + `dirty_words_addr` address a live 64-byte arena slot;
    // `cols` is exactly 16 f32 = 64 bytes.
    let bytes = unsafe { std::slice::from_raw_parts(cols.as_ptr() as *const u8, 64) };
    unsafe {
        foreign_write(binding, dirty_words_addr, bytes);
    }
}

/// Unproject an NDC click through the current camera onto the tabletop plane,
/// clamped inside the rails so the ball always lands on the felt. `None` for a
/// grazing ray or a click on the sky.
fn unproject_to_table(
    cam: &OrbitCamera,
    canvas: &OffscreenCanvas,
    floor_he: [f32; 3],
    floor_tr: [f32; 3],
    ball_radius: f32,
    ndc_x: f32,
    ndc_y: f32,
) -> Option<(f32, f32)> {
    let view = Mat4::look_at_rh(cam.eye(), cam.look_at, Vec3::Y);
    let projection = Mat4::perspective_rh(
        CAMERA_FOV_Y_DEG.to_radians(),
        aspect(canvas),
        CAMERA_NEAR,
        CAMERA_FAR,
    );
    let inv = (projection * view).inverse();
    let p0 = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
    let p1 = inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
    let dir = p1 - p0;
    let table_top = floor_tr[1] + floor_he[1];
    if dir.y.abs() < 1e-6 {
        return None; // grazing ray
    }
    let t = (table_top - p0.y) / dir.y;
    if t <= 0.0 {
        return None; // clicked the sky
    }
    let margin = ball_radius + 0.25;
    let x = (p0.x + t * dir.x).clamp(
        floor_tr[0] - floor_he[0] + margin,
        floor_tr[0] + floor_he[0] - margin,
    );
    let z = (p0.z + t * dir.z).clamp(
        floor_tr[2] - floor_he[2] + margin,
        floor_tr[2] + floor_he[2] - margin,
    );
    Some((x, z))
}

/// Fetch + deserialize a same-origin player-bundle `scene.toml`.
async fn fetch_scene(url: &str) -> Result<awsm_renderer_scene::Scene, String> {
    // `no-cache` = revalidate: a runtime fetch isn't busted by a page refresh,
    // so a re-exported scene.toml would otherwise keep serving stale.
    let text = gloo_net::http::Request::get(url)
        .cache(web_sys::RequestCache::NoCache)
        .send()
        .await
        .map_err(|e| format!("fetch: {e}"))?
        .text()
        .await
        .map_err(|e| format!("read: {e}"))?;
    awsm_renderer_scene::project_dir::scene_from_toml(&text).map_err(|e| format!("parse: {e}"))
}

/// Depth-first search for the first node carrying renderable mesh geometry in
/// `node`'s subtree (including `node` itself). The exported "Ball" is a group;
/// its geometry sits on a child mesh node, whose flat arena slot must be driven.
fn find_mesh_node(node: &awsm_renderer_scene::EditorNode) -> Option<awsm_renderer_scene::NodeId> {
    use awsm_renderer_scene::NodeKind;
    if matches!(
        node.kind,
        NodeKind::Mesh { .. } | NodeKind::SkinnedMesh { .. } | NodeKind::ClusterMesh { .. }
    ) {
        return Some(node.id);
    }
    node.children.iter().find_map(find_mesh_node)
}

/// Walk the scene's node tree and pull out everything physics needs straight
/// from the authored `Collider` nodes: each collider in **world** space (table +
/// walls static, the ball dynamic), the ball's spawn point, and the ball mesh's
/// world scale (baked into the flat arena slot physics writes).
fn derive_physics(
    scene: &awsm_renderer_scene::Scene,
    ball_mesh_id: awsm_renderer_scene::NodeId,
) -> (Vec<ColliderInit>, [f32; 3], f32) {
    let mut colliders = Vec::new();
    let mut spawn = [0.0, 0.6, 0.0];
    let mut ball_scale = 1.0_f32;
    for node in &scene.nodes {
        let in_ball = node.name == "Ball";
        walk_node(
            node,
            Mat4::IDENTITY,
            in_ball,
            ball_mesh_id,
            &mut colliders,
            &mut spawn,
            &mut ball_scale,
        );
    }
    (colliders, spawn, ball_scale)
}

#[allow(clippy::too_many_arguments)]
fn walk_node(
    node: &awsm_renderer_scene::EditorNode,
    parent_world: Mat4,
    in_ball: bool,
    ball_mesh_id: awsm_renderer_scene::NodeId,
    colliders: &mut Vec<ColliderInit>,
    spawn: &mut [f32; 3],
    ball_scale: &mut f32,
) {
    use awsm_renderer_scene::NodeKind;
    let t = &node.transform;
    let local = Mat4::from_scale_rotation_translation(
        Vec3::from_array(t.scale),
        Quat::from_array(t.rotation), // [x, y, z, w] — matches glam
        Vec3::from_array(t.translation),
    );
    let world = parent_world * local;

    if node.id == ball_mesh_id {
        // Uniform in this scene; take X as the representative scale.
        *ball_scale = world.to_scale_rotation_translation().0.x;
    }

    if let NodeKind::Collider(shape) = &node.kind {
        let (scale, rot, tr) = world.to_scale_rotation_translation();
        let role = if in_ball {
            crate::protocol::ROLE_BALL
        } else if node.name.contains("Wall") {
            crate::protocol::ROLE_WALL
        } else {
            crate::protocol::ROLE_FLOOR
        };
        if in_ball {
            *spawn = tr.to_array();
        }
        if let Some(shape) = collider_shape_msg(shape, scale) {
            colliders.push(ColliderInit {
                shape,
                translation: tr.to_array(),
                rotation: [rot.x, rot.y, rot.z, rot.w],
                dynamic: in_ball,
                role,
            });
        }
    }

    for child in &node.children {
        walk_node(
            child,
            world,
            in_ball,
            ball_mesh_id,
            colliders,
            spawn,
            ball_scale,
        );
    }
}

/// Map a scene `ColliderShape` to the physics wire shape, folding the node's
/// accumulated per-axis world **scale** into the shape extents (a physics
/// collider has no scale of its own). Per-axis folding is exact for the
/// axis-aligned boxes this scene uses. Ellipsoid is dropped (unused).
fn collider_shape_msg(
    shape: &awsm_renderer_scene::ColliderShape,
    scale: Vec3,
) -> Option<ColliderShapeMsg> {
    use awsm_renderer_scene::ColliderShape as S;
    let s = scale.abs();
    let radial = s.x.max(s.z);
    Some(match *shape {
        S::Box { half_extents } => ColliderShapeMsg::Cuboid {
            half_extents: [
                half_extents[0] * s.x,
                half_extents[1] * s.y,
                half_extents[2] * s.z,
            ],
        },
        S::Sphere { radius } => ColliderShapeMsg::Ball {
            radius: radius * s.x.max(s.y).max(s.z),
        },
        S::Capsule {
            half_height,
            radius,
        } => ColliderShapeMsg::Capsule {
            half_height: half_height * s.y,
            radius: radius * radial,
        },
        S::Cylinder {
            half_height,
            radius,
        } => ColliderShapeMsg::Cylinder {
            half_height: half_height * s.y,
            radius: radius * radial,
        },
        S::Cone {
            half_height,
            radius,
        } => ColliderShapeMsg::Cone {
            half_height: half_height * s.y,
            radius: radius * radial,
        },
        // Unused in this scene (and the Box3D wire has no hull/ellipsoid
        // shape) — dropped, same policy as Ellipsoid always had.
        S::Ellipsoid { .. } | S::ConvexHull { .. } => return None,
    })
}

/// The tabletop's box collider: `(half_extents, translation)` — the click drop
/// zone. `None` if the scene has no recognizable floor box.
fn floor_box(colliders: &[ColliderInit]) -> Option<([f32; 3], [f32; 3])> {
    colliders.iter().find_map(|c| {
        if c.dynamic || c.role != ROLE_FLOOR {
            return None;
        }
        match c.shape {
            ColliderShapeMsg::Cuboid { half_extents } => Some((half_extents, c.translation)),
            _ => None,
        }
    })
}

/// Current backing-store aspect ratio (width / height).
fn aspect(canvas: &OffscreenCanvas) -> f32 {
    (canvas.width().max(1) as f32) / (canvas.height().max(1) as f32)
}

/// This worker's `performance` clock.
fn worker_performance() -> Result<web_sys::Performance, JsValue> {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("performance"))
        .ok()
        .and_then(|p| p.dyn_into::<web_sys::Performance>().ok())
        .ok_or_else(|| JsValue::from_str("no performance.now"))
}

/// Post a human-readable load-progress line to main (the loading screen).
fn post_progress(message: &str) {
    post_to_main(&RenderMsg::Progress {
        message: message.to_string(),
    });
}

/// Serialize a [`RenderMsg`] and post it to main.
fn post_to_main(msg: &RenderMsg) {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    match serde_wasm_bindgen::to_value(msg) {
        Ok(v) => {
            let _ = scope.post_message(&v);
        }
        Err(e) => tracing::error!("game worker: serialize RenderMsg: {e}"),
    }
}

/// Serialize an [`AudioMsg`] cue and post it to main (which owns WebAudio).
fn post_audio(msg: &AudioMsg) {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    if let Ok(v) = serde_wasm_bindgen::to_value(msg) {
        let _ = scope.post_message(&v);
    }
}
