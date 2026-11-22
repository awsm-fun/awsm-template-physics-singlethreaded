//! The game worker: one `requestAnimationFrame` loop that owns the renderer AND
//! the physics world, on a worker thread.
//!
//! Everything the game does — physics, rendering, camera, drop unprojection —
//! runs here, so the main thread stays free for DOM, input, and audio. The two
//! threads share nothing; input arrives as [`InputMsg`]/[`CameraMsg`]/… and
//! audio cues + stats go back as [`AudioMsg`]/[`StatsMsg`]. Each frame:
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
//! 6. post the frame's audio cues (and, ~1 Hz, sampled stats) to main.
//!
//! While the Stats panel asks (`StatsDetailMsg`), [`Profile`] also times the
//! frame's phases and the steps' phases; see `profile`.
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

use crate::physics::{Pose, World};
use crate::profile::{Phase, Profile};
use crate::protocol::{
    AudioMsg, CameraMsg, CameraPose, ColliderInit, ColliderShapeMsg, DropMsg, Input, InputMsg,
    QualityMsg, RenderMsg, ResizeMsg, StatsDetailMsg, StatsMsg, ROLE_FLOOR, SIM_HZ,
};

/// Player input, held on the worker and fed by [`InputMsg`]s.
type SharedInput = Rc<Input>;
/// The one orbit camera (render + physics W/A/S/D basis), driven by [`CameraMsg`].
type SharedCamera = Rc<RefCell<OrbitCamera>>;
/// The camera poses main has posted, newest last (see [`CameraQueue`]).
type SharedCameraQueue = Rc<RefCell<CameraQueue>>;
/// Pending click-drops as NDC points, drained + unprojected by the loop.
type DropQueue = Rc<RefCell<Vec<(f32, f32)>>>;
/// A pending Settings anti-aliasing change `(msaa, smaa)`, applied off the render path.
type PendingAa = Rc<Cell<Option<(bool, bool)>>>;
/// The Stats breakdowns main asks for (`DETAIL_*` bits, see [`StatsDetailMsg`]).
type StatsDetail = Rc<Cell<u32>>;
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
// ── Stats ────────────────────────────────────────────────────────────────────
/// EMA weight for the reported fps / steps-per-second (per ~1 s window).
const STATS_EMA_ALPHA: f64 = 0.35;
/// Smoothing weight of each frame gap in the display-period estimate used to
/// pick the camera pose (see [`CameraQueue`]); only needs to be roughly right.
const PERIOD_EMA_ALPHA: f64 = 0.1;
/// Frames that must present before stats counting starts — boot samples read
/// 2–3× the settled cost (cold caches, pipeline warmup).
const STATS_WARMUP_FRAMES: u32 = 60;

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
            // `?v=` names the vendored build: these files keep their names
            // across releases, so a browser that cached an older transcoder
            // (wasm included: the worker carries the query over to it) would
            // otherwise pair it with the new JS. Bump with awsm-renderer.
            format!("{base}/workers/basis-worker.js?v=0.44.2"),
            format!("{base}/vendor/basis/basis_transcoder.js?v=0.44.2"),
        ));
    }
    let get_bool = |key: &str| {
        js_sys::Reflect::get(&payload, &JsValue::from_str(key))
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    };
    let desired_aa = (get_bool("msaa"), get_bool("smaa"));

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
    desired_aa: (bool, bool),
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
    let stats_detail: StatsDetail = Rc::new(Cell::new(0));
    install_message_handler(
        canvas.clone(),
        input.clone(),
        camera_queue.clone(),
        drops.clone(),
        pending_aa.clone(),
        stats_detail.clone(),
    );

    // Report the GPU's capabilities up front so main can seed the resolution
    // scale (a software adapter can't push pixels) and cap the backing store.
    report_gpu_info().await;

    // The builder reports its coarse boot phases through this handler. Since the
    // deferred-boot renderer, `build()` compiles NO pipelines (it reserves them,
    // compiled by the labeled `ensure_config_pipelines` step below), so only
    // `Init` — device + core GPU resources — takes visible time here.
    let mut renderer = AwsmRendererBuilder::new(gpu_builder)
        .with_anti_aliasing(aa_config(desired_aa))
        // GPU timing for the Stats panel's GPU group (the CPU tier is `Off`
        // except while the renderer group is open, see `profile`): `Frame` tier
        // = timestamp queries on the renderer's passes, folded into one
        // whole-frame span (cheap; silently off without `timestamp-query`).
        .with_logging(awsm_renderer::debug::AwsmRendererLogging {
            gpu: awsm_renderer::debug::TimingTier::Frame,
            ..Default::default()
        })
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
        "WebGPU device + renderer ready (msaa {}, smaa {})",
        desired_aa.0, desired_aa.1
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
    scene.post_process.anti_aliasing = awsm_renderer_scene::AntiAliasingConfig {
        msaa: desired_aa.0,
        smaa: desired_aa.1,
    };
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
    let mut ball_bindings: Vec<awsm_renderer::buffer::shared_arena::SlotBinding> = Vec::new();
    let mut stats = StatsAcc::new(perf.now());
    let mut profile = Profile::new(perf.clone());

    *raf_init.borrow_mut() = Some(Closure::new(move |vsync_ms: f64| {
        // A pending Settings anti-aliasing change: apply it off the render path
        // (async recompile). Take the renderer OUT of the cell for the awaits;
        // `reconfiguring` keeps frames off it until it's put back.
        if !reconfiguring.get() {
            if let Some((msaa, smaa)) = pending_aa.take() {
                reconfiguring.set(true);
                let cell = cell.clone();
                let done = reconfiguring.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let Some(mut r) = cell.borrow_mut().take() else {
                        done.set(false);
                        return;
                    };
                    // Tell main first — it raises the "compiling pipelines"
                    // modal for the whole recompile (the first switch in each
                    // direction really compiles; later ones hit the variant
                    // cache and the modal only flashes).
                    post_to_main(&RenderMsg::AaCompileStart { msaa, smaa });
                    if let Err(e) = r.set_anti_aliasing(aa_config((msaa, smaa))).await {
                        tracing::error!("game worker: set_anti_aliasing failed: {e:?}");
                        post_to_main(&RenderMsg::Error {
                            message: format!("anti-aliasing change: {e:?}"),
                        });
                    } else {
                        // The actual pipeline compiles happen in commit_load —
                        // stream its progress into the modal via the renderer's
                        // shared phase label, deduped (the callback fires per
                        // resolution).
                        let mut last_line = String::new();
                        let progress = |s: awsm_renderer::loading::LoadingStats| {
                            let Some(line) = s.phase_label() else { return };
                            if line != last_line {
                                post_to_main(&RenderMsg::AaCompileProgress {
                                    message: line.clone(),
                                });
                                last_line = line;
                            }
                        };
                        if let Err(e) = r.commit_load(progress).await {
                            tracing::error!(
                                "game worker: commit_load after AA change failed: {e:?}"
                            );
                            post_to_main(&RenderMsg::Error {
                                message: format!("anti-aliasing pipelines: {e:?}"),
                            });
                        } else {
                            tracing::info!(
                                "game worker: anti-aliasing applied (msaa {msaa}, smaa {smaa})"
                            );
                        }
                    }
                    // Always lower the modal — even on failure (the error line
                    // is already on its way to the status bar).
                    post_to_main(&RenderMsg::AaCompileDone);
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
        profile.begin_frame(r, stats_detail.get());
        let mut lap_ms = work_t0;

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
        let step_t0 = perf.now();
        profile.charge(Phase::ReadApp, lap_ms, step_t0);
        let mut step_timer = profile.step_timer();
        while accumulator >= FIXED_DT_MS && substeps < MAX_SUBSTEPS {
            world.step(&input, yaw, &mut cues, step_timer.as_deref_mut());
            accumulator -= FIXED_DT_MS;
            substeps += 1;
        }
        if substeps == MAX_SUBSTEPS {
            accumulator = 0.0; // hit the cap — drop the backlog (avoid a spiral)
        }
        // The steps are physics time, not the renderer's: no phase gets it.
        lap_ms = perf.now();
        let step_ms = lap_ms - step_t0;

        // 3. ── Audio cues → main ─────────────────────────────────────────────
        for cue in cues.drain(..) {
            post_audio(&cue);
        }
        lap_ms = profile.lap(Phase::SoundCues, lap_ms);

        // 4. ── Scene clips, texture flows, particles ────────────────────────
        // Advanced by the same frame time as physics. Before the body writes
        // below, so physics has the last word on the slots it drives. (Don't
        // animate a body or its parents, though: `update_transforms` rewrites
        // the slots of a subtree a clip moved, over the physics pose.)
        let _ = r.update_animations(frame_ms);
        particles.tick(r, (frame_ms / 1000.0) as f32);
        lap_ms = profile.lap(Phase::Animate, lap_ms);

        // 5. ── Interpolate poses → write transforms ─────────────────────────
        // After the catch-up loop `accumulator` is the leftover fraction of a
        // step; alpha blends prev→curr (classic fixed-timestep interpolation).
        let alpha = (accumulator / FIXED_DT_MS) as f32;
        let (pp, pc) = world.player_poses();
        write_slot(
            binding,
            dirty_words_addr,
            ball_visual_scale,
            pose_lerp(pp, pc, alpha),
        );

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
                        Some(b) => ball_bindings.push(b),
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
        for (i, b) in ball_bindings.iter().enumerate() {
            let (bp, bc) = world.ball_poses(i);
            write_slot(
                *b,
                dirty_words_addr,
                ball_visual_scale,
                pose_lerp(bp, bc, alpha),
            );
        }
        lap_ms = profile.lap(Phase::ReadApp, lap_ms);

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
        lap_ms = profile.lap(Phase::SetCamera, lap_ms);
        r.update_transforms();
        lap_ms = profile.lap(Phase::UpdateTransforms, lap_ms);
        if let Err(err) = r.render(None) {
            tracing::warn!("game worker: render error: {err}");
        }
        profile.lap(Phase::Render, lap_ms);
        profile.end_frame();

        // 7. ── Stats + ready signal ─────────────────────────────────────────
        let now = perf.now();
        stats.record(
            now,
            vsync_ms,
            now - work_t0,
            substeps,
            step_ms,
            ball_count as u32,
            &mut profile,
        );
        frame_count = frame_count.wrapping_add(1);
        if frame_count == 3 {
            post_to_main(&RenderMsg::Ready);
        }
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

/// Accumulates per-frame timings on the worker and posts a smoothed [`StatsMsg`]
/// to main ~1 Hz (main owns the panel but not the data). Counting waits out
/// [`STATS_WARMUP_FRAMES`] so cold-start samples don't pollute the EMAs.
///
/// Renderer and physics share this thread, so the frame is split, never
/// double counted: physics is the time inside the step loop, the renderer
/// everything else the frame does.
struct StatsAcc {
    window_t0: f64,
    frames: u32,
    steps: u32,
    renderer_ms_sum: f64,
    step_ms_sum: f64,
    warmed: u32,
    fps_ema: Option<f64>,
    sps_ema: Option<f64>,
    renderer_ms_ema: Option<f64>,
    step_ms_ema: Option<f64>,
    worker_ms_ema: Option<f64>,
    /// The renderer's cumulative GPU frame timing `(total_ms, samples)` at the
    /// start of the current window (`awsm_renderer::profiling::gpu_frame_timing`).
    gpu_prev: Option<(f64, u64)>,
    gpu_ms_ema: Option<f64>,
    /// Consecutive windows with no GPU sample (see `StatsMsg::gpu_unavailable`).
    gpu_empty_windows: u32,
    balls: u32,
    /// The previous presented frame's rAF timestamp, and this window's frame
    /// span (see `StatsMsg::span_ms`).
    last_vsync: Option<f64>,
    span_frames: u32,
    span_ms: f64,
}

/// The renderer's cumulative GPU frame timing as `(total_ms, samples)`.
fn gpu_totals() -> Option<(f64, u64)> {
    awsm_renderer::profiling::gpu_frame_timing().map(|s| (s.total_ms, s.count))
}

impl StatsAcc {
    fn new(now: f64) -> Self {
        Self {
            window_t0: now,
            frames: 0,
            steps: 0,
            renderer_ms_sum: 0.0,
            step_ms_sum: 0.0,
            warmed: 0,
            fps_ema: None,
            sps_ema: None,
            renderer_ms_ema: None,
            step_ms_ema: None,
            worker_ms_ema: None,
            gpu_prev: None,
            gpu_ms_ema: None,
            gpu_empty_windows: 0,
            balls: 0,
            last_vsync: None,
            span_frames: 0,
            span_ms: 0.0,
        }
    }

    /// One frame: `frame_ms` of work on this thread, `step_ms` of it in
    /// `substeps` physics steps.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        now: f64,
        vsync_ms: f64,
        frame_ms: f64,
        substeps: u32,
        step_ms: f64,
        balls: u32,
        profile: &mut Profile,
    ) {
        self.balls = balls;
        let interval = self
            .last_vsync
            .replace(vsync_ms)
            .map(|prev| vsync_ms - prev);
        // Discard boot frames entirely (don't even start the window on them).
        if self.warmed < STATS_WARMUP_FRAMES {
            self.warmed += 1;
            self.window_t0 = now;
            self.frames = 0;
            self.steps = 0;
            self.renderer_ms_sum = 0.0;
            self.step_ms_sum = 0.0;
            self.gpu_prev = gpu_totals();
            self.span_frames = 0;
            self.span_ms = 0.0;
            profile.discard_window();
            return;
        }
        if let Some(d) = interval.filter(|d| (0.0..crate::refresh::MAX_INTERVAL_MS).contains(d)) {
            self.span_frames += 1;
            self.span_ms += d;
        }
        self.frames += 1;
        self.steps += substeps;
        self.renderer_ms_sum += frame_ms - step_ms;
        self.step_ms_sum += step_ms;
        let dt = now - self.window_t0;
        if dt < 1000.0 {
            return;
        }
        let ema = |slot: &mut Option<f64>, v: f64| {
            let e = slot.get_or_insert(v);
            *e += (v - *e) * STATS_EMA_ALPHA;
            *e
        };
        let secs = dt / 1000.0;
        let fps = ema(&mut self.fps_ema, self.frames as f64 / secs);
        let sps = ema(&mut self.sps_ema, self.steps as f64 / secs);
        let renderer_ms = ema(
            &mut self.renderer_ms_ema,
            self.renderer_ms_sum / self.frames.max(1) as f64,
        );
        let step_ms = ema(
            &mut self.step_ms_ema,
            if self.steps > 0 {
                self.step_ms_sum / self.steps as f64
            } else {
                0.0
            },
        );
        let worker_ms = ema(
            &mut self.worker_ms_ema,
            (self.renderer_ms_sum + self.step_ms_sum) / self.frames.max(1) as f64,
        );
        // Avg GPU ms per sampled frame over the window (the timestamp readback
        // skips frames while one is in flight — divide by samples, not frames).
        let gpu_now = gpu_totals();
        let gpu_window = match (gpu_now, self.gpu_prev) {
            (Some((t1, n1)), Some((t0, n0))) if n1 > n0 => Some((t1 - t0) / (n1 - n0) as f64),
            (Some((t1, n1)), None) if n1 > 0 => Some(t1 / n1 as f64),
            _ => None,
        };
        self.gpu_prev = gpu_now;
        let gpu_ms = match gpu_window {
            Some(ms) => {
                self.gpu_empty_windows = 0;
                Some(ema(&mut self.gpu_ms_ema, ms))
            }
            None => {
                self.gpu_empty_windows = self.gpu_empty_windows.saturating_add(1);
                self.gpu_ms_ema
            }
        };
        let breakdown = profile.window();
        post_stats(&StatsMsg {
            fps: fps as f32,
            gpu_ms: gpu_ms.map(|v| v as f32),
            gpu_unavailable: self.gpu_ms_ema.is_none() && self.gpu_empty_windows >= 3,
            renderer_ms: renderer_ms as f32,
            step_ms: step_ms as f32,
            worker_ms: worker_ms as f32,
            sps: sps as f32,
            balls: self.balls,
            span_frames: self.span_frames,
            span_ms: self.span_ms,
            renderer_phases: breakdown.renderer_phases,
            renderer_prep: breakdown.renderer_prep,
            renderer_passes: breakdown.renderer_passes,
            renderer_submit: breakdown.renderer_submit,
            renderer_hooks: breakdown.renderer_hooks,
            gpu_passes: breakdown.gpu_passes,
            step_phases: breakdown.step_phases,
        });
        self.span_frames = 0;
        self.span_ms = 0.0;
        self.window_t0 = now;
        self.frames = 0;
        self.steps = 0;
        self.renderer_ms_sum = 0.0;
        self.step_ms_sum = 0.0;
    }
}

/// Install this worker's `onmessage`, replacing the bootstrap's init handler
/// (which has done its job). Applies main's input/camera/drop/resize/quality
/// messages to the shared local state the render loop reads.
fn install_message_handler(
    canvas: OffscreenCanvas,
    input: SharedInput,
    camera_queue: SharedCameraQueue,
    drops: DropQueue,
    pending_aa: PendingAa,
    stats_detail: StatsDetail,
) {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    let cb = Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |e: web_sys::MessageEvent| {
        let data = e.data();
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
        if let Ok(DropMsg::Ball { ndc_x, ndc_y }) =
            serde_wasm_bindgen::from_value::<DropMsg>(data.clone())
        {
            drops.borrow_mut().push((ndc_x, ndc_y));
            return;
        }
        if let Ok(ResizeMsg::Canvas { width, height }) =
            serde_wasm_bindgen::from_value::<ResizeMsg>(data.clone())
        {
            if width > 0 && height > 0 {
                canvas.set_width(width);
                canvas.set_height(height);
            }
            return;
        }
        if let Ok(QualityMsg::AntiAlias { msaa, smaa }) =
            serde_wasm_bindgen::from_value::<QualityMsg>(data.clone())
        {
            pending_aa.set(Some((msaa, smaa)));
            return;
        }
        if let Ok(StatsDetailMsg::Set { bits }) =
            serde_wasm_bindgen::from_value::<StatsDetailMsg>(data)
        {
            stats_detail.set(bits);
        }
    });
    scope.set_onmessage(Some(cb.as_ref().unchecked_ref()));
    cb.forget();
}

/// Map the Settings `(msaa, smaa)` toggle pair onto the renderer's config:
/// MSAA is 4× or off (the only counts it supports), mipmapping always on.
/// Used both at BUILD time (the startup prefs, so boot compiles exactly the
/// variants this session needs — no reconcile recompile) and for runtime
/// Settings changes.
fn aa_config((msaa, smaa): (bool, bool)) -> awsm_renderer::anti_alias::AntiAliasing {
    awsm_renderer::anti_alias::AntiAliasing {
        msaa_sample_count: if msaa { Some(4) } else { None },
        smaa,
        mipmap: true,
    }
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

/// Probe the WebGPU adapter for the two facts main needs to size the canvas —
/// whether it's a software fallback and its max 2D texture dimension — and post
/// them. Failures degrade to safe defaults (not fallback, the guaranteed 8192).
async fn report_gpu_info() {
    use wasm_bindgen_futures::JsFuture;
    let (is_fallback, max_texture_dim) = match awsm_renderer::web_global::navigator_gpu() {
        Some(gpu) => match JsFuture::from(gpu.request_adapter()).await {
            Ok(v) if !v.is_null() && !v.is_undefined() => {
                let adapter: web_sys::GpuAdapter = v.unchecked_into();
                (
                    adapter.info().is_fallback_adapter(),
                    adapter.limits().max_texture_dimension_2d(),
                )
            }
            _ => (false, 8192),
        },
        None => (false, 8192),
    };
    tracing::info!(
        "game worker: gpu info — fallback {is_fallback}, max_texture_2d {max_texture_dim}"
    );
    post_to_main(&RenderMsg::GpuInfo {
        is_fallback,
        max_texture_dim,
    });
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

/// Serialize a [`StatsMsg`] and post it to main (the stats panel).
fn post_stats(msg: &StatsMsg) {
    let scope = js_sys::global().unchecked_into::<web_sys::DedicatedWorkerGlobalScope>();
    if let Ok(v) = serde_wasm_bindgen::to_value(msg) {
        let _ = scope.post_message(&v);
    }
}
