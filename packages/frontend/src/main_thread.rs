//! The app shell (main thread) — owns the DOM (Dominator), input, and WebAudio,
//! and spawns the game worker that does the actual rendering + physics.
//!
//! Flow:
//! 1. Build the canvas + HUD with Dominator and mount them.
//! 2. Ask the GPU what the Settings need (a software adapter, the largest
//!    texture), size the canvas backing store from the Settings' resolution,
//!    `transferControlToOffscreen`, and spawn the worker with the
//!    `OffscreenCanvas` and the anti-aliasing mode (see [`crate::bootstrap`]).
//! 3. Forward keyboard/pointer input to the worker as [`InputMsg`]/[`CameraMsg`]/
//!    [`DropMsg`]; the Settings modal forwards its changes (resolution,
//!    anti-aliasing) itself.
//! 4. Play back the worker's [`AudioMsg`] cues on the WebAudio player, and show
//!    its [`RenderMsg`]s on the loading screen; the kit's panels show its
//!    `aa-compile-*` and `stats` messages.
//!
//! The Settings modal and the Stats panel are awsm-renderer-kit's
//! (`awsm-renderer-settings`, `awsm-renderer-stats`): this thread shows them
//! and passes their messages to the worker, which applies the settings and
//! measures the stats.
//!
//! The main thread never renders or steps physics — so heavy frames in the
//! worker can't jank input or audio, and vice versa. Nothing is shared but
//! `postMessage`.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use awsm_renderer_settings::{
    probe_gpu, GpuInfo, SettingsConfig, SettingsHooks, SettingsOptions, SettingsPanel,
    SettingsStore, SettingsToWorker, SettingsUi,
};
use awsm_renderer_stats::{StatsHooks, StatsPanel, StatsToWorker};
use dominator::{clone, html};
use futures_signals::signal::Mutable;
use serde::Serialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::js_sys;
use web_sys::{HtmlCanvasElement, HtmlElement, KeyboardEvent, MessageEvent, Worker};

use crate::about::{About, Tab};
use crate::audio::AudioController;
use crate::bootstrap::spawn_worker;
use crate::protocol::{
    AudioMsg, CameraMsg, CameraPose, DropMsg, InputMsg, RenderMsg, HELD_BACK, HELD_FORWARD,
    HELD_LEFT, HELD_RIGHT,
};

/// Shared, lazily-loaded audio. `None` until the export finishes loading; stays
/// silent until a user gesture starts it (browser autoplay policy).
type Audio = Rc<RefCell<Option<AudioController>>>;

/// The game worker, once spawned. The Settings and Stats panels exist before
/// it (their messages before then are dropped; both repeat what matters).
type WorkerRef = Rc<RefCell<Option<Worker>>>;

/// The prefix of every `localStorage` key this app writes: the resolution
/// (`res_scale`), the anti-aliasing mode (`aa`), the Stats panel (`stats`)
/// and its groups (`stats_open_<id>`). Every experiment is served from the
/// same origin (awsm.fun), so each app's keys carry its own prefix; change it
/// when starting a new project from this template.
const STORAGE_PREFIX: &str = "awsm_box3d_st_";

/// The element id of the reduced-quality notice (awsm-renderer-settings
/// builds it): the About modal leaves Escape to it while it is up.
pub(crate) const QUALITY_NOTICE_ID: &str = "quality-notice";

/// The pre-0.42 per-toggle anti-aliasing keys (no app prefix), migrated to
/// `<prefix>aa` once and then removed.
const LEGACY_AA_KEYS: (&str, &str) = ("awsm_msaa", "awsm_smaa");

/// The name of the camera frame loop in the Stats panel's page group.
const PAGE_FRAME_PART: &str = "camera frame loop";
/// The name of the sound cues in the Stats panel's page group.
const PAGE_CUES_PART: &str = "sound cues";

/// The graphics Settings this app offers (awsm-renderer-settings' defaults):
///
/// - **Anti-aliasing**: Off, SMAA, MSAA 4×, TAA, and the TAA upscaling
///   presets Quality (67%), Balanced (58%) and Performance (50%). TAA is the
///   default on a desktop; a touch device and a software GPU start with Off
///   (they optimize for frame rate, and a one-time notice on a touch device
///   points at Settings). A stored choice always wins, and `?aa=<id>`
///   overrides it for one session.
/// - **Resolution** (the fill-rate lever): the canvas renders at
///   `devicePixelRatio × scale` device pixels, 50–100% in 5% steps. The
///   default is native resolution capped at a 2.8 MP pixel budget (never
///   below an effective DPR of 1), and lower on a touch device (60%) or a
///   software GPU (50%). TAA's upscaling presets scale the renderer's
///   internal size on top of it.
fn settings_config() -> SettingsConfig {
    SettingsConfig::default()
}

fn post<T: Serialize>(worker: &Worker, msg: &T) {
    if let Ok(v) = serde_wasm_bindgen::to_value(msg) {
        let _ = worker.post_message(&v);
    }
}

/// Posts one of the kit's messages (already a JS value) to the worker.
fn post_kit(worker: &Worker, message: Result<JsValue, String>) {
    match message {
        Ok(value) => {
            let _ = worker.post_message(&value);
        }
        Err(err) => tracing::error!("kit message: {err}"),
    }
}

/// A new `<tag class="class">`.
fn element(tag: &str, class: &str) -> HtmlElement {
    let el: HtmlElement = dominator::body()
        .owner_document()
        .and_then(|d| d.create_element(tag).ok())
        .expect("create_element")
        .unchecked_into();
    el.set_class_name(class);
    el
}

/// Build + mount the DOM, then spawn the game worker.
pub fn start() -> Result<(), JsValue> {
    // `?reset` in the URL forgets every stored setting (resolution,
    // anti-aliasing, the stats panel and its groups) so the load boots with
    // fresh defaults. It runs before anything reads a stored value.
    let store =
        SettingsStore::new(STORAGE_PREFIX).with_legacy_aa_keys(LEGACY_AA_KEYS.0, LEGACY_AA_KEYS.1);
    if store.reset_if_asked() {
        loading_log("?reset — stored settings cleared, booting with defaults");
    }
    store.migrate_legacy_aa();

    let status = Mutable::new("booting…".to_string());
    let about = About::default();
    let worker_ref: WorkerRef = Rc::new(RefCell::new(None));

    // The Stats panel (top right, hidden by default) and the bottom-left
    // Stats button. awsm-renderer-stats fills the panel from the worker's
    // reports; it sends the worker which groups are open (only those are
    // measured in detail) and what it measured here (its frame gaps for the
    // refresh rate, and this thread's own work).
    let stats_el = element("div", "stats");
    let stats_button = element("button", "stats-btn");
    let _ = stats_button.set_attribute("type", "button");
    stats_button.set_text_content(Some("Stats"));
    let stats = StatsPanel::mount(
        STORAGE_PREFIX,
        &stats_el,
        Some(&stats_button),
        StatsHooks {
            post: Box::new(clone!(worker_ref => move |message: &StatsToWorker| {
                if let Some(worker) = worker_ref.borrow().as_ref() {
                    post_kit(worker, message.to_js());
                }
            })),
            on_help: Some(Box::new(clone!(about => move || about.show(Tab::Stats)))),
            page_part: PAGE_FRAME_PART.to_string(),
            app_sections: None,
            section_rows: None,
            shown: None,
        },
    );
    loading_log("wasm compiled — booting");

    let app = html!("div", {
        .child(html!("canvas" => HtmlCanvasElement, {
            .class("canvas")
            .after_inserted(clone!(status, stats, worker_ref => move |canvas| {
                // The GPU probe first: the Settings' defaults depend on it (a
                // software adapter starts lower), and the canvas is sized
                // from them before it goes to the worker.
                spawn_local(async move {
                    let gpu = probe_gpu().await;
                    tracing::info!(
                        "gpu — fallback adapter {}, largest 2D texture {}",
                        gpu.is_fallback,
                        gpu.max_texture_dim
                    );
                    if let Err(e) = setup(canvas, status.clone(), stats, store, gpu, worker_ref) {
                        status.set(format!("setup error: {e:?}"));
                        tracing::error!("setup: {e:?}");
                    }
                });
            }))
        }))
        .child(html!("div", {
            .class("hud")
            .text("single-player-game-physics\nW/A/S/D or arrows: roll · Space: jump · click: drop a ball\nright-drag: orbit · wheel: zoom\ntouch: swipe to fling the ball · tap to drop\nsound starts on your first key or click\n")
            .child(html!("span", {
                .text_signal(status.signal_cloned())
            }))
        }))
        .child(dominator::DomBuilder::new(stats_el).into_dom())
        .child(dominator::DomBuilder::new(stats_button).into_dom())
        .child(about.button())
        .child(about.modal())
    });

    dominator::append_dom(&dominator::body(), app);
    Ok(())
}

/// Append a line to the loading screen's log (`#loading-log` in index.html).
pub fn loading_log(message: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(log) = document.get_element_by_id("loading-log") else {
        return;
    };
    if let Ok(line) = document.create_element("div") {
        line.set_text_content(Some(message));
        let _ = log.append_child(&line);
        while log.child_element_count() > 14 {
            if let Some(first) = log.first_element_child() {
                first.remove();
            }
        }
    }
}

/// Fade out + drop the loading overlay (first frames are on screen).
pub fn loading_done() {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(overlay) = document.get_element_by_id("loading") else {
        return;
    };
    let _ = overlay.set_attribute("class", "done");
    let remove = Closure::once_into_js(move || overlay.remove());
    if let Some(window) = web_sys::window() {
        let _ = window
            .set_timeout_with_callback_and_timeout_and_arguments_0(remove.unchecked_ref(), 600);
    }
}

/// Render a single full-screen message and nothing else. Used when the app can't
/// start (no WebGPU, worker error).
pub fn fatal(message: &str) {
    loading_log(&format!("FATAL: {message}"));
    loading_done();
    let app = html!("div", {
        .class("hud")
        .style("pointer-events", "auto")
        .style("max-width", "46em")
        .style("white-space", "pre-wrap")
        .text("single-player-game-physics — cannot start\n\n")
        .child(html!("span", { .text(message) }))
    });
    dominator::append_dom(&dominator::body(), app);
}

/// Runs once the canvas is in the DOM and the GPU has been probed: size it,
/// transfer it to the worker, and wire input, the Settings and the worker
/// message bridge.
fn setup(
    canvas: HtmlCanvasElement,
    status: Mutable<String>,
    stats: StatsPanel,
    store: SettingsStore,
    gpu: GpuInfo,
    worker_ref: WorkerRef,
) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;

    // ── Settings: resolution + anti-aliasing (awsm-renderer-settings) ───────
    // The stored choices (or `?aa=`, or the device defaults: touch and the
    // software GPU probed above start lower) give the first canvas size and
    // mode; the bottom-right Settings modal changes them live and stores
    // them. The modal, the reduced-quality notice and the "compiling
    // pipelines" overlay go into the body, after the Settings button.
    let body = dominator::body();
    let settings_button = element("button", "settings-btn");
    let _ = settings_button.set_attribute("type", "button");
    settings_button.set_text_content(Some("Settings"));
    body.append_child(&settings_button)?;
    let settings = SettingsPanel::new(
        &body,
        &canvas,
        store,
        &SettingsOptions::new(&settings_config(), Some(gpu)),
        SettingsUi::default(),
        loading_log,
    );

    // Size the backing store, then transfer the canvas to the worker.
    let (w, h) = settings.backing();
    canvas.set_width(w);
    canvas.set_height(h);
    let offscreen = canvas.transfer_control_to_offscreen()?;
    // Base URL for our same-origin asset fetches (the worker's `blob:` base
    // can't resolve them).
    let base = page_base(&window);

    // The camera pose. Main owns it: pointer drags and the wheel change it
    // here, the audio listener follows its yaw at once, and `install_camera`
    // posts it to the worker once per main frame (see `CameraMsg`).
    let camera = Rc::new(Cell::new(CameraPose::START));

    // Kick off the (async) audio load now; it stays silent until the first gesture.
    let audio: Audio = Rc::new(RefCell::new(None));
    spawn_local(clone!(audio, status, base, camera => async move {
        match AudioController::load(&base).await {
            Ok(mut controller) => {
                controller.set_camera_yaw(camera.get().yaw);
                *audio.borrow_mut() = Some(controller);
                tracing::info!("audio loaded");
                loading_log("audio project loaded (roll + hit/land voices)");
            }
            Err(e) => {
                tracing::error!("audio load failed: {e:?}");
                status.set(format!("audio load error: {e:?}"));
            }
        }
    }));

    // Payload for the worker: the transferred OffscreenCanvas, the asset base,
    // and the startup anti-aliasing mode (so the worker builds with it, with
    // no round-trip / race).
    let payload = js_sys::Object::new();
    set(&payload, "canvas", &offscreen);
    set(&payload, "origin", &JsValue::from_str(&base));
    // The APP base — the directory THIS PAGE is served from — distinct from
    // `origin`/`base` above, which in dev is the live-media server. The Basis
    // codec worker + transcoder ship with the app (Trunk copy-file → `workers/`,
    // `vendor/basis/`), so the render worker must spawn them from an ABSOLUTE URL
    // (its `blob:` base can't resolve a relative path). This must keep the PATH,
    // not just the origin: under a subpath deploy (GitHub Pages `/<repo>/`, or a
    // CDN like `/experiments/<slug>/`) the files live under that directory, so a
    // bare origin would 404. See `game.rs` configure(...).
    set(&payload, "app_base", &JsValue::from_str(&app_base(&window)));
    set(&payload, "aa", &JsValue::from_str(settings.aa().id()));
    let transfer = js_sys::Array::new();
    transfer.push(&offscreen);

    // This thread's time on sound cues since its last frame (the Stats
    // panel's page group).
    let cues_ms = Rc::new(Cell::new(0.0));
    let perf = window
        .performance()
        .ok_or_else(|| JsValue::from_str("no performance"))?;

    // Handle messages coming back from the worker: the kit's (the "compiling
    // pipelines" overlay, the stats reports) first, then ours.
    let on_msg = Closure::<dyn FnMut(MessageEvent)>::new(
        clone!(audio, status, settings, stats, cues_ms, perf => move |e: MessageEvent| {
            let data = e.data();
            if settings.handle_js(&data) || stats.handle_js(&data) {
                return;
            }
            // Audio cue → the WebAudio player.
            let t0 = perf.now();
            if let Ok(msg) = serde_wasm_bindgen::from_value::<AudioMsg>(data.clone()) {
                if let Some(c) = audio.borrow_mut().as_mut() {
                    c.on_audio(msg);
                }
                cues_ms.set(cues_ms.get() + perf.now() - t0);
                return;
            }
            match serde_wasm_bindgen::from_value::<RenderMsg>(data) {
                Ok(RenderMsg::Progress { message }) => loading_log(&message),
                Ok(RenderMsg::Ready) => {
                    loading_log("first frames rendered — ready");
                    loading_done();
                    status.set("playing — roll · Space jump · click drops a ball · right-drag orbit".into());
                    // The game is visible now — if this load applied the
                    // reduced touch-device defaults, say so.
                    settings.ready();
                }
                Ok(RenderMsg::Error { message }) => {
                    loading_log(&format!("ERROR: {message}"));
                    status.set(format!("worker error: {message}"));
                }
                Err(_) => { /* not a RenderMsg (e.g. an init-error blob) — ignore */ }
            }
        }),
    );

    loading_log("spawning game worker…");
    let worker = spawn_worker(&payload, &transfer, on_msg.as_ref().unchecked_ref())?;
    on_msg.forget();
    // Publish the handle for the Stats panel's messages.
    *worker_ref.borrow_mut() = Some(worker.clone());

    // Keyboard → InputMsg; a keypress also starts audio.
    install_keyboard(&window, worker.clone(), audio.clone())?;
    // Ask for keyboard focus up front, so W/A/S/D work without a click first.
    // At the top level this is a no-op; in a cross-origin iframe the browser
    // only allows it once the parent has focused the frame (the awsm.fun
    // shell does), and otherwise the first press on the canvas does it.
    let _ = window.focus();
    // The Settings start following the canvas's layout size (the backing
    // store is CSS size × dpr × the resolution scale) and their controls;
    // every change goes to the worker, which owns the OffscreenCanvas and the
    // renderer.
    settings.start(
        &canvas,
        SettingsHooks {
            on_resize: Box::new(clone!(worker => move |width, height| {
                post_kit(&worker, SettingsToWorker::Resize { width, height }.to_js());
            })),
            on_anti_alias: Box::new(clone!(worker => move |mode| {
                post_kit(&worker, SettingsToWorker::AntiAlias { mode }.to_js());
            })),
            open_button: Some(settings_button.into()),
            extra_rows: Vec::new(),
        },
    );
    // Camera pose → the worker, once per main frame while it changes.
    install_camera(&window, worker.clone(), camera.clone(), stats, cues_ms)?;
    // Pointer: click drops a ball, right-drag orbits, wheel zooms, touch flings.
    install_pointer(&window, &canvas, worker, audio, camera)?;

    status.set("loading scene…".into());
    Ok(())
}

fn set(obj: &js_sys::Object, key: &str, value: &JsValue) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), value);
}

// ── Touch fling tuning ───────────────────────────────────────────────────────
/// CSS px a touch must travel before it counts as a swipe — under this it's a
/// tap (→ the browser's click drops a ball).
const TAP_SLOP_PX: f32 = 12.0;
/// The release-velocity window (ms).
const FLING_WINDOW_MS: f64 = 120.0;
/// Swipe speed (CSS px/s) → ball speed (m/s).
const FLING_GAIN: f32 = 0.004;
/// Swipes slower than this (m/s) are ignored — a slow deliberate drag isn't a throw.
const FLING_MIN_SPEED: f32 = 0.25;

/// Live tracking of the (single) touch pointer that may become a fling.
struct Swipe {
    id: i32,
    origin: (f32, f32),
    samples: Vec<(f64, f32, f32)>,
}

/// Post the camera pose to the worker from main's own `requestAnimationFrame`,
/// stamped with that frame's time on the absolute clock
/// (`performance.timeOrigin + timestamp`), and only when it changed: the
/// worker keeps the last few and draws the right one each frame (see
/// [`CameraMsg`] for why it is per frame and not per event).
fn install_camera(
    window: &web_sys::Window,
    worker: Worker,
    camera: Rc<Cell<CameraPose>>,
    stats: StatsPanel,
    cues_ms: Rc<Cell<f64>>,
) -> Result<(), JsValue> {
    type FrameLoop = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;
    let perf = window
        .performance()
        .ok_or_else(|| JsValue::from_str("no performance"))?;
    let time_origin = perf.time_origin();
    let cb: FrameLoop = Rc::new(RefCell::new(None));
    let next = cb.clone();
    let win = window.clone();
    // Nothing posted yet: the worker starts at the same `START` pose.
    let mut sent = CameraPose::START;
    *cb.borrow_mut() = Some(Closure::new(move |t: f64| {
        let t0 = perf.now();
        let pose = camera.get();
        if pose != sent {
            sent = pose;
            post(
                &worker,
                &CameraMsg::Pose {
                    t: time_origin + t,
                    pose,
                },
            );
        }
        // This thread's own work, for the Stats panel's page group: the
        // camera post, and the sound cues played since the last frame.
        stats.page_frame(perf.now() - t0, Some(PAGE_FRAME_PART), true);
        stats.page_frame(cues_ms.take(), Some(PAGE_CUES_PART), false);
        if let Some(f) = next.borrow().as_ref() {
            let _ = win.request_animation_frame(f.as_ref().unchecked_ref());
        }
    }));
    if let Some(f) = cb.borrow().as_ref() {
        window.request_animation_frame(f.as_ref().unchecked_ref())?;
    }
    // The loop runs for the session; the closure keeps itself alive.
    std::mem::forget(cb);
    Ok(())
}

/// Wire pointer input. A **left click drops a ball**: the click point is posted
/// as an NDC [`DropMsg`] (the worker unprojects it onto the table). A
/// **right-button drag orbits** the camera and turns the audio listener with
/// it; the **wheel zooms**. Both change main's [`CameraPose`], which
/// [`install_camera`] posts once per frame. A **touch swipe flings the ball**
/// ([`InputMsg::Fling`]); a short still touch stays a tap → drop.
fn install_pointer(
    window: &web_sys::Window,
    canvas: &HtmlCanvasElement,
    worker: Worker,
    audio: Audio,
    camera: Rc<Cell<CameraPose>>,
) -> Result<(), JsValue> {
    // Set when a touch swipe just flung the ball: the browser still synthesizes a
    // `click` afterward, and a fling must not ALSO drop a ball.
    let suppress_click = Rc::new(Cell::new(false));

    // Click → drop a ball at the clicked table spot (posted as NDC).
    let click_canvas = canvas.clone();
    let click = Closure::<dyn FnMut(web_sys::MouseEvent)>::new(
        clone!(audio, suppress_click, worker => move |e: web_sys::MouseEvent| {
            if suppress_click.take() {
                return;
            }
            // The click is also a user gesture — the first one starts audio.
            if let Some(c) = audio.borrow_mut().as_mut() {
                c.ensure_started();
            }
            let w = click_canvas.client_width().max(1) as f32;
            let h = click_canvas.client_height().max(1) as f32;
            let ndc_x = (e.offset_x() as f32 / w) * 2.0 - 1.0;
            let ndc_y = 1.0 - (e.offset_y() as f32 / h) * 2.0;
            post(&worker, &DropMsg::Ball { ndc_x, ndc_y });
        }),
    );
    canvas.add_event_listener_with_callback("click", click.as_ref().unchecked_ref())?;
    click.forget();

    let dragging = Rc::new(Cell::new(false));
    let swipe: Rc<RefCell<Option<Swipe>>> = Rc::new(RefCell::new(None));

    // pointerdown: the RIGHT button begins an orbit drag; a touch begins a
    // possible swipe. Any button is a user gesture → start audio.
    let focus_window = window.clone();
    let down = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, audio, swipe => move |e: web_sys::PointerEvent| {
            // Keyboard focus to this page. On awsm.fun it runs inside an
            // iframe, and keys go to whichever frame has focus: a press on the
            // canvas must bring W/A/S/D here, not leave them with the parent.
            let _ = focus_window.focus();
            if e.button() == 2 {
                dragging.set(true);
            }
            if e.pointer_type() == "touch" && swipe.borrow().is_none() {
                let (x, y) = (e.client_x() as f32, e.client_y() as f32);
                *swipe.borrow_mut() = Some(Swipe {
                    id: e.pointer_id(),
                    origin: (x, y),
                    samples: vec![(e.time_stamp(), x, y)],
                });
            }
            if let Some(c) = audio.borrow_mut().as_mut() {
                c.ensure_started();
            }
        }),
    );
    canvas.add_event_listener_with_callback("pointerdown", down.as_ref().unchecked_ref())?;
    down.forget();

    // Suppress the OS context menu so the right-drag is a clean orbit gesture.
    let menu = Closure::<dyn FnMut(web_sys::Event)>::new(|e: web_sys::Event| {
        e.prevent_default();
    });
    canvas.add_event_listener_with_callback("contextmenu", menu.as_ref().unchecked_ref())?;
    menu.forget();

    // pointermove on the window so a drag keeps orbiting even off-canvas. The
    // drag turns main's camera pose (posted to the worker once per frame) and
    // the audio listener with it.
    let move_ = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, audio, camera, swipe => move |e: web_sys::PointerEvent| {
            if let Some(s) = swipe.borrow_mut().as_mut() {
                if e.pointer_id() == s.id {
                    let t = e.time_stamp();
                    s.samples.push((t, e.client_x() as f32, e.client_y() as f32));
                    while s.samples.len() > 1 && t - s.samples[1].0 > FLING_WINDOW_MS {
                        s.samples.remove(0);
                    }
                    e.prevent_default();
                }
                return;
            }
            if dragging.get() {
                let mut pose = camera.get();
                pose.orbit(e.movement_x() as f32, e.movement_y() as f32);
                camera.set(pose);
                if let Some(c) = audio.borrow_mut().as_mut() {
                    c.set_camera_yaw(pose.yaw);
                }
            }
        }),
    );
    window.add_event_listener_with_callback("pointermove", move_.as_ref().unchecked_ref())?;
    move_.forget();

    // pointerup ends the drag and resolves the swipe.
    let up = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, swipe, suppress_click, worker => move |e: web_sys::PointerEvent| {
            dragging.set(false);
            let ours = matches!(&*swipe.borrow(), Some(s) if s.id == e.pointer_id());
            if !ours {
                return;
            }
            let Some(s) = swipe.borrow_mut().take() else {
                return;
            };
            let (lx, ly) = (e.client_x() as f32, e.client_y() as f32);
            let travel = ((lx - s.origin.0).powi(2) + (ly - s.origin.1).powi(2)).sqrt();
            if travel < TAP_SLOP_PX {
                return; // a tap — the browser's click follows and drops a ball
            }
            suppress_click.set(true);
            let t = e.time_stamp();
            let &(t0, x0, y0) = s
                .samples
                .iter()
                .find(|(ts, _, _)| t - ts <= FLING_WINDOW_MS)
                .unwrap_or(&s.samples[0]);
            let dt = ((t - t0) / 1000.0) as f32;
            if dt <= 0.0 {
                return;
            }
            let vx = (lx - x0) / dt * FLING_GAIN;
            let vz = (ly - y0) / dt * FLING_GAIN;
            if (vx * vx + vz * vz).sqrt() >= FLING_MIN_SPEED {
                post(&worker, &InputMsg::Fling { vx, vz });
            }
        }),
    );
    window.add_event_listener_with_callback("pointerup", up.as_ref().unchecked_ref())?;
    up.forget();

    // A cancelled pointer abandons the swipe.
    let cancel = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, swipe => move |e: web_sys::PointerEvent| {
            dragging.set(false);
            let ours = matches!(&*swipe.borrow(), Some(s) if s.id == e.pointer_id());
            if ours {
                *swipe.borrow_mut() = None;
            }
        }),
    );
    window.add_event_listener_with_callback("pointercancel", cancel.as_ref().unchecked_ref())?;
    cancel.forget();

    // wheel on the canvas zooms (preventDefault so the page doesn't scroll).
    let wheel = Closure::<dyn FnMut(web_sys::WheelEvent)>::new(
        clone!(camera => move |e: web_sys::WheelEvent| {
            e.prevent_default();
            let mut pose = camera.get();
            pose.zoom(e.delta_y() as f32);
            camera.set(pose);
        }),
    );
    canvas.add_event_listener_with_callback("wheel", wheel.as_ref().unchecked_ref())?;
    wheel.forget();

    Ok(())
}

/// Attach `keydown`/`keyup` listeners that translate WASD/arrows into
/// [`InputMsg::Held`] and Space into [`InputMsg::Jump`].
fn install_keyboard(window: &web_sys::Window, worker: Worker, audio: Audio) -> Result<(), JsValue> {
    let down =
        Closure::<dyn FnMut(KeyboardEvent)>::new(clone!(audio, worker => move |e: KeyboardEvent| {
            if let Some(c) = audio.borrow_mut().as_mut() {
                c.ensure_started();
            }
            let key = e.key();
            if key == " " || key == "Spacebar" {
                // A button reached with the keyboard keeps Space: it presses
                // the button. (One that was clicked does not: `:focus-visible`
                // is the browser's own test for "focused from the keyboard".)
                if on_keyboard_focused_control(&e) {
                    return;
                }
                if !e.repeat() {
                    post(&worker, &InputMsg::Jump);
                }
                e.prevent_default();
                return;
            }
            if let Some(mask) = key_mask(&key) {
                post(&worker, &InputMsg::Held { mask, down: true });
            }
        }));
    window.add_event_listener_with_callback("keydown", down.as_ref().unchecked_ref())?;
    down.forget();

    let up = Closure::<dyn FnMut(KeyboardEvent)>::new(move |e: KeyboardEvent| {
        if let Some(mask) = key_mask(&e.key()) {
            post(&worker, &InputMsg::Held { mask, down: false });
        }
    });
    window.add_event_listener_with_callback("keyup", up.as_ref().unchecked_ref())?;
    up.forget();

    Ok(())
}

/// Keeps a key pressed inside a modal from the game's `window` listeners, so
/// the arrows move a slider, a select or a tab and Space presses a button.
/// Escape goes through: it closes the modal.
///
/// Attach it with `EventOptions::bubbles()`: dominator's plain `.event` listens
/// in the capture phase, where stopping the key would also keep it from the
/// controls inside the modal.
pub(crate) fn keep_keys_in_modal(e: dominator::events::KeyDown) {
    if e.key() != "Escape" {
        e.stop_propagation();
    }
}

/// Whether a key event is aimed at a control that has the keyboard's focus.
fn on_keyboard_focused_control(e: &KeyboardEvent) -> bool {
    e.target()
        .and_then(|target| target.dyn_into::<web_sys::Element>().ok())
        .is_some_and(|el| {
            el.matches("button:focus-visible, select:focus-visible, a[href]:focus-visible")
                .unwrap_or(false)
        })
}

/// The `HELD_*` bit a key drives (WASD or arrows). Unknown keys → `None`.
fn key_mask(key: &str) -> Option<u32> {
    match key {
        "w" | "W" | "ArrowUp" => Some(HELD_FORWARD),
        "s" | "S" | "ArrowDown" => Some(HELD_BACK),
        "a" | "A" | "ArrowLeft" => Some(HELD_LEFT),
        "d" | "D" | "ArrowRight" => Some(HELD_RIGHT),
        _ => None,
    }
}

/// The base URL for the media fetches (`bundle/` and `audio/`), with no trailing
/// slash. Resolution order: `?media=` query param, then the `MEDIA_BASE`
/// compile-time env (`task dev` sets it to the side media server), then — the
/// production path — the directory the page is served from ([`app_base`]).
fn page_base(window: &web_sys::Window) -> String {
    let location = window.location();
    if let Ok(search) = location.search() {
        if let Ok(params) = web_sys::UrlSearchParams::new_with_str(&search) {
            if let Some(media) = params.get("media") {
                if !media.is_empty() {
                    return media.trim_end_matches('/').to_string();
                }
            }
        }
    }
    if let Some(base) = option_env!("MEDIA_BASE") {
        if !base.is_empty() {
            return base.trim_end_matches('/').to_string();
        }
    }
    app_base(window)
}

/// The directory THIS PAGE is served from — origin + the path up to (not
/// including) the trailing `/` — with no trailing slash. This is where Trunk
/// lays the app down: `index.html`, the wasm/JS glue, and the copy-file'd
/// `workers/` + `vendor/basis/`. Keeping the path makes it correct at any
/// deployment depth: the root (`/`), a GitHub Pages project site (`/<repo>/`),
/// or a CDN subpath (`/experiments/<slug>/`). Unlike [`page_base`], it never
/// redirects to the media server, so it always points at the app's own files.
///   `https://host/experiments/foo/`           → `https://host/experiments/foo`
///   `https://host/experiments/foo/index.html` → `https://host/experiments/foo`
///   `http://127.0.0.1:9000/`                   → `http://127.0.0.1:9000`
fn app_base(window: &web_sys::Window) -> String {
    let location = window.location();
    let origin = location.origin().unwrap_or_default();
    let mut path = location.pathname().unwrap_or_default();
    if let Some(idx) = path.rfind('/') {
        path.truncate(idx);
    }
    format!("{}{}", origin.trim_end_matches('/'), path)
}
