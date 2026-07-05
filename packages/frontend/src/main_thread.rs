//! The app shell (main thread) — owns the DOM (Dominator), input, and WebAudio,
//! and spawns the game worker that does the actual rendering + physics.
//!
//! Flow:
//! 1. Build the canvas + HUD with Dominator and mount them.
//! 2. Size the canvas backing store, `transferControlToOffscreen`, and spawn the
//!    worker with the `OffscreenCanvas` (see [`crate::bootstrap`]).
//! 3. Forward keyboard/pointer input to the worker as [`InputMsg`]/[`CameraMsg`]/
//!    [`DropMsg`]; forward Settings + layout changes as [`QualityMsg`]/
//!    [`ResizeMsg`].
//! 4. Play back the worker's [`AudioMsg`] cues on the WebAudio player, and show
//!    its [`RenderMsg`]/[`StatsMsg`] on the loading screen + stats panel.
//!
//! The main thread never renders or steps physics — so heavy frames in the
//! worker can't jank input or audio, and vice versa. Nothing is shared but
//! `postMessage`.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use dominator::{clone, html, with_node};
use futures_signals::signal::{Mutable, SignalExt};
use serde::Serialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::js_sys;
use web_sys::{HtmlCanvasElement, HtmlInputElement, KeyboardEvent, MessageEvent, Worker};

use crate::audio::AudioController;
use crate::bootstrap::spawn_worker;
use crate::protocol::{
    AudioMsg, CameraMsg, DropMsg, InputMsg, QualityMsg, RenderMsg, ResizeMsg, StatsMsg,
    CAMERA_ORBIT_SENSITIVITY, HELD_BACK, HELD_FORWARD, HELD_LEFT, HELD_RIGHT,
};

/// Shared, lazily-loaded audio. `None` until the export finishes loading; stays
/// silent until a user gesture starts it (browser autoplay policy).
type Audio = Rc<RefCell<Option<AudioController>>>;

// ── Resolution scale: the one fill-rate lever ────────────────────────────────
/// Floor of the slider (50% of native). The slider's `min` in `index.html` mirrors it.
const MIN_SCALE: f64 = 0.5;
/// Ceiling: native resolution. We never supersample here.
const MAX_SCALE: f64 = 1.0;
/// Start on a touch / `pointer: coarse` device (phones, tablets): high DPR × a
/// mobile GPU is the case that tanks to ~20fps at native, so seed it down.
const COARSE_START_SCALE: f64 = 0.6;
/// Start on a software / fallback adapter — it genuinely can't push pixels, so
/// seed lower still. Applied when the worker reports the adapter.
const FALLBACK_START_SCALE: f64 = 0.5;
/// Max 2D texture dimension assumed until the worker reports the real one.
const DEFAULT_MAX_TEX: u32 = 8192;
/// `localStorage` key for the persisted user choice.
const RES_STORAGE_KEY: &str = "awsm_res_scale";

/// Everything needed to size the canvas backing store: the user's `scale`
/// fraction, the GPU's texture-dimension cap, and the current CSS size + DPR
/// (updated by the `ResizeObserver`). `Copy` so it lives in a `Cell`.
#[derive(Clone, Copy)]
struct ResState {
    scale: f64,
    max_tex: u32,
    css_w: f64,
    css_h: f64,
    dpr: f64,
}

impl ResState {
    /// Backing-store size in device pixels: `css × dpr × scale`, each axis
    /// clamped to `[1, max_tex]`.
    fn backing(&self) -> (u32, u32) {
        let s = (self.dpr * self.scale).max(0.01);
        let w = ((self.css_w * s).round() as u32).clamp(1, self.max_tex);
        let h = ((self.css_h * s).round() as u32).clamp(1, self.max_tex);
        (w, h)
    }
}

/// Percent form of a scale fraction (0.6 → 60) — the slider's unit.
fn pct_of(scale: f64) -> u32 {
    (scale * 100.0).round() as u32
}

/// The scale to start at: a stored user choice if there is one, else a lower
/// seed on a touch device, else native.
fn initial_scale(window: &web_sys::Window) -> f64 {
    if let Some(stored) = stored_scale(window) {
        return stored;
    }
    if coarse_pointer(window) {
        COARSE_START_SCALE
    } else {
        MAX_SCALE
    }
}

/// A touch / `pointer: coarse` device (phones, tablets) — the device class that
/// seeds lower graphics defaults. Only DEFAULTS — a stored user choice wins.
fn coarse_pointer(window: &web_sys::Window) -> bool {
    window
        .match_media("(pointer: coarse)")
        .ok()
        .flatten()
        .map(|m| m.matches())
        .unwrap_or(false)
}

/// The persisted resolution scale, if the user has set one (clamped to range).
fn stored_scale(window: &web_sys::Window) -> Option<f64> {
    let ls = window.local_storage().ok().flatten()?;
    let raw = ls.get_item(RES_STORAGE_KEY).ok().flatten()?;
    raw.parse::<f64>()
        .ok()
        .map(|v| v.clamp(MIN_SCALE, MAX_SCALE))
}

/// Persist the user's resolution scale (only user drags call this).
fn store_scale(window: &web_sys::Window, scale: f64) {
    if let Ok(Some(ls)) = window.local_storage() {
        let _ = ls.set_item(RES_STORAGE_KEY, &format!("{scale:.2}"));
    }
}

// ── Anti-aliasing settings ───────────────────────────────────────────────────
const MSAA_STORAGE_KEY: &str = "awsm_msaa";
const SMAA_STORAGE_KEY: &str = "awsm_smaa";
/// `localStorage` key for the stats-panel toggle (default OFF — the panel eats a
/// lot of a phone screen, so it's opt-in via the bottom-left Stats chip).
const STATS_STORAGE_KEY: &str = "awsm_stats";
/// Default anti-aliasing on a fine-pointer (desktop) device. On `pointer: coarse`
/// devices BOTH toggles default OFF (the mobile defaults optimize for frame rate).
const MSAA_DEFAULT: bool = true;
const SMAA_DEFAULT: bool = true;

/// Read a persisted boolean setting (`"1"`/`"0"`), `None` if unset.
fn stored_bool(window: &web_sys::Window, key: &str) -> Option<bool> {
    let ls = window.local_storage().ok().flatten()?;
    match ls.get_item(key).ok().flatten()?.as_str() {
        "1" => Some(true),
        "0" => Some(false),
        _ => None,
    }
}

/// Persist a boolean setting.
fn store_bool(window: &web_sys::Window, key: &str, val: bool) {
    if let Ok(Some(ls)) = window.local_storage() {
        let _ = ls.set_item(key, if val { "1" } else { "0" });
    }
}

/// `?reset` in the URL wipes every persisted app setting so the load boots with
/// fresh defaults. Must run before ANY stored value is read.
fn maybe_reset_settings(window: &web_sys::Window) {
    let has_reset = window
        .location()
        .search()
        .ok()
        .and_then(|s| web_sys::UrlSearchParams::new_with_str(&s).ok())
        .map(|p| p.get("reset").is_some())
        .unwrap_or(false);
    if !has_reset {
        return;
    }
    if let Ok(Some(ls)) = window.local_storage() {
        for key in [
            RES_STORAGE_KEY,
            MSAA_STORAGE_KEY,
            SMAA_STORAGE_KEY,
            STATS_STORAGE_KEY,
        ] {
            let _ = ls.remove_item(key);
        }
    }
    loading_log("?reset — stored settings cleared, booting with defaults");
}

/// Serialize a message and post it to the worker.
fn post<T: Serialize>(worker: &Worker, msg: &T) {
    if let Ok(v) = serde_wasm_bindgen::to_value(msg) {
        let _ = worker.post_message(&v);
    }
}

/// Post the backing-store size for `st` to the worker (it owns the OffscreenCanvas).
fn post_resize(worker: &Worker, st: &ResState) {
    let (width, height) = st.backing();
    post(worker, &ResizeMsg::Canvas { width, height });
}

/// Build + mount the DOM, then spawn the game worker.
pub fn start() -> Result<(), JsValue> {
    if let Some(window) = web_sys::window() {
        maybe_reset_settings(&window);
    }
    let status = Mutable::new("booting…".to_string());
    let stats = Mutable::new(String::new());
    let about_open = Mutable::new(false);
    // Stats panel visibility: OFF by default; the bottom-left Stats chip toggles
    // it and the choice persists (never auto-open on a coarse device).
    let stats_open = Mutable::new(
        web_sys::window()
            .map(|w| !coarse_pointer(&w) && stored_bool(&w, STATS_STORAGE_KEY).unwrap_or(false))
            .unwrap_or(false),
    );
    loading_log("wasm compiled — booting");

    let app = html!("div", {
        .child(html!("canvas" => HtmlCanvasElement, {
            .class("canvas")
            .after_inserted(clone!(status, stats => move |canvas| {
                if let Err(e) = setup(canvas, status.clone(), stats.clone()) {
                    status.set(format!("setup error: {e:?}"));
                    tracing::error!("setup: {e:?}");
                }
            }))
        }))
        .child(html!("div", {
            .class("hud")
            .text("single-player-game-physics\nW/A/S/D or arrows: roll · Space: jump · click: drop a ball\nright-drag: orbit · wheel: zoom\ntouch: swipe to fling the ball · tap to drop\nsound starts on your first key or click\n")
            .child(html!("span", {
                .text_signal(status.signal_cloned())
            }))
        }))
        .child(html!("div", {
            .class("stats")
            .visible_signal(stats_open.signal())
            .text_signal(stats.signal_cloned())
        }))
        .child(stats_button(&stats_open))
        .child(about_button(&about_open))
        .child(about_modal(&about_open))
    });

    dominator::append_dom(&dominator::body(), app);
    Ok(())
}

/// The bottom-left "Stats" chip: toggles the top-right stats panel (hidden by
/// default). Persists the choice like the graphics settings.
fn stats_button(open: &Mutable<bool>) -> dominator::Dom {
    html!("button", {
        .class("stats-btn")
        .class_signal("active", open.signal())
        .text("Stats")
        .event(clone!(open => move |_: dominator::events::Click| {
            let next = !open.get();
            open.set_neq(next);
            if let Some(w) = web_sys::window() {
                store_bool(&w, STATS_STORAGE_KEY, next);
            }
        }))
    })
}

/// The one-time reduced-quality notice for touch devices.
fn quality_notice_modal(open: &Mutable<bool>) -> dominator::Dom {
    html!("div", {
        .class("settings-overlay")
        .visible_signal(open.signal())
        .event(clone!(open => move |e: dominator::events::Click| {
            let on_backdrop = e
                .dyn_target::<web_sys::Element>()
                .map(|el| el.class_list().contains("settings-overlay"))
                .unwrap_or(false);
            if on_backdrop {
                open.set_neq(false);
            }
        }))
        .global_event(clone!(open => move |e: dominator::events::KeyDown| {
            if e.key() == "Escape" {
                open.set_neq(false);
            }
        }))
        .child(html!("div", {
            .class("settings-modal")
            .child(html!("button", {
                .class("about-close")
                .attr("aria-label", "Close")
                .text("×")
                .event(clone!(open => move |_: dominator::events::Click| open.set_neq(false)))
            }))
            .child(html!("h2", { .text("Display Quality Reduced") }))
            .child(html!("p", {
                .class("notice-text")
                .text("Display quality was reduced to keep the frame rate up on \
                       touch devices. You can adjust the resolution and \
                       anti-aliasing any time in Settings (bottom right).")
            }))
            .child(html!("button", {
                .class("notice-ok")
                .text("OK")
                .event(clone!(open => move |_: dominator::events::Click| open.set_neq(false)))
            }))
        }))
    })
}

/// The bottom-center "About" chip.
fn about_button(open: &Mutable<bool>) -> dominator::Dom {
    html!("button", {
        .class("about-btn")
        .text("About")
        .event(clone!(open => move |_: dominator::events::Click| {
            open.set_neq(true);
        }))
    })
}

/// The About overlay: what this template is + where it lives.
fn about_modal(open: &Mutable<bool>) -> dominator::Dom {
    const REPO: &str = "https://github.com/awsm-fun/awsm-template-physics-singlethreaded";
    let link = |href: &str, label: &str| {
        html!("a", {
            .attr("href", href)
            .attr("target", "_blank")
            .attr("rel", "noopener")
            .text(label)
        })
    };
    html!("div", {
        .class("about-overlay")
        .visible_signal(open.signal())
        .event(clone!(open => move |e: dominator::events::Click| {
            let on_backdrop = e
                .dyn_target::<web_sys::Element>()
                .map(|el| el.class_list().contains("about-overlay"))
                .unwrap_or(false);
            if on_backdrop {
                open.set_neq(false);
            }
        }))
        .global_event(clone!(open => move |e: dominator::events::KeyDown| {
            if e.key() == "Escape" {
                open.set_neq(false);
            }
        }))
        .child(html!("div", {
            .class("about-modal")
            .child(html!("button", {
                .class("about-close")
                .attr("aria-label", "Close")
                .text("×")
                .event(clone!(open => move |_: dominator::events::Click| {
                    open.set_neq(false);
                }))
            }))
            .child(html!("h2", { .text("Single-threaded Physics Demo") }))
            .child(html!("p", {
                .text("A copyable template from the ")
                .child(link("https://awsm.fun", "Awsm"))
                .text(" project — a WebGPU game skeleton whose simulation is \
                       single-threaded (one requestAnimationFrame loop: Box3D \
                       physics + an awsm-renderer scene), hosted on a worker so \
                       the main thread stays free for DOM, input, and live \
                       synthesized audio. The two threads share nothing — just \
                       postMessage.")
            }))
            .child(html!("ul", {
                .child(html!("li", {
                    .text("Rendering: ")
                    .child(link("https://scene.awsm.fun", "AwsmRenderer"))
                }))
                .child(html!("li", {
                    .text("Audio: ")
                    .child(link("https://audio.awsm.fun", "AwsmAudio"))
                }))
                .child(html!("li", {
                    .text("Physics: ")
                    .child(link("https://github.com/erincatto/box3d", "Box3D"))
                }))
            }))
            .child(html!("p", {
                .text("Roll the red ball with WASD, jump with Space, click the table to \
                       drop more balls — every impact is synthesized and spatialized in \
                       real time from the physics contacts. Drag with the right mouse \
                       button to orbit the camera and scroll to zoom; the controls and \
                       the stereo image stay relative to your view from any angle.")
            }))
            .child(html!("p", {
                .class("about-repo")
                .child(link(REPO, "Source on GitHub"))
            }))
            .child(html!("p", {
                .class("about-footer")
                .text("Built with ❤ by David Komer")
            }))
        }))
    })
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

/// Format a [`StatsMsg`] into the panel's text block.
fn format_stats(s: &StatsMsg) -> String {
    format!(
        "engine\n  fps           {:.0}\n  frame time    {:.1} ms\n  physics time  {:.2} ms\n  \
         physics       {:.0} steps/s\n\nballs dropped  {}",
        s.fps, s.frame_ms, s.step_ms, s.sps, s.balls
    )
}

/// Runs once the canvas is in the DOM: size it, transfer it to the worker, and
/// wire input + the worker message bridge.
fn setup(
    canvas: HtmlCanvasElement,
    status: Mutable<String>,
    stats: Mutable<String>,
) -> Result<(), JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;

    // ── Resolution scale (the fill-rate lever) ──────────────────────────────
    let res = Rc::new(Cell::new(ResState {
        scale: initial_scale(&window),
        max_tex: DEFAULT_MAX_TEX,
        css_w: canvas.client_width().max(1) as f64,
        css_h: canvas.client_height().max(1) as f64,
        dpr: window.device_pixel_ratio().max(1.0),
    }));
    let user_pref = Rc::new(Cell::new(stored_scale(&window).is_some()));
    let res_pct = Mutable::new(pct_of(res.get().scale));

    // Anti-aliasing toggles (persisted). Coarse-pointer devices default BOTH off.
    let coarse = coarse_pointer(&window);
    let msaa = Mutable::new(stored_bool(&window, MSAA_STORAGE_KEY).unwrap_or(if coarse {
        false
    } else {
        MSAA_DEFAULT
    }));
    let smaa = Mutable::new(stored_bool(&window, SMAA_STORAGE_KEY).unwrap_or(if coarse {
        false
    } else {
        SMAA_DEFAULT
    }));
    // Reduced-quality notice (touch devices): shown when the coarse seeds applied.
    let show_quality_notice = coarse
        && (stored_scale(&window).is_none()
            || stored_bool(&window, MSAA_STORAGE_KEY).is_none()
            || stored_bool(&window, SMAA_STORAGE_KEY).is_none());
    let notice_open = Mutable::new(false);
    dominator::append_dom(&dominator::body(), quality_notice_modal(&notice_open));

    // Size the backing store, then transfer the canvas to the worker.
    let (w, h) = res.get().backing();
    canvas.set_width(w);
    canvas.set_height(h);
    let offscreen = canvas.transfer_control_to_offscreen()?;
    // Base URL for our same-origin asset fetches (the worker's `blob:` base
    // can't resolve them).
    let base = page_base(&window);

    // The camera yaw main mirrors for the audio listener. The worker owns the
    // real camera; main integrates the SAME orbit deltas with the SAME
    // sensitivity, so the audio heading and the visual stay in lockstep without
    // exchanging the angle (see `CAMERA_ORBIT_SENSITIVITY`).
    let yaw = Rc::new(Cell::new(0.0_f32));

    // Kick off the (async) audio load now; it stays silent until the first gesture.
    let audio: Audio = Rc::new(RefCell::new(None));
    spawn_local(clone!(audio, status, base, yaw => async move {
        match AudioController::load(&base).await {
            Ok(mut controller) => {
                controller.set_camera_yaw(yaw.get());
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
    // and the desired startup anti-aliasing (so the worker seeds it with no
    // round-trip / race).
    let payload = js_sys::Object::new();
    set(&payload, "canvas", &offscreen);
    set(&payload, "origin", &JsValue::from_str(&base));
    set(&payload, "msaa", &JsValue::from_bool(msaa.get()));
    set(&payload, "smaa", &JsValue::from_bool(smaa.get()));
    let transfer = js_sys::Array::new();
    transfer.push(&offscreen);

    // The worker handle, published after spawn so the message handler (installed
    // as the worker's onmessage, hence built before it exists) can post a resize
    // back on `GpuInfo`. Set synchronously right after spawn.
    let worker_ref: Rc<RefCell<Option<Worker>>> = Rc::new(RefCell::new(None));

    // Handle messages coming back from the worker.
    let on_msg = Closure::<dyn FnMut(MessageEvent)>::new(
        clone!(audio, status, stats, res, user_pref, res_pct, worker_ref, notice_open => move |e: MessageEvent| {
            let data = e.data();
            // Audio cue → the WebAudio player.
            if let Ok(msg) = serde_wasm_bindgen::from_value::<AudioMsg>(data.clone()) {
                if let Some(c) = audio.borrow_mut().as_mut() {
                    c.on_audio(msg);
                }
                return;
            }
            // Sampled engine stats → the panel.
            if let Ok(msg) = serde_wasm_bindgen::from_value::<StatsMsg>(data.clone()) {
                stats.set(format_stats(&msg));
                return;
            }
            match serde_wasm_bindgen::from_value::<RenderMsg>(data) {
                Ok(RenderMsg::Progress { message }) => loading_log(&message),
                Ok(RenderMsg::Ready) => {
                    loading_log("first frames rendered — ready");
                    loading_done();
                    status.set("playing — roll · Space jump · click drops a ball · right-drag orbit".into());
                    if show_quality_notice {
                        notice_open.set_neq(true);
                    }
                }
                Ok(RenderMsg::GpuInfo { is_fallback, max_texture_dim }) => {
                    let mut st = res.get();
                    st.max_tex = max_texture_dim.max(1);
                    // A software adapter can't push pixels — seed lower, but
                    // never override a choice the user has already made.
                    if is_fallback && !user_pref.get() && st.scale > FALLBACK_START_SCALE {
                        st.scale = FALLBACK_START_SCALE;
                        res_pct.set(pct_of(FALLBACK_START_SCALE));
                        loading_log(&format!(
                            "software GPU detected — starting at {}% resolution",
                            pct_of(FALLBACK_START_SCALE)
                        ));
                    }
                    res.set(st);
                    if let Some(wk) = worker_ref.borrow().as_ref() {
                        post_resize(wk, &st);
                    }
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
    *worker_ref.borrow_mut() = Some(worker.clone());

    // Keyboard → InputMsg; a keypress also starts audio.
    install_keyboard(&window, worker.clone(), audio.clone())?;
    // Canvas layout size → ResizeMsg (backing store in device pixels).
    install_resize(&window, &canvas, worker.clone(), res.clone())?;
    // Settings modal: resolution slider (→ ResizeMsg) + MSAA/SMAA (→ QualityMsg).
    install_settings(
        window.clone(),
        worker.clone(),
        res.clone(),
        user_pref,
        res_pct,
        msaa,
        smaa,
    )?;
    // Pointer: click drops a ball, right-drag orbits, wheel zooms, touch flings.
    install_pointer(&window, &canvas, worker, audio, yaw)?;

    status.set("loading scene…".into());
    Ok(())
}

fn set(obj: &js_sys::Object, key: &str, value: &JsValue) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), value);
}

/// Observe the canvas element's layout size and relay the backing-store size to
/// the worker at the current resolution scale.
fn install_resize(
    window: &web_sys::Window,
    canvas: &HtmlCanvasElement,
    worker: Worker,
    res: Rc<Cell<ResState>>,
) -> Result<(), JsValue> {
    let win = window.clone();
    let cb = Closure::<dyn FnMut(js_sys::Array)>::new(move |entries: js_sys::Array| {
        let Ok(entry) = entries.get(0).dyn_into::<web_sys::ResizeObserverEntry>() else {
            return;
        };
        let rect = entry.content_rect();
        let mut st = res.get();
        st.dpr = win.device_pixel_ratio().max(1.0);
        st.css_w = rect.width().max(1.0);
        st.css_h = rect.height().max(1.0);
        res.set(st);
        post_resize(&worker, &st);
    });
    let observer = web_sys::ResizeObserver::new(cb.as_ref().unchecked_ref())?;
    observer.observe(canvas);
    cb.forget();
    std::mem::forget(observer);
    Ok(())
}

/// The bottom-right **Settings** button + modal: the resolution slider (→
/// [`ResizeMsg`]) plus the MSAA / SMAA toggles (→ [`QualityMsg`]). Everything
/// here trades GPU cost for image quality; changes apply live and persist.
#[allow(clippy::too_many_arguments)]
fn install_settings(
    window: web_sys::Window,
    worker: Worker,
    res: Rc<Cell<ResState>>,
    user_pref: Rc<Cell<bool>>,
    res_pct: Mutable<u32>,
    msaa: Mutable<bool>,
    smaa: Mutable<bool>,
) -> Result<(), JsValue> {
    let open = Mutable::new(false);

    let button = html!("button", {
        .class("settings-btn")
        .text("Settings")
        .event(clone!(open => move |_: dominator::events::Click| open.set_neq(true)))
    });

    let resolution_row = html!("div", {
        .class("settings-row")
        .child(html!("label", {
            .class("settings-label")
            .text_signal(res_pct.signal().map(|p| format!("Resolution — {p}%")))
        }))
        .child(html!("input" => HtmlInputElement, {
            .class("res-slider")
            .attr("type", "range")
            .attr("min", "50")
            .attr("max", "100")
            .attr("step", "5")
            .attr("aria-label", "render resolution")
            .prop_signal("value", res_pct.signal().map(|p| p.to_string()))
            .with_node!(el => {
                .event(clone!(res, user_pref, res_pct, window, worker => move |_: dominator::events::Input| {
                    let pct = el.value().parse::<f64>().unwrap_or(100.0);
                    let scale = (pct / 100.0).clamp(MIN_SCALE, MAX_SCALE);
                    let mut st = res.get();
                    st.scale = scale;
                    res.set(st);
                    user_pref.set(true);
                    res_pct.set(pct_of(scale));
                    store_scale(&window, scale);
                    post_resize(&worker, &st);
                }))
            })
        }))
    });

    // A labelled checkbox row driving one AA toggle. Both flags are re-read on
    // change so the posted `QualityMsg` always carries the current pair.
    let aa_row = |label: &str, flag: Mutable<bool>, key: &'static str, is_msaa: bool| {
        html!("div", {
            .class("settings-row")
            .child(html!("label", { .class("settings-label").text(label) }))
            .child(html!("input" => HtmlInputElement, {
                .class("settings-toggle")
                .attr("type", "checkbox")
                .attr("aria-label", label)
                .prop_signal("checked", flag.signal())
                .with_node!(el => {
                    .event(clone!(flag, msaa, smaa, window, worker => move |_: dominator::events::Change| {
                        let on = el.checked();
                        flag.set_neq(on);
                        store_bool(&window, key, on);
                        let (m, s) = if is_msaa { (on, smaa.get()) } else { (msaa.get(), on) };
                        post(&worker, &QualityMsg::AntiAlias { msaa: m, smaa: s });
                    }))
                })
            }))
        })
    };
    let msaa_row = aa_row("MSAA 4×", msaa.clone(), MSAA_STORAGE_KEY, true);
    let smaa_row = aa_row("SMAA", smaa.clone(), SMAA_STORAGE_KEY, false);

    let modal = html!("div", {
        .class("settings-overlay")
        .visible_signal(open.signal())
        .event(clone!(open => move |e: dominator::events::Click| {
            let on_backdrop = e
                .dyn_target::<web_sys::Element>()
                .map(|el| el.class_list().contains("settings-overlay"))
                .unwrap_or(false);
            if on_backdrop {
                open.set_neq(false);
            }
        }))
        .global_event(clone!(open => move |e: dominator::events::KeyDown| {
            if e.key() == "Escape" {
                open.set_neq(false);
            }
        }))
        .child(html!("div", {
            .class("settings-modal")
            .child(html!("button", {
                .class("about-close")
                .attr("aria-label", "Close")
                .text("×")
                .event(clone!(open => move |_: dominator::events::Click| open.set_neq(false)))
            }))
            .child(html!("h2", { .text("Settings") }))
            .child(resolution_row)
            .child(msaa_row)
            .child(smaa_row)
            .child(html!("p", {
                .class("settings-hint")
                .text("Adjust these settings if your framerate is low.")
            }))
        }))
    });

    dominator::append_dom(&dominator::body(), button);
    dominator::append_dom(&dominator::body(), modal);
    Ok(())
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

/// Wire pointer input. A **left click drops a ball**: the click point is posted
/// as an NDC [`DropMsg`] (the worker unprojects it onto the table). A
/// **right-button drag orbits** the camera ([`CameraMsg::Orbit`]) and updates
/// the audio listener; the **wheel zooms** ([`CameraMsg::Zoom`]). A **touch
/// swipe flings the ball** ([`InputMsg::Fling`]); a short still touch stays a
/// tap → drop.
fn install_pointer(
    window: &web_sys::Window,
    canvas: &HtmlCanvasElement,
    worker: Worker,
    audio: Audio,
    yaw: Rc<Cell<f32>>,
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
    let down = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, audio, swipe => move |e: web_sys::PointerEvent| {
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
    // orbit delta goes to the worker (the visual); its dx also advances OUR yaw
    // with the same sensitivity → the audio listener, keeping the two in lockstep.
    let move_ = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(
        clone!(dragging, worker, audio, yaw, swipe => move |e: web_sys::PointerEvent| {
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
                let dx = e.movement_x() as f32;
                let dy = e.movement_y() as f32;
                let y = yaw.get() - dx * CAMERA_ORBIT_SENSITIVITY;
                yaw.set(y);
                if let Some(c) = audio.borrow_mut().as_mut() {
                    c.set_camera_yaw(y);
                }
                post(&worker, &CameraMsg::Orbit { dx, dy });
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
        clone!(worker => move |e: web_sys::WheelEvent| {
            e.prevent_default();
            post(&worker, &CameraMsg::Zoom { dy: e.delta_y() as f32 });
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
/// compile-time env (`task dev` sets it to the side media server), then the
/// directory the page is served from (kept, so it's correct under a GitHub Pages
/// project base like `/<repo>/`).
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
    let origin = location.origin().unwrap_or_default();
    let mut path = location.pathname().unwrap_or_default();
    if let Some(idx) = path.rfind('/') {
        path.truncate(idx);
    }
    format!("{}{}", origin.trim_end_matches('/'), path)
}
