//! **single-player-game-physics (single-threaded)** — a copyable template for
//! *playing* a scene exported from `awsm-scene`, rendered with `awsm-renderer`
//! and simulated with a Box3D physics world.
//!
//! The **simulation is single-threaded**: one `requestAnimationFrame` loop, no
//! parallelism, no shared-memory coordination. That loop just doesn't run on the
//! *main* thread — it runs on a **worker**, so DOM/input/audio work on main can
//! never steal a render frame (and vice versa). The two threads share nothing;
//! they only exchange `postMessage`s on cold paths.
//!
//! - **Main thread** ([`main_thread`]): owns the DOM (Dominator), captures
//!   input, and drives the WebAudio player. Spawns the worker with a transferred
//!   `OffscreenCanvas`, forwards input as messages, and plays back the worker's
//!   audio cues.
//! - **Worker thread** ([`game`]): owns `awsm-renderer` on the `OffscreenCanvas`
//!   AND the Box3D world ([`physics`]). One loop: step physics at a fixed
//!   timestep, interpolate the poses, render — reading the forwarded input and
//!   emitting audio cues back to main.
//!
//! Because there's no shared `WebAssembly.Memory`, there are no atomics, no
//! `build-std`, no nightly, and **no COOP/COEP headers** — the worker
//! instantiates its own memory from the same module (posting a
//! `WebAssembly.Module` needs no cross-origin isolation). The simulation stays a
//! single thread of execution, which is all this scene needs.

pub mod audio;
pub mod bootstrap;
pub mod game;
pub mod main_thread;
pub mod physics;
pub mod protocol;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::js_sys;

/// `true` when running inside a `DedicatedWorkerGlobalScope`.
pub fn is_worker_scope() -> bool {
    js_sys::global()
        .dyn_into::<web_sys::DedicatedWorkerGlobalScope>()
        .is_ok()
}

/// Single entry point. `wasm-bindgen` runs this automatically on every `init()`
/// (main thread *and* the worker). On the main thread it boots the app; in the
/// worker it does nothing — the worker's real work is triggered explicitly by
/// the bootstrap JS calling [`game_worker_start`] after init returns.
#[wasm_bindgen(start)]
pub fn boot() -> Result<(), JsValue> {
    install_tracing();
    if is_worker_scope() {
        Ok(())
    } else {
        tracing::info!("single-player-game-physics (single-threaded): main-thread boot");
        main_thread::start()
    }
}

/// The worker-side entry point the bootstrap JS calls after init. `payload` is
/// `{ canvas: OffscreenCanvas, origin: String, msaa: bool, smaa: bool }`.
#[wasm_bindgen]
pub fn game_worker_start(payload: JsValue) -> Result<(), JsValue> {
    install_tracing();
    game::start(payload)
}

/// Install the browser-console tracing subscriber (idempotent — safe to call on
/// the main thread and in the worker).
pub fn install_tracing() {
    use tracing_subscriber::prelude::*;
    // Surface panic messages in the console — with `panic = abort` on wasm a
    // panic otherwise dies as an opaque `RuntimeError: unreachable`.
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&format!("PANIC: {info}")));
    }));
    // The default `fmt` time formatter calls `SystemTime::now()`, which panics
    // on wasm32; `without_time` strips it (the console prepends its own time).
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .without_time()
        .with_writer(tracing_web::MakeWebConsoleWriter::new())
        .with_target(false);
    // Unfiltered, every dependency's debug!/trace! (the renderer is chatty)
    // funnels through the console writer — console logging is expensive enough
    // to show up as frame hiccups. Keep our own crate at debug (the audio-cue
    // lines are how headless checks observe impacts — they only fire on
    // contacts, which is cheap).
    let filter = tracing_subscriber::EnvFilter::new("info,single_player_game_physics=debug");
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .try_init();
}
