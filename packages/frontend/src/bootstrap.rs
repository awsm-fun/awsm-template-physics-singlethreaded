//! Spawning the game worker — the **shared-nothing** bootstrap.
//!
//! The worker runs the whole game loop (renderer + physics) on a transferred
//! `OffscreenCanvas`. The one subtlety is getting the same wasm code running in
//! it. The recipe (from `wasm-bindgen`'s `raytrace-parallel`, minus the shared
//! memory):
//!
//! 1. Build a `Worker` from an inline blob URL (no separate `worker.js` — the
//!    source travels inside the wasm bundle).
//! 2. Post it `{ wasm_module, glue_url, payload }`:
//!    - `wasm_module` = [`wasm_bindgen::module`] — the *compiled* artifact,
//!      structured-cloneable, so the worker skips re-compiling the multi-MB
//!      binary. **Crucially, we do NOT post `wasm_bindgen::memory`** — the
//!      worker instantiates its **own** linear memory. That's the whole point:
//!      no shared memory means no `SharedArrayBuffer`, so **no cross-origin
//!      isolation (COOP/COEP) is required** — posting a `WebAssembly.Module` is
//!      not gated by it.
//!    - `payload` = the per-worker startup data (here
//!      `{ canvas, origin, msaa, smaa }`).
//! 3. Worker side: `await init({ module_or_path: wasm_module })` (no `memory`
//!    arg → fresh memory), then call [`crate::game_worker_start`].

use wasm_bindgen::prelude::*;
use web_sys::js_sys;
use web_sys::{Blob, BlobPropertyBag, Url, Worker, WorkerOptions, WorkerType};

/// Spawn the game worker, handing it `payload` (delivered to
/// [`crate::game_worker_start`]) and a structured-clone `transfer` list (the
/// `OffscreenCanvas`, moved zero-copy). `on_message` is installed as the
/// worker's `onmessage` so the spawner can observe what it posts back. The
/// returned [`Worker`] is how the spawner posts *into* the worker later.
pub fn spawn_worker(
    payload: &JsValue,
    transfer: &js_sys::Array,
    on_message: &js_sys::Function,
) -> Result<Worker, JsValue> {
    let blob_options = BlobPropertyBag::new();
    blob_options.set_type("application/javascript");
    let parts = js_sys::Array::new_with_length(1);
    parts.set(0, JsValue::from_str(WORKER_BOOTSTRAP_JS));
    let blob = Blob::new_with_str_sequence_and_options(&parts.into(), &blob_options)?;
    let blob_url = Url::create_object_url_with_blob(&blob)?;

    let opts = WorkerOptions::new();
    opts.set_type(WorkerType::Module);
    let worker = Worker::new_with_options(&blob_url, &opts)?;
    let _ = Url::revoke_object_url(&blob_url);

    worker.set_onmessage(Some(on_message));
    let onerror = Closure::<dyn FnMut(JsValue)>::new(|err: JsValue| {
        // Pull the useful fields out of the ErrorEvent — logging the bare
        // object prints as `[object ErrorEvent]`, hiding the actual failure.
        let detail = err
            .dyn_ref::<web_sys::ErrorEvent>()
            .map(|e| format!("{} ({}:{})", e.message(), e.filename(), e.lineno()))
            .unwrap_or_default();
        web_sys::console::error_2(&JsValue::from_str(&format!("worker error: {detail}")), &err);
    });
    worker.set_onerror(Some(onerror.as_ref().unchecked_ref::<js_sys::Function>()));
    onerror.forget();

    let init_msg = js_sys::Object::new();
    set(&init_msg, "kind", &JsValue::from_str("awsm-init"));
    set(&init_msg, "wasm_module", &wasm_bindgen::module());
    set(&init_msg, "glue_url", &JsValue::from_str(&bundle_url()));
    set(&init_msg, "payload", payload);
    if transfer.length() == 0 {
        worker.post_message(&init_msg)?;
    } else {
        worker.post_message_with_transfer(&init_msg, transfer)?;
    }

    Ok(worker)
}

fn set(obj: &js_sys::Object, key: &str, value: &JsValue) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), value);
}

/// Worker bootstrap JS. Instantiates the posted module with its **own** memory,
/// then runs the game.
///
/// `init({ module_or_path })` is the `wasm-bindgen` `--target web` default
/// export's options form. Passing NO `memory` is what makes the worker allocate
/// a fresh linear memory (as opposed to attaching to a shared one).
pub const WORKER_BOOTSTRAP_JS: &str = r#"
self.onmessage = async (e) => {
    const d = e.data;
    if (!d || d.kind !== "awsm-init") return;
    const { wasm_module, glue_url, payload } = d;
    try {
        const wbg = await import(glue_url);
        await wbg.default({ module_or_path: wasm_module });
        // boot() ran during init (worker scope -> no-op). Now start the game.
        wbg.game_worker_start(payload);
    } catch (err) {
        self.postMessage({ kind: "awsm-init-error", message: (err && err.message) ? err.message : String(err) });
    }
};
"#;

/// Recover the JS-glue bundle URL from the page (Trunk hashes the filename in
/// release builds, so it can't be hard-coded). Falls back to `import.meta.url`
/// outside a DOM context.
#[wasm_bindgen(inline_js = r#"
export function awsm_bundle_url() {
    if (typeof document !== "undefined") {
        const scripts = document.querySelectorAll("script[type=module]");
        for (const s of scripts) {
            const t = s.textContent || "";
            const m = t.match(/from\s+['"]([^'"]+\.js)['"]/);
            if (m) return new URL(m[1], location.href).href;
        }
    }
    return import.meta.url;
}
"#)]
extern "C" {
    fn awsm_bundle_url() -> String;
}

/// The resolved JS-glue bundle URL (see [`awsm_bundle_url`]).
pub fn bundle_url() -> String {
    awsm_bundle_url()
}
