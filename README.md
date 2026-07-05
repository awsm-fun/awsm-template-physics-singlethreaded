# AWSM Template
## Spatial audio and single-threaded physics with Box3D

## **▶ [Live site](https://awsm.fun/experiments/box3d-singlethreaded)**
_the same build also deploys to GitHub Pages on every merge to `main`._

----
A copyable **template** from the [**Awsm**](https://awsm.fun) project: it
*plays* a scene authored in the [scene editor](https://scene.awsm.fun),
rendered with [`awsm-renderer`](https://crates.io/crates/awsm-renderer), with
[Box3D](https://github.com/erincatto/box3d) physics (Erin Catto's C engine,
**compiled into the same wasm module**) and spatial sound authored in the
[audio editor](https://audio.awsm.fun) (played by
[`awsm-audio-player`](https://crates.io/crates/awsm-audio-player)).

The **simulation is single-threaded** — one `requestAnimationFrame` loop, no
parallelism, no shared-memory coordination. It just runs on a **worker** instead
of the main thread: the worker owns the renderer *and* the physics world, while
the main thread keeps the DOM, input, and audio. So a heavy render/physics frame
can never jank input or audio, and vice versa — but the two threads **share
nothing**, they only trade `postMessage`s on cold paths. That means no shared
`WebAssembly.Memory`, no atomics, no `build-std`, and **no COOP/COEP headers**
(posting a `WebAssembly.Module` needs no cross-origin isolation — only
`SharedArrayBuffer` does): it's plain stable-Rust wasm that runs on any static
host. Box3D's SIMD is still real — its SSE2 math runs on **wasm simd128**.

The shipped scene is a **red** ball on a felt table with wooden rails. It drops
in and bounces; you roll it around and make it hop, and it makes 3D sound — a
rolling rumble that tracks its speed, plus a knock when it hits a rail and a thud
when it lands. **Click anywhere on the table to drop another (silver) ball
there** — every drop, bounce, and ball-on-ball clack cues a sound too, and the
top-right panel shows what the engine is doing while it happens. The point
isn't the gameplay — it's the skeleton: editor export → player loader →
renderer, **physics colliders derived from the scene's own collider nodes**,
with a fixed-timestep sim feeding interpolated transforms to the renderer and
audio cues to the WebAudio player.

## Run it

```sh
git submodule update --init   # vendor/box3d (task dev runs this too)
task dev                      # trunk serve on http://127.0.0.1:9000
```

Besides the usual Rust-wasm toolchain (`task`, `trunk`, **stable** Rust via
`rust-toolchain.toml`), building needs a **wasm-capable clang** for the Box3D C
sources: Apple's clang has no wasm backend, so on macOS `brew install llvm`
(build + Taskfile probe the Homebrew path automatically, or point
`CC_wasm32_unknown_unknown` at one). Linux distro clang works as-is.

Open it in a browser with **WebGPU** (recent Chrome/Edge, Safari 18+). No
`SharedArrayBuffer` and no special headers required — it's single-threaded.

**Controls:**

| Input | Action |
|---|---|
| **W/A/S/D** or **arrow keys** | roll the red ball |
| **Space** | jump |
| **click** | drop a silver ball at the clicked table spot (up to 200) |
| **right-drag** | orbit the camera |
| **wheel** | zoom |
| **touch: swipe** | fling the ball · **tap** drops a ball |

W/A/S/D stay **camera-relative** at any orbit angle, and the spatial-audio
stereo image tracks the view too (see [Audio](#audio)).

Sound starts on your **first key or click** (browser autoplay policy requires a
user gesture).

Other tasks: `task build` (production build into `dist/`), `task check`
(`cargo check`, no serve), `task lint` (clippy), `task fmt` (format),
`task clean`; `cargo test -p box3d-sys` runs the native Box3D + FFI
struct-layout tests. CI (`.github/workflows/ci.yml`) runs `cargo fmt --check`, the
`box3d-sys` host tests, and `task lint` on every push and PR; on merge to
`main`, `.github/workflows/deploy.yml` builds and publishes `dist/` to GitHub
Pages.

## The loop

The **worker** runs the whole game: a single `requestAnimationFrame` loop
(`packages/frontend/src/game.rs`) owns both the renderer and the physics world.
The **main thread** only owns the DOM, input, and audio, and talks to the worker
by `postMessage` (see [Data flow](#data-flow)).

| Module | File | Thread | Responsibility |
|---|---|---|---|
| **App shell** | `packages/frontend/src/main_thread.rs` | main | Owns the DOM (built with **Dominator**) + keyboard/pointer input, the settings/stats UI, and the **WebAudio player** (`packages/frontend/src/audio.rs`). Sizes the canvas, `transferControlToOffscreen`, and spawns the worker. Forwards input as messages; plays back the worker's audio cues. |
| **Worker spawn** | `packages/frontend/src/bootstrap.rs` | main | Builds the inline-blob worker and posts it the compiled `WebAssembly.Module` (**not** memory) + the `OffscreenCanvas`. Shared-nothing, so no COOP/COEP. |
| **Game loop** | `packages/frontend/src/game.rs` | worker | Builds **awsm-renderer** on the `OffscreenCanvas`, loads `scene.toml` via `load_scene_for_player`, derives the collider list (`derive_physics`), and every frame: steps physics, **interpolates** each body's pose into the transform arena, updates the orbit camera, renders, and posts audio cues + sampled stats to main. Applies the input/camera/drop/resize/quality messages it receives. |
| **Physics** | `packages/frontend/src/physics.rs` | worker | A **Box3D** `World` built from the scene's colliders (static table + walls, dynamic ball), stepped at a **fixed step** (`SIM_HZ`, 240 Hz default) with wall-clock catch-up. Reads input, keeps a prev/curr pose per body, and emits audio cues from contact/hit events. Box3D runs `workerCount = 1` (its inline serial path). |
| **Audio** | `packages/frontend/src/audio.rs` | main | Plays the `awsm-audio` export and drives it live from the physics cues (roll / wall-hit / land / ball-clack) the worker posts back. |
| **Protocol** | `packages/frontend/src/protocol.rs` | both | The message enums (serde) both threads speak, plus the shared constants (`SIM_HZ`, `CAMERA_ORBIT_SENSITIVITY`, the `HELD_*` bits). |

### Data flow

The two threads **share nothing** — every crossing is a `postMessage` on a cold
path (input events, layout changes, audio cues, ~1 Hz stats), never per-frame
render data:

```
  main thread                                worker thread
  ───────────                                ─────────────
  DOM key/pointer ──▶ InputMsg / CameraMsg / DropMsg ──▶ Input + OrbitCamera + drop queue
  ResizeObserver  ──▶ ResizeMsg  ────────────────────▶ resize the OffscreenCanvas
  Settings UI     ──▶ QualityMsg ────────────────────▶ pending AA (applied off the render path)
  WebAudio player ◀── AudioMsg ◀──┐
  loading / status ◀── RenderMsg ◀┤          game loop, each rAF frame:
  stats panel     ◀── StatsMsg ◀──┘            drain drops → step World × N (fixed dt)
                                               → interpolate poses → write arena → render
```

Both threads integrate the **same** camera yaw from the **same** orbit deltas
with the **same** `CAMERA_ORBIT_SENSITIVITY` — the worker to aim the view, main
to orbit the audio listener — so the visual and the stereo image stay in
lockstep **without ever exchanging the angle**.

Because physics and rendering share one thread and one clock, the classic
fixed-timestep interpolation is **exact**: physics keeps just a **prev/curr
pose** per body, and the loop blends them by a single `alpha` — no jitter
buffer, no cross-thread pose handoff, no display-cursor bookkeeping.

**Why a fixed timestep.** A rigid-body solver is only stable and deterministic
at a constant `dt`. So the world always advances in `dt = 1/SIM_HZ` chunks, and
the loop runs *as many* of them as elapsed real time bought since the last frame
— a bounded accumulator (multiple sub-steps per frame, capped to avoid a spiral
of death). After the catch-up, whatever fraction of a step is left over becomes
the interpolation `alpha` that blends the previous step's pose toward the
current one, so motion stays smooth regardless of the display's refresh rate.
Pose is stored as position + quaternion (not a baked matrix) so the blend can
`lerp` translation and `slerp` rotation correctly.

**One rate knob: `SIM_HZ`.** The step rate lives in a single `protocol::SIM_HZ`
(default **240 Hz**). It's a genuine one-number tuning knob because nothing else
is written in "ticks": physical quantities (velocities, gravity, damping,
`MOVE_ACCEL` as an acceleration) are `dt`-invariant, and the few tick counts
(`ROLL_EVERY`, `IMPACT_COOLDOWN`, `MAX_SUBSTEPS`) are *derived* from `SIM_HZ`,
so the feel is identical at any rate. Raising it lowers the interpolation
latency floor (`~1000/SIM_HZ` ms: ≈4.2 ms at 240, ≈16.7 at 60) and tightens
collisions, for a linear CPU cost (~0.2 ms per step here). The rule when
extending: **express any new per-step force per second and multiply by the step
`dt`** — never bake the rate into a constant.

Audio lives on the main thread (WebAudio is main-thread-only anyway) — which is
the whole reason the game loop moved off it. Physics owns the ball's motion +
contacts, so it decides *when* sounds fire and how loud, posting `AudioMsg` cues
that main hands to `packages/frontend/src/audio.rs` as live WebAudio parameter
changes. Keeping the audio player on main matters: building a voice's WebAudio
graph on a wall-hit/landing is a synchronous ~few-ms chunk of work, and if it
shared a thread with the renderer it would show up as a frame hitch — the
split gives each side its own thread to spend.

### Colliders from the scene

`awsm-scene` can author **collider nodes** (box / sphere / capsule / …) right in
the scene. `game::derive_physics` walks the loaded tree, composes each collider's
world transform by hand (parent × local), **folds the accumulated per-axis scale
into the shape extents** (a physics collider has no scale of its own — its
placement is a rotation + translation), and hands the list to `World::new`. The
node named **`Ball`** becomes the dynamic body; everything else is static. On the
Box3D side a box becomes a convex hull (`b3MakeBoxHull`), sphere/capsule are
primitives, and cylinder/cone map to Box3D's generated hulls. Move impulses are
scaled by the ball's mass so the feel is independent of the authored ball size,
and a fall-through safety net + bullet CCD keep the ball on the low-railed table.

### Box3D — C physics inside the same wasm module

Box3D is vendored as a **git submodule** (`vendor/box3d`) and compiled by
`packages/box3d-sys/build.rs` (the `cc` crate + a wasm-capable clang) **into the same
wasm module as the Rust** — no bridge, no copies, no Emscripten. What makes that
work (all in `packages/box3d-sys/` + BOX3D.md):

- a ~5-file **shim libc** (`shim/include/`) plus stb_sprintf-backed
  `printf`/`snprintf` and a real `qsort` (`shim/wasm_libc.c`) — there is no
  sysroot on `wasm32-unknown-unknown`;
- Rust-side symbols (`packages/box3d-sys/src/wasm_shim.rs`): an allocator over the Rust
  global allocator, `libm` transcendentals, spinlock mutexes, and trap-loud
  stubs for the pthread scheduler that never runs here (Box3D's `timer.c` /
  `scheduler.c` are excluded on wasm);
- **no `-matomics`.** Single-threaded, so Box3D's `__atomic_*` builtins lower to
  plain non-atomic ops and the module stays off shared memory entirely. The C
  objects are built with `-mbulk-memory -mmutable-globals` (matching modern
  Rust's default wasm target features) and `-ffp-contract=off` for Box3D's
  cross-platform determinism;
- **SIMD**: `-msimd128 -DB3_CPU_WASM` routes Box3D onto its SSE2 solver path,
  with `shim/include/emmintrin.h` mapping those intrinsics onto **wasm
  simd128** — measured ~20% faster stepping than scalar, with **bit-identical**
  results (`BOX3D_WASM_SCALAR=1 task dev` builds the scalar variant for A/B). The
  Rust side is built `+simd128` too (`.cargo/config.toml`), so glam/renderer
  math gets the vector path as well;
- **single-threaded solver.** The world is created with `workerCount = 1`, so
  Box3D's `b3ParallelFor` takes its inline serial path — no task scheduler, no
  enqueue/finish callbacks, no worker pool.

### Click-to-drop balls

A click drops a silver ball at the clicked spot, exercising the whole stack at
runtime: the click handler turns the point into NDC and pushes it onto a shared
drop queue; the game loop (which owns the camera) **unprojects** it onto the
tabletop and calls `World::drop_ball`, which creates the body. The loop then
mints the visual as a **mesh duplicate** of the ball
(`duplicate_mesh_with_transform` — shared GPU geometry + material), driven
through its own transform-arena slot like every other body. The **player ball
wears a cloned, red-tinted material** (swapped in up-front so duplicates inherit
the original silver). Dropped balls are deliberately cheaper than the player: no
CCD, sleep allowed. Impact audio for *all* balls comes from Box3D's **hit
events** (contact point + approach speed → position + intensity), so every drop
thud, rail knock, and ball-on-ball clack is audible; the cap is 200 balls (see
the rstar note in BOX3D.md).

### Loading screen + stats

The loading overlay lives in the static `packages/frontend/index.html` (visible while the wasm
bundle itself downloads/compiles — "loading code…"), then streams every load
phase as it happens (streamed from the worker as `RenderMsg::Progress`): device
creation, the scene fetch, each `awsm-renderer` loader phase
(materials/meshes/textures/pipelines), the GPU commit stats, and the audio load.
The top-right stats panel shows fps / steps-per-second / ms-per-frame /
ms-per-step: the **worker** samples its own frame/step/CPU-µs counters ~1 Hz and
posts a `StatsMsg`; main just formats it into the panel. Toggle it with the
bottom-left **Stats** chip.

## The build (why it's plain)

This is an ordinary wasm build with a private linear memory — nothing special.
Three things a shared-memory threaded wasm build would need are simply **absent**:

- **stable Rust, no `-Z build-std`.** `rust-toolchain.toml` pins `stable`; there
  are no atomics to recompile `std` for.
- **no shared-memory / atomics link args and no env `RUSTFLAGS`.** The only wasm
  build flags live in `.cargo/config.toml` — the two dep cfgs
  (`web_sys_unstable_apis`, `getrandom` wasm backend) and `+simd128` — and they
  apply to every build (trunk, `cargo check`, rust-analyzer), so there's nothing
  to repeat or keep in sync.
- **no COOP/COEP headers and no `coi-serviceworker`.** Without a shared
  `WebAssembly.Memory` there's nothing for `crossOriginIsolated` to gate, so the
  build runs on any static host as-is.

The Box3D C objects still need a **wasm-capable clang** (`packages/box3d-sys/build.rs`
compiles them with `-msimd128 -mbulk-memory -mmutable-globals`, no `-matomics`),
so `task preflight` (a dependency of dev/build/check/lint) checks the submodule +
clang up front with an actionable error.

**Dev builds optimize dependencies.** `Cargo.toml` sets
`[profile.dev.package."*"] opt-level = 3`, so the renderer / audio crates
are compiled optimized even in `task dev`, while our own crate stays at
`opt-level = 0` for fast, debuggable incremental rebuilds. This matters: at
`opt-level = 0` the audio player's per-impact WebAudio graph build runs ~30–60 ms
on the main thread, so unoptimized you feel control lag when the ball hits rails
repeatedly; optimized it's a few ms. The one-time cost is a slower *first*
compile. (`box3d-sys`'s Rust shim is also pinned to `opt-level = 3` — it's on the
hot path of every Box3D malloc/transcendental.)

## Audio

The SFX are an `awsm-audio` export under `media/audio/` (`project.toml` + the
rolling-sound `.wasm` worklet). `packages/frontend/src/audio.rs` fetches them same-origin and drives
them from the physics `AudioMsg` cues. The three sounds are **synthesized
live** — the `.wav` bounces in the export are unused at runtime, so the only asset
actually fetched is the worklet `.wasm`. (That worklet's Rust source — the DSP for
the rolling rumble — is in `packages/audio-worklet-roll/`, compiled to the shipped `.wasm`
via the [audio editor](https://audio.awsm.fun)'s worklet toolchain.)

It uses **one `awsm-audio-player` `Player`** — a single `AudioContext` + master bus
mixing many concurrent voices (the `Player` mixer model):

- The **roll** is a sustaining DSP-worklet rumble — the player's persistent
  `play()` instance (looping), so impacts never cut it. Its loudness + timbre + 3D
  position are nudged continuously with `set_param_live`.
- **wall-hit** and **land** are **one-shots** fired as independent voices
  (`play_voice_with`). A voice doesn't stop the roll or each other, and its
  per-trigger statics (intensity → gain, hardness → filter cutoff, table position
  → stage panner) are baked in as build-time **overrides** so the sound is correct
  from its first sample. A spent one-shot self-decays to silence but its source
  nodes (oscillators) keep running, so `packages/frontend/src/audio.rs` **frees each voice** once its
  tail finishes (`stop_voice`, ~0.6 s) — otherwise the idle oscillators pile up
  into a constant hum; a `max_voices` cap is the backstop.

Spatialization uses an **orbiting listener** over a tabletop stage: source
positions sit at the ball's world (x, z) on the felt, and a virtual listener
hovers low over the table, orbited around the table center at the **camera's
azimuth** and looking inward — so left/right *and* near/far track the view from
any angle. On every orbit drag the listener is re-sent (`set_listener_live`), so
every ringing voice re-spatializes to the new heading. See `listener` +
`stage_pos` in `packages/frontend/src/audio.rs`.

**Runtime control surface** — what the game drives (the roll's columns live via
`set_param_live`; the impacts' as `play_voice_with` overrides at trigger time):

| Sound | Node (label) | Param | Driven by |
|---|---|---|---|
| roll | worklet | `speed` | normalized roll speed `0..1` (impact density + ring length) |
| roll | `roll_LEVEL` | `gain` | roll speed (`0` ⇒ silent at rest) |
| roll | `roll_PANNER` | `positionX/Y/Z` | ball position → audio stage |
| wall-hit | `hit_LEVEL` | `gain` | impact intensity `0..1` (from approach speed) |
| wall-hit | `hit_FILTER` | `frequency` | impact intensity (harder ⇒ brighter) |
| wall-hit | `hit_PANNER` | `positionX/Y/Z` | impact position → audio stage |
| land | `land_LEVEL` | `gain` | landing intensity `0..1` (from drop speed) |
| land | `land_FILTER` | `frequency` | landing intensity |
| land | `land_PANNER` | `positionX/Y/Z` | landing position → audio stage |

The control nodes are resolved by label/kind at load, so you can re-export the
audio project and the wiring still finds them. Every other DSP knob (the
worklet's `roughness` / `body_hz` / `brightness`, the impact mode tunings, …)
keeps its authored value — tweak those in the [audio editor](https://audio.awsm.fun).

**Gameplay → sound mapping** lives in `packages/frontend/src/physics.rs`: roll speed is the
ball's grounded horizontal speed; impacts are classified from Box3D's begin-touch
contact events (each shape carries a floor/wall role in its user data) plus the
world's hit events, with intensity from the approach speed, debounced by a speed
gate + cooldown so shoving the ball against a rail doesn't machine-gun the knock.

## Swapping in your own scene

1. Export a player bundle from the [scene editor](https://scene.awsm.fun) (a `scene.toml` plus any asset bins),
   and optionally a project from the [audio editor](https://audio.awsm.fun).
2. Drop them into `media/` — the scene export as `media/bundle/` (`scene.toml`
   + `assets/`), the audio project as `media/audio/`. That layout is exactly
   what gets served: the `copy-dir` links in `packages/frontend/index.html`
   put it in `dist/` same-origin, and `task dev`'s side media server serves
   `media/` as-is. External scene assets go into the `HttpAssets` source passed
   to `load_scene_for_player`.
3. Author **collider nodes** in your scene for anything physical. They're derived
   automatically (`game::derive_physics`) — the node named `Ball` is the dynamic
   body; rename/extend that convention as needed and adjust the camera framing.
4. In `packages/frontend/src/physics.rs`, tune the materials / impulses / impact mapping; in
   `packages/frontend/src/audio.rs`, point the voices at your own sample names + control-node labels.

## Dependencies

Physics is **[Box3D](https://github.com/erincatto/box3d)** (MIT), vendored as a
git submodule at `vendor/box3d` and built by the local `packages/box3d-sys` crate — see
[Box3D — C physics inside the same wasm module](#box3d--c-physics-inside-the-same-wasm-module)
and `BOX3D.md` for the integration reference.

All Rust crates are from **crates.io** — `awsm-renderer` family `0.15`,
`awsm-audio` family `2.5`, `dominator` / `futures-signals`, `glam` `0.32`,
`libm` (the wasm libc shim's transcendentals) — with **one** exception:
`dominator` is redirected via `[patch.crates-io]` to an upstream git rev. The
published `dominator 0.5.38` does not compile under `web_sys_unstable_apis` (the
cfg WebGPU needs flips mouse-coord types to `f64`); the patch is the same one the
awsm-renderer repo uses, and can be dropped once a fixed dominator is released.
