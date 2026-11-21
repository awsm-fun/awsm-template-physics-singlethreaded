//! The Stats panel's breakdowns, measured on the game worker only while main
//! asks (the `DETAIL_*` bits of a [`StatsDetailMsg`]):
//!
//! * `DETAIL_RENDERER`: on every frame, the phases of the frame function are
//!   timed (one clock read between each) and awsm-renderer's CPU scopes are on
//!   (`logging.cpu = SubFrame`, a field the renderer checks at every scope).
//!   Otherwise the tier is `Off` and nothing reads the clock.
//! * `DETAIL_GPU`: the per-pass GPU times are passed on. Measuring them is
//!   already paid for: the frame total the panel always shows comes from the
//!   same timestamp queries (`logging.gpu = Frame` times every pass).
//! * `DETAIL_PHYSICS`: the phases of each physics step are timed
//!   ([`StepTimer`], handed to `World::step` only then).
//!
//! Renderer and physics share this worker, so the renderer's phases leave the
//! physics steps out: renderer time per frame + physics time per step × steps
//! per frame = the worker's frame, with nothing counted twice.
//!
//! The means go to main with each ~1 Hz `StatsMsg`.
//!
//! [`StatsDetailMsg`]: crate::protocol::StatsDetailMsg

use std::collections::HashMap;

use awsm_renderer::debug::TimingTier;
use awsm_renderer::profiling::{self, CpuScopeKind};
use awsm_renderer::AwsmRenderer;

use crate::protocol::{TimingMean, DETAIL_GPU, DETAIL_PHYSICS, DETAIL_RENDERER};

/// The renderer's phases of one frame, in the order they run; the Stats
/// panel lists them by these names. The physics steps run between the start
/// of `ReadApp` and `SoundCues`, and are not a renderer phase.
#[derive(Clone, Copy)]
pub enum Phase {
    /// Everything before the transforms are resolved, minus the physics
    /// steps, the cues and `Animate`: the camera pose main posted, click-drops,
    /// interpolating every body and writing its transform.
    ReadApp,
    /// Posting the steps' audio cues to main.
    SoundCues,
    /// The scene's clips + texture UV flows (`update_animations`) and its
    /// particles (`ParticleDriver::tick`), before the bodies are written.
    Animate,
    SetCamera,
    UpdateTransforms,
    /// `AwsmRenderer::render`: encoding every pass and submitting.
    Render,
}

const PHASE_NAMES: [&str; 6] = [
    "read app state + interpolate",
    "post sound cues",
    "animations + particles",
    "set camera",
    "update transforms",
    "render (encode + submit)",
];

/// The phases of one physics step (`World::step`), in order.
#[derive(Clone, Copy)]
pub enum StepPhase {
    Input,
    Box3dStep,
    Contacts,
    Poses,
}

const STEP_PHASE_NAMES: [&str; 4] = [
    "apply input",
    "Box3D step",
    "contacts + sound cues",
    "update poses",
];

/// Smoothing weight of each window's mean (the same as the headline numbers).
const EMA_ALPHA: f64 = 0.35;

/// Times the phases of physics steps. Only exists while asked, so a collapsed
/// physics group costs `World::step` one `None` check.
pub struct StepTimer {
    perf: web_sys::Performance,
    last_ms: f64,
    ms: [f64; 4],
    steps: u32,
}

impl StepTimer {
    /// Starts a step.
    pub fn begin(&mut self) {
        self.last_ms = self.perf.now();
        self.steps += 1;
    }

    /// Charges the time since the last lap to `phase`.
    pub fn lap(&mut self, phase: StepPhase) {
        let now = self.perf.now();
        self.ms[phase as usize] += now - self.last_ms;
        self.last_ms = now;
    }
}

/// The breakdowns of one stats window (see the `StatsMsg` fields).
#[derive(Default)]
pub struct Breakdown {
    pub renderer_phases: Vec<TimingMean>,
    pub renderer_prep: Vec<TimingMean>,
    pub renderer_passes: Vec<TimingMean>,
    pub renderer_submit: Vec<TimingMean>,
    pub renderer_hooks: Vec<TimingMean>,
    pub gpu_passes: Vec<TimingMean>,
    pub step_phases: Vec<TimingMean>,
}

/// Per-scope `(total ms, runs)` from awsm-renderer's timing tables.
type Totals = HashMap<&'static str, (f64, u64)>;

fn totals(stats: Vec<(&'static str, profiling::TimingStat)>) -> Totals {
    stats
        .into_iter()
        .map(|(name, stat)| (name, (stat.total_ms, stat.count)))
        .collect()
}

pub struct Profile {
    perf: web_sys::Performance,
    /// What main asked for at the start of the current frame.
    detail: u32,
    /// This window's sums while the renderer group is open: the phases (ms)
    /// and the frames drawn.
    phase_ms: [f64; 6],
    frames: u32,
    steps: StepTimer,
    /// The renderer's per-scope `(total ms, runs)` at the window's start, or
    /// when the group was expanded.
    cpu_seen: Totals,
    /// The GPU's per-pass totals and sampled-frame count at the window's
    /// start (`None` when they were not being collected then).
    gpu_base: Option<(u64, Totals)>,
    /// Smoothed means by row (`kind` index, name).
    means: HashMap<(u8, &'static str), f64>,
}

impl Profile {
    pub fn new(perf: web_sys::Performance) -> Self {
        Self {
            steps: StepTimer {
                perf: perf.clone(),
                last_ms: 0.0,
                ms: [0.0; 4],
                steps: 0,
            },
            perf,
            detail: 0,
            phase_ms: [0.0; 6],
            frames: 0,
            cpu_seen: HashMap::new(),
            gpu_base: None,
            means: HashMap::new(),
        }
    }

    fn renderer(&self) -> bool {
        self.detail & DETAIL_RENDERER != 0
    }

    /// Starts a frame: picks up what main asks for and switches the
    /// renderer's CPU scopes on or off for it.
    pub fn begin_frame(&mut self, renderer: &mut AwsmRenderer, detail: u32) {
        let was_renderer = self.renderer();
        self.detail = detail;
        if self.renderer() && !was_renderer {
            self.cpu_seen = totals(profiling::cpu_timing_stats());
        }
        renderer.logging.cpu = if self.renderer() {
            TimingTier::SubFrame
        } else {
            TimingTier::Off
        };
    }

    /// While timing: charges `since_ms → now_ms` to `phase` (no clock read).
    pub fn charge(&mut self, phase: Phase, since_ms: f64, now_ms: f64) {
        if self.renderer() {
            self.phase_ms[phase as usize] += now_ms - since_ms;
        }
    }

    /// While timing: charges the time since `since_ms` to `phase` and returns
    /// the time now, for the next phase to start from.
    pub fn lap(&mut self, phase: Phase, since_ms: f64) -> f64 {
        if !self.renderer() {
            return since_ms;
        }
        let now = self.perf.now();
        self.charge(phase, since_ms, now);
        now
    }

    /// Ends a frame.
    pub fn end_frame(&mut self) {
        if self.renderer() {
            self.frames += 1;
        }
    }

    /// The step timer for this frame's physics steps, while asked.
    pub fn step_timer(&mut self) -> Option<&mut StepTimer> {
        (self.detail & DETAIL_PHYSICS != 0).then_some(&mut self.steps)
    }

    /// Drops what this window has collected (the stats warm-up).
    pub fn discard_window(&mut self) {
        if self.renderer() {
            self.cpu_seen = totals(profiling::cpu_timing_stats());
        }
        self.reset_window();
        self.gpu_base = None;
        self.means.clear();
    }

    fn reset_window(&mut self) {
        self.phase_ms = [0.0; 6];
        self.frames = 0;
        self.steps.ms = [0.0; 4];
        self.steps.steps = 0;
    }

    /// Ends a stats window: the means of what was asked for, smoothed, and a
    /// fresh window.
    pub fn window(&mut self) -> Breakdown {
        let mut out = Breakdown::default();
        let mut means = HashMap::new();
        let mut mean = |kind: u8, name: &'static str, ms: f64| {
            let smoothed = match self.means.get(&(kind, name)) {
                Some(e) => e + (ms - e) * EMA_ALPHA,
                None => ms,
            };
            means.insert((kind, name), smoothed);
            TimingMean {
                name: name.to_string(),
                ms: smoothed,
            }
        };

        if self.renderer() && self.frames > 0 {
            let frames = f64::from(self.frames);
            for (name, ms) in PHASE_NAMES.into_iter().zip(self.phase_ms) {
                out.renderer_phases.push(mean(0, name, ms / frames));
            }
            // Each scope's time this window per frame drawn; a scope that
            // did not run in it drops out.
            let stats = totals(profiling::cpu_timing_stats());
            for (&name, &(total_ms, count)) in &stats {
                let (seen_ms, seen_count) = self.cpu_seen.get(name).copied().unwrap_or_default();
                if count <= seen_count {
                    continue;
                }
                let ms = (total_ms - seen_ms) / frames;
                match profiling::cpu_scope_kind(name) {
                    // The whole of render(): our render phase already.
                    CpuScopeKind::Frame => {}
                    CpuScopeKind::Prep => {
                        let name = name.trim_start_matches(profiling::CPU_PREP_PREFIX);
                        out.renderer_prep.push(mean(1, name, ms));
                    }
                    CpuScopeKind::Pass => out.renderer_passes.push(mean(2, name, ms)),
                    CpuScopeKind::Submit => {
                        let name = name.trim_start_matches(profiling::CPU_SUBMIT_PREFIX);
                        out.renderer_submit.push(mean(3, name, ms));
                    }
                    CpuScopeKind::Hook => out.renderer_hooks.push(mean(4, name, ms)),
                }
            }
            self.cpu_seen = stats;
        }

        if self.detail & DETAIL_GPU != 0 {
            // Timestamps are read back for some frames only; the whole-frame
            // entry counts exactly those.
            let sampled = profiling::gpu_frame_timing().map_or(0, |frame| frame.count);
            let stats = totals(profiling::gpu_timing_stats());
            if let Some((sampled0, base)) = &self.gpu_base {
                let frames = sampled.saturating_sub(*sampled0) as f64;
                for (&name, &(total_ms, runs)) in &stats {
                    let (total0, runs0) = base.get(name).copied().unwrap_or_default();
                    if name != profiling::GPU_FRAME_SCOPE && frames > 0.0 && runs > runs0 {
                        out.gpu_passes
                            .push(mean(5, name, (total_ms - total0) / frames));
                    }
                }
            }
            self.gpu_base = Some((sampled, stats));
        } else {
            self.gpu_base = None;
        }

        if self.detail & DETAIL_PHYSICS != 0 && self.steps.steps > 0 {
            let steps = f64::from(self.steps.steps);
            for (name, ms) in STEP_PHASE_NAMES.into_iter().zip(self.steps.ms) {
                out.step_phases.push(mean(6, name, ms / steps));
            }
        }

        self.means = means;
        self.reset_window();
        out
    }
}
