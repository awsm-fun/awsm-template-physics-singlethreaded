//! The Stats panel's breakdowns, measured on the game worker only while main
//! asks (the `DETAIL_*` bits of a [`StatsDetailMsg`]):
//!
//! * `DETAIL_RENDERER`: the phases of the frame function are timed (one
//!   clock read between each), and awsm-renderer's per-pass CPU scopes are
//!   switched on for some frames (`logging.cpu`, a field the renderer checks
//!   at every scope).
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
//! # The per-pass scopes are costly, so they are sampled
//!
//! Each awsm-renderer scope looks the clock up through the global object at
//! both ends, which in a worker takes far longer than most passes (about
//! 80 µs a lookup: a dozen scopes cost several times the frame). So that
//! looking does not change what is looked at:
//!
//! * the scopes are on for one frame in [`PASS_TIMING_EVERY`] only, and the
//!   phases are taken from the other frames, which are undisturbed;
//! * what the scopes cost is measured, as the difference between the render
//!   phase with and without them, and reported as a row of its own. One
//!   estimate is made per stats window and the median of the last few is
//!   used, so that a frame that hitched (a pipeline compile, a collection)
//!   does not pass for the scopes' doing;
//! * each pass has the scope's share of its reading taken out. A scope's
//!   reading runs from the end of its first lookup to the end of its second,
//!   so it contains one of the two: half of what the scope costs.
//!
//! [`StatsDetailMsg`]: crate::protocol::StatsDetailMsg

use std::collections::{HashMap, VecDeque};

use awsm_renderer::debug::TimingTier;
use awsm_renderer::{profiling, AwsmRenderer};

use crate::protocol::{TimingMean, DETAIL_GPU, DETAIL_PHYSICS, DETAIL_RENDERER};

/// The renderer's phases of one frame, in the order they run; the Stats
/// panel lists them by these names.
#[derive(Clone, Copy)]
pub enum Phase {
    /// Everything before the transforms are resolved, minus the physics
    /// steps: the camera pose main posted, click-drops, interpolating every
    /// body and writing its transform.
    ReadApp,
    SetCamera,
    UpdateTransforms,
    /// `AwsmRenderer::render`: encoding every pass and submitting.
    Render,
    /// Posting the steps' audio cues to main.
    SoundCues,
}

const PHASE_NAMES: [&str; 5] = [
    "read app state + interpolate",
    "set camera",
    "update transforms",
    "render (encode + submit)",
    "post sound cues",
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

/// awsm-renderer's own scope around the whole of `render()`: the same time
/// as our render phase, so it is not listed among the passes.
const WHOLE_RENDER_SCOPE: &str = "Render";
/// The row the cost of the scopes is reported under.
const OVERHEAD_NAME: &str = "timing the passes (only while open)";
/// The per-pass scopes are on for one frame in this many.
const PASS_TIMING_EVERY: u64 = 16;
/// The scope cost in use is the median of this many estimates, one per
/// stats window (about a second) that had frames of both kinds.
const SCOPE_ESTIMATES: usize = 5;

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
    pub renderer_passes: Vec<TimingMean>,
    pub gpu_passes: Vec<TimingMean>,
    pub step_phases: Vec<TimingMean>,
}

/// Per-pass `(total ms, runs)` from awsm-renderer's timing tables.
type Totals = HashMap<&'static str, (f64, u64)>;

fn totals(stats: Vec<(&'static str, profiling::TimingStat)>) -> Totals {
    stats
        .into_iter()
        .map(|(name, stat)| (name, (stat.total_ms, stat.count)))
        .collect()
}

/// The middle value (the upper of the two for an even count), or 0 for none.
fn median(values: &VecDeque<f64>) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
}

pub struct Profile {
    perf: web_sys::Performance,
    /// What main asked for at the start of the current frame, and whether
    /// the per-pass scopes are on for it.
    detail: u32,
    timing_passes: bool,
    /// Frames drawn while the renderer group was open (the 1-in-N schedule).
    renderer_frames: u64,
    /// This window's sums while asked, over the frames WITHOUT the scopes:
    /// the phases (ms) and the frames counted.
    phase_ms: [f64; 5],
    plain_frames: u32,
    /// The same for the render phase over the frames WITH them.
    timed_render_ms: f64,
    timed_frames: u32,
    steps: StepTimer,
    /// The renderer's per-pass `(total ms, runs)` as last seen.
    cpu_seen: Totals,
    /// The latest estimates of what one scope costs (ms).
    scope_estimates: VecDeque<f64>,
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
            timing_passes: false,
            renderer_frames: 0,
            phase_ms: [0.0; 5],
            plain_frames: 0,
            timed_render_ms: 0.0,
            timed_frames: 0,
            cpu_seen: HashMap::new(),
            scope_estimates: VecDeque::new(),
            gpu_base: None,
            means: HashMap::new(),
        }
    }

    fn renderer(&self) -> bool {
        self.detail & DETAIL_RENDERER != 0
    }

    /// Starts a frame: picks up what main asks for and switches the
    /// renderer's per-pass CPU scopes on or off for it.
    pub fn begin_frame(&mut self, renderer: &mut AwsmRenderer, detail: u32) {
        self.detail = detail;
        self.timing_passes =
            self.renderer() && self.renderer_frames.is_multiple_of(PASS_TIMING_EVERY);
        renderer.logging.cpu = if self.timing_passes {
            TimingTier::SubFrame
        } else {
            TimingTier::Off
        };
    }

    /// While timing: charges `since_ms → now_ms` to `phase` (no clock read).
    pub fn charge(&mut self, phase: Phase, since_ms: f64, now_ms: f64) {
        if !self.renderer() {
            return;
        }
        if !self.timing_passes {
            self.phase_ms[phase as usize] += now_ms - since_ms;
        } else if matches!(phase, Phase::Render) {
            self.timed_render_ms += now_ms - since_ms;
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
        if self.timing_passes {
            self.timed_frames += 1;
        } else if self.renderer() {
            self.plain_frames += 1;
        }
        if self.renderer() {
            self.renderer_frames += 1;
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
        self.phase_ms = [0.0; 5];
        self.plain_frames = 0;
        self.timed_render_ms = 0.0;
        self.timed_frames = 0;
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

        if self.renderer() {
            // What the scopes ran and read this window. They only run while
            // asked, so a scope never seen before ran entirely in it.
            let stats = totals(profiling::cpu_timing_stats());
            let mut window = Vec::with_capacity(stats.len());
            let mut scope_runs = 0.0;
            for (&name, &(total_ms, count)) in &stats {
                let (seen_ms, seen_count) = self.cpu_seen.get(name).copied().unwrap_or_default();
                let runs = count.saturating_sub(seen_count) as f64;
                scope_runs += runs;
                window.push((name, runs, total_ms - seen_ms));
            }
            self.cpu_seen = stats;

            // What one scope costs: what a frame with the scopes took over
            // one without, shared among the scopes it ran.
            if self.plain_frames > 0 && self.timed_frames > 0 && scope_runs > 0.0 {
                let plain_ms = self.phase_ms[Phase::Render as usize] / f64::from(self.plain_frames);
                let timed_ms = self.timed_render_ms / f64::from(self.timed_frames);
                let estimate = (timed_ms - plain_ms) * f64::from(self.timed_frames) / scope_runs;
                if self.scope_estimates.len() == SCOPE_ESTIMATES {
                    self.scope_estimates.pop_front();
                }
                self.scope_estimates.push_back(estimate);
            }
            let scope_ms = median(&self.scope_estimates).max(0.0);

            // The phases come from the undisturbed frames, the passes from
            // the timed ones, each pass less the half of a scope its reading
            // contains.
            let mut overhead_ms = 0.0;
            for (name, runs, read_ms) in window {
                overhead_ms += runs * scope_ms;
                if name != WHOLE_RENDER_SCOPE && runs > 0.0 && self.timed_frames > 0 {
                    let net_ms = (read_ms - runs * scope_ms / 2.0).max(0.0);
                    out.renderer_passes
                        .push(mean(1, name, net_ms / f64::from(self.timed_frames)));
                }
            }
            if self.plain_frames > 0 {
                let frames = f64::from(self.plain_frames);
                for (name, ms) in PHASE_NAMES.into_iter().zip(self.phase_ms) {
                    out.renderer_phases.push(mean(0, name, ms / frames));
                }
                // Averaged over every frame: what looking adds to the
                // renderer's time per frame.
                let all_frames = frames + f64::from(self.timed_frames);
                out.renderer_phases
                    .push(mean(0, OVERHEAD_NAME, overhead_ms / all_frames));
            }
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
                            .push(mean(2, name, (total_ms - total0) / frames));
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
                out.step_phases.push(mean(3, name, ms / steps));
            }
        }

        self.means = means;
        self.reset_window();
        out
    }
}
