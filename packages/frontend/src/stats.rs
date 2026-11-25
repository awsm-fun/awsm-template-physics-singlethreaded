//! What the game worker tells the page's Stats panel.
//!
//! awsm-renderer-stats does the measuring and the formatting: frame windows,
//! smoothing, the frame budget, the display group (from the page's own frame
//! gaps), the renderer's CPU scopes and GPU passes, and the page group (from
//! the page's own work times). Its `StatsCollector` runs here, on the thread
//! that renders, and posts a report about once a second; the page's
//! `StatsPanel` only shows it, and sends back which groups are open.
//!
//! This file adds what is particular to this project: the names of the
//! frame's phases, and the physics group. Renderer and physics share this
//! worker, so the frame is split, never double counted: the physics steps
//! are physics time, the renderer everything else the frame does. The
//! frame budget has one lane for the whole worker (renderer + physics),
//! carried by the physics group.

use awsm_renderer_stats::{
    duration_row, step_headline, AppSection, NamedMeans, SectionSpec, StatRow, StatsCollector,
    StatsConfig, Window, WARMUP_FRAMES,
};

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

/// Not a renderer phase: `StatsCollector::lap` with this index only moves
/// the phase clock on (the physics steps are physics time).
pub const NOT_RENDERER: usize = usize::MAX;

/// The physics group's id (the page stores whether it is open under it).
pub const PHYSICS: &str = "physics";
const PHYSICS_LABEL: &str = "physics (Box3D, game worker)";
/// The frame budget's one lane for this worker.
const WORKER_LANE_LABEL: &str = "game worker (renderer + physics)";

/// The Stats panel's configuration: the frame's renderer phases, and the
/// physics group (the game worker's lane, green) between the renderer and
/// the page. The renderer is not a lane of the budget on its own: the
/// worker's lane is renderer + physics.
pub fn config() -> StatsConfig {
    StatsConfig {
        renderer_label: "renderer (game worker)".to_string(),
        phases: PHASE_NAMES.map(String::from).to_vec(),
        render_phase: Some(Phase::Render as usize),
        sections: vec![SectionSpec::new(PHYSICS, PHYSICS_LABEL, Some("sim"))],
        renderer_in_budget: false,
        // The worker draws on every refresh: no pacing row.
        pacing_row: false,
        ..StatsConfig::default()
    }
}

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

/// Times the phases of physics steps. Handed to `World::step` only while the
/// physics group is open, so a collapsed group costs a step one `None` check.
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

    fn take(&mut self) -> ([f64; 4], u32) {
        let out = (self.ms, self.steps);
        self.ms = [0.0; 4];
        self.steps = 0;
        out
    }
}

/// The physics group's sums since the last report.
pub struct PhysicsStats {
    timer: StepTimer,
    /// Frames since `ready` (the collector's warm-up ends at
    /// [`WARMUP_FRAMES`]; the sums start over there).
    frames_since_ready: Option<u32>,
    frames: u32,
    steps: u32,
    step_ms: f64,
    worker_ms: f64,
    phases: NamedMeans,
}

impl PhysicsStats {
    pub fn new(perf: web_sys::Performance) -> Self {
        Self {
            timer: StepTimer {
                perf,
                last_ms: 0.0,
                ms: [0.0; 4],
                steps: 0,
            },
            frames_since_ready: None,
            frames: 0,
            steps: 0,
            step_ms: 0.0,
            worker_ms: 0.0,
            phases: NamedMeans::default(),
        }
    }

    /// The step timer for this frame's physics steps, while the physics
    /// group is open.
    pub fn step_timer(&mut self, stats: &StatsCollector) -> Option<&mut StepTimer> {
        stats.is_open(PHYSICS).then_some(&mut self.timer)
    }

    /// The scene is on screen (the collector's `mark_ready`): start over.
    pub fn ready(&mut self) {
        self.frames_since_ready = Some(0);
        self.clear();
    }

    fn clear(&mut self) {
        self.frames = 0;
        self.steps = 0;
        self.step_ms = 0.0;
        self.worker_ms = 0.0;
        self.timer.take();
    }

    /// One frame: `substeps` physics steps that took `step_ms`, out of the
    /// worker's `worker_ms` for the whole frame. Call it after the
    /// collector's `end_frame` for the same frame.
    pub fn frame(&mut self, substeps: u32, step_ms: f64, worker_ms: f64) {
        self.frames += 1;
        self.steps += substeps;
        self.step_ms += step_ms;
        self.worker_ms += worker_ms;
        // The collector's first window starts after the warm-up's last frame.
        if let Some(n) = self.frames_since_ready.as_mut() {
            *n += 1;
            if *n == WARMUP_FRAMES {
                self.frames_since_ready = None;
                self.clear();
            }
        }
    }

    /// The physics group for one window: its time per fixed step, and the
    /// whole worker's time per frame as the budget lane. Its rows: steps per
    /// second, the step's phases (while open), and the balls dropped.
    pub fn section(
        &mut self,
        stats: &mut StatsCollector,
        window: &Window,
        balls: usize,
    ) -> AppSection {
        let steps_per_s = stats.smooth("steps", f64::from(self.steps) / window.seconds);
        let per_step = (self.steps > 0)
            .then(|| stats.smooth("step", self.step_ms / f64::from(self.steps)))
            .or(stats.smoothed("step"));
        let worker = (self.frames > 0)
            .then(|| stats.smooth("worker", self.worker_ms / f64::from(self.frames)))
            .or(stats.smoothed("worker"));

        let (phase_ms, phase_steps) = self.timer.take();
        if stats.is_open(PHYSICS) && phase_steps > 0 {
            let steps = f64::from(phase_steps);
            self.phases.update(
                STEP_PHASE_NAMES
                    .iter()
                    .zip(phase_ms)
                    .map(|(name, ms)| (name.to_string(), ms / steps)),
            );
        } else {
            self.phases.clear();
        }

        let mut rows = vec![StatRow::new("steps", format!("{steps_per_s:.0} steps/s"))];
        rows.extend(
            self.phases
                .entries()
                .iter()
                .map(|(name, ms)| duration_row(name.clone(), *ms, "step")),
        );
        rows.push(StatRow::new("balls dropped", balls.to_string()));
        self.frames = 0;
        self.steps = 0;
        self.step_ms = 0.0;
        self.worker_ms = 0.0;

        SectionSpec::new(PHYSICS, PHYSICS_LABEL, Some("sim"))
            .group()
            .headline(step_headline(per_step))
            .budget_ms(worker)
            .budget_label(WORKER_LANE_LABEL)
            .rows(rows)
    }
}
