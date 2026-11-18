//! The Stats panel (top right) and the Stats button (bottom left).
//!
//! One group per thread, so every number sits under the thread that spent
//! it, and every number says its unit and what it is per:
//!
//! - `frame budget`: the refresh period, and each thread's and the GPU's
//!   time per drawn frame against it.
//! - `display`: the refresh rate of the browser's frame clock (measured here,
//!   see `refresh`), the frames the worker presented per second, and the
//!   refreshes that went by with no new frame.
//! - `GPU`: the GPU's time per frame (renderer timestamp queries; `n/a`
//!   without them). Its rows are the passes; on GPUs that run passes
//!   overlapped (Apple's) they add up to more than the frame, which is
//!   expected, not a bug.
//! - `renderer (game worker)` and `physics (Box3D, game worker)`: the game
//!   worker's CPU time, split so nothing is counted twice: physics is the
//!   time inside the fixed-step loop (per step), the renderer everything
//!   else the worker's frame does (per frame). Expanded, the renderer shows
//!   its phases and awsm-renderer's CPU scopes inside `render()`.
//! - `page (main thread)`: this thread's own per-frame work.
//!
//! A group's header always shows its headline. Expanding a group shows its
//! breakdown, and only then is the breakdown measured: main posts the
//! expanded groups' `DETAIL_*` bits ([`StatsDetailMsg`]) and the worker times
//! its phases and passes only while a bit is set. A collapsed group costs
//! nothing.
//!
//! The worker samples its own counters and posts a smoothed [`StatsMsg`]
//! about once a second (see `game::StatsAcc`); main measures the display and
//! its own work, and lays it all out.
//!
//! [`StatsDetailMsg`]: crate::protocol::StatsDetailMsg

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use dominator::{clone, html, Dom};
use futures_signals::signal::{Mutable, Signal, SignalExt};
use futures_signals::signal_vec::{MutableVec, SignalVecExt};

use crate::main_thread::{coarse_pointer, store_bool, stored_bool, STATS_STORAGE_KEY};
use crate::protocol::{StatsMsg, TimingMean, DETAIL_GPU, DETAIL_PHYSICS, DETAIL_RENDERER};
use crate::refresh::{Missed, Refresh};

/// Smoothing weight of each new sample of this thread's own work (the
/// worker smooths its figures the same way).
const EMA_ALPHA: f64 = 0.35;
/// The renderer phase whose time its scope rows break down.
const RENDER_PHASE: &str = "render (encode + submit)";
/// Scopes under this mean (ms per frame) are folded into one line.
const FOLD_BELOW_MS: f64 = 0.001;

/// The groups, top to bottom.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Budget,
    Display,
    Gpu,
    Renderer,
    Physics,
    Page,
}

impl Group {
    const ALL: [Group; 6] = [
        Group::Budget,
        Group::Display,
        Group::Gpu,
        Group::Renderer,
        Group::Physics,
        Group::Page,
    ];

    /// Storage / element id.
    fn id(self) -> &'static str {
        match self {
            Group::Budget => "budget",
            Group::Display => "display",
            Group::Gpu => "gpu",
            Group::Renderer => "renderer",
            Group::Physics => "physics",
            Group::Page => "page",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Group::Budget => "frame budget",
            Group::Display => "display",
            Group::Gpu => "GPU",
            Group::Renderer => "renderer (game worker)",
            Group::Physics => "physics (Box3D, game worker)",
            Group::Page => "page (main thread)",
        }
    }

    /// The bit that asks the worker to measure this group's breakdown.
    fn detail(self) -> u32 {
        match self {
            Group::Gpu => DETAIL_GPU,
            Group::Renderer => DETAIL_RENDERER,
            Group::Physics => DETAIL_PHYSICS,
            Group::Budget | Group::Display | Group::Page => 0,
        }
    }
}

/// The `localStorage` key of a group's open/closed state.
fn open_key(group: Group) -> String {
    format!("{STATS_STORAGE_KEY}_open_{}", group.id())
}

/// Every group's open/closed key, for `?reset`.
pub fn open_keys() -> impl Iterator<Item = String> {
    Group::ALL.into_iter().map(open_key)
}

/// A duration with its unit: microseconds below a tenth of a millisecond.
fn duration(ms: f64) -> String {
    if ms < 0.09995 {
        let us = ms * 1000.0;
        if us < 9.95 {
            format!("{us:.1} µs")
        } else {
            format!("{us:.0} µs")
        }
    } else if ms < 9.995 {
        format!("{ms:.2} ms")
    } else {
        format!("{ms:.1} ms")
    }
}

fn per_frame(ms: f64) -> String {
    format!("{} / frame", duration(ms))
}

fn per_step(ms: f64) -> String {
    format!("{} / step", duration(ms))
}

/// Milliseconds with one decimal, for the frame budget: `<0.1` for a
/// nonzero time that would print as 0.0.
fn ms_1dp(ms: f64) -> String {
    if ms > 0.0 && ms < 0.05 {
        "<0.1".to_string()
    } else {
        format!("{ms:.1}")
    }
}

/// A frame budget row: `7.2 / 16.7 ms (43%)`.
fn budget_share(used_ms: f64, budget_ms: f64) -> String {
    let pct = used_ms / budget_ms * 100.0;
    let pct = if pct > 0.0 && pct < 1.0 {
        "<1%".to_string()
    } else {
        format!("{pct:.0}%")
    };
    format!("{} / {budget_ms:.1} ms ({pct})", ms_1dp(used_ms))
}

fn smooth(ema: Option<f64>, value: f64) -> f64 {
    match ema {
        Some(e) => e + (value - e) * EMA_ALPHA,
        None => value,
    }
}

/// One row of a group: label left, value right. `depth` indents it under
/// the row above (a breakdown of it), up to two levels.
#[derive(Clone, PartialEq)]
struct Row {
    label: String,
    value: String,
    depth: u8,
}

fn row(label: impl Into<String>, value: impl Into<String>) -> Row {
    Row {
        label: label.into(),
        value: value.into(),
        depth: 0,
    }
}

impl Row {
    fn nested(mut self) -> Self {
        self.depth = 1;
        self
    }

    fn nested_2(mut self) -> Self {
        self.depth = 2;
        self
    }
}

struct GroupView {
    group: Group,
    open: Mutable<bool>,
    headline: Mutable<String>,
    /// Turns the header red (over the frame budget).
    bad: Mutable<bool>,
    rows: MutableVec<Row>,
}

/// The panel's state: whether it is shown, and per group whether it is
/// expanded and what it shows.
pub struct Panel {
    visible: Mutable<bool>,
    groups: Vec<GroupView>,
    /// The `DETAIL_*` bits of the expanded groups, 0 while the panel is hidden.
    detail: Mutable<u32>,
}

impl Panel {
    /// Hidden by default: on a phone the panel covers a lot of the screen, so
    /// a stored "open" is only honoured on a desktop. Every group starts
    /// collapsed unless the user expanded it before.
    pub fn new() -> Rc<Self> {
        let window = web_sys::window();
        let stored = |key: &str| {
            window
                .as_ref()
                .and_then(|w| stored_bool(w, key))
                .unwrap_or(false)
        };
        let visible = window
            .as_ref()
            .is_some_and(|w| !coarse_pointer(w) && stored(STATS_STORAGE_KEY));
        let groups = Group::ALL
            .into_iter()
            .map(|group| GroupView {
                group,
                open: Mutable::new(stored(&open_key(group))),
                headline: Mutable::new(String::new()),
                bad: Mutable::new(false),
                rows: MutableVec::new(),
            })
            .collect();
        let panel = Rc::new(Panel {
            visible: Mutable::new(visible),
            groups,
            detail: Mutable::new(0),
        });
        panel.update_detail();
        panel.placeholder("starting…");
        panel
    }

    fn view(&self, group: Group) -> &GroupView {
        &self.groups[group as usize]
    }

    fn update_detail(&self) {
        let bits = if self.visible.get() {
            self.groups
                .iter()
                .filter(|g| g.open.get())
                .fold(0, |bits, g| bits | g.group.detail())
        } else {
            0
        };
        self.detail.set_neq(bits);
    }

    /// The `DETAIL_*` bits the worker should measure right now.
    pub fn detail(&self) -> u32 {
        self.detail.get()
    }

    pub fn detail_signal(&self) -> impl Signal<Item = u32> {
        self.detail.signal()
    }

    fn set(&self, group: Group, headline: impl Into<String>, rows: Vec<Row>) {
        self.set_bad(group, headline, rows, false);
    }

    fn set_bad(&self, group: Group, headline: impl Into<String>, rows: Vec<Row>, bad: bool) {
        let view = self.view(group);
        view.headline.set_neq(headline.into());
        view.bad.set_neq(bad);
        let mut current = view.rows.lock_mut();
        if current[..] != rows[..] {
            current.replace_cloned(rows);
        }
    }

    /// What every group says until there are numbers.
    fn placeholder(&self, text: &str) {
        for group in Group::ALL {
            let headline = if group == Group::Display { "—" } else { text };
            self.set(group, headline, Vec::new());
        }
    }

    /// The panel: per group, a header button (label left, headline right)
    /// and the rows it shows while expanded.
    pub fn dom(self: &Rc<Self>) -> Dom {
        let panel = self.clone();
        html!("div", {
            .class("stats")
            .visible_signal(self.visible.signal())
            .children(self.groups.iter().enumerate().flat_map(|(index, view)| {
                let rows_id = format!("stats-rows-{}", view.group.id());
                [
                    html!("button", {
                        .attr("type", "button")
                        .class("stats-group")
                        .class_signal("stats-bad", view.bad.signal())
                        .attr("aria-controls", &rows_id)
                        .attr_signal("aria-expanded", view.open.signal().map(|open| if open { "true" } else { "false" }))
                        .child(html!("span", {
                            .text_signal(view.open.signal().map(|open| if open { "▾ " } else { "▸ " }))
                            .text(view.group.label())
                        }))
                        .text(" ") // for a screen reader; the layout ignores it
                        .child(html!("span", {
                            .text_signal(view.headline.signal_cloned())
                        }))
                        .event(clone!(panel => move |_: dominator::events::Click| {
                            let view = &panel.groups[index];
                            let open = !view.open.get();
                            view.open.set(open);
                            if let Some(w) = web_sys::window() {
                                store_bool(&w, &open_key(view.group), open);
                            }
                            panel.update_detail();
                        }))
                    }),
                    html!("div", {
                        .class("stats-rows")
                        .attr("id", &rows_id)
                        .prop_signal("hidden", view.open.signal().map(|open| !open))
                        .children_signal_vec(view.rows.signal_vec_cloned().map(|row| {
                            html!("div", {
                                .class("stats-row")
                                .apply_if(row.depth == 1, |dom| dom.class("stats-nested"))
                                .apply_if(row.depth == 2, |dom| dom.class("stats-nested-2"))
                                .child(html!("span", { .text(&row.label) }))
                                .child(html!("span", { .text(&row.value) }))
                            })
                        }))
                    }),
                ]
            }).collect::<Vec<_>>())
        })
    }

    /// The bottom-left "Stats" chip: shows/hides the panel, persisted.
    pub fn button(self: &Rc<Self>) -> Dom {
        let panel = self.clone();
        html!("button", {
            .attr("type", "button")
            .class("stats-btn")
            .class_signal("active", self.visible.signal())
            .text("Stats")
            .event(move |_: dominator::events::Click| {
                let next = !panel.visible.get();
                panel.visible.set_neq(next);
                if let Some(w) = web_sys::window() {
                    store_bool(&w, STATS_STORAGE_KEY, next);
                }
                panel.update_detail();
            })
        })
    }
}

/// This thread's own per-frame work, summed between stats messages: its
/// camera frame loop, and the sound cues it plays for the worker.
#[derive(Default)]
pub struct PageWork {
    frame_ms: Cell<f64>,
    cues_ms: Cell<f64>,
    frames: Cell<u32>,
}

impl PageWork {
    /// One animation frame of the main thread, which took `ms`.
    pub fn frame(&self, ms: f64) {
        self.frame_ms.set(self.frame_ms.get() + ms);
        self.frames.set(self.frames.get() + 1);
    }

    /// Time spent handling sound cues.
    pub fn cues(&self, ms: f64) {
        self.cues_ms.set(self.cues_ms.get() + ms);
    }

    fn take(&self) -> (f64, f64, u32) {
        (
            self.frame_ms.take(),
            self.cues_ms.take(),
            self.frames.take(),
        )
    }
}

/// Turns the worker's [`StatsMsg`]s, the display measurement and this
/// thread's own work into the panel.
pub struct Stats {
    panel: Rc<Panel>,
    page: Rc<PageWork>,
    refresh: Rc<RefCell<Refresh>>,
    missed: RefCell<Missed>,
    /// This thread's work per frame: total, camera loop, sound cues.
    page_ema: Cell<[Option<f64>; 3]>,
    received: Cell<bool>,
}

impl Stats {
    pub fn new(panel: Rc<Panel>, page: Rc<PageWork>, refresh: Rc<RefCell<Refresh>>) -> Self {
        Self {
            panel,
            page,
            refresh,
            missed: RefCell::new(Missed::default()),
            page_ema: Cell::new([None; 3]),
            received: Cell::new(false),
        }
    }

    /// The worker is drawing: its first numbers follow its warm-up.
    pub fn ready(&self) {
        if !self.received.get() {
            self.panel.placeholder("warming up…");
        }
    }

    pub fn on_stats(&self, msg: &StatsMsg) {
        self.received.set(true);
        let panel = &self.panel;
        let refresh = self.refresh.borrow();
        let mut missed = self.missed.borrow_mut();
        missed.window(msg.span_frames, msg.span_ms, refresh.period_ms());

        let hz = refresh.line();
        panel.set(
            Group::Display,
            format!("{hz} · {:.0} fps", msg.fps),
            vec![
                row("refresh rate", hz),
                row("frames presented", format!("{:.0} / s", msg.fps)),
                row("missed frames", missed.line()),
            ],
        );

        let gpu = match msg.gpu_ms {
            Some(ms) => per_frame(f64::from(ms)),
            None if msg.gpu_unavailable => "n/a".to_string(),
            None => "—".to_string(),
        };
        // On GPUs that overlap passes (Apple's), these add up to more than the
        // frame total above; that is expected.
        let gpu_rows = slowest_first(&msg.gpu_passes)
            .into_iter()
            .map(|pass| row(&pass.name, per_frame(pass.ms)))
            .collect();
        panel.set(Group::Gpu, gpu, gpu_rows);

        // The renderer's phases in the order they run, with awsm-renderer's
        // scopes inside render() grouped by kind under the render phase.
        let mut renderer_rows = Vec::new();
        for phase in &msg.renderer_phases {
            renderer_rows.push(row(&phase.name, per_frame(phase.ms)));
            if phase.name != RENDER_PHASE {
                continue;
            }
            let mut in_scopes = None;
            for (label, scopes) in [
                ("prepare", &msg.renderer_prep),
                ("passes", &msg.renderer_passes),
                ("submit", &msg.renderer_submit),
                ("hooks", &msg.renderer_hooks),
            ] {
                if scopes.is_empty() {
                    continue;
                }
                let sum: f64 = scopes.iter().map(|scope| scope.ms).sum();
                *in_scopes.get_or_insert(0.0) += sum;
                renderer_rows.push(row(label, per_frame(sum)).nested());
                let (shown, folded): (Vec<_>, Vec<_>) = slowest_first(scopes)
                    .into_iter()
                    .partition(|scope| scope.ms >= FOLD_BELOW_MS);
                for scope in shown {
                    renderer_rows.push(row(&scope.name, per_frame(scope.ms)).nested_2());
                }
                if !folded.is_empty() {
                    let folded_ms: f64 = folded.iter().map(|scope| scope.ms).sum();
                    renderer_rows.push(
                        row(
                            format!("{} more under 1 µs", folded.len()),
                            per_frame(folded_ms),
                        )
                        .nested_2(),
                    );
                }
            }
            // The glue between the scopes.
            if let Some(in_scopes) = in_scopes {
                renderer_rows.push(
                    row(
                        "outside the scopes",
                        per_frame((phase.ms - in_scopes).max(0.0)),
                    )
                    .nested(),
                );
            }
        }
        panel.set(
            Group::Renderer,
            per_frame(f64::from(msg.renderer_ms)),
            renderer_rows,
        );

        let mut physics_rows = vec![row("steps", format!("{:.0} steps/s", msg.sps))];
        for phase in &msg.step_phases {
            physics_rows.push(row(&phase.name, per_step(phase.ms)));
        }
        physics_rows.push(row("balls dropped", msg.balls.to_string()));
        let step = if msg.sps > 0.0 {
            per_step(f64::from(msg.step_ms))
        } else {
            "—".to_string()
        };
        panel.set(Group::Physics, step, physics_rows);

        // This thread's work since the previous message, per main frame.
        let (frame_ms, cues_ms, frames) = self.page.take();
        let mut ema = self.page_ema.get();
        if frames > 0 {
            let per = |ms: f64| ms / f64::from(frames);
            for (slot, ms) in ema.iter_mut().zip([frame_ms + cues_ms, frame_ms, cues_ms]) {
                *slot = Some(smooth(*slot, per(ms)));
            }
            self.page_ema.set(ema);
        }
        let page_rows = match ema {
            [_, Some(frame), Some(cues)] => vec![
                row("camera frame loop", per_frame(frame)),
                row("sound cues", per_frame(cues)),
            ],
            _ => Vec::new(),
        };
        panel.set(
            Group::Page,
            ema[0].map_or("—".to_string(), per_frame),
            page_rows,
        );

        self.show_budget(msg, refresh.period_ms(), ema[0]);
    }

    /// The `frame budget` group: one lane per thread plus the GPU, each its
    /// time per drawn frame, against the refresh period. CPU and GPU work
    /// run in parallel, so each must fit the budget on its own; they do not
    /// add up, except when the CPU waits on the GPU, which shows as missed
    /// frames.
    fn show_budget(&self, msg: &StatsMsg, period_ms: Option<f64>, page_ms: Option<f64>) {
        let Some(budget) = period_ms else {
            self.panel.set(Group::Budget, "measuring…", Vec::new());
            return;
        };
        let gpu = msg.gpu_ms.map(f64::from);
        // Renderer and physics share the game worker: one lane, its whole
        // frame.
        let lanes: Vec<(&str, f64)> = [
            ("GPU", gpu),
            (
                "game worker (renderer + physics)",
                Some(f64::from(msg.worker_ms)),
            ),
            (Group::Page.label(), page_ms),
        ]
        .into_iter()
        .filter_map(|(label, ms)| Some((label, ms?)))
        .collect();

        let mut headline = format!("{budget:.1} ms ({:.0} Hz)", 1000.0 / budget);
        let mut over = false;
        if let Some(largest) = lanes.iter().map(|(_, ms)| *ms).reduce(f64::max) {
            let headroom = budget - largest;
            over = headroom < 0.0;
            if over {
                headline += &format!(" · over by {} ms", ms_1dp(-headroom));
            } else {
                headline += &format!(" · headroom {} ms", ms_1dp(headroom));
            }
        }
        let mut rows: Vec<Row> = lanes
            .iter()
            .map(|&(label, ms)| row(label, budget_share(ms, budget)))
            .collect();
        if let Some(gpu) = gpu {
            rows.push(row(
                "latency (renderer + GPU), reference only",
                format!("{} ms", ms_1dp(f64::from(msg.renderer_ms) + gpu)),
            ));
        }
        self.panel.set_bad(Group::Budget, headline, rows, over);
    }
}

/// Rows, slowest first.
fn slowest_first(entries: &[TimingMean]) -> Vec<&TimingMean> {
    let mut sorted: Vec<&TimingMean> = entries.iter().collect();
    sorted.sort_by(|a, b| b.ms.total_cmp(&a.ms));
    sorted
}
