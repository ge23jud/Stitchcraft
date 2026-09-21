import os
import sys
import numpy as np
import h5py
from scipy.optimize import curve_fit, brentq
from scipy.signal import fftconvolve
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QFileDialog, QMessageBox, QFrame,
    QScrollArea, QStyle, QDoubleSpinBox, QCheckBox, QDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plotting import (
    PGCanvas, MultiLinePlotter, CategoricalScheme, DraggableSpan,
    make_pg_toolbar, _COMPACT_BTN_STYLE,
)
from io_utils import _read_trpl_h5_file

# Colors for each non-final "fixing" stage's DraggableSpan / confirmed
# overlay, keyed by stage index (0 = slowest). The final (fastest, freely
# fit) stage always uses green — see _add_stage_span(). At most
# 2 non-final stages ever exist (n_exp is capped at 3), so this never needs
# a 3rd entry.
_FIXED_STAGE_COLORS = {0: (220, 140, 0), 1: (160, 60, 200)}   # slow: orange, medium: purple


def _rank_name(rank, n_exp):
    """Human name for the component at decay-speed rank `rank` (0 =
    slowest) out of n_exp total components."""
    if n_exp == 1:
        return ""
    if n_exp == 2:
        return ("slow", "fast")[rank]
    if n_exp == 3:
        return ("slow", "medium", "fast")[rank]
    raise ValueError(f"unsupported n_exp={n_exp}")


def _guess_slot_for_stage(idx, n_exp):
    """Map a fit stage index to one of the 3 fixed "slow"/"medium"/"fast"
    initial-guess slots the Settings dialog exposes (see TRPLTab._guess_
    widgets), independent of n_exp — unlike _rank_name's labels (which
    shift meaning with n_exp, e.g. rank 0 is "slow" for n_exp=2 but the
    *only* component for n_exp=1), the slot a given stage reads its guess
    override from stays fixed: the final stage (freely fit, whatever its
    index) always reads "fast", stage 0 (only present for n_exp >= 2)
    always reads "slow", and stage 1 (only present for n_exp == 3) always
    reads "medium". This lets a user's typed-in guesses for e.g. the fast
    component survive switching the Exponentials combo back and forth."""
    if idx == n_exp - 1:
        return "fast"
    if idx == 0:
        return "slow"
    return "medium"


def _multiexp(t, *params):
    """Sum of exponential decays evaluated at t: sum_i a_i * exp(-b_i * t).
    params is the flattened [a_1, b_1, a_2, b_2, ...] — the variadic shape
    scipy.optimize.curve_fit requires for its model-function argument."""
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(t)
    for i in range(len(params) // 2):
        a, b = params[2 * i], params[2 * i + 1]
        y = y + a * np.exp(-b * t)
    return y


def _multiexp_initial_guess(x, y, n):
    """Seed n components with distinct, geometrically-spaced decay rates
    spanning the selected window's width (so they don't all converge on the
    same local optimum), and an amplitude guess that reproduces the
    window's first sample when the n components are summed at that point —
    curve_fit refines both from there."""
    t_ref = float(x[0])
    width = max(float(x[-1]) - t_ref, 1e-9)
    taus = np.geomspace(width / (2 * n), width * 2, n)
    b0 = 1.0 / taus

    y_ref = float(y[0])
    if y_ref <= 0:
        y_max = float(np.max(y))
        y_ref = y_max if y_max > 0 else 1.0
    # a * exp(-b * t_ref) = y_ref / n  =>  a = (y_ref / n) * exp(b * t_ref)
    a0 = (y_ref / n) * np.exp(np.clip(b0 * t_ref, -700, 700))

    p0 = np.empty(2 * n)
    p0[0::2] = a0
    p0[1::2] = b0
    return p0


def _fixed_components_model(fixed_params):
    """Return a 2-parameter (a, b) curve_fit model: one free exponential
    plus a sum of already-fixed exponential terms added in as constants —
    used for every stage past the first in the iterative fit, where all
    slower components were already determined independently and only the
    current (faster) one is actually being optimized."""
    fixed_arr = np.asarray(fixed_params, dtype=float)   # (k, 2)

    def _model(t, a, b):
        y = a * np.exp(-b * t)
        for af, bf in fixed_arr:
            y = y + af * np.exp(-bf * t)
        return y
    return _model


def _convolved_model(t_full, irf_norm, mask, fixed_params=None):
    """Return a 2-parameter (a, b) curve_fit model — same free/fixed-
    component convention as _multiexp/_fixed_components_model — but for
    "Include IRF convolution": Counts(t) = IRF(t) (*) D(t), where D is the
    sum of the free component being fit plus every already-fixed (slower)
    component, and (*) is convolution.

    A convolution's value at any t depends on D's entire history back to
    t=0 (the IRF's own time origin), not just the fit window — unlike a
    bare exponential, D can't be evaluated locally over just the picked
    range. So D is always evaluated over the FULL time axis `t_full`
    (self._times), convolved with `irf_norm` (the sum-normalized,
    offset/crop/baseline-adjusted AND offset-aligned-to-t_full working
    IRF kernel — see _irf_normalized(), which does the sample-index shift
    that actually makes "IRF Time Offset" affect this convolution, since
    fftconvolve() itself only ever sees indices, never time values), and
    only then sliced down to `mask` (the boolean mask over `t_full`
    selecting this stage's fit window) to compare against the windowed
    data curve_fit is actually given. This implicitly assumes the IRF and
    the main data share the same channel width (ns/sample) — true for
    both loaded from the same instrument settings, which is the normal
    case this feature is meant for.

    Uses scipy.signal.fftconvolve rather than np.convolve: identical
    'full'-mode result, but FFT-based (O(N log N) vs np.convolve's
    O(N*M)) — necessary for interactive use against TRPL's ~65536-sample
    histograms, since curve_fit calls this model many times per fit."""
    fixed_arr = np.asarray(fixed_params, dtype=float) if fixed_params else None

    def _model(_t_window, a, b):
        params = [a, b]
        if fixed_arr is not None:
            params.extend(fixed_arr.ravel())
        D = _multiexp(t_full, *params)
        conv = fftconvolve(irf_norm, D, mode="full")[: len(t_full)]
        return conv[mask]
    return _model


def _free_component_initial_guess(x, y_resid, fixed_params):
    """Seed the current stage's free component from the residual left after
    subtracting every already-fixed (slower) component, biased toward a
    distinctly shorter lifetime than the fastest one fixed so far (and than
    the window itself) — otherwise curve_fit tends to converge it onto the
    same timescale as an already-fixed component instead of separating out
    a genuinely faster decay."""
    t_ref = float(x[0])
    width = max(float(x[-1]) - t_ref, 1e-9)
    b_fixed_max = max((b for _a, b in fixed_params), default=0.0)
    tau_ref = 1.0 / b_fixed_max if b_fixed_max > 0 else width
    tau_free = max(min(width / 10.0, tau_ref / 5.0), 1e-9)
    b0 = 1.0 / tau_free

    y_ref = float(y_resid[0])
    if y_ref <= 0:
        y_max = float(np.max(y_resid))
        y_ref = y_max if y_max > 0 else 1.0
    a0 = y_ref * np.exp(np.clip(b0 * t_ref, -700, 700))
    return np.array([a0, b0])


def _solve_total_decay_time(b_values):
    """Numerically solve 1/e = (1/n) * sum_i exp(-b_i * t) for t: the time at
    which the *unweighted* average of each component's own normalized
    (amplitude-1) decay curve has fallen to 1/e — a single "overall decay
    time" summarizing an N>=2 multi-exponential fit's b_i's alone (no a_i
    dependence), distinct from any individual component's own 1/b_i
    lifetime. f(t) = mean_i(exp(-b_i*t)) - 1/e is strictly decreasing from
    f(0) = 1 - 1/e > 0 to f(t->inf) -> -1/e < 0, so there's always exactly
    one root; bracket it by doubling an upper bound (seeded from the
    slowest component's own 1/e time) until the sign flips, then refine
    with brentq."""
    b = np.asarray(b_values, dtype=float)

    def f(t):
        return np.mean(np.exp(-b * t)) - 1.0 / np.e

    t_hi = 1.0 / np.min(b)
    while f(t_hi) > 0:
        t_hi *= 2.0
    return brentq(f, 0.0, t_hi)


class TRPLTab(QWidget):
    """"TRPL": load a TRPL histogram HDF5 file (as written by the Convert
    tab), display Counts vs. Times on a log-y axis, and fit 1, 2, or 3
    exponential decays (a combo box, capped at 3 — not a free-form N) to a
    user-picked time window, Counts = sum_i a_i * exp(-b_i * Times)
    (nonlinear least squares via scipy.optimize.curve_fit), to extract
    per-component lifetimes (1/b_i). Results are saved into the file's
    "analysis" HDF5 subgroup, mirroring the Analysis tab's save convention.

    N >= 2 is an iterative, N-stage fit, never one joint 2N-parameter
    optimization: PL decays here are typically ordered slow/medium/fast
    where, at long enough times, faster components have already died out
    and only the slower ones remain. So stage 0 drags a tail range where
    only the slowest decay remains and fits it alone; each subsequent
    stage k drags a wider range and fits ONLY the next-fastest component
    freely, with every previously-determined (slower) component *fixed* as
    a constant term in the model (_fixed_components_model) — always just 2
    free parameters per stage, never more, regardless of N. The final stage
    (the fastest component) is fit over the full time axis by default. This
    avoids components trading off against each other the way an
    unconstrained joint fit easily can, since each slower component is
    already well-constrained by data where faster ones contribute
    essentially nothing.

    Every stage's dragged span stays live and adjustable for the rest of
    the attempt (self._stage_progress) — moving on to add the next stage's
    span never freezes or removes an earlier one, so e.g. the slow-decay
    span can still be dragged while the fast (final) span is being picked.
    Dragging any span re-fits that stage and cascades the result forward
    through every later stage already added, since each stage's model
    bakes in every earlier stage's fitted params as a fixed constant.
    Canceling at any point still aborts the whole attempt — there's no
    undo of just one stage, only "start over" or "keep going forward".

    Optional "Include IRF convolution" (requires an IRF loaded — see
    below): instead of fitting the bare exponential sum directly, each
    stage fits Counts(t) = IRF(t) convolved with that sum (the sum-
    normalized, currently offset/crop/baseline-adjusted working IRF —
    see _convolved_model()/_irf_normalized()). Read from the checkbox
    once at the start of each attempt (self._fit_use_irf_convolution,
    same convention as self._n_exp), so toggling it mid-attempt has no
    effect until the next "Select fit range…". Every displayed fit curve
    (the live per-stage preview and the final confirmed overlay) reflects
    whichever mode was actually used (_eval_fit_curve()).

    Optional baseline subtraction, done before any lifetime fitting: drag a
    span over a flat region (e.g. the long-time dark-count tail), a constant
    is fit to it (least-squares constant = mean Counts over the range), and
    that constant is subtracted from every Counts sample — the lifetime
    fit above then runs on this baseline-subtracted working Counts array.
    Applying or resetting the baseline invalidates any already-confirmed
    lifetime fit, since it'd otherwise be stale relative to the new
    Counts. If used, "Save" additionally writes the fitted constant as
    "Baseline" (scalar) and the subtracted array as "CountsBased" into the
    same "analysis" subgroup.

    Optional per-stage initial-guess override (added 2026-09): by default
    every stage's free-component p0 for curve_fit comes from
    _multiexp_initial_guess() (stage 0) or _free_component_initial_guess()
    (later stages) — both heuristics, not always a good starting point for
    unusual data. The Settings dialog (see below) offers 3 fixed slots,
    "slow"/"medium"/"fast" (_guess_slot_for_stage() maps a stage index to
    one of these independent of n_exp — the final stage always reads
    "fast", stage 0 always reads "slow", stage 1, n_exp==3 only, always
    reads "medium" — so a typed guess survives switching the Exponentials
    combo), each an optional (a0, b0) pair via its own checkbox + two
    spinboxes. Unchecked (the default) leaves that stage on the automatic
    heuristic; checked, its (a0, b0) is used verbatim as p0 instead — read
    once into self._fit_initial_guesses at _on_start_selecting(), same
    "settings are read at attempt-start, not live" convention as
    self._n_exp/self._fit_use_irf_convolution, so edits made mid-attempt
    apply only to the next "Select fit range…".

    All fitting-procedure settings — Exponentials count, "Include IRF
    convolution", and the initial-guess overrides above — live in a
    separate non-modal "TRPL Fit Settings" dialog (added 2026-09,
    self._fit_settings_dialog, opened via the sidebar's "Settings…"
    button), keeping the sidebar itself down to "Select fit range…",
    "Reset fit", and the status label. The dialog's widgets are still
    plain instance attributes (self._n_exp_combo, self._chk_irf_
    convolution, self._guess_widgets) read the same way as before this
    change — only their parent widget moved, not how/when they're read.

    Optional IRF (Instrument Response Function) overlay: load a second TRPL
    histogram from its own HDF5 file, plotted on the same axes alongside
    Counts. Entirely independent of the Counts/fit workflow above — never
    fit, never written into "Save fit results to HDF5" — and available even
    before/without a main file being loaded. "Hide IRF"/"Show IRF" toggles
    its visibility without discarding it. Its own baseline subtraction
    (same drag-a-span/constant-fit/subtract pattern as the main Counts
    baseline) is available, independently of whether the main Counts
    baseline has been applied. "IRF Time Offset" shifts the whole IRF curve
    along the time axis by a constant, nudged by two arrow buttons (+/- a
    user-set step size) since there's a single value to pick rather than a
    range — typically used to line its rising edge up with the main Counts
    curve's own, since the two don't necessarily share a common t=0. "Crop
    IRF" (same drag-a-span/Confirm pattern) discards data points outside a
    picked range entirely, e.g. to drop a stray reflection peak or a noisy
    tail. The offset, baseline, and crop are all in-memory only: none of
    them is ever written back into the loaded IRF's own file, only cached
    in this session's own state (_irf_times/_irf_counts, recomputed from
    the immutable _irf_times_raw/_irf_counts_raw by
    _recompute_irf_working_arrays() whenever any of the three changes) for
    display/analysis.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── State ────────────────────────────────────────────────
        self._path  = None
        self._times_raw = None   # ns, exactly as loaded — never modified
        self._times = None   # working — raw plus the current data time offset
        self._counts_raw = None   # ns-indexed raw Counts, as loaded
        self._counts = None        # working Counts — raw, or raw minus baseline
        self._meta  = {}

        # Data time offset (added 2026-09): a constant (ns) added to
        # _times_raw to shift the whole Counts curve's own time axis —
        # mirrors the IRF's own time offset below (same "raw immutable
        # array + a confirmed/pending offset, recomputed into a working
        # array" convention — see _recompute_data_working_times()). Useful
        # e.g. to correct a fixed acquisition/trigger delay in the file's
        # own Times convention (io_utils._parse_trpl_dat's "0, ns/channel,
        # …", which just means "start of acquisition", not necessarily
        # "start of the actual decay") without needing an IRF loaded at
        # all. Deliberately in-memory only, like the IRF's own offset:
        # never written back into the loaded file's own Times dataset —
        # only cached here as self._times, which every fit range/baseline
        # range below is picked against (not _times_raw). Because of that,
        # the confirmed value is recorded alongside any saved fit as
        # "DataTimeOffset" so a saved FitRange/SlowRange/MediumRange/
        # baseline range can be related back to the file's own raw Times
        # dataset later (see _on_save()) — and the Visualizer tab's TRPL
        # overlay, which only ever reads a file's raw Times, reverses this
        # same offset before drawing the saved fit against it.
        self._data_time_offset   = 0.0    # confirmed offset (ns)
        self._data_offset_pending = None  # live value while adjusting, else None

        self._mode = "idle"   # "idle" | "selecting" | "baseline_selecting"
                               # | "data_offset_selecting"
                               # | "irf_baseline_selecting" | "irf_offset_selecting"
                               # | "irf_crop_selecting"
        self._n_exp     = 0      # 1, 2, or 3 exponential components
        # Whether the current (or most-recently-confirmed) fit attempt
        # convolves the exponential sum with the IRF before comparing to
        # data — read from _chk_irf_convolution once at _on_start_selecting(),
        # same "settings are read at attempt-start, not live" convention as
        # _n_exp above, so it stays frozen at whatever an already-confirmed
        # fit actually used even if the checkbox is toggled afterward.
        self._fit_use_irf_convolution = False
        self._stage_idx = None   # 0-based: the highest-index stage added so far
                                  # (rank order: 0 = slowest ... n_exp-1 = fastest, the last/final stage)
        self._fit_range  = None   # (xmin, xmax) — the final stage's (fastest component's) fit range
        self._fit_params = None   # (n_exp, 2) ndarray: [a_i, b_i], ordered fastest (row 0) to slowest
        self._fit_cov    = None   # (2, 2) — covariance from the LAST stage's own free-component fit only

        # Per-stage initial-guess (p0) overrides, read once from the
        # Settings dialog's checkboxes/spinboxes at _on_start_selecting() —
        # {"slow"|"medium"|"fast": (a0, b0)}, only for slots whose checkbox
        # was checked. Empty means every stage uses its usual automatic
        # heuristic (_multiexp_initial_guess/_free_component_initial_guess).
        # See _guess_slot_for_stage() and _fit_stage()'s use of this.
        self._fit_initial_guesses = {}

        # Populated only once the whole multi-stage attempt is finally
        # confirmed (see _finalize_fit()): every non-final ("fixing")
        # stage's result, in the order they were added (slowest first),
        # frozen for save/display — {"range": (xmin,xmax), "params": (a,b),
        # "cov": (2,2)}. While an attempt is still in progress, its state
        # instead lives in _stage_progress below, where every stage stays
        # live/adjustable simultaneously.
        self._fixed_stages = []

        # In-progress multi-stage fit: one entry per stage 0..stage_idx,
        # each {"span": DraggableSpan, "range": (xmin,xmax), "params":
        # (a,b) or None, "cov": (2,2) or None, "error": str or None,
        # "preview_item": PlotDataItem or None} — every entry's span stays
        # movable and connected for as long as the attempt is in progress,
        # not just the newest one. See _add_stage_span()/_on_stage_span_changed().
        self._stage_progress = []

        # Baseline subtraction: a constant fit (mean Counts) over a
        # user-picked range, subtracted from every Counts sample.
        self._baseline_range   = None   # (xmin, xmax) — None until confirmed
        self._baseline_value   = None   # fitted constant — None until confirmed
        self._baseline_span    = None   # DraggableSpan, active during selection
        self._baseline_pending_range = None
        self._baseline_pending_value = None
        self._baseline_preview_item  = None  # live dashed baseline-level line

        # IRF (Instrument Response Function): an optional second TRPL
        # histogram overlaid on the same plot for reference (e.g. system
        # timing-jitter characterization), loaded from its own HDF5 file —
        # entirely independent of the main Counts/fit workflow above and of
        # whether a main file is even loaded yet.
        self._irf_path  = None
        self._irf_times_raw  = None   # ns, exactly as loaded — never modified
        self._irf_times      = None   # working — raw plus the current time offset
        self._irf_counts_raw = None   # as loaded
        self._irf_counts     = None   # working — raw, or raw minus its own baseline
        self._irf_meta  = {}
        self._irf_visible = True   # toggled by the "Hide IRF"/"Show IRF" button

        # IRF baseline subtraction: same "drag a span, constant = mean
        # Counts over it, subtract" pattern as the main Counts baseline
        # above, but entirely independent (its own range/value/span) since
        # the IRF is its own separate dataset.
        self._irf_baseline_range = None
        self._irf_baseline_value = None
        self._irf_baseline_span  = None
        self._irf_baseline_pending_range = None
        self._irf_baseline_pending_value = None
        self._irf_baseline_preview_item  = None

        # IRF time offset: a constant (ns) added to _irf_times_raw to shift
        # the whole IRF curve along the time axis — e.g. to line up its
        # rising edge with the main Counts curve's own rising edge, since
        # the IRF and the sample measurement don't necessarily share a
        # common t=0. Adjusted via two arrow buttons that nudge it by a
        # user-set step size (see _btn_irf_offset_dec/_inc and
        # _irf_offset_step_spin) rather than a DraggableSpan, since there's
        # no "range" to pick — the value itself is what's being chosen.
        # Deliberately in-memory only: never written back into the loaded
        # IRF's own HDF5 file, only cached here as _irf_times for use by
        # this session's own display/analysis (baseline subtraction, any
        # later deconvolution-style use).
        self._irf_time_offset   = 0.0    # confirmed offset (ns)
        self._irf_offset_pending = None  # live value while adjusting, else None

        # IRF crop: a (xmin, xmax) ns range outside of which IRF data
        # points are discarded from the working arrays entirely, e.g. to
        # drop a stray reflection peak or a noisy tail. Same drag-a-
        # DraggableSpan/Confirm pattern as the baseline selectors, but
        # nothing is computed from the picked range — it's applied as-is
        # as a keep-window. Stored in RAW-time coordinates (converted at
        # confirm time from whatever was picked on the currently-displayed,
        # offset-shifted axis — see _on_irf_crop_confirm()), specifically
        # so it stays anchored to the same physical raw samples regardless
        # of the offset being changed later — see _recompute_irf_working_
        # arrays() for why applying it against the offset-shifted axis
        # instead would make the kept window silently slide whenever the
        # offset changes after the crop was picked. Deliberately in-memory
        # only, like the offset and baseline above: never written back
        # into the loaded IRF's own file.
        self._irf_crop_range = None   # (xmin, xmax) RAW ns — None until confirmed
        self._irf_crop_pending_range = None
        self._irf_crop_span  = None   # DraggableSpan, active during selection

        # ── Build UI ────────────────────────────────────────────────
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        scroll = QScrollArea()
        scrollbar_w = scroll.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        scroll.setFixedWidth(300 + scrollbar_w)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar = QWidget()
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # File group
        g_file = QGroupBox("TRPL HDF5 file")
        gfl = QVBoxLayout(g_file)
        self._file_lbl = QLabel("No file loaded")
        self._file_lbl.setWordWrap(True)
        gfl.addWidget(self._file_lbl)
        btn_load = QPushButton("Load HDF5…")
        btn_load.clicked.connect(self._on_load_file)
        gfl.addWidget(btn_load)
        sl.addWidget(g_file)

        # Data time offset group (added 2026-09) — same drag-free,
        # arrow-button-nudge UX as "IRF Time Offset" below, but shifts the
        # main Counts curve's own time axis instead of the IRF's. Placed
        # ahead of baseline/fit so it's the first adjustment made to a
        # newly-loaded file, since everything after it (baseline ranges,
        # fit ranges, and — if an IRF is also loaded — its own offset
        # alignment) is picked against the resulting shifted self._times.
        g_data_offset = QGroupBox("Data time offset")
        dol = QVBoxLayout(g_data_offset)
        self._btn_data_offset = QPushButton("Data Time Offset")
        self._btn_data_offset.setEnabled(False)
        self._btn_data_offset.clicked.connect(self._on_start_data_offset)
        dol.addWidget(self._btn_data_offset)

        data_step_row = QHBoxLayout()
        data_step_row.addWidget(QLabel("Step:"))
        self._data_offset_step_spin = QDoubleSpinBox()
        self._data_offset_step_spin.setDecimals(4)
        self._data_offset_step_spin.setRange(0.0001, 1000.0)
        self._data_offset_step_spin.setSingleStep(0.01)
        self._data_offset_step_spin.setValue(0.1)
        self._data_offset_step_spin.setSuffix(" ns")
        data_step_row.addWidget(self._data_offset_step_spin)
        dol.addLayout(data_step_row)

        data_arrow_row = QHBoxLayout()
        self._btn_data_offset_dec = QPushButton("◀")
        self._btn_data_offset_dec.setToolTip("Shift the data earlier by the step size")
        self._btn_data_offset_dec.setEnabled(False)
        self._btn_data_offset_dec.clicked.connect(lambda: self._on_data_offset_step(-1))
        data_arrow_row.addWidget(self._btn_data_offset_dec)
        self._btn_data_offset_inc = QPushButton("▶")
        self._btn_data_offset_inc.setToolTip("Shift the data later by the step size")
        self._btn_data_offset_inc.setEnabled(False)
        self._btn_data_offset_inc.clicked.connect(lambda: self._on_data_offset_step(+1))
        data_arrow_row.addWidget(self._btn_data_offset_inc)
        dol.addLayout(data_arrow_row)

        self._btn_data_offset_reset = QPushButton("Reset data offset")
        self._btn_data_offset_reset.setEnabled(False)
        self._btn_data_offset_reset.clicked.connect(self._on_reset_data_offset)
        dol.addWidget(self._btn_data_offset_reset)

        self._data_offset_status_lbl = QLabel("No data time offset applied.")
        self._data_offset_status_lbl.setWordWrap(True)
        dol.addWidget(self._data_offset_status_lbl)
        g_data_offset.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_data_offset)

        # IRF (Instrument Response Function) group — deliberately minimal
        # (added 2026-09, same convention as the Lifetime fit group below):
        # every widget for IRF baseline subtraction, time offset, and crop
        # lives in the separate "TRPL IRF Settings" dialog (see
        # _build_irf_settings_dialog()), opened via "Settings…" below — only
        # loading the IRF and toggling its visibility stay in the sidebar.
        g_irf = QGroupBox("IRF (Instrument Response Function)")
        irfl = QVBoxLayout(g_irf)
        self._irf_file_lbl = QLabel("No IRF loaded")
        self._irf_file_lbl.setWordWrap(True)
        irfl.addWidget(self._irf_file_lbl)

        btn_irf_load = QPushButton("Load IRF…")
        btn_irf_load.clicked.connect(self._on_load_irf)
        irfl.addWidget(btn_irf_load)

        self._btn_irf_hide = QPushButton("Hide IRF")
        self._btn_irf_hide.setEnabled(False)
        self._btn_irf_hide.clicked.connect(self._on_toggle_irf_visibility)
        irfl.addWidget(self._btn_irf_hide)

        self._irf_settings_dialog = self._build_irf_settings_dialog()
        btn_irf_settings = QPushButton("Settings…")
        btn_irf_settings.clicked.connect(self._on_open_irf_settings)
        irfl.addWidget(btn_irf_settings)

        g_irf.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_irf)

        # Baseline subtraction group
        g_base = QGroupBox("Baseline subtraction")
        basel = QVBoxLayout(g_base)
        self._btn_baseline_select = QPushButton("Select baseline range…")
        self._btn_baseline_select.setEnabled(False)
        self._btn_baseline_select.clicked.connect(self._on_start_baseline_selecting)
        basel.addWidget(self._btn_baseline_select)

        self._btn_baseline_reset = QPushButton("Reset baseline")
        self._btn_baseline_reset.setEnabled(False)
        self._btn_baseline_reset.clicked.connect(self._on_reset_baseline)
        basel.addWidget(self._btn_baseline_reset)

        self._baseline_status_lbl = QLabel("No baseline applied.")
        self._baseline_status_lbl.setWordWrap(True)
        basel.addWidget(self._baseline_status_lbl)
        g_base.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_base)

        # Fit group — deliberately minimal (added 2026-09): every widget
        # that configures *how* a fit runs (exponential count, IRF
        # convolution, initial-guess overrides) lives in the separate
        # "TRPL Fit Settings" dialog (see _build_fit_settings_dialog()),
        # opened via "Settings…" below — only the two actions that drive
        # the fit-range-selection workflow itself, plus its status readout,
        # stay in the sidebar.
        g_fit = QGroupBox("Lifetime fit  (Σ aᵢ·exp(−bᵢ·t))")
        fitl = QVBoxLayout(g_fit)

        self._fit_settings_dialog = self._build_fit_settings_dialog()

        btn_fit_settings = QPushButton("Settings…")
        btn_fit_settings.clicked.connect(self._on_open_fit_settings)
        fitl.addWidget(btn_fit_settings)

        self._btn_start = QPushButton("Select fit range…")
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._on_start_selecting)
        fitl.addWidget(self._btn_start)

        self._btn_reset_fits = QPushButton("Reset fit")
        self._btn_reset_fits.setEnabled(False)
        self._btn_reset_fits.clicked.connect(self._on_reset_fits)
        fitl.addWidget(self._btn_reset_fits)

        self._fit_status_lbl = QLabel("")
        self._fit_status_lbl.setWordWrap(True)
        fitl.addWidget(self._fit_status_lbl)
        g_fit.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_fit)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        self._btn_save = QPushButton("Save fit results to HDF5")
        bold = QFont(); bold.setBold(True)
        self._btn_save.setFont(bold)
        self._btn_save.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; }"
            "QPushButton:disabled { background-color: #bbbbbb; color: #666666; }"
        )
        self._btn_save.setMinimumHeight(32)
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        sl.addWidget(self._btn_save)

        sl.addStretch(1)
        scroll.setWidget(sidebar)
        layout.addWidget(scroll)

        # ── Right: toolbar + canvas + action bar ─────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self._canvas  = PGCanvas(right, welcome_msg="Load a TRPL HDF5 file to begin.")
        self._toolbar = make_pg_toolbar(self._canvas, right)
        rl.addWidget(self._toolbar)
        rl.addWidget(self._canvas, stretch=1)

        self._action_bar = QFrame()
        self._action_bar.setFrameShape(QFrame.StyledPanel)
        ab = QHBoxLayout(self._action_bar)
        ab.setContentsMargins(8, 4, 8, 4)
        self._action_lbl = QLabel("")
        self._btn_act_confirm = QPushButton("Confirm fit")
        self._btn_act_cancel  = QPushButton("Cancel selection")
        self._btn_act_confirm.clicked.connect(self._on_action_confirm)
        self._btn_act_cancel.clicked.connect(self._on_action_cancel)
        ab.addWidget(self._action_lbl, stretch=1)
        ab.addWidget(self._btn_act_confirm)
        ab.addWidget(self._btn_act_cancel)
        self._action_bar.setVisible(False)
        rl.addWidget(self._action_bar)

        layout.addWidget(right, stretch=1)

    # ── Fit settings dialog ─────────────────────────────────────
    # Added 2026-09: every widget that configures how the next fit attempt
    # runs — Exponentials count, "Include IRF convolution", and the
    # per-stage initial-guess overrides — lives here rather than inline in
    # the sidebar. The dialog is built once (lazily reused, not
    # recreated) and shown non-modally so it can stay open alongside the
    # plot while the user adjusts settings and clicks "Select fit range…".

    def _build_fit_settings_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("TRPL Fit Settings")
        dlg.setModal(False)
        dlg.setStyleSheet(_COMPACT_BTN_STYLE)
        vlay = QVBoxLayout(dlg)

        g_cfg = QGroupBox("Fit configuration")
        cfgl = QVBoxLayout(g_cfg)
        r_n = QHBoxLayout()
        r_n.addWidget(QLabel("Exponentials:"))
        self._n_exp_combo = QComboBox()
        self._n_exp_combo.addItem("1 — single exponential", 1)
        self._n_exp_combo.addItem("2 — double (fast + slow)", 2)
        self._n_exp_combo.addItem("3 — triple (fast + medium + slow)", 3)
        r_n.addWidget(self._n_exp_combo)
        cfgl.addLayout(r_n)

        self._chk_irf_convolution = QCheckBox("Include IRF convolution")
        self._chk_irf_convolution.setEnabled(False)
        self._chk_irf_convolution.setToolTip(
            "Fit Counts(t) = IRF(t) ⊛ Σᵢ aᵢ·exp(−bᵢ·t) "
            "instead of the bare sum — requires an IRF to be loaded."
        )
        cfgl.addWidget(self._chk_irf_convolution)
        vlay.addWidget(g_cfg)

        # Per-stage initial-guess (p0) overrides — 3 fixed slots regardless
        # of the current Exponentials selection (see _guess_slot_for_stage()
        # for why the mapping from stage index to slot is n_exp-independent
        # and so a typed guess survives switching the combo). Each slot
        # starts unchecked/disabled: the stage falls back to its automatic
        # heuristic (_multiexp_initial_guess/_free_component_initial_guess)
        # unless the user explicitly opts in.
        g_guess = QGroupBox("Initial guesses (optional)")
        g_guess.setToolTip(
            "Override the automatic p0 seed scipy.optimize.curve_fit starts "
            "from for a stage's free component. Leave unchecked to use the "
            "automatic guess. Read once when \"Select fit range…\" starts a "
            "new attempt — changes here don't affect a fit already in "
            "progress."
        )
        guess_grid = QGridLayout(g_guess)
        guess_grid.addWidget(QLabel("Component"), 0, 0)
        guess_grid.addWidget(QLabel("a₀ (counts)"), 0, 1, 1, 2)
        guess_grid.addWidget(QLabel("b₀ (1/ns)"), 0, 3, 1, 2)

        self._guess_widgets = {}
        rows = [
            ("slow",   "Slow (1st stage, N≥2)"),
            ("medium", "Medium (2nd stage, N=3)"),
            ("fast",   "Fast (final stage)"),
        ]
        for row, (slot, label) in enumerate(rows, start=1):
            chk = QCheckBox(label)
            a_spin = QDoubleSpinBox()
            a_spin.setDecimals(3)
            a_spin.setRange(0.0, 1.0e12)
            a_spin.setSingleStep(10.0)
            a_spin.setValue(1.0)
            a_spin.setEnabled(False)
            b_spin = QDoubleSpinBox()
            b_spin.setDecimals(6)
            b_spin.setRange(0.0, 1.0e6)
            b_spin.setSingleStep(0.01)
            b_spin.setValue(1.0)
            b_spin.setEnabled(False)
            # Bind a=a_spin, b=b_spin at connect-time so each checkbox only
            # ever toggles its own row's spinboxes.
            chk.toggled.connect(lambda checked, a=a_spin, b=b_spin: (
                a.setEnabled(checked), b.setEnabled(checked)))

            guess_grid.addWidget(chk, row, 0)
            guess_grid.addWidget(a_spin, row, 1, 1, 2)
            guess_grid.addWidget(b_spin, row, 3, 1, 2)
            self._guess_widgets[slot] = (chk, a_spin, b_spin)
        vlay.addWidget(g_guess)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.close)
        vlay.addWidget(btn_close)
        return dlg

    def _on_open_fit_settings(self):
        self._fit_settings_dialog.show()
        self._fit_settings_dialog.raise_()
        self._fit_settings_dialog.activateWindow()

    # ── IRF settings dialog ──────────────────────────────────────
    # Added 2026-09, same convention as the fit-settings dialog above:
    # every IRF baseline-subtraction/time-offset/crop widget lives in this
    # separate non-modal dialog (built once, reused) rather than inline in
    # the sidebar — only "Load IRF…" and "Hide IRF"/"Show IRF" (which need
    # to stay reachable without opening anything) remain there. Being
    # non-modal, it can stay open alongside the plot while a
    # baseline/offset/crop selection is dragged on the canvas and confirmed
    # via the shared action bar, exactly as before this change.

    def _build_irf_settings_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("TRPL IRF Settings")
        dlg.setModal(False)
        dlg.setStyleSheet(_COMPACT_BTN_STYLE)
        vlay = QVBoxLayout(dlg)

        g_base = QGroupBox("IRF baseline subtraction")
        basel = QVBoxLayout(g_base)
        self._btn_irf_baseline_select = QPushButton("Select IRF baseline range…")
        self._btn_irf_baseline_select.setEnabled(False)
        self._btn_irf_baseline_select.clicked.connect(self._on_start_irf_baseline_selecting)
        basel.addWidget(self._btn_irf_baseline_select)

        self._btn_irf_baseline_reset = QPushButton("Reset IRF baseline")
        self._btn_irf_baseline_reset.setEnabled(False)
        self._btn_irf_baseline_reset.clicked.connect(self._on_reset_irf_baseline)
        basel.addWidget(self._btn_irf_baseline_reset)

        self._irf_baseline_status_lbl = QLabel("No IRF baseline applied.")
        self._irf_baseline_status_lbl.setWordWrap(True)
        basel.addWidget(self._irf_baseline_status_lbl)
        vlay.addWidget(g_base)

        g_offset = QGroupBox("IRF time offset")
        offl = QVBoxLayout(g_offset)
        self._btn_irf_offset = QPushButton("IRF Time Offset")
        self._btn_irf_offset.setEnabled(False)
        self._btn_irf_offset.clicked.connect(self._on_start_irf_offset)
        offl.addWidget(self._btn_irf_offset)

        # Step size applied per arrow-button click below — a plain user
        # preference, not itself part of the offset state, so it's left
        # alone (not reset) whenever a new IRF is loaded or the offset is
        # reset. 4 decimals gives sub-picosecond control if ever needed.
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step:"))
        self._irf_offset_step_spin = QDoubleSpinBox()
        self._irf_offset_step_spin.setDecimals(4)
        self._irf_offset_step_spin.setRange(0.0001, 1000.0)
        self._irf_offset_step_spin.setSingleStep(0.01)
        self._irf_offset_step_spin.setValue(0.1)
        self._irf_offset_step_spin.setSuffix(" ns")
        step_row.addWidget(self._irf_offset_step_spin)
        offl.addLayout(step_row)

        # Arrow buttons nudge the pending offset by +/- the step size above
        # and redraw immediately — enabled only while a "IRF Time Offset"
        # attempt is in progress (_on_start_irf_offset/_on_irf_offset_confirm
        # /_cancel), same lifecycle the slider they replaced had.
        arrow_row = QHBoxLayout()
        self._btn_irf_offset_dec = QPushButton("◀")
        self._btn_irf_offset_dec.setToolTip("Shift the IRF earlier by the step size")
        self._btn_irf_offset_dec.setEnabled(False)
        self._btn_irf_offset_dec.clicked.connect(lambda: self._on_irf_offset_step(-1))
        arrow_row.addWidget(self._btn_irf_offset_dec)
        self._btn_irf_offset_inc = QPushButton("▶")
        self._btn_irf_offset_inc.setToolTip("Shift the IRF later by the step size")
        self._btn_irf_offset_inc.setEnabled(False)
        self._btn_irf_offset_inc.clicked.connect(lambda: self._on_irf_offset_step(+1))
        arrow_row.addWidget(self._btn_irf_offset_inc)
        offl.addLayout(arrow_row)

        self._btn_irf_offset_reset = QPushButton("Reset IRF offset")
        self._btn_irf_offset_reset.setEnabled(False)
        self._btn_irf_offset_reset.clicked.connect(self._on_reset_irf_offset)
        offl.addWidget(self._btn_irf_offset_reset)

        self._irf_offset_status_lbl = QLabel("No IRF time offset applied.")
        self._irf_offset_status_lbl.setWordWrap(True)
        offl.addWidget(self._irf_offset_status_lbl)
        vlay.addWidget(g_offset)

        g_crop = QGroupBox("IRF crop")
        cropl = QVBoxLayout(g_crop)
        self._btn_irf_crop = QPushButton("Crop IRF")
        self._btn_irf_crop.setEnabled(False)
        self._btn_irf_crop.clicked.connect(self._on_start_irf_crop)
        cropl.addWidget(self._btn_irf_crop)

        self._btn_irf_crop_reset = QPushButton("Reset IRF crop")
        self._btn_irf_crop_reset.setEnabled(False)
        self._btn_irf_crop_reset.clicked.connect(self._on_reset_irf_crop)
        cropl.addWidget(self._btn_irf_crop_reset)

        self._irf_crop_status_lbl = QLabel("No IRF crop applied.")
        self._irf_crop_status_lbl.setWordWrap(True)
        cropl.addWidget(self._irf_crop_status_lbl)
        vlay.addWidget(g_crop)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.close)
        vlay.addWidget(btn_close)
        return dlg

    def _on_open_irf_settings(self):
        self._irf_settings_dialog.show()
        self._irf_settings_dialog.raise_()
        self._irf_settings_dialog.activateWindow()

    # ── Action bar dispatch ──────────────────────────────────────
    # The action bar (Confirm/Cancel) is shared between the lifetime-fit
    # range selector (itself possibly several stages, for N >= 2) and the
    # baseline-range selector — only one of those is ever active at a time,
    # so route to whichever is live.

    def _on_action_confirm(self):
        if self._mode == "selecting":
            self._on_confirm_fit()
        elif self._mode == "baseline_selecting":
            self._on_baseline_confirm()
        elif self._mode == "data_offset_selecting":
            self._on_data_offset_confirm()
        elif self._mode == "irf_baseline_selecting":
            self._on_irf_baseline_confirm()
        elif self._mode == "irf_offset_selecting":
            self._on_irf_offset_confirm()
        elif self._mode == "irf_crop_selecting":
            self._on_irf_crop_confirm()

    def _on_action_cancel(self):
        if self._mode == "selecting":
            self._on_reset_fits()
        elif self._mode == "baseline_selecting":
            self._on_baseline_cancel()
        elif self._mode == "data_offset_selecting":
            self._on_data_offset_cancel()
        elif self._mode == "irf_baseline_selecting":
            self._on_irf_baseline_cancel()
        elif self._mode == "irf_offset_selecting":
            self._on_irf_offset_cancel()
        elif self._mode == "irf_crop_selecting":
            self._on_irf_crop_cancel()

    # ── File loading ─────────────────────────────────────────────

    def _on_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load TRPL HDF5 file", "",
            "HDF5 files (*.h5);;All files (*.*)"
        )
        if not path:
            return
        try:
            data = _read_trpl_h5_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Could not read:\n{path}\n\n{exc}")
            return

        self._path   = path
        self._times_raw = np.asarray(data["times"], dtype=float)
        self._data_time_offset = 0.0
        self._data_offset_pending = None
        self._recompute_data_working_times()
        self._counts_raw = np.asarray(data["counts"], dtype=float)
        self._counts = self._counts_raw.copy()
        self._meta   = data

        self._reset_fit_state()
        self._reset_baseline_state()
        self._action_bar.setVisible(False)
        self._mode = "idle"

        label = os.path.basename(path)
        info = f"{label}\n{len(self._times)} channel(s)"
        if data.get("power_mW") is not None:
            info += f"\n{data['power_mW']:.4g} mW"
        self._file_lbl.setText(info)
        self._btn_save.setEnabled(False)
        self._fit_status_lbl.setText("")
        self._btn_baseline_reset.setEnabled(False)
        self._baseline_status_lbl.setText("No baseline applied.")
        self._btn_data_offset_reset.setEnabled(False)
        self._data_offset_status_lbl.setText("No data time offset applied.")
        self._set_exclusive_buttons_enabled(True)
        self._draw_base_plot()

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: loaded {label} ({len(self._times)} channels)."
            )

    # ── Data time offset ────────────────────────────────────────────
    # Added 2026-09, mirroring the IRF's own "IRF Time Offset" below:
    # shifts the main Counts curve's own time axis by a constant (ns),
    # nudged by two arrow buttons (+/- a user-set step size) rather than a
    # DraggableSpan, since there's a single value to pick rather than a
    # range. Every fit range and baseline range is picked against
    # self._times (the working, offset-applied array) elsewhere in this
    # tab, so applying or resetting this offset invalidates any already-
    # confirmed lifetime fit (_invalidate_lifetime_fits()) — its saved
    # range would otherwise silently select the wrong samples once
    # self._times shifts under it. The confirmed baseline is left alone:
    # its value is just a constant already subtracted from self._counts,
    # not tied to any particular time axis, so a later data-offset change
    # can't invalidate it the way it can a range-dependent fit.

    def _recompute_data_working_times(self, offset=None):
        """Rebuild self._times from the immutable self._times_raw plus the
        confirmed (or, if given, a live pending) data time offset — same
        convention as _recompute_irf_working_arrays(). self._counts/
        self._counts_raw are untouched: adding a constant to every time
        LABEL doesn't change which count value belongs to which sample."""
        if self._times_raw is None:
            return
        eff_offset = self._data_time_offset if offset is None else offset
        self._times = self._times_raw + eff_offset

    def _on_start_data_offset(self):
        if self._times_raw is None or self._mode != "idle":
            return

        self._mode = "data_offset_selecting"
        self._set_exclusive_buttons_enabled(False)

        self._data_offset_pending = self._data_time_offset
        self._btn_data_offset_dec.setEnabled(True)
        self._btn_data_offset_inc.setEnabled(True)
        self._data_offset_step_spin.setEnabled(True)
        self._data_offset_status_lbl.setText(f"Offset: {self._data_time_offset:+.4g} ns")

        self._btn_act_confirm.setEnabled(True)   # any value is valid, unlike a range pick
        self._btn_act_confirm.setText("Confirm data offset")
        self._action_bar.setVisible(True)

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                "TRPL: use the ◀/▶ buttons to shift the data's time axis, "
                "then Confirm."
            )

    def _on_data_offset_step(self, direction):
        if self._mode != "data_offset_selecting":
            return
        offset = self._data_offset_pending + direction * self._data_offset_step_spin.value()
        self._data_offset_pending = offset
        self._recompute_data_working_times(offset=offset)
        self._data_offset_status_lbl.setText(f"Offset: {offset:+.4g} ns")
        self._action_lbl.setText(f"Data time offset: {offset:+.4g} ns")
        self._redraw_preserving_view()

    def _on_data_offset_confirm(self):
        if self._mode != "data_offset_selecting":
            return
        self._data_time_offset = self._data_offset_pending
        self._data_offset_pending = None
        # Recompute from _times_raw rather than trusting the
        # click-accumulated working array verbatim, so the confirmed value
        # is exactly _times_raw + _data_time_offset with no accumulated
        # float drift from repeated clicks.
        self._recompute_data_working_times()

        # Any already-confirmed lifetime fit was picked against the old
        # self._times — its saved range would now select different
        # samples than the user actually chose, so it can't be trusted.
        self._invalidate_lifetime_fits()

        self._btn_data_offset_dec.setEnabled(False)
        self._btn_data_offset_inc.setEnabled(False)
        self._data_offset_step_spin.setEnabled(False)
        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._set_exclusive_buttons_enabled(True)
        self._btn_data_offset_reset.setEnabled(self._data_time_offset != 0.0)
        self._data_offset_status_lbl.setText(
            f"Data time offset: {self._data_time_offset:+.4g} ns"
        )
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: data time offset {self._data_time_offset:+.4g} ns confirmed."
            )

    def _on_data_offset_cancel(self):
        self._data_offset_pending = None
        # Revert the working array back to the last confirmed offset — it
        # may have been live-shifted by arrow-button clicks since.
        self._recompute_data_working_times()

        self._btn_data_offset_dec.setEnabled(False)
        self._btn_data_offset_inc.setEnabled(False)
        self._data_offset_step_spin.setEnabled(False)
        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._set_exclusive_buttons_enabled(True)
        self._data_offset_status_lbl.setText(
            f"Data time offset: {self._data_time_offset:+.4g} ns"
            if self._data_time_offset != 0.0 else "No data time offset applied."
        )
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: data time offset selection canceled.")

    def _on_reset_data_offset(self):
        if self._data_time_offset == 0.0:
            return
        self._data_time_offset = 0.0
        self._recompute_data_working_times()
        self._invalidate_lifetime_fits()
        self._btn_data_offset_reset.setEnabled(False)
        self._data_offset_status_lbl.setText("No data time offset applied.")
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: data time offset reset.")

    # ── IRF (Instrument Response Function) ──────────────────────────
    # An optional second TRPL histogram loaded from its own file and
    # overlaid on the same plot, entirely independent of the main Counts
    # workflow above — its own visibility toggle and baseline subtraction,
    # no fitting of its own, never written into "Save fit results to HDF5".

    def _on_load_irf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load IRF HDF5 file", "",
            "HDF5 files (*.h5);;All files (*.*)"
        )
        if not path:
            return
        try:
            data = _read_trpl_h5_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Could not read:\n{path}\n\n{exc}")
            return

        # Loading new IRF data out from under an in-progress IRF-baseline,
        # IRF-offset, or IRF-crop adjustment would leave that span/pending
        # value pointing at stale arrays — cancel it first rather than
        # leave the app inconsistent.
        if self._mode == "irf_baseline_selecting":
            self._on_irf_baseline_cancel()
        elif self._mode == "irf_offset_selecting":
            self._on_irf_offset_cancel()
        elif self._mode == "irf_crop_selecting":
            self._on_irf_crop_cancel()

        self._irf_path = path
        self._irf_times_raw = np.asarray(data["times"], dtype=float)
        self._irf_counts_raw = np.asarray(data["counts"], dtype=float)
        self._irf_meta = data
        self._reset_irf_baseline_state()
        self._irf_time_offset = 0.0
        self._irf_offset_pending = None
        self._reset_irf_crop_state()
        self._recompute_irf_working_arrays()
        self._btn_irf_offset_reset.setEnabled(False)
        self._irf_offset_status_lbl.setText("No IRF time offset applied.")
        self._btn_irf_crop_reset.setEnabled(False)
        self._irf_crop_status_lbl.setText("No IRF crop applied.")
        self._irf_visible = True
        self._btn_irf_hide.setText("Hide IRF")
        self._irf_baseline_status_lbl.setText("No IRF baseline applied.")
        self._chk_irf_convolution.setEnabled(True)

        label = os.path.basename(path)
        info = f"{label}\n{len(self._irf_times)} channel(s)"
        self._irf_file_lbl.setText(info)

        # Only touch the shared plot/button state while idle — if another
        # selection mode is active, _draw_base_plot()'s full-clear rebuild
        # would corrupt its live DraggableSpan; the new IRF data still
        # picks up automatically once that mode returns to idle (every
        # exit path redraws via _draw_base_plot()).
        if self._mode == "idle":
            self._set_exclusive_buttons_enabled(True)
            self._draw_base_plot()   # safe with no main file loaded — see its own guard

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: loaded IRF {label} ({len(self._irf_times)} channels)."
            )

    def _on_toggle_irf_visibility(self):
        if self._irf_times is None or self._mode != "idle":
            return
        self._irf_visible = not self._irf_visible
        self._btn_irf_hide.setText("Hide IRF" if self._irf_visible else "Show IRF")
        self._redraw_preserving_view()

    def _redraw_preserving_view(self):
        """Rebuild the base plot without disturbing whatever pan/zoom the
        user currently has — _draw_base_plot() does a full plot_item.clear()
        -based rebuild (reset_axes()), which resets the view to a small
        known range before re-enabling auto-range. Used for redraws that
        don't change the main Counts data (e.g. toggling IRF visibility),
        where jumping the view would just be an unwanted side effect."""
        vb = self._canvas.plot_item.getViewBox()
        prev_range = vb.viewRange()
        self._draw_base_plot()
        vb.setRange(xRange=prev_range[0], yRange=prev_range[1], padding=0)

    def _set_exclusive_buttons_enabled(self, enabled):
        """Enable/disable every button that starts one of the six
        mutually-exclusive interactive modes (lifetime-fit range, Counts
        baseline, data time offset, IRF baseline, IRF time offset, IRF
        crop) plus the IRF visibility toggle — all of them either add a
        DraggableSpan (or, for the data/IRF offset arrow buttons, an
        equivalent live-adjust control) wired to the single shared action
        bar, or (Hide/Show IRF) would corrupt an in-progress control's
        state via _draw_base_plot()'s full-clear rebuild. Call with False
        when entering any of the six modes; call with True when returning
        to idle, which re-enables each button only if its own prerequisite
        data is actually loaded."""
        self._btn_start.setEnabled(enabled and self._times is not None)
        self._btn_baseline_select.setEnabled(enabled and self._times is not None)
        self._btn_data_offset.setEnabled(enabled and self._times is not None)
        self._btn_irf_baseline_select.setEnabled(enabled and self._irf_times is not None)
        self._btn_irf_hide.setEnabled(enabled and self._irf_times is not None)
        self._btn_irf_offset.setEnabled(enabled and self._irf_times is not None)
        self._btn_irf_crop.setEnabled(enabled and self._irf_times is not None)

    def _recompute_irf_working_arrays(self, offset=None):
        """Rebuild the full working self._irf_times/self._irf_counts from
        the immutable _irf_times_raw/_irf_counts_raw plus whatever
        offset/crop/baseline are currently confirmed — the single place
        that combines all three per-IRF adjustments, so each one's own
        confirm/cancel/reset just updates its own state and calls this
        rather than re-deriving the combination ad hoc (which is what let
        an earlier version of the offset code silently drop any applied
        crop/baseline whenever the offset changed).

        `offset` optionally overrides self._irf_time_offset for a live
        preview of a not-yet-confirmed value (used while nudging via the
        IRF Time Offset arrow buttons) without touching the confirmed
        state itself.

        Order: crop is applied to the RAW (un-offset) times first —
        self._irf_crop_range is stored in raw-time coordinates (converted
        from whatever was picked on-screen at confirm time; see
        _on_irf_crop_confirm()) specifically so the same physical raw
        samples stay cropped regardless of what the offset is later
        changed to. Applying crop against the offset-shifted times instead
        (an earlier version of this method did that) made the kept window
        silently slide whenever the offset changed after the fact — crop
        picked before a later offset adjustment no longer meant what it
        looked like when you picked it. Offset is applied after crop
        (shifting only the surviving samples), then baseline subtracts its
        constant from whatever samples remain — baseline doesn't interact
        with either (a per-element subtract doesn't care about x-position
        or which elements survived the crop)."""
        eff_offset = self._irf_time_offset if offset is None else offset
        times_raw = self._irf_times_raw
        counts = self._irf_counts_raw
        if self._irf_crop_range is not None:
            lo, hi = self._irf_crop_range   # raw-time bounds, offset-independent
            mask = (times_raw >= lo) & (times_raw <= hi)
            times_raw = times_raw[mask]
            counts = counts[mask]
        times = times_raw + eff_offset
        if self._irf_baseline_value is not None:
            counts = counts - self._irf_baseline_value
        self._irf_times = times
        self._irf_counts = counts

    def _irf_normalized(self):
        """Sum-normalized, alignment-corrected working IRF for use as the
        convolution kernel — see _convolved_model()/_eval_fit_curve().

        Sum-normalizing self._irf_counts alone is NOT enough: _convolved_
        model()'s fftconvolve() only ever sees sample INDICES, never time
        values, so by itself it implicitly assumes index 0 of the kernel
        lines up with index 0 of self._times (t_full) — i.e. that the IRF
        and the data share the same t=0. That's exactly the misalignment
        "IRF Time Offset" exists to correct, but self._irf_times is only
        ever used for display/range-picking (crop, baseline) elsewhere —
        never fed into the convolution itself. So this method also shifts
        the normalized kernel by however many samples self._irf_times[0]
        currently sits from self._times[0] (self._irf_times already
        reflects the confirmed offset and any crop — see
        _recompute_irf_working_arrays()), assuming a common channel width
        (ns/sample) between the two, which the offset/crop feature exists
        to be robust to sub-channel error in but not a wholesale binning
        mismatch. Zero-padded (not circular) shift: samples pushed off
        one end are dropped, not wrapped to the other.

        Returns None if no IRF is loaded, there's no main data to align
        against, or the IRF's counts sum to <= 0 (e.g. an over-subtracted
        baseline), signaling the caller to fail that fit with a clear
        message rather than dividing by zero or silently ignoring the
        offset."""
        if self._irf_counts is None or self._times is None or len(self._irf_counts) == 0:
            return None
        total = float(np.sum(self._irf_counts))
        if total <= 0:
            return None
        kernel = self._irf_counts / total

        dt = float(np.median(np.diff(self._times)))
        if dt <= 0:
            return kernel
        n_shift = int(round(float(self._irf_times[0] - self._times[0]) / dt))
        if n_shift == 0:
            return kernel

        shifted = np.zeros_like(kernel)
        if n_shift > 0:
            if n_shift < len(kernel):
                shifted[n_shift:] = kernel[: len(kernel) - n_shift]
        else:
            n = -n_shift
            if n < len(kernel):
                shifted[: len(kernel) - n] = kernel[n:]
        return shifted

    def _reset_irf_baseline_state(self):
        self._irf_baseline_range = None
        self._irf_baseline_value = None
        self._irf_baseline_pending_range = None
        self._irf_baseline_pending_value = None
        if self._irf_baseline_span is not None:
            self._irf_baseline_span.deactivate()
            self._irf_baseline_span = None
        self._remove_irf_baseline_preview_item()

    def _remove_irf_baseline_preview_item(self):
        if self._irf_baseline_preview_item is not None:
            try:
                self._canvas.plot_item.removeItem(self._irf_baseline_preview_item)
            except Exception:
                pass
            self._irf_baseline_preview_item = None

    def _default_irf_tail_bounds(self):
        """Seed a span over the tail 20% of the IRF's own time axis — same
        fixed-fraction convention as the main Counts baseline's
        _default_tail_bounds(), but against the IRF's own (independent)
        array."""
        t_min, t_max = float(self._irf_times[0]), float(self._irf_times[-1])
        return (t_min + 0.8 * (t_max - t_min), t_max)

    def _on_start_irf_baseline_selecting(self):
        if self._irf_times is None or self._mode != "idle":
            return

        self._mode = "irf_baseline_selecting"
        self._set_exclusive_buttons_enabled(False)

        ax = self._draw_base_plot()
        t_min, t_max = float(self._irf_times[0]), float(self._irf_times[-1])
        x_lo, x_hi = self._default_irf_tail_bounds()

        self._irf_baseline_span = DraggableSpan(ax, color=(220, 30, 180, 60), movable=True)
        self._irf_baseline_span.activate(initial_range=(x_lo, x_hi), bounds=(t_min, t_max))
        self._irf_baseline_span.sigRegionSelected.connect(self._on_irf_baseline_span_changed)

        self._btn_act_confirm.setEnabled(False)
        self._btn_act_confirm.setText("Confirm IRF baseline")
        self._action_bar.setVisible(True)
        self._on_irf_baseline_span_changed(x_lo, x_hi)   # initial preview immediately

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                "TRPL: drag the IRF baseline range, then Confirm."
            )

    def _on_irf_baseline_span_changed(self, xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        self._irf_baseline_pending_range = (xmin, xmax)

        mask = (self._irf_times >= xmin) & (self._irf_times <= xmax)
        n_pts = int(np.sum(mask))
        if n_pts < 1:
            self._irf_baseline_pending_value = None
            self._remove_irf_baseline_preview_item()
            self._btn_act_confirm.setEnabled(False)
            self._action_lbl.setText("IRF baseline range: no points in range — widen it.")
            return

        value = float(np.mean(self._irf_counts[mask]))
        self._irf_baseline_pending_value = value

        self._remove_irf_baseline_preview_item()
        if value > 0:
            x_curve = np.array([self._irf_times[0], self._irf_times[-1]])
            y_curve = np.full(2, value)
            self._irf_baseline_preview_item = self._canvas.plot_item.plot(
                x_curve, y_curve, pen=pg.mkPen("magenta", width=1.5, style=Qt.DashLine)
            )
        self._canvas.draw_idle()

        msg = f"IRF baseline range: constant={value:.4g} counts  ({n_pts} pts)"
        self._action_lbl.setText(msg)
        self._btn_act_confirm.setEnabled(True)

    def _on_irf_baseline_confirm(self):
        if self._mode != "irf_baseline_selecting" or self._irf_baseline_pending_value is None:
            return
        self._irf_baseline_range = self._irf_baseline_pending_range
        self._irf_baseline_value = self._irf_baseline_pending_value
        self._recompute_irf_working_arrays()

        if self._irf_baseline_span is not None:
            self._irf_baseline_span.deactivate()
            self._irf_baseline_span = None
        self._remove_irf_baseline_preview_item()
        self._irf_baseline_pending_range = None
        self._irf_baseline_pending_value = None

        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._set_exclusive_buttons_enabled(True)
        self._btn_irf_baseline_reset.setEnabled(True)

        xmin, xmax = self._irf_baseline_range
        self._irf_baseline_status_lbl.setText(
            f"IRF baseline = {self._irf_baseline_value:.4g} counts "
            f"(fit over {xmin:.4g}–{xmax:.4g} ns). Subtracted from all IRF Counts."
        )
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: IRF baseline {self._irf_baseline_value:.4g} counts subtracted."
            )

    def _on_irf_baseline_cancel(self):
        if self._irf_baseline_span is not None:
            self._irf_baseline_span.deactivate()
            self._irf_baseline_span = None
        self._remove_irf_baseline_preview_item()
        self._irf_baseline_pending_range = None
        self._irf_baseline_pending_value = None

        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._set_exclusive_buttons_enabled(True)
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: IRF baseline selection canceled.")

    def _on_reset_irf_baseline(self):
        if self._irf_baseline_value is None:
            return
        self._irf_baseline_range = None
        self._irf_baseline_value = None
        self._recompute_irf_working_arrays()
        self._btn_irf_baseline_reset.setEnabled(False)
        self._irf_baseline_status_lbl.setText("No IRF baseline applied.")
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: IRF baseline subtraction reset.")

    # ── IRF time offset ──────────────────────────────────────────────
    # Shifts the whole IRF curve along the time axis by a constant (ns) so
    # its rising edge can be lined up against the main Counts curve's own
    # rising edge — unlike every other interactive control here, there's no
    # "range" to drag, just a single value, nudged by two arrow buttons
    # (+/- a user-set step size) rather than a DraggableSpan. In-memory
    # only: _irf_times (the working array _draw_base_plot()/baseline
    # masking already read) is recomputed from _irf_times_raw + the
    # offset, and neither the offset nor the shifted array is ever written
    # into the loaded IRF file itself.

    def _on_start_irf_offset(self):
        if self._irf_times_raw is None or self._mode != "idle":
            return

        self._mode = "irf_offset_selecting"
        self._set_exclusive_buttons_enabled(False)

        self._irf_offset_pending = self._irf_time_offset
        self._btn_irf_offset_dec.setEnabled(True)
        self._btn_irf_offset_inc.setEnabled(True)
        self._irf_offset_step_spin.setEnabled(True)
        self._irf_offset_status_lbl.setText(f"Offset: {self._irf_time_offset:+.4g} ns")

        self._btn_act_confirm.setEnabled(True)   # any value is valid, unlike a range pick
        self._btn_act_confirm.setText("Confirm IRF offset")
        self._action_bar.setVisible(True)

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                "TRPL: use the ◀/▶ buttons to align the IRF's rising edge "
                "with the data's, then Confirm."
            )

    def _on_irf_offset_step(self, direction):
        if self._mode != "irf_offset_selecting":
            return
        offset = self._irf_offset_pending + direction * self._irf_offset_step_spin.value()
        self._irf_offset_pending = offset
        self._recompute_irf_working_arrays(offset=offset)
        self._irf_offset_status_lbl.setText(f"Offset: {offset:+.4g} ns")
        self._action_lbl.setText(f"IRF time offset: {offset:+.4g} ns")
        self._redraw_preserving_view()

    def _on_irf_offset_confirm(self):
        if self._mode != "irf_offset_selecting":
            return
        self._irf_time_offset = self._irf_offset_pending
        self._irf_offset_pending = None
        # Recompute from _irf_times_raw rather than trusting the
        # click-accumulated working array verbatim, so the confirmed value
        # is exactly _irf_times_raw + _irf_time_offset with no accumulated
        # float drift from repeated clicks.
        self._recompute_irf_working_arrays()
        self._refresh_irf_crop_status_label()   # its displayed range is offset-relative

        self._btn_irf_offset_dec.setEnabled(False)
        self._btn_irf_offset_inc.setEnabled(False)
        self._irf_offset_step_spin.setEnabled(False)
        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._set_exclusive_buttons_enabled(True)
        self._btn_irf_offset_reset.setEnabled(self._irf_time_offset != 0.0)
        self._irf_offset_status_lbl.setText(
            f"IRF time offset: {self._irf_time_offset:+.4g} ns"
        )
        self._redraw_preserving_view()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: IRF time offset {self._irf_time_offset:+.4g} ns confirmed."
            )

    def _on_irf_offset_cancel(self):
        self._irf_offset_pending = None
        # Revert the working arrays back to the last confirmed offset —
        # they may have been live-shifted by arrow-button clicks since.
        self._recompute_irf_working_arrays()

        self._btn_irf_offset_dec.setEnabled(False)
        self._btn_irf_offset_inc.setEnabled(False)
        self._irf_offset_step_spin.setEnabled(False)
        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._set_exclusive_buttons_enabled(True)
        self._irf_offset_status_lbl.setText(
            f"IRF time offset: {self._irf_time_offset:+.4g} ns"
            if self._irf_time_offset != 0.0 else "No IRF time offset applied."
        )
        self._redraw_preserving_view()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: IRF time offset selection canceled.")

    def _on_reset_irf_offset(self):
        if self._irf_time_offset == 0.0:
            return
        self._irf_time_offset = 0.0
        self._recompute_irf_working_arrays()
        self._refresh_irf_crop_status_label()   # its displayed range is offset-relative
        self._btn_irf_offset_reset.setEnabled(False)
        self._irf_offset_status_lbl.setText("No IRF time offset applied.")
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: IRF time offset reset.")

    # ── IRF crop ─────────────────────────────────────────────────────
    # Discards IRF data points outside a user-picked (xmin, xmax) ns range
    # entirely from the working arrays — e.g. to drop a stray reflection
    # peak or a noisy tail — rather than just visually clipping the view.
    # Same drag-a-DraggableSpan/Confirm pattern as the baseline selectors,
    # but nothing is computed from the picked range; it's applied as-is via
    # _recompute_irf_working_arrays(). In-memory only, like the offset and
    # baseline above: never written back into the loaded IRF's own file.

    def _reset_irf_crop_state(self):
        self._irf_crop_range = None
        self._irf_crop_pending_range = None
        if self._irf_crop_span is not None:
            self._irf_crop_span.deactivate()
            self._irf_crop_span = None

    def _refresh_irf_crop_status_label(self):
        """Keep the crop status text in sync with the currently-displayed
        axis. It reports the surviving range in DISPLAY (offset-shifted)
        coordinates — see _on_irf_crop_confirm() — which needs refreshing
        whenever the offset changes after a crop was already applied, even
        though the underlying raw-anchored _irf_crop_range itself doesn't
        need to change (that's the whole point of anchoring it to raw
        time). No-op if no crop is currently applied."""
        if self._irf_crop_range is None:
            return
        disp_lo, disp_hi = float(self._irf_times[0]), float(self._irf_times[-1])
        self._irf_crop_status_lbl.setText(
            f"IRF cropped to {disp_lo:.4g}–{disp_hi:.4g} ns "
            f"({len(self._irf_times)} point(s) kept)."
        )

    def _on_start_irf_crop(self):
        if self._irf_times is None or self._mode != "idle":
            return

        self._mode = "irf_crop_selecting"
        self._set_exclusive_buttons_enabled(False)

        ax = self._draw_base_plot()
        t_min, t_max = float(self._irf_times[0]), float(self._irf_times[-1])
        # Default to the full currently-kept range — a crop narrows inward
        # from "keep everything", unlike the lifetime-fit spans which
        # default to a small window and get widened.
        x_lo, x_hi = t_min, t_max

        self._irf_crop_span = DraggableSpan(ax, color=(0, 170, 170, 60), movable=True)
        self._irf_crop_span.activate(initial_range=(x_lo, x_hi), bounds=(t_min, t_max))
        self._irf_crop_span.sigRegionSelected.connect(self._on_irf_crop_span_changed)

        self._btn_act_confirm.setEnabled(True)   # the full-axis default is itself always valid
        self._btn_act_confirm.setText("Confirm IRF crop")
        self._action_bar.setVisible(True)
        self._on_irf_crop_span_changed(x_lo, x_hi)   # initial status text immediately

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                "TRPL: drag the range of IRF data to keep, then Confirm — "
                "points outside it will be discarded."
            )

    def _on_irf_crop_span_changed(self, xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        self._irf_crop_pending_range = (xmin, xmax)

        mask = (self._irf_times >= xmin) & (self._irf_times <= xmax)
        n_keep = int(np.sum(mask))
        n_drop = len(self._irf_times) - n_keep
        if n_keep < 1:
            self._btn_act_confirm.setEnabled(False)
            self._action_lbl.setText("Crop range: no points would remain — widen it.")
            return

        self._btn_act_confirm.setEnabled(True)
        self._action_lbl.setText(
            f"Crop range: {n_keep} point(s) kept, {n_drop} discarded"
        )

    def _on_irf_crop_confirm(self):
        if self._mode != "irf_crop_selecting" or self._irf_crop_pending_range is None:
            return
        # Convert from what was picked on the currently-displayed
        # (offset-shifted) axis to raw-time coordinates by removing the
        # offset that's in effect right now — see _irf_crop_range's own
        # comment / _recompute_irf_working_arrays() for why storing it
        # raw-anchored (rather than the display value verbatim) is what
        # keeps the crop meaning the same physical samples if the offset
        # is changed again later.
        disp_xmin, disp_xmax = self._irf_crop_pending_range
        self._irf_crop_range = (
            disp_xmin - self._irf_time_offset,
            disp_xmax - self._irf_time_offset,
        )
        self._irf_crop_pending_range = None

        if self._irf_crop_span is not None:
            self._irf_crop_span.deactivate()
            self._irf_crop_span = None
        self._recompute_irf_working_arrays()

        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._set_exclusive_buttons_enabled(True)
        self._btn_irf_crop_reset.setEnabled(True)

        # Report the surviving range in the currently-displayed frame
        # (self._irf_times, post-recompute) rather than the raw-anchored
        # _irf_crop_range — matches what the user actually sees on screen.
        disp_lo, disp_hi = float(self._irf_times[0]), float(self._irf_times[-1])
        self._refresh_irf_crop_status_label()
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: IRF cropped to {disp_lo:.4g}–{disp_hi:.4g} ns."
            )

    def _on_irf_crop_cancel(self):
        self._irf_crop_pending_range = None
        if self._irf_crop_span is not None:
            self._irf_crop_span.deactivate()
            self._irf_crop_span = None

        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._set_exclusive_buttons_enabled(True)
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: IRF crop selection canceled.")

    def _on_reset_irf_crop(self):
        if self._irf_crop_range is None:
            return
        self._irf_crop_range = None
        self._recompute_irf_working_arrays()
        self._btn_irf_crop_reset.setEnabled(False)
        self._irf_crop_status_lbl.setText("No IRF crop applied.")
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: IRF crop reset.")

    # ── Plotting ─────────────────────────────────────────────────

    def _draw_base_plot(self):
        """Full rebuild: raw histogram (log-y, downsampled for smooth pan/zoom
        even at ~65k points — same technique as the Convert tab's preview)
        plus any already-confirmed fit overlays."""
        ax = self._canvas.reset_axes()
        ax.setLogMode(y=True)
        ax.setClipToView(True)
        ax.setDownsampling(auto=True, mode="peak")
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        # A legend is only worth showing once there's a second curve to
        # distinguish (the IRF) — an unlabeled single Counts curve otherwise
        # keeps the plot exactly as uncluttered as before this feature.
        if self._irf_times is not None:
            ax.addLegend(labelTextSize="7pt")
        # self._times can be None here — the IRF group's own workflows
        # (baseline, time offset) only require an IRF to be loaded, not a
        # main file, so this rebuild has to tolerate "IRF only" too.
        if self._times is not None:
            mlp.plot(self._times, self._counts, index=0, width=1.0, antialias=False, label="Counts")
        if self._irf_times is not None and self._irf_visible:
            mlp.plot(self._irf_times, self._irf_counts, index=1, width=1.0,
                      antialias=False, label="IRF")
        ax.setLabel("bottom", "Time (ns)")
        ax.setLabel("left", "Counts")
        title = os.path.basename(self._path) if self._path else ""
        if self._meta.get("power_mW") is not None:
            title += f"  —  {self._meta['power_mW']:.4g} mW"
        ax.setTitle(title)
        ax.showGrid(x=True, y=True, alpha=0.3)

        if self._fit_range is not None and self._fit_params is not None:
            self._draw_confirmed_overlay(ax, self._fit_range, self._fit_params)

        # Each already-confirmed fixing stage is drawn as soon as it's
        # locked in, even while a later stage is still being dragged.
        for stage_idx, fs in enumerate(self._fixed_stages):
            rgb = _FIXED_STAGE_COLORS.get(stage_idx, (220, 140, 0))
            self._draw_confirmed_fixed_stage_overlay(ax, fs["range"], fs["params"], rgb)

        if self._baseline_range is not None:
            self._draw_confirmed_baseline_overlay(ax, self._baseline_range)

        self._canvas.draw_idle()
        return ax

    def _eval_fit_curve(self, xmin, xmax, params):
        """Evaluate the combined (summed) fit curve over [xmin, xmax] for
        display. Normally a smooth bare sum-of-exponentials on a 200-point
        linspace grid — but when self._fit_use_irf_convolution is set (the
        currently-displayed fit was actually fit with IRF convolution —
        see _convolved_model()), a convolution's result only has meaning
        at the exact grid it was computed on, so this instead evaluates
        the real convolved model over the full time axis and returns just
        the real sample points within [xmin, xmax] — not a synthetic
        interpolated curve."""
        params_arr = np.asarray(params, dtype=float)
        if self._fit_use_irf_convolution:
            irf_norm = self._irf_normalized()
            if irf_norm is not None:
                mask = (self._times >= xmin) & (self._times <= xmax)
                D = _multiexp(self._times, *params_arr.ravel())
                conv = fftconvolve(irf_norm, D, mode="full")[: len(self._times)]
                return self._times[mask], conv[mask]
        x_curve = np.linspace(xmin, xmax, 200)
        y_curve = _multiexp(x_curve, *params_arr.ravel())
        return x_curve, y_curve

    def _draw_confirmed_overlay(self, ax, rng, params):
        xmin, xmax = rng
        region = pg.LinearRegionItem(
            values=(xmin, xmax), orientation="vertical",
            brush=pg.mkBrush(200, 30, 30, 30), movable=False,
        )
        region.setZValue(5)
        ax.addItem(region)
        x_curve, y_curve = self._eval_fit_curve(xmin, xmax, params)
        ax.plot(x_curve, y_curve, pen=pg.mkPen("red", width=1.5))

    def _draw_confirmed_fixed_stage_overlay(self, ax, rng, params, rgb):
        """Mark a fixing stage's own range and the single exponential fixed
        over it — a distinct color per stage so the sequence of fixed
        components stays visually distinguishable from the final overlay."""
        xmin, xmax = rng
        region = pg.LinearRegionItem(
            values=(xmin, xmax), orientation="vertical",
            brush=pg.mkBrush(*rgb, 30), movable=False,
        )
        region.setZValue(4)
        ax.addItem(region)
        a, b = params
        x_curve = np.linspace(xmin, xmax, 200)
        y_curve = a * np.exp(-b * x_curve)
        ax.plot(x_curve, y_curve, pen=pg.mkPen(pg.mkColor(*rgb), width=1.5, style=Qt.DashLine))

    def _draw_confirmed_baseline_overlay(self, ax, rng):
        """Mark the range the currently-applied baseline was fit over. The
        working Counts are already baseline-subtracted at this point, so
        drawing the constant's own level (near/at 0) wouldn't read as
        anything on a log-y axis — just shade the source range instead."""
        xmin, xmax = rng
        region = pg.LinearRegionItem(
            values=(xmin, xmax), orientation="vertical",
            brush=pg.mkBrush(30, 100, 220, 30), movable=False,
        )
        region.setZValue(4)
        ax.addItem(region)

    def _remove_baseline_preview_item(self):
        if self._baseline_preview_item is not None:
            try:
                self._canvas.plot_item.removeItem(self._baseline_preview_item)
            except Exception:
                pass
            self._baseline_preview_item = None

    # ── Fit-range selection loop ──────────────────────────────────
    # n_exp == 1: one dragged span, one free exponential fit to it (stage 0
    # is immediately also the final stage).
    # n_exp >= 2: n_exp dragged spans in sequence, slowest component first —
    # every stage before the last fits ONE free component (with every
    # previously-fixed, slower component held fixed in the model) and then
    # fixes it for later stages; the last stage fits the fastest component,
    # with every slower one now fixed, and finalizes the whole fit.
    # "Confirm fit"/"Cancel selection" drive it via the shared action bar
    # (see _on_action_confirm/_on_action_cancel above).

    def _reset_fit_state(self):
        self._mode = "idle"
        self._n_exp = 0
        self._stage_idx = None
        self._fixed_stages = []
        self._fit_range = None
        self._fit_params = None
        self._fit_cov = None
        self._clear_stage_progress()

    def _clear_stage_progress(self):
        for entry in self._stage_progress:
            if entry["span"] is not None:
                entry["span"].deactivate()
            if entry["preview_item"] is not None:
                try:
                    self._canvas.plot_item.removeItem(entry["preview_item"])
                except Exception:
                    pass
        self._stage_progress = []

    def _on_start_selecting(self):
        if self._times is None:
            return

        # _draw_base_plot() below does a full plot_item.clear()-based rebuild
        # (via reset_axes()), which resets the view to a small known range
        # before re-enabling auto-range — that would silently discard
        # whatever pan/zoom the user was already looking at. Capture it here
        # and restore it verbatim once the rebuild (and the new stage span)
        # is in place, so clicking this button never moves the view.
        vb = self._canvas.plot_item.getViewBox()
        prev_range = vb.viewRange()

        self._n_exp      = self._n_exp_combo.currentData()
        self._fit_use_irf_convolution = self._chk_irf_convolution.isChecked()
        self._fit_initial_guesses = {
            slot: (float(a_spin.value()), float(b_spin.value()))
            for slot, (chk, a_spin, b_spin) in self._guess_widgets.items()
            if chk.isChecked()
        }
        self._fit_range  = None
        self._fit_params = None
        self._fit_cov    = None
        self._fixed_stages = []
        self._clear_stage_progress()
        self._stage_idx = 0
        self._mode = "selecting"
        self._btn_save.setEnabled(False)
        self._btn_reset_fits.setEnabled(True)
        self._set_exclusive_buttons_enabled(False)   # mutually exclusive with baseline/IRF selection
        self._draw_base_plot()   # one full rebuild for the whole multi-stage attempt
        self._add_stage_span(0)
        vb.setRange(xRange=prev_range[0], yRange=prev_range[1], padding=0)

    def _default_fit_bounds(self):
        """Seed the final stage's span over the fixed 0-5 ns window — just a
        starting point; the user drags it to the region the fastest
        component's fit should actually cover."""
        return self._default_window_bounds()

    def _default_stage_bounds(self, stage_idx, n_exp):
        """Seed a non-final fixing stage's span over the same fixed 0-5 ns
        window as the final stage — just a starting point; the user drags
        it to the region that stage's component should actually be fixed
        from."""
        return self._default_window_bounds()

    def _default_window_bounds(self):
        """Shared 0-5 ns default span, clamped to the loaded file's actual
        time axis (e.g. a file shorter than 5 ns, or one not starting at
        0 ns) so the initial span is always a valid, draggable sub-range."""
        t_min, t_max = float(self._times[0]), float(self._times[-1])
        lo, hi = max(0.0, t_min), min(5.0, t_max)
        if hi <= lo:
            hi = t_max
        return (lo, hi)

    def _default_tail_bounds(self):
        """Seed a span over the tail 20% of the time axis — used for
        baseline-range selection (always this fixed fraction, unlike the
        progressively-widening _default_stage_bounds above)."""
        t_min, t_max = float(self._times[0]), float(self._times[-1])
        return (t_min + 0.8 * (t_max - t_min), t_max)

    def _stage_label_for(self, idx):
        """Short label for status/error messages identifying stage `idx`:
        the generic "Fit range" for N=1 or the final stage, else the named
        fixing stage, e.g. "Slow-decay fit"."""
        if self._n_exp == 1 or idx == self._n_exp - 1:
            return "Fit range"
        return f"{_rank_name(idx, self._n_exp).capitalize()}-decay fit"

    def _stage_point_count(self, idx):
        xmin, xmax = self._stage_progress[idx]["range"]
        mask = (self._times >= xmin) & (self._times <= xmax)
        return int(np.sum(mask))

    def _add_stage_span(self, idx):
        """Add stage `idx`'s DraggableSpan on top of whatever spans are
        already active for this attempt — never removes or freezes an
        earlier stage's span, so every stage picked so far stays draggable
        for the rest of the multi-stage attempt."""
        ax = self._canvas.plot_item
        t_min, t_max = float(self._times[0]), float(self._times[-1])
        is_last = idx == self._n_exp - 1

        if is_last:
            x_lo, x_hi = self._default_fit_bounds()
            color = (0, 150, 0, 60)
        else:
            x_lo, x_hi = self._default_stage_bounds(idx, self._n_exp)
            rgb = _FIXED_STAGE_COLORS.get(idx, (220, 140, 0))
            color = (*rgb, 60)

        span = DraggableSpan(ax, color=color, movable=True)
        span.activate(initial_range=(x_lo, x_hi), bounds=(t_min, t_max))
        # i=idx binds this stage's index at connect-time, not lookup-time —
        # otherwise every span's callback would see whatever _stage_idx
        # happens to be current when it fires, not the stage it belongs to.
        span.sigRegionSelected.connect(lambda xmin, xmax, i=idx: self._on_stage_span_changed(i, xmin, xmax))

        self._stage_progress.append({
            "span": span, "range": (x_lo, x_hi),
            "params": None, "cov": None, "error": None, "preview_item": None,
        })

        self._btn_act_confirm.setEnabled(False)
        if is_last:
            self._btn_act_confirm.setText("Confirm fit")
        else:
            next_name = _rank_name(idx + 1, self._n_exp)
            self._btn_act_confirm.setText(f"Add {next_name} fit  ▶")
        self._action_bar.setVisible(True)
        self._on_stage_span_changed(idx, x_lo, x_hi)   # show an initial preview immediately

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            if self._n_exp == 1:
                msg = "TRPL: drag the fit range for the single exponential, then Confirm."
            elif is_last:
                prior = ", ".join(_rank_name(r, self._n_exp) for r in range(idx))
                msg = (
                    f"TRPL: drag the range for the {_rank_name(idx, self._n_exp)} "
                    f"component (fit {idx + 1}/{self._n_exp}; {prior} fixed — "
                    "still adjustable above), then Confirm fit."
                )
            else:
                name = _rank_name(idx, self._n_exp)
                msg = (
                    f"TRPL: drag the tail range where the {name} decay "
                    f"dominates (fit {idx + 1}/{self._n_exp}). Earlier ranges "
                    "stay adjustable — drag them again any time, then "
                    "Confirm to add the next component."
                )
            parent.statusBar().showMessage(msg)

    def _on_stage_span_changed(self, idx, xmin, xmax):
        """Fired when any stage's span is dragged — including an
        already-added earlier stage's, which stays live for the whole
        attempt (e.g. the medium-decay span can still be moved while the
        fast/final span is being picked). Refits stage `idx` and cascades
        forward through every later stage already added, since each one's
        model bakes in every earlier stage's fitted params as a fixed
        constant — an earlier stage's own fit never depends on a later
        one, so there's nothing to recompute backward."""
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        self._stage_progress[idx]["range"] = (xmin, xmax)
        self._recompute_from(idx)

    def _recompute_from(self, start_idx):
        for j in range(start_idx, len(self._stage_progress)):
            entry = self._stage_progress[j]
            fixed_params = [self._stage_progress[m]["params"] for m in range(j)]
            if any(p is None for p in fixed_params):
                entry["params"] = None
                entry["cov"] = None
                entry["error"] = "blocked by an earlier stage that hasn't converged."
                self._remove_stage_preview(j)
                continue

            xmin, xmax = entry["range"]
            mask = (self._times >= xmin) & (self._times <= xmax)
            n_pts = int(np.sum(mask))
            x = self._times[mask]
            y = self._counts[mask]
            self._fit_stage(j, x, y, mask, n_pts, fixed_params)

        self._refresh_stage_ui()

    def _fit_stage(self, j, x, y, mask, n_pts, fixed_params):
        """Fit stage `j`'s own free component (with every earlier stage's
        params baked in as fixed constants) and store the result on its
        _stage_progress entry — the same 2-free-parameter model regardless
        of N, as in the original one-span-at-a-time design. `mask` is the
        boolean mask over self._times this stage's window (x, y) was
        selected with — only used when self._fit_use_irf_convolution (see
        _convolved_model()), which needs it to slice its full-axis
        convolution result back down to this window."""
        entry = self._stage_progress[j]
        n_free_params = 2   # always 2: every stage fits exactly one free component
        if n_pts <= n_free_params:
            entry["params"] = None
            entry["cov"] = None
            entry["error"] = f"only {n_pts} point(s) in range — widen it."
            self._remove_stage_preview(j)
            return False

        if not fixed_params:
            p0 = _multiexp_initial_guess(x, y, 1)
        else:
            y_resid = y.copy()
            for af, bf in fixed_params:
                y_resid = y_resid - af * np.exp(-bf * x)
            p0 = _free_component_initial_guess(x, y_resid, fixed_params)

        # User-supplied (a0, b0) override for this stage's slot (see
        # _guess_slot_for_stage()/the Settings dialog), read once at
        # _on_start_selecting() — takes the place of the automatic
        # heuristic above entirely rather than blending with it.
        override = self._fit_initial_guesses.get(_guess_slot_for_stage(j, self._n_exp))
        if override is not None:
            p0 = np.array(override, dtype=float)

        if self._fit_use_irf_convolution:
            irf_norm = self._irf_normalized()
            if irf_norm is None:
                entry["params"] = None
                entry["cov"] = None
                entry["error"] = "IRF convolution enabled but no valid IRF is loaded."
                self._remove_stage_preview(j)
                return False
            model = _convolved_model(self._times, irf_norm, mask, fixed_params)
        elif not fixed_params:
            model = _multiexp
        else:
            model = _fixed_components_model(fixed_params)

        try:
            popt, cov = curve_fit(model, x, y, p0=p0, bounds=([0, 0], [np.inf, np.inf]))
        except (RuntimeError, ValueError):
            entry["params"] = None
            entry["cov"] = None
            entry["error"] = "did not converge over this range — try adjusting it."
            self._remove_stage_preview(j)
            return False

        entry["params"] = (float(popt[0]), float(popt[1]))
        entry["cov"] = cov
        entry["error"] = None
        self._draw_stage_preview(j, fixed_params)
        return True

    def _draw_stage_preview(self, j, fixed_params):
        entry = self._stage_progress[j]
        xmin, xmax = entry["range"]
        # This stage's free component first, then every already-fixed
        # component in reverse (most-recently-fixed = next slower first) —
        # keeps rows ordered fastest-to-slowest, matching the final
        # FitParameters convention.
        combined = [entry["params"]] + list(reversed(fixed_params))
        params = np.array(combined)
        x_curve, y_curve = self._eval_fit_curve(xmin, xmax, params)
        is_last = j == self._n_exp - 1
        pen_color = "yellow" if is_last else pg.mkColor(*_FIXED_STAGE_COLORS.get(j, (220, 140, 0)))
        self._remove_stage_preview(j)
        entry["preview_item"] = self._canvas.plot_item.plot(
            x_curve, y_curve, pen=pg.mkPen(pen_color, width=1.5, style=Qt.DashLine)
        )

    def _remove_stage_preview(self, j):
        entry = self._stage_progress[j]
        if entry["preview_item"] is not None:
            try:
                self._canvas.plot_item.removeItem(entry["preview_item"])
            except Exception:
                pass
            entry["preview_item"] = None

    def _refresh_stage_ui(self):
        """Update the action bar / status text from the newest stage's
        current state — even though every earlier stage stays adjustable,
        only the newest one gates "Confirm"/"Add next", since it's the
        only one that hasn't been superseded by a further stage yet."""
        idx = self._stage_idx
        entry = self._stage_progress[idx]
        if entry["params"] is None:
            self._btn_act_confirm.setEnabled(False)
            broken = next(
                (j for j in range(idx + 1) if self._stage_progress[j]["params"] is None),
                idx,
            )
            reason = self._stage_progress[broken]["error"] or "did not converge."
            msg = f"{self._stage_label_for(broken)}: {reason}"
            self._action_lbl.setText(msg)
            self._fit_status_lbl.setText(msg)
            self._canvas.draw_idle()
            return

        self._btn_act_confirm.setEnabled(True)
        msg = self._format_all_stages_message()
        self._action_lbl.setText(msg)
        self._fit_status_lbl.setText(msg)
        self._canvas.draw_idle()

    def _format_all_stages_message(self):
        """Unlike the single-stage message the wizard used to show, this
        lists every stage added so far (all still adjustable), not just
        the newest one — so the user can see at a glance that e.g. the
        medium-decay fit is still live while the fast fit is being picked."""
        idx = self._stage_idx
        n_pts = self._stage_point_count(idx)
        conv_tag = " [IRF-convolved]" if self._fit_use_irf_convolution else ""
        if self._n_exp == 1:
            a, b = self._stage_progress[0]["params"]
            tau = 1.0 / b if b != 0 else float("nan")
            return f"Fit range ({n_pts} pts){conv_tag}: a={a:.4g}, b={b:.4g} 1/ns, τ={tau:.4g} ns"

        is_last = idx == self._n_exp - 1
        if is_last:
            header = f"Fit range ({n_pts} pts){conv_tag} — {self._n_exp}-exponential fit:"
        else:
            header = f"{self._stage_label_for(idx)} ({n_pts} pts){conv_tag}:"
        lines = [header]
        for j in range(idx, -1, -1):
            a, b = self._stage_progress[j]["params"]
            name = _rank_name(j, self._n_exp)
            tag = "(fit)" if j == idx else "(fixed, adjustable)"
            tau = 1.0 / b if b != 0 else float("nan")
            lines.append(f"  {name} {tag}: a={a:.4g}, b={b:.4g} 1/ns, τ={tau:.4g} ns")
        return "\n".join(lines)

    def _on_confirm_fit(self):
        if self._mode != "selecting":
            return
        idx = self._stage_idx
        if self._stage_progress[idx]["params"] is None:
            return

        if idx < self._n_exp - 1:
            self._stage_idx += 1
            self._add_stage_span(self._stage_idx)
            return

        self._finalize_fit()

    def _finalize_fit(self):
        """Lock in the whole multi-stage attempt: freeze every stage's
        current (possibly just-readjusted) params into _fixed_stages/
        _fit_params, tear down all the live spans/previews, and hand off
        to the normal confirmed-fit display (_draw_base_plot())."""
        idx = self._stage_idx
        final_entry = self._stage_progress[idx]

        params_rows = [final_entry["params"]]
        for j in range(idx - 1, -1, -1):
            params_rows.append(self._stage_progress[j]["params"])
        self._fit_params = np.array(params_rows)
        self._fit_range  = final_entry["range"]
        self._fit_cov    = final_entry["cov"]

        self._fixed_stages = [
            {"range": self._stage_progress[j]["range"],
             "params": self._stage_progress[j]["params"],
             "cov": self._stage_progress[j]["cov"]}
            for j in range(idx)
        ]

        self._clear_stage_progress()
        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._draw_base_plot()
        self._btn_save.setEnabled(True)
        self._set_exclusive_buttons_enabled(True)
        self._fit_status_lbl.setText(
            f"Fit confirmed: {self._n_exp} exponential(s). Ready to save."
        )
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: fit confirmed ({self._n_exp} exponential(s)). Ready to save."
            )

    def _on_reset_fits(self):
        self._reset_fit_state()
        self._btn_reset_fits.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._set_exclusive_buttons_enabled(True)
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._fit_status_lbl.setText("")
        if self._times is not None:
            self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: fit selection reset.")

    # ── Baseline-range selection loop ─────────────────────────────
    # Same "drag a span, live-preview, Confirm" pattern as the lifetime-fit
    # selector above, but there's only ever one range: a constant is fit
    # (its least-squares value is just the mean) over the picked range and
    # subtracted from every Counts sample.

    def _reset_baseline_state(self):
        self._baseline_range = None
        self._baseline_value = None
        self._baseline_pending_range = None
        self._baseline_pending_value = None
        if self._baseline_span is not None:
            self._baseline_span.deactivate()
            self._baseline_span = None
        self._remove_baseline_preview_item()

    def _on_start_baseline_selecting(self):
        if self._times is None or self._mode != "idle":
            return

        self._mode = "baseline_selecting"
        self._set_exclusive_buttons_enabled(False)   # mutually exclusive with fit/IRF selection

        ax = self._draw_base_plot()
        t_min, t_max = float(self._times[0]), float(self._times[-1])
        x_lo, x_hi = self._default_tail_bounds()

        self._baseline_span = DraggableSpan(ax, color=(30, 100, 220, 60), movable=True)
        self._baseline_span.activate(initial_range=(x_lo, x_hi), bounds=(t_min, t_max))
        self._baseline_span.sigRegionSelected.connect(self._on_baseline_span_changed)

        self._btn_act_confirm.setEnabled(False)
        self._btn_act_confirm.setText("Confirm baseline")
        self._action_bar.setVisible(True)
        self._on_baseline_span_changed(x_lo, x_hi)   # initial preview immediately

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                "TRPL: drag the baseline range, then Confirm."
            )

    def _on_baseline_span_changed(self, xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        self._baseline_pending_range = (xmin, xmax)

        mask = (self._times >= xmin) & (self._times <= xmax)
        n_pts = int(np.sum(mask))
        if n_pts < 1:
            self._baseline_pending_value = None
            self._remove_baseline_preview_item()
            self._btn_act_confirm.setEnabled(False)
            self._action_lbl.setText("Baseline range: no points in range — widen it.")
            return

        value = float(np.mean(self._counts[mask]))
        self._baseline_pending_value = value

        self._remove_baseline_preview_item()
        if value > 0:
            x_curve = np.array([self._times[0], self._times[-1]])
            y_curve = np.full(2, value)
            self._baseline_preview_item = self._canvas.plot_item.plot(
                x_curve, y_curve, pen=pg.mkPen("cyan", width=1.5, style=Qt.DashLine)
            )
        self._canvas.draw_idle()

        msg = f"Baseline range: constant={value:.4g} counts  ({n_pts} pts)"
        self._action_lbl.setText(msg)
        self._btn_act_confirm.setEnabled(True)

    def _on_baseline_confirm(self):
        if self._mode != "baseline_selecting" or self._baseline_pending_value is None:
            return
        self._baseline_range = self._baseline_pending_range
        self._baseline_value = self._baseline_pending_value
        self._counts = self._counts_raw - self._baseline_value

        if self._baseline_span is not None:
            self._baseline_span.deactivate()
            self._baseline_span = None
        self._remove_baseline_preview_item()
        self._baseline_pending_range = None
        self._baseline_pending_value = None

        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._set_exclusive_buttons_enabled(True)
        self._btn_baseline_reset.setEnabled(True)

        # Any already-confirmed lifetime fit was computed on the stale,
        # pre-subtraction Counts — invalidate it rather than save/plot a
        # mismatch between the two.
        self._invalidate_lifetime_fits()

        xmin, xmax = self._baseline_range
        self._baseline_status_lbl.setText(
            f"Baseline = {self._baseline_value:.4g} counts "
            f"(fit over {xmin:.4g}–{xmax:.4g} ns). Subtracted from all Counts."
        )
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: baseline {self._baseline_value:.4g} counts subtracted."
            )

    def _on_baseline_cancel(self):
        if self._baseline_span is not None:
            self._baseline_span.deactivate()
            self._baseline_span = None
        self._remove_baseline_preview_item()
        self._baseline_pending_range = None
        self._baseline_pending_value = None

        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._set_exclusive_buttons_enabled(True)
        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: baseline selection canceled.")

    def _on_reset_baseline(self):
        if self._baseline_value is None:
            return
        self._counts = self._counts_raw.copy()
        self._baseline_range = None
        self._baseline_value = None
        self._btn_baseline_reset.setEnabled(False)
        self._baseline_status_lbl.setText("No baseline applied.")

        # The lifetime fit (if any) was computed on the baseline-subtracted
        # Counts — invalidate it along with the baseline itself.
        self._invalidate_lifetime_fits()

        self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: baseline subtraction reset.")

    def _invalidate_lifetime_fits(self):
        """Discard any confirmed lifetime fit — call whenever the working
        Counts change (baseline applied or reset) or the working time axis
        shifts (data time offset applied or reset) so a stale fit computed
        against different data/axis can't be saved or displayed as
        current."""
        self._reset_fit_state()
        self._btn_reset_fits.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._action_bar.setVisible(False)
        self._btn_act_confirm.setText("Confirm fit")
        self._fit_status_lbl.setText("")

    # ── Save ─────────────────────────────────────────────────────

    def _on_save(self):
        if (self._path is None or self._n_exp == 0
                or self._fit_range is None or self._fit_params is None):
            QMessageBox.warning(self, "Nothing to save",
                                "Confirm a fit range first.")
            return

        n = self._n_exp
        params_arr   = np.asarray(self._fit_params, dtype=float)   # (n, 2): [a, b]
        range_arr    = np.asarray(self._fit_range, dtype=float)     # (2,)
        lifetime_arr = np.array([1.0 / b for _a, b in self._fit_params],
                                 dtype=float)                        # (n,)
        total_decay_time = _solve_total_decay_time(params_arr[:, 1])

        if n == 1:
            cov_arr = np.asarray(self._fit_cov, dtype=float)   # (2, 2)
            fit_function_desc = (
                "a * exp(-b * x)  —  nonlinear least squares "
                "(scipy.optimize.curve_fit) over a single fit range, "
                "x = Times (ns)"
            )
            params_desc = "1 x 2: [a, b], Counts = a * exp(-b * Times)"
            cov_desc = "2 x 2: covariance matrix of [a, b] (scipy.optimize.curve_fit)"
        else:
            # Iterative fit: each row past the first (fastest, jointly fit
            # over FitRange) came from its own independent single-free-
            # -component fit over its own range and was held fixed in every
            # later stage — so there's no jointly-estimated covariance
            # across stages. Embed each stage's own 2x2 covariance as a
            # block-diagonal (2n x 2n) matrix rather than imply a
            # cross-covariance that was never actually computed.
            cov_arr = np.zeros((2 * n, 2 * n))
            cov_arr[0:2, 0:2] = np.asarray(self._fit_cov, dtype=float)
            for i, fs in enumerate(reversed(self._fixed_stages)):
                s = 2 * (i + 1)
                cov_arr[s:s + 2, s:s + 2] = np.asarray(fs["cov"], dtype=float)

            if n == 2:
                fit_function_desc = (
                    "Two-stage fit: (1) a_slow * exp(-b_slow * x) fit alone "
                    "over the tail range 'SlowRange', where only the slow "
                    "decay remains; (2) with a_slow, b_slow then fixed at "
                    "those values, a_fast * exp(-b_fast * x) is optimized "
                    "over 'FitRange'. Counts = a_fast*exp(-b_fast*x) + "
                    "a_slow*exp(-b_slow*x), x = Times (ns). FitParameters "
                    "row 0 = fast (fit), row 1 = slow (fixed)."
                )
                params_desc = (
                    "2 x 2: [a, b] per component — row 0 = fast (jointly "
                    "fit over FitRange with the slow term fixed), row 1 = "
                    "slow (independently fixed over SlowRange)"
                )
                cov_desc = (
                    "4 x 4, block-diagonal: [0:2,0:2] is the fast "
                    "component's covariance (from the FitRange fit), "
                    "[2:4,2:4] is the slow component's (from the "
                    "independent SlowRange fit); the off-diagonal "
                    "cross-blocks are zero because the two components "
                    "were never jointly optimized, not because they're "
                    "actually uncorrelated"
                )
            else:  # n == 3
                fit_function_desc = (
                    "Three-stage iterative fit: (1) a_slow * exp(-b_slow * "
                    "x) fit alone over the tail range 'SlowRange', where "
                    "only the slow decay remains; (2) with a_slow, b_slow "
                    "fixed, a_medium * exp(-b_medium * x) is fit jointly "
                    "with that fixed term over 'MediumRange'; (3) with "
                    "a_slow, b_slow, a_medium, b_medium all fixed, "
                    "a_fast * exp(-b_fast * x) is optimized over "
                    "'FitRange'. Counts = a_fast*exp(-b_fast*x) + "
                    "a_medium*exp(-b_medium*x) + a_slow*exp(-b_slow*x), "
                    "x = Times (ns). FitParameters row 0 = fast (fit), "
                    "row 1 = medium (fixed), row 2 = slow (fixed)."
                )
                params_desc = (
                    "3 x 2: [a, b] per component — row 0 = fast (jointly "
                    "fit over FitRange with medium and slow fixed), row 1 "
                    "= medium (fixed, from the MediumRange fit), row 2 = "
                    "slow (fixed, from the SlowRange fit)"
                )
                cov_desc = (
                    "6 x 6, block-diagonal: [0:2,0:2] is the fast "
                    "component's covariance (from the FitRange fit), "
                    "[2:4,2:4] is the medium component's (from the "
                    "MediumRange fit), [4:6,4:6] is the slow component's "
                    "(from the SlowRange fit); the off-diagonal "
                    "cross-blocks are zero because each stage was fixed "
                    "before the next was optimized, not because the "
                    "components are actually uncorrelated"
                )

        if self._fit_use_irf_convolution:
            # Every stage above was actually fit as IRF(x) convolved with
            # the described exponential sum, not the bare sum itself — but
            # the IRF (and its offset/crop/baseline) is never saved to any
            # file (see the TRPL tab's IRF controls), so this can only be
            # recorded as a note on what was done, not reproduced from this
            # file alone.
            fit_function_desc = (
                fit_function_desc + " Fit with IRF convolution: the actual "
                "model compared to Counts was IRF(x) convolved with the "
                "exponential sum described above (scipy.signal.fftconvolve, "
                "'full' mode, truncated to len(Times)), using the "
                "sum-normalized IRF as loaded/adjusted in this session — "
                "the IRF itself is not stored in this file, so the exact "
                "fit model can't be reconstructed from FitParameters alone."
            )

        try:
            with h5py.File(self._path, "a") as f:
                if "analysis" in f:
                    del f["analysis"]
                grp = f.create_group("analysis")

                grp.attrs["FitFunction"] = fit_function_desc
                grp.attrs["IRFConvolutionApplied"] = bool(self._fit_use_irf_convolution)

                # Every range/x-value below (FitRange, SlowRange,
                # MediumRange, Baseline's own FitRange attribute) was
                # picked against self._times = this file's own raw Times
                # dataset + this offset (see "Data time offset" in the
                # sidebar / _recompute_data_working_times()) — 0.0 if it
                # was never applied. Saved unconditionally (like
                # IRFConvolutionApplied above) so any of those saved
                # ranges can always be related back to the file's own
                # unmodified Times: raw_Times = Times_in_saved_ranges -
                # DataTimeOffset. The Visualizer tab's TRPL overlay, which
                # only ever reads the file's raw Times, reverses this
                # offset before drawing the saved fit against it.
                grp.attrs["DataTimeOffset"] = float(self._data_time_offset)
                grp.attrs["DataTimeOffset_units"] = "ns"

                ds_p = grp.create_dataset("FitParameters", data=params_arr)
                ds_p.attrs["description"] = params_desc

                ds_c = grp.create_dataset("FitCovarianceMatrix", data=cov_arr)
                ds_c.attrs["description"] = cov_desc

                ds_r = grp.create_dataset("FitRange", data=range_arr)
                ds_r.attrs["units"] = "ns"
                ds_r.attrs["description"] = (
                    "[xmin, xmax]: the range the fastest (freely fit) component was fit/evaluated over"
                )

                # SlowRange = the fixing stage confirmed first (rank 0);
                # MediumRange = the one confirmed second (rank 1, N=3 only).
                range_ds_names = ["SlowRange", "MediumRange"]
                for i, fs in enumerate(self._fixed_stages):
                    ds_fr = grp.create_dataset(range_ds_names[i], data=np.asarray(fs["range"], dtype=float))
                    ds_fr.attrs["units"] = "ns"
                    ds_fr.attrs["description"] = (
                        f"[xmin, xmax]: tail range used to independently fix "
                        f"the {_rank_name(i, n)} component (FitParameters "
                        f"row {n - 1 - i}) before fitting the next component"
                    )

                ds_l = grp.create_dataset("Lifetime", data=lifetime_arr)
                ds_l.attrs["units"] = "ns"
                ds_l.attrs["description"] = "n: 1/b per exponential component"

                ds_td = grp.create_dataset("Total_Decay_Time", data=float(total_decay_time))
                ds_td.attrs["units"] = "ns"
                ds_td.attrs["description"] = (
                    "t solving 1/e = (1/n) * sum_i exp(-b_i * t) over this "
                    "fit's n FitParameters b_i's — the time at which the "
                    "unweighted average of each component's own normalized "
                    "decay curve has fallen to 1/e, independent of the a_i "
                    "amplitudes. Solved numerically (scipy.optimize.brentq). "
                    "For n == 1 this is identical to Lifetime (saved "
                    "separately anyway for schema consistency across n)."
                )

                if self._baseline_value is not None:
                    ds_bl = grp.create_dataset("Baseline", data=float(self._baseline_value))
                    ds_bl.attrs["units"] = "counts"
                    ds_bl.attrs["description"] = (
                        "constant fit (mean Counts) over the user-picked "
                        "baseline range, subtracted from every Counts sample "
                        "to give CountsBased"
                    )
                    if self._baseline_range is not None:
                        ds_bl.attrs["FitRange"] = list(self._baseline_range)
                        ds_bl.attrs["FitRange_units"] = "ns"

                    ds_cb = grp.create_dataset("CountsBased", data=self._counts)
                    ds_cb.attrs["description"] = (
                        "Counts with the fitted Baseline constant subtracted"
                    )
        except Exception as exc:
            QMessageBox.critical(self, "Save error", f"Could not save:\n{exc}")
            return

        msg = (
            f"Saved a {n}-exponential fit to the 'analysis' group in\n"
            f"{os.path.basename(self._path)}"
        )
        if self._baseline_value is not None:
            msg += "\n(including Baseline and CountsBased)"
        if self._data_time_offset != 0.0:
            msg += f"\n(fit ranges use a {self._data_time_offset:+.4g} ns data time offset — see DataTimeOffset)"
        QMessageBox.information(self, "Saved", msg)
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: saved {n}-exponential fit to {os.path.basename(self._path)}."
            )
