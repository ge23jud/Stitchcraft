import os
import sys
import numpy as np
import h5py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QFileDialog, QMessageBox, QFrame,
    QScrollArea, QStyle,
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


class TRPLTab(QWidget):
    """"TRPL": load a TRPL histogram HDF5 file (as written by the Convert
    tab), display Counts vs. Times on a log-y axis, and fit N user-picked
    time intervals with a single-exponential decay ln(Counts) = a*Times + b
    (equivalently Counts = exp(b) * exp(a*Times)) to extract lifetimes
    (-1/a). Results are saved into the file's "analysis" HDF5 subgroup,
    mirroring the Analysis tab's save convention (FitWindows-style
    (n, 2) range array, etc.).
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── State ────────────────────────────────────────────────
        self._path  = None
        self._times = None   # ns
        self._counts = None
        self._meta  = {}

        self._mode = "idle"   # "idle" | "selecting"
        self._n_fits    = 0
        self._fit_idx   = 0
        self._fit_ranges = []   # [(xmin, xmax), ...] — None until confirmed
        self._fit_params = []   # [(a, b), ...]
        self._fit_cov    = []   # [2x2 ndarray, ...]

        self._span         = None   # DraggableSpan, active during selection
        self._pending_range = None
        self._pending_fit    = None
        self._preview_item   = None  # live (unconfirmed) fit-curve PlotDataItem

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

        # Fit group
        g_fit = QGroupBox("Lifetime fit  (ln(Counts) = a·Times + b)")
        fitl = QVBoxLayout(g_fit)
        r_n = QHBoxLayout()
        r_n.addWidget(QLabel("Number of fits:"))
        self._n_fits_input = QLineEdit("1")
        r_n.addWidget(self._n_fits_input)
        fitl.addLayout(r_n)

        self._btn_start = QPushButton("Select fit ranges…")
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._on_start_selecting)
        fitl.addWidget(self._btn_start)

        self._btn_reset_fits = QPushButton("Reset fits")
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
        self._btn_act_confirm = QPushButton("Confirm & next range  ▶")
        self._btn_act_cancel  = QPushButton("Cancel selection")
        self._btn_act_confirm.clicked.connect(self._on_confirm_and_next)
        self._btn_act_cancel.clicked.connect(self._on_reset_fits)
        ab.addWidget(self._action_lbl, stretch=1)
        ab.addWidget(self._btn_act_confirm)
        ab.addWidget(self._btn_act_cancel)
        self._action_bar.setVisible(False)
        rl.addWidget(self._action_bar)

        layout.addWidget(right, stretch=1)

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
        self._times  = data["times"]
        self._counts = data["counts"]
        self._meta   = data

        self._reset_fit_state()
        self._action_bar.setVisible(False)
        self._mode = "idle"

        label = os.path.basename(path)
        info = f"{label}\n{len(self._times)} channel(s)"
        if data.get("power_mW") is not None:
            info += f"\n{data['power_mW']:.4g} mW"
        self._file_lbl.setText(info)
        self._btn_start.setEnabled(True)
        self._btn_save.setEnabled(False)
        self._fit_status_lbl.setText("")
        self._draw_base_plot()

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: loaded {label} ({len(self._times)} channels)."
            )

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
        mlp.plot(self._times, self._counts, index=0, width=1.0, antialias=False)
        ax.setLabel("bottom", "Time (ns)")
        ax.setLabel("left", "Counts")
        title = os.path.basename(self._path) if self._path else ""
        if self._meta.get("power_mW") is not None:
            title += f"  —  {self._meta['power_mW']:.4g} mW"
        ax.setTitle(title)
        ax.showGrid(x=True, y=True, alpha=0.3)

        for rng, params in zip(self._fit_ranges, self._fit_params):
            if rng is None or params is None:
                continue
            self._draw_confirmed_overlay(ax, rng, params)

        self._canvas.draw_idle()
        return ax

    def _draw_confirmed_overlay(self, ax, rng, params):
        xmin, xmax = rng
        a, b = params
        region = pg.LinearRegionItem(
            values=(xmin, xmax), orientation="vertical",
            brush=pg.mkBrush(200, 30, 30, 30), movable=False,
        )
        region.setZValue(5)
        ax.addItem(region)
        x_curve = np.linspace(xmin, xmax, 200)
        y_curve = np.exp(b) * np.exp(a * x_curve)
        ax.plot(x_curve, y_curve, pen=pg.mkPen("red", width=1.5))

    def _remove_preview_item(self):
        if self._preview_item is not None:
            try:
                self._canvas.plot_item.removeItem(self._preview_item)
            except Exception:
                pass
            self._preview_item = None

    # ── Fit-range selection loop ──────────────────────────────────

    def _reset_fit_state(self):
        self._mode = "idle"
        self._n_fits = 0
        self._fit_idx = 0
        self._fit_ranges = []
        self._fit_params = []
        self._fit_cov = []
        self._pending_range = None
        self._pending_fit = None
        if self._span is not None:
            self._span.deactivate()
            self._span = None
        self._remove_preview_item()

    def _on_start_selecting(self):
        if self._times is None:
            return
        text = self._n_fits_input.text().strip()
        try:
            n = int(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid input",
                                "Number of fits must be an integer.")
            return
        if n < 1:
            QMessageBox.warning(self, "Invalid input",
                                "Number of fits must be at least 1.")
            return

        self._n_fits     = n
        self._fit_idx    = 0
        self._fit_ranges = [None] * n
        self._fit_params = [None] * n
        self._fit_cov    = [None] * n
        self._mode = "selecting"
        self._btn_save.setEnabled(False)
        self._btn_reset_fits.setEnabled(True)
        self._show_range_selector(0)

    def _default_bounds(self, i):
        """Seed the i-th span at the i-th equal slice of the time axis —
        just a starting point; the user drags it to the actual decay
        region they want to fit."""
        t_min, t_max = float(self._times[0]), float(self._times[-1])
        seg = (t_max - t_min) / self._n_fits
        return (t_min + i * seg, t_min + (i + 1) * seg)

    def _show_range_selector(self, i):
        ax = self._draw_base_plot()

        t_min, t_max = float(self._times[0]), float(self._times[-1])
        x_lo, x_hi = self._default_bounds(i)

        self._span = DraggableSpan(ax, color=(0, 150, 0, 60), movable=True)
        self._span.activate(initial_range=(x_lo, x_hi), bounds=(t_min, t_max))
        self._span.sigRegionSelected.connect(self._on_span_changed)

        self._btn_act_confirm.setEnabled(False)
        self._action_bar.setVisible(True)
        self._on_span_changed(x_lo, x_hi)   # show an initial preview immediately

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: drag the region for fit range {i + 1}/{self._n_fits}, "
                "then Confirm."
            )

    def _on_span_changed(self, xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        self._pending_range = (xmin, xmax)

        mask = (self._times >= xmin) & (self._times <= xmax) & (self._counts > 0)
        n_pts = int(np.sum(mask))
        if n_pts < 2:
            self._pending_fit = None
            self._remove_preview_item()
            self._btn_act_confirm.setEnabled(False)
            self._action_lbl.setText(
                f"Fit range {self._fit_idx + 1}/{self._n_fits}: only {n_pts} "
                "positive-count point(s) in range — widen it."
            )
            self._fit_status_lbl.setText(self._action_lbl.text())
            return

        x = self._times[mask]
        y = self._counts[mask]
        if n_pts >= 3:
            coeffs, cov = np.polyfit(x, np.log(y), 1, cov=True)
        else:
            # np.polyfit's cov=True is undefined (rank-deficient) with only
            # as many points as parameters — fit without it and report NaNs
            # instead of letting numpy raise a RankWarning (mirrors the
            # Analysis tab's threshold fit, which does the same for <3 pts).
            coeffs = np.polyfit(x, np.log(y), 1)
            cov = np.full((2, 2), np.nan)
        a, b = float(coeffs[0]), float(coeffs[1])
        self._pending_fit = (a, b, cov)

        self._remove_preview_item()
        x_curve = np.linspace(xmin, xmax, 200)
        y_curve = np.exp(b) * np.exp(a * x_curve)
        self._preview_item = self._canvas.plot_item.plot(
            x_curve, y_curve, pen=pg.mkPen("yellow", width=1.5, style=Qt.DashLine)
        )
        self._canvas.draw_idle()

        lifetime = -1.0 / a if a != 0 else float("nan")
        msg = (
            f"Fit range {self._fit_idx + 1}/{self._n_fits}: "
            f"a={a:.4g} 1/ns, b={b:.4g}, lifetime={lifetime:.4g} ns  ({n_pts} pts)"
        )
        self._action_lbl.setText(msg)
        self._fit_status_lbl.setText(msg)
        self._btn_act_confirm.setEnabled(True)

    def _on_confirm_and_next(self):
        if self._mode != "selecting" or self._pending_fit is None:
            return
        i = self._fit_idx
        a, b, cov = self._pending_fit
        self._fit_ranges[i] = self._pending_range
        self._fit_params[i] = (a, b)
        self._fit_cov[i]    = cov

        if self._span is not None:
            self._span.deactivate()
            self._span = None
        self._remove_preview_item()
        self._pending_range = None
        self._pending_fit   = None

        i += 1
        self._fit_idx = i
        if i < self._n_fits:
            self._show_range_selector(i)
        else:
            self._finish_fits()

    def _finish_fits(self):
        self._mode = "idle"
        self._action_bar.setVisible(False)
        self._draw_base_plot()
        self._btn_save.setEnabled(True)
        self._fit_status_lbl.setText(
            f"All {self._n_fits} fit(s) confirmed. Ready to save."
        )
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: {self._n_fits} fit(s) confirmed. Ready to save."
            )

    def _on_reset_fits(self):
        self._reset_fit_state()
        self._btn_reset_fits.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._action_bar.setVisible(False)
        self._fit_status_lbl.setText("")
        if self._times is not None:
            self._draw_base_plot()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("TRPL: fit selection reset.")

    # ── Save ─────────────────────────────────────────────────────

    def _on_save(self):
        if (self._path is None or self._n_fits == 0
                or any(r is None for r in self._fit_ranges)
                or any(p is None for p in self._fit_params)):
            QMessageBox.warning(self, "Nothing to save",
                                "Confirm all fit ranges first.")
            return

        n = self._n_fits
        params_arr   = np.array(self._fit_params, dtype=float)              # (n, 2): [a, b]
        cov_arr      = np.array(self._fit_cov, dtype=float)                  # (n, 2, 2)
        ranges_arr   = np.array(self._fit_ranges, dtype=float)               # (n, 2)
        lifetime_arr = np.array([-1.0 / a for a, _b in self._fit_params],
                                 dtype=float)                                # (n,)

        try:
            with h5py.File(self._path, "a") as f:
                if "analysis" in f:
                    del f["analysis"]
                grp = f.create_group("analysis")

                grp.attrs["FitFunction"] = (
                    "exp(b) * exp(a * x)  —  from the linear fit "
                    "ln(y) = a*x + b, where x = Times (ns) and y = Counts"
                )

                ds_p = grp.create_dataset("FitParameters", data=params_arr)
                ds_p.attrs["description"] = (
                    "n x 2: [a, b] per fit, from ln(Counts) = a*Times + b"
                )

                ds_c = grp.create_dataset("FitCovarianceMatrix", data=cov_arr)
                ds_c.attrs["description"] = (
                    "n x 2 x 2: covariance matrix of [a, b] per fit "
                    "(np.polyfit(..., cov=True))"
                )

                ds_r = grp.create_dataset("FitRanges", data=ranges_arr)
                ds_r.attrs["units"] = "ns"
                ds_r.attrs["description"] = "n x 2: [xmin, xmax] fit range per fit"

                ds_l = grp.create_dataset("Lifetime", data=lifetime_arr)
                ds_l.attrs["units"] = "ns"
                ds_l.attrs["description"] = "n: -1/a per fit"
        except Exception as exc:
            QMessageBox.critical(self, "Save error", f"Could not save:\n{exc}")
            return

        QMessageBox.information(
            self, "Saved",
            f"Saved {n} fit(s) to the 'analysis' group in\n{os.path.basename(self._path)}"
        )
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"TRPL: saved {n} fit(s) to {os.path.basename(self._path)}."
            )
