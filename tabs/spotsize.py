import os
import re
import numpy as np
import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QRadioButton, QFileDialog, QMessageBox, QSizePolicy,
    QCheckBox, QLineEdit, QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plotting import (
    PGCanvas, MultiLinePlotter, CategoricalScheme, TAB10,
    DraggableSpan, FilledRegion, MeasurementArrow, make_pg_toolbar,
)

from pl import _HC_EV_NM, _gaussian, _find_symmetric_95_bounds

_SPAN_COLORS = [
    (*pg.mkColor(TAB10[2]).getRgb()[:3], 60),   # Span 1 — green
    (*pg.mkColor(TAB10[1]).getRgb()[:3], 60),   # Span 2 — orange
]


class SpotsizeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Spotsize state ────────────────────────────────────────
        self._ss_x            = None
        self._ss_deriv        = None
        self._ss_deriv_raw    = None           # derivative before sign flip
        self._ss_spans        = [None, None]
        self._ss_span_widgets = [None, None]   # DraggableSpan instances

        # ── Build UI (body of original _build_spotsize_tab) ──────
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Sidebar ───────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(270)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        g_file = QGroupBox("Input data")
        fl = QVBoxLayout(g_file)
        self._ss_file_label = QLabel("No file loaded")
        self._ss_file_label.setWordWrap(True)
        fl.addWidget(self._ss_file_label)
        btn_load = QPushButton("Load Excel…")
        btn_load.clicked.connect(self._ss_load_file)
        fl.addWidget(btn_load)
        self._ss_is_deriv = QCheckBox("Column 2 is already a derivative")
        fl.addWidget(self._ss_is_deriv)
        self._ss_reverse_sign = QCheckBox("Reverse derivative sign")
        self._ss_reverse_sign.stateChanged.connect(self._ss_on_reverse_sign)
        fl.addWidget(self._ss_reverse_sign)
        sl.addWidget(g_file)

        g_par = QGroupBox("Parameters")
        pl2 = QVBoxLayout(g_par)
        r_ang = QHBoxLayout()
        r_ang.addWidget(QLabel("Angle (°):"))
        self._ss_angle = QLineEdit("45.0")
        r_ang.addWidget(self._ss_angle)
        pl2.addLayout(r_ang)
        sl.addWidget(g_par)

        g_span = QGroupBox("Span selection")
        spl2 = QVBoxLayout(g_span)
        spl2.addWidget(QLabel(
            "Drag either shaded region's edges to set its span.\n"
            "Both spans are needed before analysis."
        ))
        r_sp = QHBoxLayout()
        self._ss_rb_span1 = QRadioButton("Span 1")
        self._ss_rb_span2 = QRadioButton("Span 2")
        self._ss_rb_span1.setChecked(True)
        r_sp.addWidget(self._ss_rb_span1)
        r_sp.addWidget(self._ss_rb_span2)
        spl2.addLayout(r_sp)
        self._ss_span1_lbl = QLabel("Span 1: not set")
        self._ss_span2_lbl = QLabel("Span 2: not set")
        spl2.addWidget(self._ss_span1_lbl)
        spl2.addWidget(self._ss_span2_lbl)
        btn_clear = QPushButton("Clear spans")
        btn_clear.clicked.connect(self._ss_clear_spans)
        spl2.addWidget(btn_clear)
        sl.addWidget(g_span)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        self._ss_btn_analyze = QPushButton("Analyze")
        self._ss_btn_analyze.setEnabled(False)
        bold = QFont()
        bold.setBold(True)
        self._ss_btn_analyze.setFont(bold)
        self._ss_btn_analyze.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; }"
            "QPushButton:disabled { background-color: #bbbbbb; color: #666666; }"
        )
        self._ss_btn_analyze.setMinimumHeight(32)
        self._ss_btn_analyze.clicked.connect(self._ss_analyze)
        sl.addWidget(self._ss_btn_analyze)

        self._ss_result_lbl = QLabel("")
        self._ss_result_lbl.setWordWrap(True)
        sl.addWidget(self._ss_result_lbl)

        sl.addStretch(1)
        layout.addWidget(sidebar)

        # ── Right: canvas ─────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self._ss_canvas = PGCanvas(right, welcome_msg="Load an Excel file to begin.")
        self._ss_toolbar = make_pg_toolbar(self._ss_canvas, right)
        rl.addWidget(self._ss_toolbar)
        rl.addWidget(self._ss_canvas, stretch=1)
        layout.addWidget(right, stretch=1)

    # ── Spotsize: helpers ─────────────────────────────────────────

    def _ss_signed_deriv(self):
        if self._ss_deriv_raw is None:
            return None
        return -self._ss_deriv_raw if self._ss_reverse_sign.isChecked() else self._ss_deriv_raw

    def _ss_on_reverse_sign(self):
        if self._ss_deriv_raw is None:
            return
        self._ss_deriv = self._ss_signed_deriv()
        self._ss_result_lbl.setText("")
        self._ss_setup_canvas()

    def _ss_default_span_ranges(self):
        """Initial (draggable, not-yet-committed) span positions shown when
        data is first loaded or spans are cleared."""
        lo, hi = float(self._ss_x.min()), float(self._ss_x.max())
        width = (hi - lo) * 0.12
        c1 = lo + (hi - lo) * 0.25
        c2 = lo + (hi - lo) * 0.75
        return (c1 - width / 2, c1 + width / 2), (c2 - width / 2, c2 + width / 2)

    def _ss_setup_canvas(self):
        """Plot current derivative on canvas and install two draggable spans."""
        ax = self._ss_canvas.reset_axes()
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        mlp.plot(self._ss_x, self._ss_deriv, index=0, width=1.5)
        ax.setLabel("bottom", "x")
        ax.setLabel("left", "dy/dx")
        ax.setTitle("Drag the shaded regions to select two spans")
        ax.showGrid(x=True, y=True, alpha=0.4)

        default_ranges = self._ss_default_span_ranges()
        self._ss_span_widgets = [None, None]
        for i in range(2):
            widget = DraggableSpan(ax, color=_SPAN_COLORS[i], movable=True)
            initial = self._ss_spans[i] if self._ss_spans[i] is not None else default_ranges[i]
            widget.activate(initial_range=initial)
            widget.sigRegionSelected.connect(
                lambda xmin, xmax, idx=i: self._ss_on_span_select(idx, xmin, xmax)
            )
            self._ss_span_widgets[i] = widget

        self._ss_canvas.draw_idle()

    def _ss_on_span_select(self, idx, xmin, xmax):
        self._ss_spans[idx] = (xmin, xmax)
        lbls = [self._ss_span1_lbl, self._ss_span2_lbl]
        lbls[idx].setText(f"Span {idx + 1}: [{xmin:.5g}, {xmax:.5g}]")

        self._ss_canvas.draw_idle()
        if all(s is not None for s in self._ss_spans):
            self._ss_btn_analyze.setEnabled(True)

    # ── Spotsize: actions ─────────────────────────────────────────

    def _ss_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select scan file", "",
            "Scan files (*.xlsx *.xls *.txt *.dat *.csv);;All files (*.*)"
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".xlsx", ".xls"):
                import pandas as pd
                df = pd.read_excel(path, header=None)
                df = df.apply(pd.to_numeric, errors="coerce").dropna()
                x = df.iloc[:, 0].to_numpy(dtype=float)
                y = df.iloc[:, 1].to_numpy(dtype=float)
            else:
                data = np.loadtxt(path)
                x = data[:, 0]
                y = data[:, 1]
            order = np.argsort(x)
            x, y = x[order], y[order]
            if x.size < 10:
                QMessageBox.warning(
                    self, "Too few data points",
                    f"Only {x.size} data points after cleaning. Check the file."
                )
                return
        except Exception as exc:
            QMessageBox.critical(self, "Load error",
                                 f"Could not read:\n{path}\n\n{exc}")
            return

        self._ss_x         = x
        self._ss_deriv_raw = y if self._ss_is_deriv.isChecked() else np.gradient(y, x)
        self._ss_deriv     = self._ss_signed_deriv()
        self._ss_spans = [None, None]
        self._ss_span1_lbl.setText("Span 1: not set")
        self._ss_span2_lbl.setText("Span 2: not set")
        self._ss_result_lbl.setText("")
        self._ss_btn_analyze.setEnabled(False)
        self._ss_file_label.setText(
            f"{os.path.basename(path)}\n"
            f"{x.size} points  x = [{x.min():.4g}, {x.max():.4g}]"
        )
        self._ss_setup_canvas()

    def _ss_clear_spans(self):
        self._ss_spans = [None, None]
        self._ss_span1_lbl.setText("Span 1: not set")
        self._ss_span2_lbl.setText("Span 2: not set")
        self._ss_btn_analyze.setEnabled(False)
        self._ss_result_lbl.setText("")
        if self._ss_x is not None and all(w is not None for w in self._ss_span_widgets):
            for widget, rng in zip(self._ss_span_widgets, self._ss_default_span_ranges()):
                widget.set_range(*rng)
        self._ss_canvas.draw_idle()

    def _ss_analyze(self):
        from scipy.optimize import curve_fit
        from scipy.integrate import trapezoid

        x, deriv, spans = self._ss_x, self._ss_deriv, self._ss_spans
        if x is None or any(s is None for s in spans):
            return

        try:
            angle = float(self._ss_angle.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Invalid angle",
                                "Please enter a valid angle in degrees.")
            return

        combined_mask = np.zeros(x.size, dtype=bool)
        for xmin, xmax in spans:
            combined_mask |= (x >= xmin) & (x <= xmax)

        xs, ds = x[combined_mask], deriv[combined_mask]
        if xs.size < 5:
            QMessageBox.warning(
                self, "Too few points",
                f"Combined spans contain only {xs.size} point(s) — "
                "widen the spans."
            )
            return

        peak_idx  = int(np.argmax(np.abs(ds)))
        span_width = max(s[1] - s[0] for s in spans)
        p0 = [ds[peak_idx], xs[peak_idx], span_width / 4.0]
        try:
            popt, _ = curve_fit(_gaussian, xs, ds, p0=p0, maxfev=10_000)
        except RuntimeError:
            QMessageBox.warning(self, "Fit failed",
                                "Gaussian fit failed to converge. "
                                "Try adjusting the selected spans.")
            return

        amp, mu, sigma = popt[0], popt[1], abs(popt[2])
        x_lo, x_hi    = _find_symmetric_95_bounds(x, deriv, mu)
        delta          = x_hi - x_lo
        delta_corr     = delta * np.sin(np.radians(angle))

        integral_total  = trapezoid(deriv, x)
        bounds_mask     = (x >= x_lo) & (x <= x_hi)
        integral_bounds = trapezoid(deriv[bounds_mask], x[bounds_mask])

        # ── Draw result ───────────────────────────────────────────
        ax = self._ss_canvas.reset_axes()
        legend = ax.addLegend(labelTextSize="8pt")

        mlp = MultiLinePlotter(ax, CategoricalScheme())
        mlp.plot(x, deriv, index=0, label="dy/dx", width=1.5)

        self._ss_fill = FilledRegion(
            ax, x[bounds_mask], deriv[bounds_mask],
            brush=(*pg.mkColor(TAB10[2]).getRgb()[:3], 80),
        )

        x_fit = np.linspace(x_lo, x_hi, 500)
        fit_pen = pg.mkPen(TAB10[2], width=2, style=Qt.DashLine)
        ax.plot(x_fit, _gaussian(x_fit, amp, mu, sigma), pen=fit_pen,
                name=f"Gaussian  μ={mu:.5g}  σ={sigma:.5g}")

        mu_line = pg.InfiniteLine(pos=mu, angle=90,
                                   pen=pg.mkPen(TAB10[2], width=1.2, style=Qt.DotLine))
        ax.addItem(mu_line)

        # Arrow annotation — placed above the data, in the headroom carved
        # out below by expanding the y-range by 25%.
        ax.getViewBox().autoRange()
        (_x_lo_v, _x_hi_v), (y_lo, y_hi) = ax.getViewBox().viewRange()
        span_y = y_hi - y_lo
        new_y_hi = y_hi + 0.25 * span_y
        ax.setYRange(y_lo, new_y_hi, padding=0)
        span_y = new_y_hi - y_lo
        y_arrow = new_y_hi - 0.08 * span_y

        self._ss_arrow = MeasurementArrow(
            ax, x_lo, x_hi, y_arrow,
            label=f"Δx = {delta:.5g}  |  Δx × sin({angle}°) = {delta_corr:.5g}",
            color="r",
        )

        # Reapply span regions on the new axes (still draggable; identified by
        # their color-coded position in the sidebar rather than in the legend —
        # pyqtgraph's LegendItem doesn't support LinearRegionItem entries).
        self._ss_span_widgets = [None, None]
        for i, sp in enumerate(spans):
            widget = DraggableSpan(ax, color=_SPAN_COLORS[i], movable=True)
            widget.activate(initial_range=sp)
            widget.sigRegionSelected.connect(
                lambda xmin, xmax, idx=i: self._ss_on_span_select(idx, xmin, xmax)
            )
            self._ss_span_widgets[i] = widget

        ax.setLabel("bottom", "x")
        ax.setLabel("left", "dy/dx")
        ax.setTitle(f"Δx × sin({angle}°) = {delta_corr:.5g}")
        ax.showGrid(x=True, y=True, alpha=0.4)
        self._ss_canvas.draw_idle()

        self._ss_result_lbl.setText(
            f"μ = {mu:.6g}\n"
            f"σ = {sigma:.6g}\n"
            f"95.4 % bounds: [{x_lo:.5g}, {x_hi:.5g}]\n"
            f"Δx = {delta:.6g}\n"
            f"Δx × sin({angle}°) = {delta_corr:.6g}\n"
            f"∫ total = {integral_total:.4g}\n"
            f"∫ window = {integral_bounds:.4g}"
        )
