import os
import sys
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QFileDialog, QMessageBox, QDialog,
    QFormLayout, QFrame,
)
import pyqtgraph as pg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plotting import (
    PGCanvas, MultiLinePlotter, CategoricalScheme, TAB10,
    DraggableSpan, make_pg_toolbar,
)
from io_utils import _read_h5_spectrum
from pl import _gaussian

_SPAN_COLOR = (*pg.mkColor(TAB10[2]).getRgb()[:3], 60)

# Registry of selectable fit functions. Add new entries here as more fit
# functions are needed — each just needs a callable and its parameter names.
FIT_FUNCTIONS = {
    "Gaussian": {
        "func": _gaussian,
        "param_labels": ["Amplitude", "Center (eV)", "Sigma (eV)"],
    },
}


class FitTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Fit state ─────────────────────────────────────────────
        self._fit_data        = None   # dict from _read_h5_spectrum
        self._fit_energy      = None
        self._fit_counts      = None   # 1D SpectraDiff for the plotted power step
        self._fit_span        = None
        self._fit_span_widget = None   # DraggableSpan instance

        # ── Build UI ────────────────────────────────────────────────
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
        self._fit_file_label = QLabel("No file loaded")
        self._fit_file_label.setWordWrap(True)
        fl.addWidget(self._fit_file_label)
        btn_load = QPushButton("Load Powerseries HDF5…")
        btn_load.clicked.connect(self._fit_load_file)
        fl.addWidget(btn_load)
        sl.addWidget(g_file)

        g_func = QGroupBox("Fit function")
        gfl = QVBoxLayout(g_func)
        self._fit_func_combo = QComboBox()
        self._fit_func_combo.addItems(list(FIT_FUNCTIONS.keys()))
        gfl.addWidget(self._fit_func_combo)
        sl.addWidget(g_func)

        g_span = QGroupBox("Span selection")
        spl = QVBoxLayout(g_span)
        spl.addWidget(QLabel("Drag the shaded region's edges to set the fit span."))
        self._fit_span_lbl = QLabel("Span: not set")
        spl.addWidget(self._fit_span_lbl)
        btn_clear = QPushButton("Clear span")
        btn_clear.clicked.connect(self._fit_clear_span)
        spl.addWidget(btn_clear)
        sl.addWidget(g_span)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        self._fit_btn_fit = QPushButton("Fit")
        self._fit_btn_fit.setEnabled(False)
        bold = QFont()
        bold.setBold(True)
        self._fit_btn_fit.setFont(bold)
        self._fit_btn_fit.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; }"
            "QPushButton:disabled { background-color: #bbbbbb; color: #666666; }"
        )
        self._fit_btn_fit.setMinimumHeight(32)
        self._fit_btn_fit.clicked.connect(self._fit_run)
        sl.addWidget(self._fit_btn_fit)

        self._fit_result_lbl = QLabel("")
        self._fit_result_lbl.setWordWrap(True)
        sl.addWidget(self._fit_result_lbl)

        sl.addStretch(1)
        layout.addWidget(sidebar)

        # ── Right: canvas ─────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self._fit_canvas = PGCanvas(right, welcome_msg="Load a Powerseries HDF5 file to begin.")
        self._fit_toolbar = make_pg_toolbar(self._fit_canvas, right)
        rl.addWidget(self._fit_toolbar)
        rl.addWidget(self._fit_canvas, stretch=1)
        layout.addWidget(right, stretch=1)

    # ── Fit: helpers ────────────────────────────────────────────────

    def _fit_default_span_range(self):
        """Initial (draggable, not-yet-committed) span position shown when
        data is first loaded or the span is cleared."""
        lo, hi = float(self._fit_energy.min()), float(self._fit_energy.max())
        width = (hi - lo) * 0.12
        center = lo + (hi - lo) * 0.5
        return center - width / 2, center + width / 2

    def _fit_setup_canvas(self):
        """Plot the loaded spectrum on canvas and install a draggable span."""
        ax = self._fit_canvas.reset_axes()
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        mlp.plot(self._fit_energy, self._fit_counts, index=0, width=1.5)
        ax.setLabel("bottom", "Energy (eV)")
        ax.setLabel("left", "SpectraDiff (counts)")
        ax.setTitle("Drag the shaded region to select the fit span")
        ax.showGrid(x=True, y=True, alpha=0.4)

        span_range = self._fit_span if self._fit_span is not None else self._fit_default_span_range()
        widget = DraggableSpan(ax, color=_SPAN_COLOR, movable=True)
        widget.activate(initial_range=span_range)
        widget.sigRegionSelected.connect(self._fit_on_span_select)
        self._fit_span_widget = widget

        self._fit_canvas.draw_idle()

    def _fit_on_span_select(self, xmin, xmax):
        self._fit_span = (xmin, xmax)
        self._fit_span_lbl.setText(f"Span: [{xmin:.5g}, {xmax:.5g}]")
        self._fit_btn_fit.setEnabled(self._fit_energy is not None)

    # ── Fit: actions ────────────────────────────────────────────────

    def _fit_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Powerseries HDF5 file", "",
            "HDF5 files (*.h5);;All files (*.*)"
        )
        if not path:
            return
        try:
            data = _read_h5_spectrum(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error",
                                 f"Could not read:\n{path}\n\n{exc}")
            return

        self._fit_data   = data
        self._fit_energy = data["energy"]
        self._fit_counts = data["counts"][:, -1]     # highest power step
        self._fit_span   = None
        self._fit_span_lbl.setText("Span: not set")
        self._fit_result_lbl.setText("")
        self._fit_btn_fit.setEnabled(False)
        n_powers = data["counts"].shape[1]
        self._fit_file_label.setText(
            f"{data['label']}\n"
            f"{n_powers} power step(s) — showing highest ({data['power_mW'][-1]:.4g} mW)"
        )
        self._fit_setup_canvas()

    def _fit_clear_span(self):
        self._fit_span = None
        self._fit_span_lbl.setText("Span: not set")
        self._fit_btn_fit.setEnabled(False)
        self._fit_result_lbl.setText("")
        if self._fit_energy is not None and self._fit_span_widget is not None:
            self._fit_span_widget.set_range(*self._fit_default_span_range())
        self._fit_canvas.draw_idle()

    def _fit_run(self):
        from scipy.optimize import curve_fit

        x, y, span = self._fit_energy, self._fit_counts, self._fit_span
        if x is None or span is None:
            return

        xmin, xmax = span
        mask = (x >= xmin) & (x <= xmax)
        xs, ys = x[mask], y[mask]
        if xs.size < 5:
            QMessageBox.warning(
                self, "Too few points",
                f"Selected span contains only {xs.size} point(s) — widen the span."
            )
            return

        func_name = self._fit_func_combo.currentText()
        spec = FIT_FUNCTIONS[func_name]
        func, param_labels = spec["func"], spec["param_labels"]

        peak_idx = int(np.argmax(ys))
        p0 = [float(ys[peak_idx]), float(xs[peak_idx]), (xmax - xmin) / 4.0]
        try:
            popt, pcov = curve_fit(func, xs, ys, p0=p0, maxfev=10_000)
        except RuntimeError:
            QMessageBox.warning(
                self, "Fit failed",
                f"{func_name} fit failed to converge. Try adjusting the selected span."
            )
            return

        perr = np.sqrt(np.diag(pcov))
        residuals = ys - func(xs, *popt)
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # ── Redraw with fit overlay ─────────────────────────────────
        ax = self._fit_canvas.reset_axes()
        ax.addLegend(labelTextSize="8pt")
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        mlp.plot(x, y, index=0, label="SpectraDiff", width=1.5)

        x_fit = np.linspace(xmin, xmax, 500)
        fit_pen = pg.mkPen(TAB10[2], width=2, style=Qt.DashLine)
        ax.plot(x_fit, func(x_fit, *popt), pen=fit_pen, name=f"{func_name} fit")

        self._fit_span_widget = DraggableSpan(ax, color=_SPAN_COLOR, movable=True)
        self._fit_span_widget.activate(initial_range=span)
        self._fit_span_widget.sigRegionSelected.connect(self._fit_on_span_select)

        ax.setLabel("bottom", "Energy (eV)")
        ax.setLabel("left", "SpectraDiff (counts)")
        ax.setTitle(f"{func_name} fit  —  R² = {r_squared:.4g}")
        ax.showGrid(x=True, y=True, alpha=0.4)
        self._fit_canvas.draw_idle()

        summary = "\n".join(
            f"{label} = {val:.6g} ± {err:.4g}"
            for label, val, err in zip(param_labels, popt, perr)
        )
        self._fit_result_lbl.setText(f"R² = {r_squared:.5g}\n{summary}")

        self._fit_show_results_dialog(func_name, param_labels, popt, perr, r_squared)

    def _fit_show_results_dialog(self, func_name, param_labels, popt, perr, r_squared):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"{func_name} fit parameters")
        dlg.resize(360, 220)
        dl = QVBoxLayout(dlg)

        form = QFormLayout()
        for label, val, err in zip(param_labels, popt, perr):
            form.addRow(label + ":", QLabel(f"{val:.6g} ± {err:.4g}"))
        form.addRow("R²:", QLabel(f"{r_squared:.6g}"))
        dl.addLayout(form)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        dl.addWidget(btn_close)
        dlg.exec_()
