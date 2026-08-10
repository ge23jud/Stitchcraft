import os
import sys
import numpy as np
from scipy.optimize import curve_fit

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QSpinBox, QFileDialog, QMessageBox, QDialog,
    QFormLayout, QFrame, QScrollArea,
)
import pyqtgraph as pg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plotting import (
    PGCanvas, MultiLinePlotter, CategoricalScheme, TAB10,
    DraggableSpan, make_pg_toolbar,
)

_SPAN_COLOR    = (*pg.mkColor(TAB10[2]).getRgb()[:3], 60)
_BOUND_PEN     = pg.mkPen((150, 150, 150), width=1.0, style=Qt.DotLine)
_CLICK_PEN     = pg.mkPen("red", width=1.0, style=Qt.DashLine)
_FIT_PEN       = pg.mkPen(TAB10[2], width=2, style=Qt.DashLine)


def _gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def _lorentzian(x, a, x0, gamma):
    return a * (gamma / 2.0) / ((x - x0) ** 2 + (gamma / 2.0) ** 2)


def _pseudo_voigt(x, a, x0, gamma, eta):
    sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
    return eta * _gaussian(x, a, x0, sigma) + (1 - eta) * _lorentzian(x, a, x0, gamma)


class XrdTab(QWidget):
    """Multi-peak pseudo-Voigt fitting for XRD line scans: load a two-column
    (angle, intensity) text file, drag a span to pick the fit range, click
    once per peak to seed initial centers, then fit all peaks together
    (single shared background offset — the pasted script summed one offset
    per peak, which silently scaled the baseline with peak count)."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── State ─────────────────────────────────────────────────
        self._xrd_x          = None
        self._xrd_y          = None
        self._xrd_file        = None
        self._xrd_span        = None   # DraggableSpan (movable, range picker)
        self._xrd_span_range  = None
        self._xrd_mode        = "idle"  # idle | picking
        self._xrd_click_conn  = None
        self._xrd_click_x     = []

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
        self._xrd_file_label = QLabel("No file loaded")
        self._xrd_file_label.setWordWrap(True)
        fl.addWidget(self._xrd_file_label)
        btn_load = QPushButton("Load XRD data…")
        btn_load.clicked.connect(self._xrd_load_file)
        fl.addWidget(btn_load)
        sl.addWidget(g_file)

        g_span = QGroupBox("Fit range")
        spl = QVBoxLayout(g_span)
        spl.addWidget(QLabel("Drag the shaded region's edges to set the fit range."))
        self._xrd_span_lbl = QLabel("Range: not set")
        spl.addWidget(self._xrd_span_lbl)
        sl.addWidget(g_span)

        g_peaks = QGroupBox("Peaks")
        pkl = QVBoxLayout(g_peaks)
        row = QHBoxLayout()
        row.addWidget(QLabel("Number of peaks:"))
        self._xrd_npeaks_spin = QSpinBox()
        self._xrd_npeaks_spin.setRange(1, 10)
        self._xrd_npeaks_spin.setValue(2)
        row.addWidget(self._xrd_npeaks_spin)
        pkl.addLayout(row)
        self._xrd_btn_pick = QPushButton("Select peak positions…")
        self._xrd_btn_pick.setEnabled(False)
        self._xrd_btn_pick.clicked.connect(self._xrd_start_picking)
        pkl.addWidget(self._xrd_btn_pick)
        sl.addWidget(g_peaks)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        self._xrd_result_lbl = QLabel("")
        self._xrd_result_lbl.setWordWrap(True)
        sl.addWidget(self._xrd_result_lbl)

        sl.addStretch(1)
        layout.addWidget(sidebar)

        # ── Right: toolbar + canvas + action bar ───────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self._xrd_canvas  = PGCanvas(right, welcome_msg="Load an XRD data file to begin.")
        self._xrd_toolbar = make_pg_toolbar(self._xrd_canvas, right)
        rl.addWidget(self._xrd_toolbar)
        rl.addWidget(self._xrd_canvas, stretch=1)

        self._xrd_action_bar = QFrame()
        self._xrd_action_bar.setFrameShape(QFrame.StyledPanel)
        ab = QHBoxLayout(self._xrd_action_bar)
        ab.setContentsMargins(8, 4, 8, 4)
        self._xrd_action_lbl  = QLabel("")
        self._xrd_btn_reset   = QPushButton("Reset clicks")
        self._xrd_btn_cancel  = QPushButton("Cancel")
        self._xrd_btn_reset.clicked.connect(self._xrd_reset_picking)
        self._xrd_btn_cancel.clicked.connect(self._xrd_cancel_picking)
        ab.addWidget(self._xrd_action_lbl)
        ab.addWidget(self._xrd_btn_reset)
        ab.addWidget(self._xrd_btn_cancel)
        ab.addStretch(1)
        self._xrd_action_bar.setVisible(False)
        rl.addWidget(self._xrd_action_bar)

        layout.addWidget(right, stretch=1)

    # ── Helpers ───────────────────────────────────────────────────

    def _xrd_default_span_range(self):
        lo, hi = float(self._xrd_x.min()), float(self._xrd_x.max())
        width = (hi - lo) * 0.12
        center = lo + (hi - lo) * 0.5
        return center - width / 2, center + width / 2

    def _xrd_show_preview(self, span_range=None):
        """(Re)draw the full spectrum with a movable fit-range span."""
        ax = self._xrd_canvas.reset_axes()
        ax.setLogMode(y=True)
        # clipToView + auto-downsampling + antialias=False: same fix as the
        # Convert tab's TRPL preview — a non-default pen width under the
        # app-wide antialias=True default is slow for large scans, and
        # without downsampling every pan/zoom repaints every point.
        ax.setClipToView(True)
        ax.setDownsampling(auto=True, mode="peak")
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        mlp.plot(self._xrd_x, self._xrd_y, index=0, width=1.0, antialias=False)
        ax.setLabel("bottom", "2θ (°)")
        ax.setLabel("left", "Intensity (a.u.)")
        ax.setTitle("Drag the shaded region to select the fit range")
        ax.showGrid(x=True, y=True, alpha=0.3)

        rng = span_range or self._xrd_span_range or self._xrd_default_span_range()
        self._xrd_span = DraggableSpan(ax, color=_SPAN_COLOR, movable=True)
        self._xrd_span.activate(initial_range=rng)
        self._xrd_span.sigRegionSelected.connect(self._xrd_on_span_select)
        self._xrd_span_range = rng
        self._xrd_span_lbl.setText(f"Range: [{rng[0]:.5g}, {rng[1]:.5g}]")
        self._xrd_canvas.draw_idle()

    def _xrd_on_span_select(self, xmin, xmax):
        self._xrd_span_range = (xmin, xmax)
        self._xrd_span_lbl.setText(f"Range: [{xmin:.5g}, {xmax:.5g}]")

    # ── Actions: load ─────────────────────────────────────────────

    def _xrd_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select XRD data file", "",
            "Text files (*.txt *.dat *.xy *.csv);;All files (*.*)"
        )
        if not path:
            return
        try:
            data = np.loadtxt(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error",
                                 f"Could not read:\n{path}\n\n{exc}")
            return
        if data.ndim != 2 or data.shape[1] < 2:
            QMessageBox.critical(self, "Load error",
                                 "Expected a two-column (angle, intensity) file.")
            return

        self._xrd_file       = path
        self._xrd_x          = data[:, 0]
        self._xrd_y          = data[:, 1]
        self._xrd_span_range = None
        self._xrd_mode        = "idle"
        self._xrd_action_bar.setVisible(False)
        self._xrd_result_lbl.setText("")
        self._xrd_file_label.setText(
            f"{os.path.basename(path)}\n{data.shape[0]} points"
        )
        self._xrd_btn_pick.setEnabled(True)
        self._xrd_npeaks_spin.setEnabled(True)
        self._xrd_show_preview()

    # ── Actions: peak picking ─────────────────────────────────────

    def _xrd_start_picking(self):
        if self._xrd_x is None or self._xrd_span_range is None:
            return
        xmin, xmax = self._xrd_span_range
        mask = (self._xrd_x >= xmin) & (self._xrd_x <= xmax)
        if np.count_nonzero(mask) < 5:
            QMessageBox.warning(self, "Range too small",
                                "Selected fit range contains too few points.")
            return

        self._xrd_n_peaks = int(self._xrd_npeaks_spin.value())
        self._xrd_click_x = []
        if self._xrd_span is not None:
            self._xrd_span.deactivate()
            self._xrd_span = None

        ax = self._xrd_canvas.reset_axes()
        ax.setLogMode(y=True)
        ax.setClipToView(True)
        ax.setDownsampling(auto=True, mode="peak")
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        mlp.plot(self._xrd_x, self._xrd_y, index=0, width=1.0, antialias=False)
        for b in (xmin, xmax):
            ax.addItem(pg.InfiniteLine(pos=b, angle=90, pen=_BOUND_PEN))
        ax.setXRange(xmin, xmax, padding=0.05)
        ax.setLabel("bottom", "2θ (°)")
        ax.setLabel("left", "Intensity (a.u.)")
        ax.setTitle(f"Click {self._xrd_n_peaks} peak center(s), left to right")
        ax.showGrid(x=True, y=True, alpha=0.3)
        self._xrd_canvas.draw_idle()

        self._xrd_mode = "picking"
        self._xrd_btn_pick.setEnabled(False)
        self._xrd_npeaks_spin.setEnabled(False)
        self._xrd_action_lbl.setText(f"0 of {self._xrd_n_peaks} peaks selected")
        self._xrd_action_bar.setVisible(True)
        self._xrd_click_conn = self._xrd_canvas.sigDataClicked.connect(
            self._xrd_on_peak_click
        )

    def _xrd_on_peak_click(self, x, y, button):
        if self._xrd_mode != "picking" or button != Qt.LeftButton:
            return
        self._xrd_click_x.append(x)
        n = len(self._xrd_click_x)
        self._xrd_action_lbl.setText(f"{n} of {self._xrd_n_peaks} peaks selected")
        self._xrd_canvas.plot_item.addItem(
            pg.InfiniteLine(pos=x, angle=90, pen=_CLICK_PEN)
        )
        self._xrd_canvas.draw_idle()
        if n == self._xrd_n_peaks:
            self._xrd_disconnect_clicks()
            self._xrd_run_fit()

    def _xrd_disconnect_clicks(self):
        if self._xrd_click_conn is not None:
            try:
                self._xrd_canvas.sigDataClicked.disconnect(self._xrd_on_peak_click)
            except (TypeError, RuntimeError):
                pass
            self._xrd_click_conn = None

    def _xrd_reset_picking(self):
        if self._xrd_mode != "picking":
            return
        self._xrd_disconnect_clicks()
        self._xrd_start_picking()

    def _xrd_cancel_picking(self):
        self._xrd_disconnect_clicks()
        self._xrd_mode = "idle"
        self._xrd_action_bar.setVisible(False)
        self._xrd_btn_pick.setEnabled(True)
        self._xrd_npeaks_spin.setEnabled(True)
        self._xrd_show_preview(self._xrd_span_range)

    # ── Fit ───────────────────────────────────────────────────────

    def _xrd_run_fit(self):
        xmin, xmax = self._xrd_span_range
        mask = (self._xrd_x >= xmin) & (self._xrd_x <= xmax)
        xs, ys = self._xrd_x[mask], self._xrd_y[mask]
        n_peaks = self._xrd_n_peaks
        centers = sorted(self._xrd_click_x)

        offset_guess = float(np.min(ys))
        amp_guess    = float(np.max(ys) - offset_guess)
        gamma_guess  = (xmax - xmin) / (4.0 * n_peaks)

        p0 = [offset_guess]
        for c in centers:
            p0 += [amp_guess, c, gamma_guess, 0.5]

        def model(x, offset, *params):
            y = np.full_like(x, offset, dtype=float)
            for i in range(n_peaks):
                a, x0, gamma, eta = params[i * 4:i * 4 + 4]
                y = y + _pseudo_voigt(x, a, x0, gamma, eta)
            return y

        try:
            popt, pcov = curve_fit(model, xs, ys, p0=p0, maxfev=20000)
        except RuntimeError:
            QMessageBox.warning(
                self, "Fit failed",
                "Multi-peak fit failed to converge. Reset and try different "
                "peak positions or a wider/narrower fit range."
            )
            self._xrd_reset_picking()
            return

        perr = np.sqrt(np.diag(pcov))
        residuals = ys - model(xs, *popt)
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # ── Redraw with fit overlay ─────────────────────────────────
        ax = self._xrd_canvas.reset_axes()
        ax.setLogMode(y=True)
        ax.addLegend(labelTextSize="8pt")
        ax.setClipToView(True)
        ax.setDownsampling(auto=True, mode="peak")
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        mlp.plot(self._xrd_x, self._xrd_y, index=0, label="Data",
                width=1.0, antialias=False)
        x_fit = np.linspace(xmin, xmax, 500)
        ax.plot(x_fit, model(x_fit, *popt), pen=_FIT_PEN, name="Pseudo-Voigt fit")

        self._xrd_span = DraggableSpan(ax, color=_SPAN_COLOR, movable=True)
        self._xrd_span.activate(initial_range=(xmin, xmax))
        self._xrd_span.sigRegionSelected.connect(self._xrd_on_span_select)

        ax.setLabel("bottom", "2θ (°)")
        ax.setLabel("left", "Intensity (a.u.)")
        ax.setTitle(f"Pseudo-Voigt fit — {n_peaks} peak(s), R² = {r_squared:.4g}")
        ax.showGrid(x=True, y=True, alpha=0.3)
        self._xrd_canvas.draw_idle()

        self._xrd_mode = "idle"
        self._xrd_action_bar.setVisible(False)
        self._xrd_btn_pick.setEnabled(True)
        self._xrd_npeaks_spin.setEnabled(True)

        offset, offset_err = popt[0], perr[0]
        peaks = []
        for i in range(n_peaks):
            a, x0, gamma, eta = popt[1 + i * 4: 1 + i * 4 + 4]
            ea, ex0, egamma, eeta = perr[1 + i * 4: 1 + i * 4 + 4]
            sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
            peaks.append(dict(a=a, ea=ea, x0=x0, ex0=ex0, gamma=gamma, egamma=egamma,
                               sigma=sigma, eta=eta, eeta=eeta))
        peaks.sort(key=lambda p: p["x0"])

        print(f"\nOffset: {offset:.4f} ± {offset_err:.4f}")
        for i, p in enumerate(peaks):
            print(f"\nPeak {i+1}:")
            print(f"  Amplitude:    {p['a']:.4f} ± {p['ea']:.4f}")
            print(f"  Center (2θ): {p['x0']:.4f} ± {p['ex0']:.4f}")
            print(f"  Gamma:        {p['gamma']:.4f} ± {p['egamma']:.4f}")
            print(f"  Sigma:        {p['sigma']:.4f}")
            print(f"  Eta:          {p['eta']:.4f} ± {p['eeta']:.4f}")
        print(f"\nR² = {r_squared:.5f}")

        summary = "\n".join(f"Peak {i+1}: 2θ = {p['x0']:.5g} ± {p['ex0']:.3g}"
                             for i, p in enumerate(peaks))
        self._xrd_result_lbl.setText(f"R² = {r_squared:.5g}\n{summary}")

        self._xrd_show_results_dialog(peaks, offset, offset_err, r_squared)

    def _xrd_show_results_dialog(self, peaks, offset, offset_err, r_squared):
        dlg = QDialog(self)
        dlg.setWindowTitle("Pseudo-Voigt fit parameters")
        dlg.resize(380, 420)
        dl = QVBoxLayout(dlg)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        cl = QVBoxLayout(content)

        for i, p in enumerate(peaks):
            box = QGroupBox(f"Peak {i + 1}")
            form = QFormLayout(box)
            form.addRow("Amplitude:", QLabel(f"{p['a']:.6g} ± {p['ea']:.4g}"))
            form.addRow("Center (2θ):", QLabel(f"{p['x0']:.6g} ± {p['ex0']:.4g}"))
            form.addRow("Gamma (FWHM):", QLabel(f"{p['gamma']:.6g} ± {p['egamma']:.4g}"))
            form.addRow("Sigma:", QLabel(f"{p['sigma']:.6g}"))
            form.addRow("Eta:", QLabel(f"{p['eta']:.6g} ± {p['eeta']:.4g}"))
            cl.addWidget(box)

        cl.addStretch(1)
        scroll.setWidget(content)
        dl.addWidget(scroll)

        form_bottom = QFormLayout()
        form_bottom.addRow("Offset:", QLabel(f"{offset:.6g} ± {offset_err:.4g}"))
        form_bottom.addRow("R²:", QLabel(f"{r_squared:.6g}"))
        dl.addLayout(form_bottom)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        dl.addWidget(btn_close)
        dlg.exec_()
