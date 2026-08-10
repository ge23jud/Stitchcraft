import os
import sys
import re
import math
import numpy as np
import h5py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QRadioButton, QComboBox, QCheckBox, QLineEdit,
    QFileDialog, QMessageBox, QSizePolicy, QScrollArea, QFrame,
    QApplication, QDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plotting import (
    PGCanvas, MultiLinePlotter, SequentialScheme, PLASMA, TAB10, GradientLegend,
    DraggableSpan, DraggableRect, RecolorableScatter, make_pg_toolbar,
)

from pl import _HC_EV_NM

try:
    import nw_analysis as nwa
    _NWA_AVAILABLE = True
    _NWA_ERROR = ""
except Exception as _nwa_import_exc:
    _NWA_AVAILABLE = False
    _NWA_ERROR = str(_nwa_import_exc)


class AnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Analysis state ────────────────────────────────────────
        self._ana_file = None
        self._ana_nw   = None   # nw_analysis SimpleNamespace
        self._ana_mode = "idle"
        # peak clicking
        self._ana_sc_ref_idx    = 0
        self._ana_sc_clicks_x   = []
        self._ana_sc_click_conn = None
        self._ana_sc_span       = None
        self._ana_sc_span_sel   = [None]
        self._ana_sc_windows    = []
        self._ana_sc_peak_idx   = 0
        # thresholds
        self._ana_thr_queue    = []
        self._ana_thr_idx      = 0
        self._ana_thr_results  = {}
        self._ana_thr_rect_sel = None
        self._ana_thr_sel_pts  = []
        self._ana_thr_coeffs   = None
        self._ana_thr_power    = None
        self._ana_thr_values   = None
        self._ana_thr_title    = ""
        self._ana_thr_scatter  = None
        # inspect fits
        self._ana_insp_peak      = 0
        self._ana_insp_power_idx = 0
        self._ana_insp_span      = None

        # ── Build UI (body of original _build_analysis_tab) ───────
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        if not _NWA_AVAILABLE:
            msg = QLabel(
                f"nw_analysis could not be imported:\n{_NWA_ERROR}"
            )
            msg.setWordWrap(True)
            layout.addWidget(msg)
            return

        # ── Scrollable sidebar ────────────────────────────────────
        scroll = QScrollArea()
        scroll.setFixedWidth(305)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_content = QWidget()
        sl = QVBoxLayout(sidebar_content)
        sl.setContentsMargins(4, 4, 4, 4)
        sl.setSpacing(6)

        # File group
        g_file = QGroupBox("HDF5 file")
        gfl = QVBoxLayout(g_file)
        self._ana_file_label = QLabel("No file loaded")
        self._ana_file_label.setWordWrap(True)
        gfl.addWidget(self._ana_file_label)
        btn_load_ana = QPushButton("Load HDF5…")
        btn_load_ana.clicked.connect(self._ana_load_file)
        gfl.addWidget(btn_load_ana)
        sl.addWidget(g_file)

        btn_settings = QPushButton("⚙  Settings…")
        btn_settings.clicked.connect(self._ana_open_settings)
        sl.addWidget(btn_settings)

        # set_startconditions group
        g_sc = QGroupBox("1  Select peaks  (set_startconditions)")
        scl = QVBoxLayout(g_sc)

        self._ana_btn_sc = QPushButton("1a  Click peaks in plot…")
        self._ana_btn_sc.setEnabled(False)
        self._ana_btn_sc.clicked.connect(self._ana_start_peak_clicking)
        scl.addWidget(self._ana_btn_sc)
        self._ana_sc_lbl = QLabel("")
        self._ana_sc_lbl.setWordWrap(True)
        scl.addWidget(self._ana_sc_lbl)
        sl.addWidget(g_sc)

        # fit_nw group
        g_fit = QGroupBox("2  Fit peaks  (fit_nw)")
        fitl = QVBoxLayout(g_fit)

        self._ana_btn_fit = QPushButton("Fit peaks")
        self._ana_btn_fit.setEnabled(False)
        self._ana_btn_fit.clicked.connect(self._ana_fit)
        fitl.addWidget(self._ana_btn_fit)
        self._ana_fit_lbl = QLabel("")
        self._ana_fit_lbl.setWordWrap(True)
        fitl.addWidget(self._ana_fit_lbl)
        sl.addWidget(g_fit)

        # thresholds group
        g_thr = QGroupBox("3  Find thresholds")
        thrl = QVBoxLayout(g_thr)
        self._ana_btn_thr = QPushButton("Find thresholds…")
        self._ana_btn_thr.setEnabled(False)
        self._ana_btn_thr.clicked.connect(self._ana_thresholds)
        thrl.addWidget(self._ana_btn_thr)
        self._ana_thr_lbl = QLabel("")
        self._ana_thr_lbl.setWordWrap(True)
        thrl.addWidget(self._ana_thr_lbl)
        sl.addWidget(g_thr)

        # Inspect fits group
        g_insp = QGroupBox("Inspect fits")
        insp_l = QVBoxLayout(g_insp)

        r_insp_peak = QHBoxLayout()
        r_insp_peak.addWidget(QLabel("Peak:"))
        self._ana_insp_peak_sel = QComboBox()
        self._ana_insp_peak_sel.currentIndexChanged.connect(
            self._ana_inspect_peak_changed
        )
        r_insp_peak.addWidget(self._ana_insp_peak_sel)
        insp_l.addLayout(r_insp_peak)

        self._ana_btn_insp = QPushButton("Inspect…")
        self._ana_btn_insp.setEnabled(False)
        self._ana_btn_insp.clicked.connect(self._ana_inspect_start)
        insp_l.addWidget(self._ana_btn_insp)
        self._ana_insp_lbl = QLabel("")
        self._ana_insp_lbl.setWordWrap(True)
        insp_l.addWidget(self._ana_insp_lbl)
        sl.addWidget(g_insp)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        btn_plot_ll = QPushButton("Plot L-L curve")
        btn_plot_ll.clicked.connect(self._ana_plot_ll)
        sl.addWidget(btn_plot_ll)

        self._ana_btn_save = QPushButton("Save results to HDF5")
        self._ana_btn_save.setEnabled(False)
        self._ana_btn_save.clicked.connect(self._ana_save)
        sl.addWidget(self._ana_btn_save)

        sl.addStretch(1)
        scroll.setWidget(sidebar_content)
        layout.addWidget(scroll)

        # ── Right: toolbar + canvas + action bar ─────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self._ana_canvas  = PGCanvas(right)
        self._ana_toolbar = make_pg_toolbar(self._ana_canvas, right)
        rl.addWidget(self._ana_toolbar)
        rl.addWidget(self._ana_canvas, stretch=1)

        # Action bar (shown during interactive phases)
        self._ana_action_bar = QFrame()
        self._ana_action_bar.setFrameShape(QFrame.StyledPanel)
        ab = QHBoxLayout(self._ana_action_bar)
        ab.setContentsMargins(8, 4, 8, 4)
        self._ana_action_lbl    = QLabel("")
        self._ana_btn_act_done  = QPushButton("✓  Done selecting peaks")
        self._ana_btn_act_conf  = QPushButton("Confirm window")
        self._ana_btn_act_skip  = QPushButton("Skip")
        self._ana_btn_act_comp  = QPushButton("Compute threshold")
        self._ana_btn_act_next  = QPushButton("Next  ▶")
        self._ana_btn_act_insp_prev  = QPushButton("◀  Prev spectrum")
        self._ana_btn_act_insp_next  = QPushButton("Next spectrum  ▶")
        self._ana_btn_act_insp_close = QPushButton("✓  Close inspect")
        self._ana_btn_act_done.clicked.connect(self._ana_done_clicking_peaks)
        self._ana_btn_act_conf.clicked.connect(self._ana_confirm_window)
        self._ana_btn_act_skip.clicked.connect(self._ana_skip_window)
        self._ana_btn_act_comp.clicked.connect(self._ana_compute_threshold)
        self._ana_btn_act_next.clicked.connect(self._ana_next_threshold)
        self._ana_btn_act_insp_prev.clicked.connect(self._ana_inspect_prev)
        self._ana_btn_act_insp_next.clicked.connect(self._ana_inspect_next)
        self._ana_btn_act_insp_close.clicked.connect(self._ana_inspect_close)
        for w in (self._ana_action_lbl, self._ana_btn_act_done,
                  self._ana_btn_act_conf, self._ana_btn_act_skip,
                  self._ana_btn_act_comp, self._ana_btn_act_next,
                  self._ana_btn_act_insp_prev, self._ana_btn_act_insp_next,
                  self._ana_btn_act_insp_close):
            ab.addWidget(w)
        self._ana_action_bar.setVisible(False)
        rl.addWidget(self._ana_action_bar)

        layout.addWidget(right, stretch=1)

        self._ana_build_settings_dialog()

    # ── Analysis: settings dialog ─────────────────────────────────

    def _ana_build_settings_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Analysis Settings")
        dl = QVBoxLayout(dlg)

        # Select peaks settings
        g_sc = QGroupBox("Select peaks")
        scl = QVBoxLayout(g_sc)

        r_win = QHBoxLayout()
        r_win.addWidget(QLabel("Window width:"))
        self._ana_window_width = QLineEdit("0.05")
        r_win.addWidget(self._ana_window_width)
        scl.addLayout(r_win)

        r_xu = QHBoxLayout()
        r_xu.addWidget(QLabel("X unit:"))
        self._ana_rb_ev = QRadioButton("eV")
        self._ana_rb_nm = QRadioButton("nm")
        self._ana_rb_ev.setChecked(True)
        r_xu.addWidget(self._ana_rb_ev)
        r_xu.addWidget(self._ana_rb_nm)
        scl.addLayout(r_xu)

        r_sel = QHBoxLayout()
        r_sel.addWidget(QLabel("Spectrum:"))
        self._ana_spectrum_sel = QComboBox()
        self._ana_spectrum_sel.addItems(["last", "maxpeak", "maxsum"])
        r_sel.addWidget(self._ana_spectrum_sel)
        scl.addLayout(r_sel)

        r_ys = QHBoxLayout()
        r_ys.addWidget(QLabel("Y scale:"))
        self._ana_rb_log = QRadioButton("log")
        self._ana_rb_lin = QRadioButton("linear")
        self._ana_rb_log.setChecked(True)
        r_ys.addWidget(self._ana_rb_log)
        r_ys.addWidget(self._ana_rb_lin)
        scl.addLayout(r_ys)

        dl.addWidget(g_sc)

        # fit_nw settings
        g_fit = QGroupBox("Fit peaks")
        fitl = QVBoxLayout(g_fit)

        r_ff = QHBoxLayout()
        r_ff.addWidget(QLabel("Function:"))
        self._ana_fitfunc = QComboBox()
        self._ana_fitfunc.addItems(
            ["gauss1", "gauss2", "gauss3", "gauss4", "lorentz1", "lorentz2"]
        )
        r_ff.addWidget(self._ana_fitfunc)
        fitl.addLayout(r_ff)

        r_bg = QHBoxLayout()
        r_bg.addWidget(QLabel("Background:"))
        self._ana_fitbg = QComboBox()
        self._ana_fitbg.addItems(["linear", "none", "constant", "raw"])
        r_bg.addWidget(self._ana_fitbg)
        fitl.addLayout(r_bg)

        r_fw = QHBoxLayout()
        r_fw.addWidget(QLabel("Fix window below spectrum index:"))
        self._ana_fitfix_idx = QLineEdit("")
        self._ana_fitfix_idx.setPlaceholderText("blank = full tracking")
        self._ana_fitfix_idx.setToolTip(
            "0-based spectrum index, ordered by ascending power "
            "(0 = lowest power). Spectra with index below this value keep "
            "a fit window frozen at whatever position it had reached at "
            "this index, instead of re-centering on the tracked peak.\n"
            "Leave blank to track the peak across the full power series "
            "(original behavior)."
        )
        r_fw.addWidget(self._ana_fitfix_idx)
        fitl.addLayout(r_fw)

        dl.addWidget(g_fit)

        # thresholds settings
        g_thr = QGroupBox("Find thresholds — mode")
        thrl = QVBoxLayout(g_thr)
        self._ana_thr_area  = QCheckBox("Fit area")
        self._ana_thr_int   = QCheckBox("Integral")
        self._ana_thr_max   = QCheckBox("Maximum")
        self._ana_thr_total = QCheckBox("Total area")
        self._ana_thr_area.setChecked(True)
        for cb in (self._ana_thr_area, self._ana_thr_int,
                   self._ana_thr_max, self._ana_thr_total):
            thrl.addWidget(cb)

        dl.addWidget(g_thr)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        dl.addWidget(btn_close)

        self._ana_settings_dialog = dlg

    def _ana_open_settings(self):
        self._ana_settings_dialog.exec_()

    # ── Analysis: file loading ────────────────────────────────────

    def _ana_load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load HDF5 spectrum", "",
            "HDF5 files (*.h5);;All files (*.*)"
        )
        if not path:
            return
        try:
            with h5py.File(path, "r") as f:
                if "Energy" in f:
                    wl = _HC_EV_NM / f["Energy"][:]     # eV → nm for internal use
                elif "Wavelength" in f:
                    wl = f["Wavelength"][:]
                else:
                    wl = f["wavelength_nm"][:]
                # SpectraDiff = dark-subtracted; Spectra_raw = no dark sub
                spec_diff = (f["SpectraDiff"]        if "SpectraDiff"        in f
                             else f["counts"])        [:]
                spec_raw  = (f["Spectra_raw"]        if "Spectra_raw"        in f
                             else spec_diff)          [:]  # fallback: same
                if "Power" in f:
                    pwr_ana_ds = f["Power"]
                else:
                    uncal_ds = (f["Power_uncalibrated"] if "Power_uncalibrated" in f
                                else f.get("powers_W"))
                    if uncal_ds is None:
                        QMessageBox.critical(
                            self, "Load error",
                            f"No power dataset found in:\n{path}"
                        )
                        return
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("No calibrated power")
                    msg.setText(
                        "This file has no calibrated 'Power' dataset "
                        "(transmission-corrected power at the sample).\n\n"
                        "Use the uncalibrated power instead, or cancel loading?"
                    )
                    use_btn    = msg.addButton("Use uncalibrated power", QMessageBox.AcceptRole)
                    cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
                    msg.setDefaultButton(cancel_btn)
                    msg.exec_()
                    if msg.clickedButton() is not use_btn:
                        return
                    pwr_ana_ds = uncal_ds
                powers_W   = pwr_ana_ds[:].astype(float)
                _pwr_units = pwr_ana_ds.attrs.get("units", "W")
                spot_diam_um = float(f.attrs["spot_diameter_um"]) \
                    if "spot_diameter_um" in f.attrs else None
                rep_rate_mhz = float(f.attrs["rep_rate_mhz"]) \
                    if "rep_rate_mhz" in f.attrs else None
        except Exception as exc:
            QMessageBox.critical(self, "Load error",
                                 f"Could not read:\n{path}\n\n{exc}")
            return

        nw = nwa._new_nanowire()
        nw.name            = os.path.splitext(os.path.basename(path))[0]
        nw.wavelength      = wl.copy()
        nw.wavelength_unit = "nm"
        nw.spectra_raw     = spec_raw.copy()
        nw.spectra_diff    = spec_diff.copy()
        powers_mW          = powers_W if _pwr_units == "mW" else powers_W * 1e3
        nw.power           = powers_mW
        nw.specsum         = []
        if spot_diam_um is not None:
            nw.spot_radius_short = spot_diam_um / 2.0
        if rep_rate_mhz is not None:
            nw.rep_rate = rep_rate_mhz * 1e6

        self._ana_file = path
        self._ana_nw   = nw
        self._ana_mode = "idle"
        self._ana_thr_results = {}
        self._ana_action_bar.setVisible(False)
        self._ana_sc_cleanup()

        n_wl, n_p = spec_diff.shape
        label = os.path.basename(path)
        self._ana_file_label.setText(
            f"{label}\n{n_wl} px, {n_p} powers\n"
            f"{wl.min():.1f}–{wl.max():.1f} nm\n"
            f"{powers_mW.min():.4g}–{powers_mW.max():.4g} mW"
        )
        self._ana_btn_sc.setEnabled(True)
        self._ana_btn_fit.setEnabled(False)
        self._ana_btn_thr.setEnabled(False)
        self._ana_btn_save.setEnabled(True)
        self._ana_btn_insp.setEnabled(False)
        self._ana_insp_peak_sel.clear()
        self._ana_sc_lbl.setText("")
        self._ana_fit_lbl.setText("")
        self._ana_thr_lbl.setText("")
        self._ana_insp_lbl.setText("")
        self._ana_plot_spectrum()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Analysis: loaded {label} ({n_wl} px, {n_p} power steps)"
            )

    # ── Analysis: spectrum overview ──────────────────────────────

    def _ana_x_of_wl(self, wl):
        """Convert nm → display units (eV or nm) for the analysis tab."""
        return _HC_EV_NM / wl if self._ana_rb_ev.isChecked() else wl

    def _ana_wl_of_x(self, x_val):
        """Convert display unit → nm for the analysis tab."""
        if self._ana_rb_ev.isChecked():
            return _HC_EV_NM / x_val
        return x_val

    def _ana_display_spectra(self, nw):
        """(n_wl, n_powers) spectra to plot for the human — dark-subtracted
        if available, falling back to raw counts otherwise. Numerics (fit_nw,
        _ana_integrate) pick their own spectra independently of this."""
        return nw.spectra_diff if nw.spectra_diff is not None else nw.spectra_raw

    def _ana_plot_spectrum(self):
        nw = self._ana_nw
        if nw is None:
            return
        ax = self._ana_canvas.reset_axes()
        wl     = nw.wavelength
        spectra = self._ana_display_spectra(nw)
        n_p    = spectra.shape[1]
        x   = self._ana_x_of_wl(wl)
        s   = np.argsort(x)
        vmin, vmax = nw.power[0], nw.power[-1]
        mlp = MultiLinePlotter(ax, SequentialScheme(vmin=vmin, vmax=vmax, cmap=PLASMA))
        for p in range(n_p):
            item = mlp.plot(x[s], spectra[s, p], value=nw.power[p], width=0.7)
            item.setOpacity(0.8)
        self._ana_canvas.add_colorbar_legend(
            GradientLegend(cmap=PLASMA, vmin=vmin, vmax=vmax, label="Power (mW)")
        )
        x_label = "Energy (eV)" if self._ana_rb_ev.isChecked() else "Wavelength (nm)"
        ax.setLabel("bottom", x_label)
        ax.setLabel("left", "Counts")
        ax.setTitle(nw.name)
        ax.showGrid(x=True, y=True, alpha=0.3)
        self._ana_canvas.draw_idle()

    # ── Analysis: peak clicking (phase 1 of set_startconditions) ─

    def _ana_start_peak_clicking(self):
        nw = self._ana_nw
        if nw is None:
            return
        self._ana_sc_cleanup()

        spectra = self._ana_display_spectra(nw)

        # Choose reference spectrum
        sel = self._ana_spectrum_sel.currentText()
        if sel == "last":
            ref = spectra.shape[1] - 1
        elif sel == "maxpeak":
            ref = int(np.argmax(np.max(spectra, axis=0)))
        else:
            ref = int(np.argmax(np.sum(spectra, axis=0)))
        self._ana_sc_ref_idx = ref

        # Draw reference spectrum
        ax = self._ana_canvas.reset_axes()
        ax.setLogMode(y=self._ana_rb_log.isChecked())
        wl   = nw.wavelength
        x    = self._ana_x_of_wl(wl)
        s    = np.argsort(x)
        spec = spectra[:, ref]
        ax.plot(x[s], spec[s], pen=pg.mkPen("steelblue", width=1.0))
        x_label = "Energy (eV)" if self._ana_rb_ev.isChecked() else "Wavelength (nm)"
        ax.setLabel("bottom", x_label)
        ax.setLabel("left", "Counts")
        ax.setTitle(f"{nw.name} — left-click peaks, then ✓ Done")
        ax.showGrid(x=True, y=True, alpha=0.3)
        self._ana_canvas.draw_idle()

        # Connect click handler
        self._ana_sc_clicks_x = []  # x-positions (data units)
        self._ana_sc_click_conn = self._ana_canvas.sigDataClicked.connect(
            self._ana_on_peak_click
        )
        self._ana_mode = "sc_click_peaks"

        # Show action bar — only Done button
        self._ana_action_lbl.setText("0 peaks selected")
        self._ana_btn_act_done.setVisible(True)
        self._ana_btn_act_conf.setVisible(False)
        self._ana_btn_act_skip.setVisible(False)
        self._ana_btn_act_comp.setVisible(False)
        self._ana_btn_act_next.setVisible(False)
        self._ana_btn_act_insp_prev.setVisible(False)
        self._ana_btn_act_insp_next.setVisible(False)
        self._ana_btn_act_insp_close.setVisible(False)
        self._ana_action_bar.setVisible(True)
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                "Left-click peak positions in the plot, then click ✓ Done."
            )

    def _ana_on_peak_click(self, x, y, button):
        if button != Qt.LeftButton:
            return
        if self._ana_mode != "sc_click_peaks":
            return
        self._ana_sc_clicks_x.append(x)
        n = len(self._ana_sc_clicks_x)
        self._ana_action_lbl.setText(f"{n} peak(s) selected")
        # Mark the click on the canvas
        line = pg.InfiniteLine(pos=x, angle=90,
                                pen=pg.mkPen("red", width=1.0, style=Qt.DashLine))
        self._ana_canvas.plot_item.addItem(line)
        self._ana_canvas.draw_idle()

    def _ana_sc_cleanup(self):
        """Disconnect any active click handler or DraggableSpan."""
        if self._ana_sc_click_conn is not None:
            try:
                self._ana_canvas.sigDataClicked.disconnect(self._ana_on_peak_click)
            except (TypeError, RuntimeError):
                pass
            self._ana_sc_click_conn = None
        if self._ana_sc_span is not None:
            self._ana_sc_span.deactivate()
            self._ana_sc_span = None

    def _ana_done_clicking_peaks(self):
        if self._ana_mode != "sc_click_peaks":
            return
        self._ana_sc_cleanup()

        if not self._ana_sc_clicks_x:
            self._ana_sc_lbl.setText("No peaks clicked — try again.")
            self._ana_action_bar.setVisible(False)
            self._ana_mode = "idle"
            return

        # Sort clicks in increasing x
        self._ana_sc_clicks_x = sorted(self._ana_sc_clicks_x)
        self._ana_sc_windows  = [None] * len(self._ana_sc_clicks_x)
        self._ana_sc_peak_idx = 0
        self._ana_mode = "sc_window"
        self._ana_show_window_for_peak(0)

    # ── Analysis: fit-window selection (phase 2) ─────────────────

    def _ana_show_window_for_peak(self, j):
        nw   = self._ana_nw
        wl   = nw.wavelength
        x    = self._ana_x_of_wl(wl)
        s    = np.argsort(x)
        ref  = self._ana_sc_ref_idx
        spec = self._ana_display_spectra(nw)[:, ref]

        x_peak = self._ana_sc_clicks_x[j]
        try:
            window_half = float(self._ana_window_width.text()) / 2
        except ValueError:
            window_half = 0.025

        # Zoom window for display
        x_lo = x_peak - window_half * 3
        x_hi = x_peak + window_half * 3

        ax = self._ana_canvas.reset_axes()
        ax.setLogMode(y=self._ana_rb_log.isChecked())
        mask = (x[s] >= x_lo) & (x[s] <= x_hi)
        ax.plot(x[s][mask], spec[s][mask], pen=pg.mkPen("steelblue", width=1.0),
                symbol="x", symbolSize=6, symbolPen="steelblue", symbolBrush="steelblue")
        x_label = "Energy (eV)" if self._ana_rb_ev.isChecked() else "Wavelength (nm)"
        ax.setLabel("bottom", x_label)
        ax.setLabel("left", "Counts")
        n_tot = len(self._ana_sc_clicks_x)
        ax.setTitle(
            f"{nw.name} — peak {j+1}/{n_tot}\n"
            "Drag the shaded region's edges to select fit window, then Confirm"
        )
        ax.showGrid(x=True, y=True, alpha=0.3)
        ax.setXRange(x_lo, x_hi, padding=0)
        self._ana_canvas.draw_idle()

        # DraggableSpan (SpanSelector replacement)
        self._ana_sc_span_sel = [None]  # stores (xmin, xmax)
        self._ana_sc_span = DraggableSpan(ax, color=(0, 150, 0, 60), movable=True)
        self._ana_sc_span.activate(
            initial_range=(x_peak - window_half, x_peak + window_half),
            bounds=(x_lo, x_hi),
        )
        self._ana_sc_span.sigRegionSelected.connect(
            lambda xmin, xmax: self._ana_sc_span_sel.__setitem__(0, (xmin, xmax))
        )

        n_done = j
        self._ana_action_lbl.setText(
            f"Peak {j+1} of {n_tot}  ({n_done} confirmed)"
        )
        self._ana_btn_act_done.setVisible(False)
        self._ana_btn_act_conf.setVisible(True)
        self._ana_btn_act_skip.setVisible(True)
        self._ana_btn_act_comp.setVisible(False)
        self._ana_btn_act_next.setVisible(False)
        self._ana_btn_act_insp_prev.setVisible(False)
        self._ana_btn_act_insp_next.setVisible(False)
        self._ana_btn_act_insp_close.setVisible(False)
        self._ana_action_bar.setVisible(True)
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Drag to select fit window for peak {j+1}, then Confirm."
            )

    def _ana_confirm_window(self):
        if self._ana_mode != "sc_window":
            return
        j = self._ana_sc_peak_idx
        sel = self._ana_sc_span_sel[0] if hasattr(self, "_ana_sc_span_sel") else None
        if sel is not None and abs(sel[1] - sel[0]) > 1e-12:
            self._ana_sc_windows[j] = sel
        else:
            # Fallback: use window_width centred on the click
            try:
                hw = float(self._ana_window_width.text()) / 2
            except ValueError:
                hw = 0.025
            cx = self._ana_sc_clicks_x[j]
            self._ana_sc_windows[j] = (cx - hw, cx + hw)
        self._ana_sc_cleanup()
        self._ana_advance_window()

    def _ana_skip_window(self):
        if self._ana_mode != "sc_window":
            return
        j = self._ana_sc_peak_idx
        try:
            hw = float(self._ana_window_width.text()) / 2
        except ValueError:
            hw = 0.025
        cx = self._ana_sc_clicks_x[j]
        self._ana_sc_windows[j] = (cx - hw, cx + hw)
        self._ana_sc_cleanup()
        self._ana_advance_window()

    def _ana_advance_window(self):
        j = self._ana_sc_peak_idx + 1
        self._ana_sc_peak_idx = j
        n_tot = len(self._ana_sc_clicks_x)
        if j < n_tot:
            self._ana_show_window_for_peak(j)
        else:
            self._ana_finish_startconditions()

    def _ana_finish_startconditions(self):
        """Store start_conditions in the nw object and update UI."""
        self._ana_sc_cleanup()
        self._ana_action_bar.setVisible(False)
        self._ana_mode = "idle"

        nw   = self._ana_nw
        wl   = nw.wavelength
        x    = self._ana_x_of_wl(wl)
        n    = len(self._ana_sc_clicks_x)
        ref  = self._ana_sc_ref_idx

        sc = np.full((3, n), np.nan)
        for j in range(n):
            cx = self._ana_sc_clicks_x[j]
            peak_idx = int(np.argmin(np.abs(x - cx)))
            xmin, xmax = self._ana_sc_windows[j]
            i1 = int(np.argmin(np.abs(x - xmin)))
            i2 = int(np.argmin(np.abs(x - xmax)))
            fitwindow = max(abs(i2 - i1), 4)
            sc[:, j] = [peak_idx, fitwindow, ref]

        nw.start_conditions = sc
        nw.n_sel_peaks      = n
        # Store the (xmin, xmax) bounds in display units so _ana_integrate
        # can use them instead of the manual center/width fields.
        nw.fit_windows = list(self._ana_sc_windows)

        self._ana_sc_lbl.setText(f"{n} peak(s) selected")
        self._ana_btn_fit.setEnabled(True)
        self._ana_btn_thr.setEnabled(False)
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Analysis: {n} peak(s) configured. Ready to fit."
            )
        self._ana_draw_peaks()

    def _ana_draw_peaks(self):
        nw = self._ana_nw
        if nw is None or nw.start_conditions is None:
            return
        ax = self._ana_canvas.reset_axes()
        wl  = nw.wavelength
        x   = self._ana_x_of_wl(wl)
        s   = np.argsort(x)
        ref = int(nw.start_conditions[2, 0])
        ax.plot(x[s], self._ana_display_spectra(nw)[s, ref],
                pen=pg.mkPen("steelblue", width=1.0))
        for k in range(nw.n_sel_peaks):
            pi = int(nw.start_conditions[0, k])
            line = pg.InfiniteLine(
                pos=x[pi], angle=90, pen=pg.mkPen("red", width=1.0, style=Qt.DashLine),
                label=f"peak {k+1}: {x[pi]:.4g}",
                labelOpts={"position": max(0.05, 0.9 - 0.08 * k), "color": (200, 60, 60)},
            )
            ax.addItem(line)
        x_label = "Energy (eV)" if self._ana_rb_ev.isChecked() else "Wavelength (nm)"
        ax.setLabel("bottom", x_label)
        ax.setLabel("left", "Counts")
        ax.setTitle(f"{nw.name} — {nw.n_sel_peaks} selected peak(s)")
        ax.showGrid(x=True, y=True, alpha=0.3)
        self._ana_canvas.draw_idle()

    # ── Analysis: fit_nw ─────────────────────────────────────────

    def _ana_fit(self):
        nw = self._ana_nw
        if nw is None or nw.start_conditions is None or nw.n_sel_peaks == 0:
            QMessageBox.warning(self, "No peaks",
                                "Run set_startconditions first.")
            return
        fitfunction = self._ana_fitfunc.currentText()
        subtract_bg = self._ana_fitbg.currentText()

        fix_text = self._ana_fitfix_idx.text().strip()
        fixed_window_below = None
        if fix_text:
            try:
                fixed_window_below = int(fix_text)
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid input",
                    "'Fix window below spectrum index' must be an integer "
                    "(or left blank for full tracking)."
                )
                return
            if fixed_window_below < 0:
                QMessageBox.warning(
                    self, "Invalid input",
                    "'Fix window below spectrum index' must be 0 or greater."
                )
                return

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("Fitting… please wait.")
        QApplication.processEvents()
        try:
            nwa.fit_nw(nw,
                       subtract_fit_background=subtract_bg,
                       fitfunction=fitfunction,
                       show_progress=False,
                       fixed_window_below_index=fixed_window_below)
        except Exception as exc:
            QMessageBox.critical(self, "fit_nw error", str(exc))
            return

        n_ok = sum(
            1 for j in range(nw.n_sel_peaks)
            for i in range(len(nw.power))
            if nw.fits[j][i] is not None
        )
        n_total = nw.n_sel_peaks * len(nw.power)
        self._ana_fit_lbl.setText(
            f"{fitfunction} fit done\n{n_ok}/{n_total} converged"
        )
        self._ana_btn_thr.setEnabled(True)
        self._ana_btn_insp.setEnabled(True)
        self._ana_insp_peak_sel.blockSignals(True)
        self._ana_insp_peak_sel.clear()
        self._ana_insp_peak_sel.addItems(
            [f"Peak {k+1}" for k in range(nw.n_sel_peaks)]
        )
        self._ana_insp_peak_sel.blockSignals(False)
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Analysis: fit complete ({n_ok}/{n_total} converged). Integrating…"
            )
        QApplication.processEvents()
        self._ana_integrate()

    # ── Analysis: integrate_spectra ──────────────────────────────

    def _ana_integrate(self):
        nw = self._ana_nw
        if nw is None:
            return
        spectype = "no_background"
        method   = "trapz"

        # Reset specsum so repeated clicks don't accumulate entries.
        nw.specsum = []

        if nw.start_conditions is not None:
            # ── Per-peak tracking path (mirrors MATLAB fit_nw) ───────
            # Window width is fixed per peak; center follows the peak as
            # it shifts spectrally across power steps.
            n_peaks  = nw.n_sel_peaks
            n_wl     = nw.wavelength.shape[0]
            n_powers = len(nw.power)
            wl       = nw.wavelength  # nm, shape (n_wl,)

            spectra = nw.spectra_raw if spectype == 'raw' else nw.spectra_diff
            if spectra is None:
                return

            peak_integral = np.full((n_peaks, n_powers), np.nan)

            for j in range(n_peaks):
                center_idx = int(round(float(nw.start_conditions[0, j])))
                fitwindow  = int(round(float(nw.start_conditions[1, j])))
                fitwindow  = max(2 * (fitwindow // 2), 2)   # keep even, min 2
                ref_idx    = int(round(float(nw.start_conditions[2, j])))
                ref_idx    = max(0, min(ref_idx, n_powers - 1))

                peakindex = np.zeros(n_powers, dtype=int)
                peakindex[ref_idx] = center_idx
                if ref_idx + 1 < n_powers:
                    peakindex[ref_idx + 1] = center_idx

                backward = list(range(ref_idx, -1, -1))
                forward  = list(range(ref_idx + 1, n_powers))

                for i in backward + forward:
                    i1 = max(0, peakindex[i] - fitwindow // 2)
                    i2 = min(n_wl, peakindex[i] + fitwindow // 2)
                    if i2 <= i1:
                        continue

                    wl_seg = wl[i1:i2]
                    y_seg  = spectra[i1:i2, i].astype(float)

                    if method == 'sum' or len(wl_seg) < 2:
                        val = float(np.sum(y_seg))
                    elif method == 'trapz':
                        val = float(np.trapz(y_seg, wl_seg))
                    else:
                        val = float(np.sum(y_seg * np.gradient(wl_seg)))
                    peak_integral[j, i] = np.nan if val == 0 else val

                    next_center = i1 + int(np.argmax(y_seg))
                    if 0 < i <= ref_idx:
                        peakindex[i - 1] = next_center
                    elif i > ref_idx and i + 1 < n_powers:
                        peakindex[i + 1] = next_center

                i1_r = max(0, center_idx - fitwindow // 2)
                i2_r = min(n_wl - 1, center_idx + fitwindow // 2)
                nw.specsum.append(dict(
                    values=peak_integral[j, :].copy(),
                    center=float(wl[center_idx]) if 0 <= center_idx < n_wl else 0.0,
                    width=float(abs(wl[i2_r] - wl[i1_r])) if i2_r > i1_r else 0.0,
                    spectrumtype=spectype,
                ))

            nw.peak_integral = peak_integral
            n_valid = int(np.sum(~np.isnan(peak_integral)))
            msg = (f"Analysis: fit + integration done "
                   f"({n_peaks} peak(s), {n_valid}/{n_peaks * n_powers} valid).")

        else:
            msg = "Analysis: integration skipped (no start conditions)."

        self._ana_btn_thr.setEnabled(True)
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(msg)
        self._ana_plot_ll()

    # ── Analysis: thresholds (native Qt version) ─────────────────

    def _ana_thresholds(self):
        nw = self._ana_nw
        if nw is None:
            return
        mode = []
        if self._ana_thr_area.isChecked():  mode.append("area")
        if self._ana_thr_int.isChecked():   mode.append("integral")
        if self._ana_thr_max.isChecked():   mode.append("max")
        if self._ana_thr_total.isChecked(): mode.append("total")
        if not mode:
            QMessageBox.warning(self, "No mode",
                                "Select at least one threshold mode.")
            return

        n_pk = nw.n_sel_peaks
        n_pw = len(nw.power)

        def _cell_to_mat(cell):
            mat = np.full((n_pk, n_pw), np.nan)
            for j in range(n_pk):
                for i in range(n_pw):
                    v = cell[j][i] if cell is not None else None
                    if v is not None and np.any(~np.isnan(v)):
                        mat[j, i] = float(np.nanmax(v))
            return mat

        # Build queue: list of (title, power_arr, values_arr)
        queue = []
        if "area" in mode and nw.peak_area is not None:
            mat = _cell_to_mat(nw.peak_area)
            for j in range(n_pk):
                vals = mat[j, :]
                if np.any(~np.isnan(vals)):
                    queue.append((f"peak {j+1} fit-area", nw.power, vals))
        if "integral" in mode and nw.peak_integral is not None:
            for j in range(n_pk):
                vals = nw.peak_integral[j, :]
                if np.any(~np.isnan(vals)):
                    queue.append((f"peak {j+1} integral", nw.power, vals))
        if "max" in mode and nw.peak_maximum is not None:
            for j in range(n_pk):
                vals = nw.peak_maximum[j, :]
                if np.any(~np.isnan(vals)):
                    queue.append((f"peak {j+1} maximum", nw.power, vals))
        if "total" in mode and nw.total_peak_area is not None:
            vals = nw.total_peak_area
            if np.any(~np.isnan(vals)):
                queue.append(("total peak area", nw.power, vals))

        if not queue:
            QMessageBox.information(self, "Nothing to fit",
                                    "No fit results available yet.\n"
                                    "Run Fit peaks or Integrate first.")
            return

        self._ana_thr_queue   = queue
        self._ana_thr_idx     = 0
        self._ana_thr_results = {}      # {title: (thr, thr_err, slope, slope_err)}
        self._ana_thr_sel_pts = []      # selected indices for current item
        self._ana_thr_coeffs  = None    # np.polyfit result for current item
        self._ana_mode = "thr_select"
        self._ana_show_thr_item(0)

    def _ana_show_thr_item(self, idx):
        title, power, values = self._ana_thr_queue[idx]
        n_tot = len(self._ana_thr_queue)

        # Clean up any previous rect selector
        if self._ana_thr_rect_sel is not None:
            self._ana_thr_rect_sel.deactivate()
            self._ana_thr_rect_sel = None

        ax = self._ana_canvas.reset_axes()
        valid = ~np.isnan(values)
        self._ana_thr_scatter = RecolorableScatter(ax, default_color="steelblue", size=8)
        self._ana_thr_scatter.set_data(power[valid], values[valid])
        ax.setLabel("bottom", "Power (mW)")
        ax.setLabel("left", "Intensity (arb.u.)")
        ax.setTitle(
            f"{title}  ({idx+1}/{n_tot})\n"
            "Drag the rectangle's handles to select the ASE region, then Compute threshold"
        )
        ax.showGrid(x=True, y=True, alpha=0.3)
        self._ana_canvas.draw_idle()

        # Store current item's data for use in compute/next
        self._ana_thr_power  = power[valid]
        self._ana_thr_values = values[valid]
        self._ana_thr_sel_pts = list(range(len(self._ana_thr_power)))  # default=all
        self._ana_thr_coeffs  = None
        self._ana_thr_title   = title

        p_arr, v_arr = self._ana_thr_power, self._ana_thr_values
        if len(p_arr):
            default_rect = (float(np.min(p_arr)), float(np.min(v_arr)),
                            float(np.max(p_arr)), float(np.max(v_arr)))
        else:
            default_rect = (0.0, 0.0, 1.0, 1.0)
        self._ana_thr_rect_sel = DraggableRect(ax)
        self._ana_thr_rect_sel.activate(default_rect)
        self._ana_thr_rect_sel.sigRectSelected.connect(self._ana_on_thr_rect)

        self._ana_action_lbl.setText(f"Threshold {idx+1}/{n_tot}: {title}")
        self._ana_btn_act_done.setVisible(False)
        self._ana_btn_act_conf.setVisible(False)
        self._ana_btn_act_skip.setVisible(False)
        self._ana_btn_act_comp.setVisible(True)
        self._ana_btn_act_next.setVisible(True)
        self._ana_btn_act_next.setText(
            "Skip" if idx + 1 < len(self._ana_thr_queue) else "Finish"
        )
        self._ana_btn_act_insp_prev.setVisible(False)
        self._ana_btn_act_insp_next.setVisible(False)
        self._ana_btn_act_insp_close.setVisible(False)
        self._ana_action_bar.setVisible(True)
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Drag to select ASE region for {title}, then Compute threshold."
            )

    def _ana_on_thr_rect(self, x0, y0, x1, y1):
        p, v = self._ana_thr_power, self._ana_thr_values
        inside = np.where(
            (p >= x0) & (p <= x1) & (v >= y0) & (v <= y1)
        )[0]
        self._ana_thr_sel_pts = inside.tolist()
        # Highlight selected points
        colors = ["steelblue"] * len(p)
        for i in self._ana_thr_sel_pts:
            colors[i] = "tomato"
        self._ana_thr_scatter.set_point_colors(colors)
        self._ana_canvas.draw_idle()

    def _ana_compute_threshold(self):
        if self._ana_mode != "thr_select":
            return
        sel = self._ana_thr_sel_pts
        p_all, v_all = self._ana_thr_power, self._ana_thr_values
        if len(sel) < 2:
            sel = list(range(len(p_all)))
        p_sel = p_all[sel]
        v_sel = v_all[sel]
        valid = ~(np.isnan(p_sel) | np.isnan(v_sel))
        p_sel, v_sel = p_sel[valid], v_sel[valid]
        if len(p_sel) < 2:
            parent = self.parent()
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage("Not enough points for fit.")
            return

        if len(p_sel) >= 3:
            coeffs, pcov_fit = np.polyfit(p_sel, v_sel, 1, cov=True)
            a, b = float(coeffs[0]), float(coeffs[1])
            sigma_a = float(np.sqrt(pcov_fit[0, 0]))
            sigma_b = float(np.sqrt(pcov_fit[1, 1]))
            if a != 0:
                # Error propagation: T = -b/a, dT/da = b/a², dT/db = -1/a
                thr_err   = 2.0 * float(np.sqrt((sigma_b / a)**2 + (sigma_a * b / a**2)**2))
            else:
                thr_err   = np.nan
            slope_err = 2.0 * sigma_a
        else:
            coeffs    = np.polyfit(p_sel, v_sel, 1)
            a, b      = float(coeffs[0]), float(coeffs[1])
            thr_err   = np.nan
            slope_err = np.nan

        thresh = -b / a if a != 0 else np.nan
        self._ana_thr_coeffs = coeffs
        self._ana_thr_results[self._ana_thr_title] = {
            "threshold":     thresh,
            "threshold_err": thr_err,
            "slope":         a,
            "slope_err":     slope_err,
            "intercept":     b,
            "sel_power":     p_sel.tolist(),
            "sel_values":    v_sel.tolist(),
        }

        # Redraw with fit line
        ax = self._ana_canvas.plot_item
        ax.addLegend(labelTextSize="8pt")
        x_fit = np.linspace(0, max(p_all) * 1.1, 200)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), pen=pg.mkPen("r", width=1.5), name="fit")
        if not np.isnan(thresh):
            err_str = f" ± {thr_err:.2g}" if not np.isnan(thr_err) else ""
            thr_line = pg.InfiniteLine(
                pos=thresh, angle=90, pen=pg.mkPen("black", width=1.2, style=Qt.DashLine),
                label=f"Threshold = {thresh:.4g}{err_str} mW", labelOpts={"position": 0.95},
            )
            ax.addItem(thr_line)
        ax.setTitle(
            f"{self._ana_thr_title}\n"
            f"Threshold = {thresh:.4g} mW  |  slope = {a:.3g}"
        )
        self._ana_canvas.draw_idle()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Threshold ({self._ana_thr_title}): {thresh:.4g} mW"
            )

    def _ana_next_threshold(self):
        if self._ana_mode != "thr_select":
            return
        if self._ana_thr_rect_sel is not None:
            self._ana_thr_rect_sel.deactivate()
            self._ana_thr_rect_sel = None

        idx = self._ana_thr_idx + 1
        self._ana_thr_idx = idx
        if idx < len(self._ana_thr_queue):
            self._ana_show_thr_item(idx)
        else:
            self._ana_finish_thresholds()

    def _ana_finish_thresholds(self):
        if self._ana_thr_rect_sel is not None:
            self._ana_thr_rect_sel.deactivate()
            self._ana_thr_rect_sel = None
        self._ana_action_bar.setVisible(False)
        self._ana_mode = "idle"

        # Store results back into nw
        nw = self._ana_nw
        results = self._ana_thr_results
        nw.thr_results = results   # persist full data for Save
        parts = []
        for title, data in results.items():
            thr   = data["threshold"]
            slope = data["slope"]
            if not np.isnan(thr):
                parts.append(f"{title}: {thr:.4g} mW")
                # Try to map into nw fields by title prefix
                if "integral" in title:
                    nw.threshold_integral = thr
                    nw.slope_integral     = slope
                elif "maximum" in title or "max" in title:
                    nw.threshold_max = thr
                    nw.slope_max     = slope
                elif "total" in title:
                    nw.threshold = np.array([thr])
                    nw.slope     = np.array([slope])
                else:  # fit-area
                    nw.threshold = np.array(
                        [thr] if nw.threshold is None
                        else np.append(np.atleast_1d(nw.threshold), thr)
                    )

        self._ana_thr_lbl.setText("\n".join(parts) if parts else "Done")
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("Analysis: thresholds done.")
        self._ana_plot_ll()

    # ── Analysis: L-L plot ───────────────────────────────────────

    def _ana_plot_ll(self):
        nw = self._ana_nw
        if nw is None:
            return
        ax = self._ana_canvas.reset_axes()
        power   = nw.power
        plotted = False
        color_idx = 0
        ax.addLegend(labelTextSize="7pt")

        if nw.peak_integral is not None:
            for j in range(nw.n_sel_peaks):
                vals  = nw.peak_integral[j, :]
                valid = ~np.isnan(vals)
                if valid.any():
                    color = TAB10[color_idx % 10]
                    ax.plot(power[valid], vals[valid], pen=pg.mkPen(color, width=1.2),
                            symbol="o", symbolSize=6, symbolBrush=color,
                            name=f"peak {j+1} integral")
                    color_idx += 1
                    plotted = True

        if nw.specsum:
            for entry in nw.specsum:
                vals  = entry["values"]
                valid = ~np.isnan(vals)
                if valid.any():
                    c = entry["center"]; w = entry["width"]
                    color = TAB10[color_idx % 10]
                    ax.plot(power[valid], vals[valid],
                            pen=pg.mkPen(color, width=1.2, style=Qt.DashLine),
                            symbol="s", symbolSize=6, symbolBrush=color,
                            name=f"integrate @ {c:.4g} ±{w/2:.3g}")
                    color_idx += 1
                    plotted = True

        def _vline(thr, lbl):
            if thr is not None:
                t = float(np.nanmean(np.atleast_1d(thr)))
                if not np.isnan(t):
                    line = pg.InfiniteLine(
                        pos=t, angle=90, pen=pg.mkPen("red", width=1.2, style=Qt.DotLine),
                        label=lbl, labelOpts={"position": 0.95},
                    )
                    ax.addItem(line)

        _vline(nw.threshold,          "thr (area)")
        _vline(nw.threshold_integral, "thr (int)")
        _vline(nw.threshold_max,      "thr (max)")

        if not plotted:
            vb = ax.getViewBox()
            vb.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
            vb.setMouseEnabled(x=False, y=False)
            text = pg.TextItem("No results yet.\nRun Fit or Integrate first.",
                                color="#888888", anchor=(0.5, 0.5))
            text.setPos(0.5, 0.5)
            vb.addItem(text)
        else:
            ax.setLabel("bottom", "Power (mW)")
            ax.setLabel("left", "Intensity (arb.u.)")
            ax.setTitle(f"{nw.name} — L-L curve")
            ax.showGrid(x=True, y=True, alpha=0.3)

        self._ana_canvas.draw_idle()

    # ── Analysis: inspect fits ────────────────────────────────────

    def _ana_inspect_start(self):
        nw = self._ana_nw
        if nw is None or nw.fits is None:
            return
        self._ana_insp_peak = max(0, self._ana_insp_peak_sel.currentIndex())
        # Start on the reference power step for this peak — it's the one
        # the fit was seeded from, so it's the most likely to have converged.
        ref_idx = 0
        if nw.start_conditions is not None:
            ref_idx = int(round(float(nw.start_conditions[2, self._ana_insp_peak])))
        self._ana_insp_power_idx = max(0, min(ref_idx, len(nw.power) - 1))
        self._ana_mode = "inspect"

        self._ana_btn_act_done.setVisible(False)
        self._ana_btn_act_conf.setVisible(False)
        self._ana_btn_act_skip.setVisible(False)
        self._ana_btn_act_comp.setVisible(False)
        self._ana_btn_act_next.setVisible(False)
        self._ana_btn_act_insp_prev.setVisible(True)
        self._ana_btn_act_insp_next.setVisible(True)
        self._ana_btn_act_insp_close.setVisible(True)
        self._ana_action_bar.setVisible(True)

        self._ana_inspect_draw()

    def _ana_inspect_peak_changed(self, idx):
        if self._ana_mode != "inspect" or idx < 0:
            return
        nw = self._ana_nw
        if nw is None or nw.fits is None:
            return
        self._ana_insp_peak = idx
        ref_idx = 0
        if nw.start_conditions is not None:
            ref_idx = int(round(float(nw.start_conditions[2, idx])))
        self._ana_insp_power_idx = max(0, min(ref_idx, len(nw.power) - 1))
        self._ana_inspect_draw()

    def _ana_inspect_prev(self):
        if self._ana_mode != "inspect":
            return
        self._ana_insp_power_idx = max(0, self._ana_insp_power_idx - 1)
        self._ana_inspect_draw()

    def _ana_inspect_next(self):
        if self._ana_mode != "inspect":
            return
        n_powers = len(self._ana_nw.power)
        self._ana_insp_power_idx = min(n_powers - 1, self._ana_insp_power_idx + 1)
        self._ana_inspect_draw()

    def _ana_inspect_close(self):
        if self._ana_mode != "inspect":
            return
        self._ana_insp_span = None
        self._ana_action_bar.setVisible(False)
        self._ana_mode = "idle"
        self._ana_plot_ll()

    def _ana_inspect_draw(self):
        """Show the single spectrum at the current power step, with the
        fit interval shaded and the fitted lineshape (if converged)
        overlaid on top of the dark-subtracted counts (falls back to raw
        if no dark-subtracted spectra are available)."""
        nw = self._ana_nw
        j  = self._ana_insp_peak
        i  = self._ana_insp_power_idx
        n_powers = len(nw.power)

        ax = self._ana_canvas.reset_axes()
        ax.setLogMode(y=self._ana_rb_log.isChecked())

        wl     = nw.wavelength
        x_full = self._ana_x_of_wl(wl)
        s_full = np.argsort(x_full)
        ax.plot(x_full[s_full], self._ana_display_spectra(nw)[s_full, i],
                pen=pg.mkPen("steelblue", width=1.0))

        fit_ns = nw.fits[j][i] if nw.fits is not None else None
        fdata  = nw.fit_data[j][i] if nw.fit_data is not None else None
        converged = fit_ns is not None

        if fdata is not None and len(fdata):
            fdata = np.asarray(fdata, dtype=float)
            X_bg, Y_bg, bg = fdata[:, 0], fdata[:, 1], fdata[:, 2]
            x_win = self._ana_x_of_wl(X_bg)
            order_win = np.argsort(x_win)
            x_win_s   = x_win[order_win]
            y_raw_win = (Y_bg + bg)[order_win]

            # Highlight the windowed raw data on top of the full spectrum
            ax.plot(x_win_s, y_raw_win, pen=pg.mkPen("orange", width=2.0),
                    symbol="o", symbolSize=5, symbolBrush="orange")

            # Shade the fit interval (static replay — not draggable here)
            lo_disp, hi_disp = float(np.min(x_win_s)), float(np.max(x_win_s))
            self._ana_insp_span = DraggableSpan(ax, color=(0, 150, 0, 60),
                                                movable=False)
            self._ana_insp_span.set_range(lo_disp, hi_disp)

            if converged:
                x_fit_nm = np.linspace(float(X_bg.min()), float(X_bg.max()), 200)
                model_fn = (nwa._lorentz_n_model(fit_ns.n) if fit_ns.is_lorentz
                            else nwa._gauss_n_model(fit_ns.n))
                y_fit_bgsub = model_fn(x_fit_nm, *fit_ns.popt)
                y_fit_raw   = y_fit_bgsub + np.interp(x_fit_nm, X_bg, bg)
                x_fit_disp  = self._ana_x_of_wl(x_fit_nm)
                order_fit   = np.argsort(x_fit_disp)
                ax.plot(x_fit_disp[order_fit], y_fit_raw[order_fit],
                        pen=pg.mkPen("red", width=1.5))
        else:
            self._ana_insp_span = None

        x_label = "Energy (eV)" if self._ana_rb_ev.isChecked() else "Wavelength (nm)"
        ax.setLabel("bottom", x_label)
        ax.setLabel("left", "Counts")
        status = "fit converged" if converged else "fit did NOT converge / no window data"
        power_i = nw.power[i]
        ax.setTitle(
            f"{nw.name} — peak {j+1}/{nw.n_sel_peaks}, "
            f"power step {i+1}/{n_powers}  (P = {power_i:.4g} mW)\n{status}"
        )
        ax.showGrid(x=True, y=True, alpha=0.3)
        self._ana_canvas.draw_idle()

        self._ana_insp_lbl.setText(
            f"Peak {j+1}, step {i+1}/{n_powers} — {status}"
        )
        self._ana_action_lbl.setText(
            f"Inspecting peak {j+1}, power step {i+1}/{n_powers}"
        )

    # ── Save results ─────────────────────────────────────────────

    def _ana_save(self):
        nw = self._ana_nw
        if nw is None or self._ana_file is None:
            return

        try:
            with h5py.File(self._ana_file, "a") as f:
                # Remove stale group, recreate fresh
                if "analysis" in f:
                    del f["analysis"]
                grp = f.create_group("analysis")

                # ── Metadata attributes ───────────────────────────
                # nw.fit_model (not "fit_function") is the real attribute set
                # by nw_analysis.fit_nw() — reading the wrong name here meant
                # this always silently saved "gaussian" regardless of which
                # fit function was actually used; same bug for background_type
                # below (nw_analysis.fit_nw() now sets it too, see 2026-08 fix).
                fit_fn = getattr(nw, "fit_model", "gauss1") or "gauss1"
                grp.attrs["FitFunction"] = fit_fn
                bg_type = getattr(nw, "background_type", "linear") or "linear"
                grp.attrs["FitBackground"] = bg_type

                # ── StartConditions ───────────────────────────────
                if nw.start_conditions is not None:
                    sc = grp.create_dataset(
                        "StartConditions", data=np.array(nw.start_conditions, dtype=float)
                    )
                    sc.attrs["description"] = (
                        "3 x n_peaks: [peak_pixel_index, fitwindow_pixels, ref_power_step_idx]"
                    )

                # ── FitWindows ────────────────────────────────────
                fw = getattr(nw, "fit_windows", None)
                if fw is not None:
                    fw_arr = np.array(fw, dtype=float)
                    ds = grp.create_dataset("FitWindows", data=fw_arr)
                    ds.attrs["units"] = "pixels"
                    ds.attrs["description"] = "Half-window in pixels for each peak"

                # ── PeakArea / PeakAreaErr ────────────────────────
                def _cell_to_mat_max(cell, n_pk, n_pw):
                    arr = np.full((n_pk, n_pw), np.nan)
                    for j, row in enumerate(cell):
                        for i, val in enumerate(row):
                            v = np.atleast_1d(val)
                            if v.size > 0 and not np.all(np.isnan(v)):
                                arr[j, i] = float(np.nanmax(v))
                    return arr

                def _cell_to_mat_first(cell, n_pk, n_pw):
                    arr = np.full((n_pk, n_pw), np.nan)
                    for j, row in enumerate(cell):
                        for i, val in enumerate(row):
                            v = np.atleast_1d(val)
                            if v.size > 0 and not np.all(np.isnan(v)):
                                arr[j, i] = float(v[0])
                    return arr

                pa = getattr(nw, "peak_area", None)
                if pa is not None:
                    try:
                        n_pk = len(pa); n_pw = len(pa[0]) if n_pk > 0 else 0
                        grp.create_dataset("PeakArea", data=_cell_to_mat_max(pa, n_pk, n_pw))
                    except Exception:
                        pass

                pae = getattr(nw, "peak_area_err", None)
                if pae is not None:
                    try:
                        n_pk = len(pae); n_pw = len(pae[0]) if n_pk > 0 else 0
                        grp.create_dataset("PeakAreaErr", data=_cell_to_mat_max(pae, n_pk, n_pw))
                    except Exception:
                        pass

                # ── PeakPos / PeakPosErr (eV) ─────────────────────
                # Fitted centers come out of fit_nw in nm (fits are run
                # against nw.wavelength); convert to energy before saving.
                pp = getattr(nw, "peak_pos", None)
                pos_nm = None
                if pp is not None:
                    try:
                        n_pk = len(pp); n_pw = len(pp[0]) if n_pk > 0 else 0
                        pos_nm = _cell_to_mat_first(pp, n_pk, n_pw)
                        ds = grp.create_dataset("PeakPos", data=_HC_EV_NM / pos_nm)
                        ds.attrs["units"] = "eV"
                    except Exception:
                        pass

                ppe = getattr(nw, "peak_pos_err", None)
                if ppe is not None and pos_nm is not None:
                    try:
                        n_pk = len(ppe); n_pw = len(ppe[0]) if n_pk > 0 else 0
                        err_nm = _cell_to_mat_first(ppe, n_pk, n_pw)
                        # dE = (hc / lambda^2) * dlambda
                        ds = grp.create_dataset(
                            "PeakPosErr", data=_HC_EV_NM / pos_nm**2 * err_nm
                        )
                        ds.attrs["units"] = "eV"
                    except Exception:
                        pass

                # ── FWHM / FWHMErr ────────────────────────────────
                fwhm = getattr(nw, "fwhm", None) or getattr(nw, "findpeaks_fwhm", None)
                if fwhm is not None:
                    try:
                        fwhm_arr = np.array(fwhm, dtype=float)
                        ds = grp.create_dataset("FWHM", data=fwhm_arr)
                        ds.attrs["units"] = "nm"
                    except Exception:
                        pass

                fwhm_e = getattr(nw, "fwhm_err", None)
                if fwhm_e is not None:
                    try:
                        fwhm_e_arr = np.array(fwhm_e, dtype=float)
                        ds = grp.create_dataset("FWHMErr", data=fwhm_e_arr)
                        ds.attrs["units"] = "nm"
                    except Exception:
                        pass

                # ── FitParameters: (n_peaks, n_powers, n_params) ──
                fits = getattr(nw, "fits", None)
                if fits is not None:
                    try:
                        n_pk = len(fits); n_pw = len(fits[0]) if n_pk > 0 else 0
                        n_par = 0
                        for peak_fits in fits:
                            for fit_ns in peak_fits:
                                if fit_ns is not None:
                                    popt = getattr(fit_ns, "popt", fit_ns)
                                    if popt is not None:
                                        n_par = max(n_par, len(np.atleast_1d(popt)))
                        if n_par > 0:
                            fp_arr = np.full((n_pk, n_pw, n_par), np.nan)
                            for j, peak_fits in enumerate(fits):
                                for i, fit_ns in enumerate(peak_fits):
                                    if fit_ns is not None:
                                        popt = getattr(fit_ns, "popt", fit_ns)
                                        if popt is not None:
                                            popt = np.atleast_1d(popt).astype(float)
                                            fp_arr[j, i, :len(popt)] = popt
                            ds = grp.create_dataset("FitParameters", data=fp_arr)
                            ds.attrs["description"] = (
                                "peaks x powers x parameters, from the fit model's popt "
                                "(e.g. gauss: amplitude, center, sigma)"
                            )
                    except Exception:
                        pass

                # ── FitWindowX / FitWindowRawY / BackgroundData: (n_peaks, n_powers, n_datapoints) ──
                # fit_data[j][i] holds 3 columns per (peak, power step): the
                # window's x-positions (nm), the background-subtracted y used
                # for fitting, and the local background curve. All 3 are
                # saved (not just the background, as before 2026-08) so an
                # external reader — e.g. the Visualizer tab's Inspect peaks,
                # which has no access to this session's live `nw` object —
                # can replicate this tab's Inspect-fits overlay exactly:
                # FitWindowX + FitWindowRawY draw the highlighted windowed
                # data, FitWindowX + BackgroundData + FitParameters draw the
                # fitted curve with its background added back. Named
                # "BackgroundData", not "FitBackground", to avoid colliding
                # with the group-level "FitBackground" attribute (the
                # background-subtraction mode string, e.g. "linear").
                fit_data = getattr(nw, "fit_data", None)
                if fit_data is not None:
                    try:
                        n_pk = len(fit_data); n_pw = len(fit_data[0]) if n_pk > 0 else 0
                        n_pts = 0
                        for peak_data in fit_data:
                            for data_arr in peak_data:
                                if data_arr is not None:
                                    arr = np.atleast_2d(data_arr)
                                    if arr.shape[1] >= 3:
                                        n_pts = max(n_pts, arr.shape[0])
                        if n_pts > 0:
                            x_arr   = np.full((n_pk, n_pw, n_pts), np.nan)
                            rawy_arr = np.full((n_pk, n_pw, n_pts), np.nan)
                            bg_arr  = np.full((n_pk, n_pw, n_pts), np.nan)
                            for j, peak_data in enumerate(fit_data):
                                for i, data_arr in enumerate(peak_data):
                                    if data_arr is not None:
                                        arr = np.atleast_2d(data_arr)
                                        if arr.shape[1] >= 3:
                                            n = arr.shape[0]
                                            x_arr[j, i, :n]    = arr[:, 0]
                                            rawy_arr[j, i, :n] = arr[:, 1] + arr[:, 2]
                                            bg_arr[j, i, :n]   = arr[:, 2]
                            ds = grp.create_dataset("FitWindowX", data=x_arr)
                            ds.attrs["units"] = "nm"
                            ds.attrs["description"] = (
                                "peaks x powers x datapoints: fit window x-positions"
                            )
                            ds = grp.create_dataset("FitWindowRawY", data=rawy_arr)
                            ds.attrs["description"] = (
                                "peaks x powers x datapoints: raw (background-included) "
                                "counts in the fit window"
                            )
                            ds = grp.create_dataset("BackgroundData", data=bg_arr)
                            ds.attrs["description"] = (
                                "peaks x powers x datapoints: local background curve "
                                "subtracted before fitting"
                            )
                    except Exception:
                        pass

                # ── PeakIntegral ──────────────────────────────────
                pi = getattr(nw, "peak_integral", None)
                if pi is not None:
                    grp.create_dataset("PeakIntegral", data=np.array(pi, dtype=float))

                # ── Specsum: (n_entries, n_powers), values only ───
                specsum = getattr(nw, "specsum", None)
                if specsum:
                    n_entries = len(specsum)
                    n_pw = max((len(entry.get("values", [])) for entry in specsum),
                               default=0)
                    ss_arr = np.full((n_entries, n_pw), np.nan)
                    for k, entry in enumerate(specsum):
                        vals = np.array(entry.get("values", []), dtype=float)
                        ss_arr[k, :len(vals)] = vals
                    grp.create_dataset("Specsum", data=ss_arr)

                # ── Thresholds ─────────────────────────────────────
                # Merge persisted results (set once the multi-item wizard is
                # finished) with whatever is currently computed in-session,
                # so a threshold computed but not yet stepped through to
                # "Finish" is still included. Saved flat into the analysis
                # group — the threshold-determination mode (fit-area,
                # integral, ...) isn't recorded, only the resulting per-peak
                # numbers.
                thr_results = dict(getattr(nw, "thr_results", None) or {})
                thr_results.update(self._ana_thr_results or {})
                if thr_results:
                    spot_r   = getattr(nw, "spot_radius_short", None)
                    rep_rate = getattr(nw, "rep_rate", None)
                    n_pk     = getattr(nw, "n_sel_peaks", None) or 0

                    def _thr_to_fluence(thr_mW, thr_err_mW):
                        if (spot_r is not None and rep_rate is not None
                                and not np.isnan(thr_mW)
                                and float(rep_rate) > 0 and float(spot_r) > 0):
                            d_cm = 2.0 * float(spot_r) * 1e-4
                            P_W  = float(thr_mW) * 1e-3
                            f_Hz = float(rep_rate)
                            conv = 4.0 / f_Hz / math.pi / d_cm**2 * 1e6  # mW→µJ/cm²
                            err = (conv * float(thr_err_mW) * 1e-3
                                   if not np.isnan(thr_err_mW) else np.nan)
                            return conv * P_W, err, "uJ/cm^2"
                        return thr_mW, thr_err_mW, "mW"

                    # Split "peak N <mode>" titles (e.g. "peak 1 fit-area") from
                    # peak-less ones (e.g. "total peak area"). If more than one
                    # mode was computed for the same peak, the last one in
                    # thr_results wins (mode identity is discarded either way).
                    by_peak = {}   # peak_idx -> data
                    other   = {}   # title -> data
                    for title, data in thr_results.items():
                        m = re.match(r"^peak (\d+) (.+)$", title)
                        if m:
                            by_peak[int(m.group(1)) - 1] = data
                        else:
                            other[title] = data

                    if by_peak:
                        n_pk_out = max(n_pk, max(by_peak) + 1)
                        thr_arr       = np.full(n_pk_out, np.nan)
                        thr_err_arr   = np.full(n_pk_out, np.nan)
                        slope_arr     = np.full(n_pk_out, np.nan)
                        slope_err_arr = np.full(n_pk_out, np.nan)
                        intcpt_arr    = np.full(n_pk_out, np.nan)
                        p_sel_list    = [None] * n_pk_out
                        v_sel_list    = [None] * n_pk_out
                        units = "mW"
                        for peak_idx, data in by_peak.items():
                            thr_val, err_val, units = _thr_to_fluence(
                                data.get("threshold", np.nan),
                                data.get("threshold_err", np.nan)
                            )
                            thr_arr[peak_idx]       = thr_val
                            thr_err_arr[peak_idx]   = err_val
                            slope_arr[peak_idx]     = data.get("slope", np.nan)
                            slope_err_arr[peak_idx] = data.get("slope_err", np.nan)
                            intcpt_arr[peak_idx]    = data.get("intercept", np.nan)
                            p_sel_list[peak_idx] = np.array(data.get("sel_power",  []), dtype=float)
                            v_sel_list[peak_idx] = np.array(data.get("sel_values", []), dtype=float)

                        grp.create_dataset("Threshold",    data=thr_arr)
                        grp.create_dataset("ThresholdErr", data=thr_err_arr)
                        grp.attrs["ThresholdUnits"] = units
                        grp.create_dataset("Slope",        data=slope_arr)
                        grp.create_dataset("SlopeErr",     data=slope_err_arr)
                        grp.create_dataset("Intercept",    data=intcpt_arr)

                        # FitIntervalPower/Values: (n_peaks, n_points), padded
                        # with NaN for peaks with fewer selected points.
                        n_pts = max((len(a) for a in p_sel_list if a is not None),
                                    default=0)
                        if n_pts > 0:
                            fip = np.full((n_pk_out, n_pts), np.nan)
                            fiv = np.full((n_pk_out, n_pts), np.nan)
                            for peak_idx in range(n_pk_out):
                                p_sel = p_sel_list[peak_idx]
                                v_sel = v_sel_list[peak_idx]
                                if p_sel is not None and p_sel.size:
                                    fip[peak_idx, :len(p_sel)] = p_sel
                                if v_sel is not None and v_sel.size:
                                    fiv[peak_idx, :len(v_sel)] = v_sel
                            grp.create_dataset("FitIntervalPower",  data=fip)
                            grp.create_dataset("FitIntervalValues", data=fiv)

                    for title, data in other.items():
                        tg = grp.create_group(title)
                        thr_val, err_val, units = _thr_to_fluence(
                            data.get("threshold", np.nan),
                            data.get("threshold_err", np.nan)
                        )
                        tg.attrs["Threshold"]      = thr_val
                        tg.attrs["ThresholdErr"]   = err_val
                        tg.attrs["ThresholdUnits"] = units
                        tg.attrs["slope"]     = data.get("slope", np.nan)
                        tg.attrs["slope_err"] = data.get("slope_err", np.nan)
                        tg.attrs["intercept"] = data.get("intercept", np.nan)
                        p_sel = np.array(data.get("sel_power",  []), dtype=float)
                        v_sel = np.array(data.get("sel_values", []), dtype=float)
                        if p_sel.size:
                            tg.create_dataset("FitIntervalPower",  data=p_sel)
                        if v_sel.size:
                            tg.create_dataset("FitIntervalValues", data=v_sel)

        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        QMessageBox.information(
            self, "Saved",
            f"Analysis results written to\n{os.path.basename(self._ana_file)}"
        )
