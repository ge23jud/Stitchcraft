import os
import sys
import re
import numpy as np
import h5py
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QCheckBox, QStackedWidget,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QAbstractItemView, QFrame, QDialog,
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plotting import (
    PGCanvas, _COMPACT_BTN_STYLE, MultiLinePlotter, CategoricalScheme,
    SequentialScheme, LogAlphaRamp, DraggableSpan, PLASMA, TAB10, make_pg_toolbar,
)
from io_utils import _h5_contents_summary, _read_trpl_h5_file

from pl import _HC_EV_NM

try:
    import nw_analysis as nwa
    _NWA_AVAILABLE = True
    _NWA_ERROR = ""
except Exception as _nwa_import_exc:
    _NWA_AVAILABLE = False
    _NWA_ERROR = str(_nwa_import_exc)


# ── Powerseries Y/X data-selector option sets ──────────────────────
# Every option is grouped into exactly one of two mutually-exclusive
# "modes": a spectrum (per-wavelength/energy curve, one per power step) or
# a scalar-per-power-step series (peak-fit-derived quantity vs. some power
# quantity). Picking either Y or X pins the mode, which is what makes the
# two dropdowns mutually filter each other down to compatible options.
_VIS_Y_OPTIONS = ["SpectraDiff", "Spectra_raw", "PeakArea", "PeakIntegral", "Specsum"]
_VIS_X_OPTIONS = ["Wavelength", "Energy", "Power", "Uncalibrated Power", "Pump Fluence"]
_VIS_Y_GROUP = {
    "SpectraDiff": "spectra", "Spectra_raw": "spectra",
    "PeakArea": "scalar", "PeakIntegral": "scalar", "Specsum": "scalar",
}
_VIS_X_GROUP = {
    "Wavelength": "spectra", "Energy": "spectra",
    "Power": "scalar", "Uncalibrated Power": "scalar", "Pump Fluence": "scalar",
}
_VIS_X_DEFAULT_ORDER = {
    "spectra": ["Energy", "Wavelength"],
    "scalar":  ["Power", "Uncalibrated Power", "Pump Fluence"],
}
_VIS_X_LABEL = {
    "Wavelength": "Wavelength (nm)", "Energy": "Energy (eV)",
    "Power": "Power (mW)", "Uncalibrated Power": "Uncalibrated power (mW)",
    "Pump Fluence": "Pump fluence (mJ/cm²)",
}


class VisualizerTab(QWidget):
    """"Visualizer": one tab, two independent viewers selected by a "File
    type" dropdown — "Powerseries" (the original tab: overlay PL spectra —
    or, now, any saved fit-derived quantity — across loaded HDF5 files) and
    "TRPL" (overlay TRPL histograms, with optional fit-interval/lifetime
    overlay from the analysis subgroup a TRPL fit tab may have saved).
    Structured the same way tabs/stitch.py merges its two converters:
    `_ps_*`/`_trpl_*`-namespaced state and widgets, one shared canvas.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ══════════════════════════════════════════════════════════
        # Powerseries state
        # ══════════════════════════════════════════════════════════
        self._vis_ps_files: list = []
        self._vis_ps_data:  list = []   # per-file dict, see _vis_ps_load_file()
        self._vis_ps_mode = "idle"      # "idle" | "inspect"

        # Inspect-peaks state (populated by _vis_ps_on_inspect_start)
        self._vis_ps_insp_file_idx  = 0
        self._vis_ps_insp_peak      = 0
        self._vis_ps_insp_power_idx = 0
        self._vis_ps_insp_fit_params = None   # (n_peaks, n_powers, n_params)
        self._vis_ps_insp_win_x      = None   # (n_peaks, n_powers, n_pts) nm
        self._vis_ps_insp_win_rawy   = None
        self._vis_ps_insp_bg         = None
        self._vis_ps_insp_bg_ref_x   = None    # (n_peaks, n_powers, 2) nm, or None (older files)
        self._vis_ps_insp_fit_fn_per_step = None  # (n_peaks, n_powers) str, or None (older files)
        self._vis_ps_insp_is_lorentz = False
        self._vis_ps_insp_n_sub      = 1
        self._vis_ps_insp_span       = None

        # ══════════════════════════════════════════════════════════
        # TRPL state
        # ══════════════════════════════════════════════════════════
        self._vis_trpl_files: list = []
        self._vis_trpl_data:  list = []   # per-file dict, see _vis_trpl_load_file()

        # ── Build UI ────────────────────────────────────────────────
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        sidebar = QWidget()
        sidebar.setFixedWidth(270)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # File type selector — shared, sits above whichever page is active
        g_type = QGroupBox("File type")
        tl = QHBoxLayout(g_type)
        tl.addWidget(QLabel("View:"))
        self._type_combo = QComboBox()
        self._type_combo.addItem("Powerseries", "ps")
        self._type_combo.addItem("TRPL", "trpl")
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        tl.addWidget(self._type_combo, stretch=1)
        sl.addWidget(g_type)

        self._stack = QStackedWidget()
        self._page_ps   = self._build_ps_page()
        self._page_trpl = self._build_trpl_page()
        self._stack.addWidget(self._page_ps)
        self._stack.addWidget(self._page_trpl)
        sl.addWidget(self._stack, stretch=1)

        layout.addWidget(sidebar)

        # ── Canvas, shared by both file types ─────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self._vis_canvas  = PGCanvas(right, welcome_msg="Add HDF5 files to begin.")
        self._vis_toolbar = make_pg_toolbar(self._vis_canvas, right)
        rl.addWidget(self._vis_toolbar)
        rl.addWidget(self._vis_canvas, stretch=1)

        # Powerseries-only inspect-peaks action bar
        self._vis_ps_action_bar = QFrame()
        self._vis_ps_action_bar.setFrameShape(QFrame.StyledPanel)
        ab = QHBoxLayout(self._vis_ps_action_bar)
        ab.setContentsMargins(8, 4, 8, 4)
        ab.addWidget(QLabel("Peak:"))
        self._vis_ps_insp_peak_sel = QComboBox()
        self._vis_ps_insp_peak_sel.currentIndexChanged.connect(self._vis_ps_inspect_peak_changed)
        ab.addWidget(self._vis_ps_insp_peak_sel)
        self._vis_ps_insp_lbl = QLabel("")
        ab.addWidget(self._vis_ps_insp_lbl, stretch=1)
        btn_insp_prev  = QPushButton("◀  Prev spectrum")
        btn_insp_next  = QPushButton("Next spectrum  ▶")
        btn_insp_close = QPushButton("✓  Close inspect")
        btn_insp_prev.clicked.connect(self._vis_ps_inspect_prev)
        btn_insp_next.clicked.connect(self._vis_ps_inspect_next)
        btn_insp_close.clicked.connect(self._vis_ps_inspect_close)
        ab.addWidget(btn_insp_prev)
        ab.addWidget(btn_insp_next)
        ab.addWidget(btn_insp_close)
        self._vis_ps_action_bar.setVisible(False)
        rl.addWidget(self._vis_ps_action_bar)

        layout.addWidget(right, stretch=1)

        self._vis_ps_refresh_combos()

    # ── File-type switch ─────────────────────────────────────────

    def _on_type_changed(self, _index=None):
        kind = self._type_combo.currentData()
        if kind == "ps":
            self._stack.setCurrentWidget(self._page_ps)
            if self._vis_ps_mode == "inspect":
                self._vis_ps_inspect_draw()
            else:
                self._vis_ps_plot()
        else:
            self._vis_ps_mode = "idle"
            self._vis_ps_action_bar.setVisible(False)
            self._stack.setCurrentWidget(self._page_trpl)
            self._vis_trpl_plot()

    # ══════════════════════════════════════════════════════════════
    # Powerseries page
    # ══════════════════════════════════════════════════════════════

    def _build_ps_page(self):
        page = QWidget()
        sl = QVBoxLayout(page)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # Files group
        g_files = QGroupBox("Files")
        fl = QVBoxLayout(g_files)
        self._vis_ps_file_list = QListWidget()
        self._vis_ps_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        fl.addWidget(self._vis_ps_file_list)
        fb = QHBoxLayout()
        self._vis_ps_btn_add    = QPushButton("Add…")
        self._vis_ps_btn_remove = QPushButton("Remove")
        self._vis_ps_btn_clear  = QPushButton("Clear")
        self._vis_ps_btn_add.clicked.connect(self._vis_ps_on_add)
        self._vis_ps_btn_remove.clicked.connect(self._vis_ps_on_remove)
        self._vis_ps_btn_clear.clicked.connect(self._vis_ps_on_clear)
        for b in (self._vis_ps_btn_add, self._vis_ps_btn_remove, self._vis_ps_btn_clear):
            fb.addWidget(b)
        fl.addLayout(fb)
        self._vis_ps_btn_show_data = QPushButton("Show Data")
        self._vis_ps_btn_show_data.clicked.connect(self._vis_ps_on_show_data)
        fl.addWidget(self._vis_ps_btn_show_data)
        g_files.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_files, stretch=1)

        # Thresholds group — shows analysis/Threshold, per loaded file, if present
        g_thr = QGroupBox("Thresholds")
        thl = QVBoxLayout(g_thr)
        self._vis_ps_thr_lbl = QLabel("No files loaded.")
        self._vis_ps_thr_lbl.setWordWrap(True)
        thl.addWidget(self._vis_ps_thr_lbl)
        sl.addWidget(g_thr)

        # Plot group — Y/X data selectors
        g_plot = QGroupBox("Plot")
        pl = QVBoxLayout(g_plot)
        r_y = QHBoxLayout()
        r_y.addWidget(QLabel("Y data:"))
        self._vis_ps_y_combo = QComboBox()
        self._vis_ps_y_combo.currentIndexChanged.connect(self._vis_ps_on_y_changed)
        r_y.addWidget(self._vis_ps_y_combo, stretch=1)
        pl.addLayout(r_y)
        r_x = QHBoxLayout()
        r_x.addWidget(QLabel("X data:"))
        self._vis_ps_x_combo = QComboBox()
        self._vis_ps_x_combo.currentIndexChanged.connect(self._vis_ps_on_x_changed)
        r_x.addWidget(self._vis_ps_x_combo, stretch=1)
        pl.addLayout(r_x)
        sl.addWidget(g_plot)

        self._vis_ps_btn_inspect = QPushButton("Inspect peaks…")
        self._vis_ps_btn_inspect.clicked.connect(self._vis_ps_on_inspect_start)
        sl.addWidget(self._vis_ps_btn_inspect)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        btn_plot = QPushButton("Update plot")
        btn_plot.setMinimumHeight(32)
        bold = QFont(); bold.setBold(True)
        btn_plot.setFont(bold)
        btn_plot.clicked.connect(self._vis_ps_plot)
        sl.addWidget(btn_plot)

        return page

    # ── PS: file management ──────────────────────────────────────

    def _vis_ps_on_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select HDF5 files", "",
            "HDF5 files (*.h5);;All files (*.*)"
        )
        for p in paths:
            if p not in self._vis_ps_files:
                d = self._vis_ps_load_file(p)
                if d is not None:
                    self._vis_ps_files.append(p)
                    self._vis_ps_data.append(d)
                    item = QListWidgetItem(d["label"])
                    item.setToolTip(p)
                    self._vis_ps_file_list.addItem(item)
        self._vis_ps_update_thresholds()
        self._vis_ps_refresh_combos()
        self._vis_ps_plot()

    def _vis_ps_load_file(self, path: str):
        try:
            with h5py.File(path, "r") as f:
                wl = None
                if "Energy" in f:
                    wl = _HC_EV_NM / f["Energy"][:]     # eV → nm for internal use
                elif "Wavelength" in f:
                    wl = f["Wavelength"][:]
                elif "wavelength_nm" in f:
                    wl = f["wavelength_nm"][:]
                if wl is None:
                    raise ValueError("No Energy/Wavelength dataset found.")

                spectra_diff = None
                if "SpectraDiff" in f:
                    spectra_diff = f["SpectraDiff"][:]
                elif "counts" in f:
                    spectra_diff = f["counts"][:]
                spectra_raw = f["Spectra_raw"][:] if "Spectra_raw" in f else None

                pwr_ds = f["Power_uncalibrated"] if "Power_uncalibrated" in f else f.get("powers_W")
                if pwr_ds is None:
                    raise ValueError("No Power_uncalibrated/powers_W dataset found.")
                powers_W = pwr_ds[:].astype(float)
                if pwr_ds.attrs.get("units", "W") == "mW":
                    powers_W = powers_W * 1e-3   # normalise to W internally

                power_cal = None
                if "Power" in f:
                    pc_ds     = f["Power"]
                    power_cal = pc_ds[:].astype(float)
                    if pc_ds.attrs.get("units", "W") == "mW":
                        power_cal = power_cal * 1e-3

                pump_fluence = None
                if "Pump_fluence" in f:
                    fl_ds        = f["Pump_fluence"]
                    pump_fluence = fl_ds[:].astype(float)
                    if fl_ds.attrs.get("units", "mJ/cm^2") in ("uJ/cm^2", "µJ/cm^2"):
                        pump_fluence = pump_fluence * 1e-3   # old files: µJ → mJ

                threshold = threshold_err = threshold_units = None
                peak_area = peak_integral = specsum = None
                if "analysis" in f:
                    agrp = f["analysis"]
                    if "Threshold" in agrp:
                        threshold       = agrp["Threshold"][:].astype(float)
                        threshold_units = agrp.attrs.get("ThresholdUnits", "mW")
                        if "ThresholdErr" in agrp:
                            threshold_err = agrp["ThresholdErr"][:].astype(float)
                    if "PeakArea" in agrp:
                        peak_area = agrp["PeakArea"][:].astype(float)
                    if "PeakIntegral" in agrp:
                        peak_integral = agrp["PeakIntegral"][:].astype(float)
                    if "Specsum" in agrp:
                        specsum = agrp["Specsum"][:].astype(float)
        except Exception as exc:
            QMessageBox.warning(self, "Load error",
                                f"Could not load:\n{path}\n\n{exc}")
            return None

        y_avail = set()
        if spectra_diff   is not None: y_avail.add("SpectraDiff")
        if spectra_raw    is not None: y_avail.add("Spectra_raw")
        if peak_area      is not None: y_avail.add("PeakArea")
        if peak_integral  is not None: y_avail.add("PeakIntegral")
        if specsum        is not None: y_avail.add("Specsum")

        x_avail = {"Wavelength", "Energy", "Uncalibrated Power"}
        if power_cal    is not None: x_avail.add("Power")
        if pump_fluence is not None: x_avail.add("Pump Fluence")

        return {
            "label":           os.path.splitext(os.path.basename(path))[0],
            "path":            path,
            "wl":              wl,
            "spectra_diff":    spectra_diff,
            "spectra_raw":     spectra_raw,
            "powers_W":        powers_W,
            "power_cal":       power_cal,
            "pump_fluence":    pump_fluence,
            "threshold":       threshold,
            "threshold_err":   threshold_err,
            "threshold_units": threshold_units,
            "peak_area":       peak_area,
            "peak_integral":   peak_integral,
            "specsum":         specsum,
            "y_avail":         y_avail,
            "x_avail":         x_avail,
        }

    def _vis_ps_on_remove(self):
        # Removing files can shift/invalidate the file index an active
        # Inspect view is holding onto — close it first rather than risk an
        # IndexError on the next Prev/Next/redraw.
        if self._vis_ps_mode == "inspect":
            self._vis_ps_inspect_close()
        rows = sorted(
            [self._vis_ps_file_list.row(i)
             for i in self._vis_ps_file_list.selectedItems()],
            reverse=True,
        )
        for row in rows:
            self._vis_ps_file_list.takeItem(row)
            self._vis_ps_files.pop(row)
            self._vis_ps_data.pop(row)
        self._vis_ps_update_thresholds()
        self._vis_ps_refresh_combos()
        self._vis_ps_plot()

    def _vis_ps_on_clear(self):
        if self._vis_ps_mode == "inspect":
            self._vis_ps_inspect_close()
        self._vis_ps_files.clear()
        self._vis_ps_data.clear()
        self._vis_ps_file_list.clear()
        self._vis_ps_update_thresholds()
        self._vis_ps_refresh_combos()
        if self._type_combo.currentData() == "ps":
            self._vis_canvas.set_welcome("Add HDF5 files to begin.")

    def _vis_ps_update_thresholds(self):
        """Refresh the Thresholds panel from analysis/Threshold in each
        loaded file, if present (written by the Analysis tab's Save)."""
        if not self._vis_ps_data:
            self._vis_ps_thr_lbl.setText("No files loaded.")
            return

        lines = []
        for d in self._vis_ps_data:
            thr = d.get("threshold")
            if thr is None:
                lines.append(f"{d['label']}: —")
                continue
            units = d.get("threshold_units") or "mW"
            err   = d.get("threshold_err")
            parts = []
            for i, val in enumerate(thr):
                if np.isnan(val):
                    continue
                if err is not None and i < len(err) and not np.isnan(err[i]):
                    parts.append(f"peak {i+1}: {val:.4g} ± {err[i]:.2g} {units}")
                else:
                    parts.append(f"peak {i+1}: {val:.4g} {units}")
            lines.append(f"{d['label']}: " + ("; ".join(parts) if parts else "—"))

        self._vis_ps_thr_lbl.setText("\n".join(lines))

    def _vis_ps_on_show_data(self):
        row = self._vis_ps_file_list.currentRow()
        if row < 0 or row >= len(self._vis_ps_files):
            QMessageBox.information(self, "No selection",
                                    "Select a file in the list first.")
            return
        path = self._vis_ps_files[row]
        try:
            lines = _h5_contents_summary(path)
        except Exception as exc:
            QMessageBox.critical(self, "Read error",
                                 f"Could not read:\n{path}\n\n{exc}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Data in {os.path.basename(path)}")
        dlg.resize(560, 420)
        dl = QVBoxLayout(dlg)
        listw = QListWidget()
        mono = QFont("Consolas")
        listw.setFont(mono)
        for line in lines:
            listw.addItem(QListWidgetItem(line))
        dl.addWidget(listw)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        dl.addWidget(btn_close)
        dlg.exec_()

    # ── PS: Y/X combo cross-filtering ────────────────────────────

    def _vis_ps_y_avail(self):
        return set().union(*(d["y_avail"] for d in self._vis_ps_data)) if self._vis_ps_data else set()

    def _vis_ps_x_avail(self):
        return set().union(*(d["x_avail"] for d in self._vis_ps_data)) if self._vis_ps_data else set()

    @staticmethod
    def _vis_ps_fill_combo(combo, opts, selected):
        combo.blockSignals(True)
        combo.clear()
        for o in opts:
            combo.addItem(o, o)
        if selected in opts:
            combo.setCurrentIndex(opts.index(selected))
        combo.blockSignals(False)

    def _vis_ps_refresh_combos(self, driver=None):
        """Repopulate both combos from file availability. Only the *passive*
        combo (the one the user didn't just change) gets narrowed to options
        compatible with the active one's current selection — "spectra"
        options (SpectraDiff/Spectra_raw + Wavelength/Energy) and "scalar"
        options (PeakArea/PeakIntegral/Specsum + Power/Uncalibrated
        Power/Pump Fluence) are mutually exclusive groups, so pinning
        either combo's group is enough to filter the other down to just
        that group's available options.

        The active combo always keeps its FULL availability-based list —
        narrowing it too (i.e. both sides simultaneously, as an earlier
        version of this did) creates a dead end: once scalar-mode narrowed
        Y down to just PeakArea/PeakIntegral/Specsum, "SpectraDiff" was no
        longer offered anywhere to switch back with.
        """
        y_avail = self._vis_ps_y_avail()
        x_avail = self._vis_ps_x_avail()
        y_full = [o for o in _VIS_Y_OPTIONS if o in y_avail]
        x_full = [o for o in _VIS_X_OPTIONS if o in x_avail]

        cur_y = self._vis_ps_y_combo.currentData()
        cur_x = self._vis_ps_x_combo.currentData()

        if driver == "x" and cur_x in x_avail:
            mode, active = _VIS_X_GROUP[cur_x], "x"
        elif driver == "y" and cur_y in y_avail:
            mode, active = _VIS_Y_GROUP[cur_y], "y"
        elif cur_y in y_avail:
            mode, active = _VIS_Y_GROUP[cur_y], "y"
        elif cur_x in x_avail:
            mode, active = _VIS_X_GROUP[cur_x], "x"
        else:
            mode, active = None, None

        if mode is None:
            y_opts, x_opts = y_full, x_full
        elif active == "y":
            y_opts = y_full
            x_opts = [o for o in x_full if _VIS_X_GROUP[o] == mode] or x_full
        else:
            x_opts = x_full
            y_opts = [o for o in y_full if _VIS_Y_GROUP[o] == mode] or y_full

        def _pick(cur, opts, default_order):
            if cur in opts:
                return cur
            for o in default_order:
                if o in opts:
                    return o
            return opts[0] if opts else None

        new_y = _pick(cur_y, y_opts, _VIS_Y_OPTIONS)
        x_default_order = _VIS_X_DEFAULT_ORDER.get(_VIS_Y_GROUP.get(new_y, mode), _VIS_X_OPTIONS)
        new_x = _pick(cur_x, x_opts, x_default_order)

        self._vis_ps_fill_combo(self._vis_ps_y_combo, y_opts, new_y)
        self._vis_ps_fill_combo(self._vis_ps_x_combo, x_opts, new_x)

    def _vis_ps_on_y_changed(self, _index=None):
        self._vis_ps_refresh_combos(driver="y")
        self._vis_ps_plot()

    def _vis_ps_on_x_changed(self, _index=None):
        self._vis_ps_refresh_combos(driver="x")
        self._vis_ps_plot()

    # ── PS: plot dispatch ────────────────────────────────────────

    @staticmethod
    def _vis_fmt_power(p_W: float) -> str:
        p_mW = p_W * 1e3
        if p_mW >= 1.0:
            return f"{p_mW:.4g} mW"
        p_uW = p_W * 1e6
        if p_uW >= 1.0:
            return f"{p_uW:.4g} µW"
        return f"{p_W * 1e9:.4g} nW"

    def _vis_ps_legend_label(self, d, idx):
        """Pump fluence if the file has it, else calibrated power, else
        uncalibrated power."""
        fluence = d.get("pump_fluence")
        if fluence is not None:
            return f"{fluence[idx]:.4g} mJ/cm²"
        cal = d.get("power_cal")
        if cal is not None:
            return self._vis_fmt_power(cal[idx])
        return self._vis_fmt_power(d["powers_W"][idx])

    @staticmethod
    def _vis_ps_y_matrix(d, y_key):
        return {
            "SpectraDiff":   d["spectra_diff"],
            "Spectra_raw":   d["spectra_raw"],
            "PeakArea":      d["peak_area"],
            "PeakIntegral":  d["peak_integral"],
            "Specsum":       d["specsum"],
        }.get(y_key)

    @staticmethod
    def _vis_ps_x_array(d, x_key):
        if x_key == "Wavelength":
            return d["wl"]
        if x_key == "Energy":
            return _HC_EV_NM / d["wl"]
        if x_key == "Power":
            cal = d["power_cal"]
            return cal * 1e3 if cal is not None else None
        if x_key == "Uncalibrated Power":
            return d["powers_W"] * 1e3
        if x_key == "Pump Fluence":
            return d["pump_fluence"]
        return None

    def _vis_ps_plot(self):
        if self._vis_ps_mode == "inspect":
            return   # the inspect view owns the canvas right now
        if self._type_combo.currentData() != "ps":
            return
        if not self._vis_ps_data:
            self._vis_canvas.set_welcome("Add HDF5 files to begin.")
            return
        y_key = self._vis_ps_y_combo.currentData()
        x_key = self._vis_ps_x_combo.currentData()
        if y_key is None or x_key is None:
            self._vis_canvas.set_welcome("No compatible data in the loaded file(s).")
            return
        if _VIS_Y_GROUP.get(y_key) == "spectra":
            self._vis_ps_plot_spectra(y_key, x_key)
        else:
            self._vis_ps_plot_scalar(y_key, x_key)

    def _vis_ps_plot_spectra(self, y_key, x_key):
        files = [d for d in self._vis_ps_data
                 if y_key in d["y_avail"] and x_key in d["x_avail"]]
        if not files:
            self._vis_canvas.set_welcome(f"No loaded file has both {y_key} and {x_key}.")
            return

        p_sorted = sorted(set(float(p) for d in files for p in d["powers_W"]))
        if not p_sorted:
            self._vis_canvas.reset_axes()
            self._vis_canvas.draw_idle()
            return

        ax = self._vis_canvas.reset_axes()
        n_lines = len(files) * len(p_sorted)
        ax.addLegend(labelTextSize=f"{max(4, 7 - n_lines // 5)}pt",
                     colCount=max(1, n_lines // 20))

        if len(files) == 1:
            mlp = MultiLinePlotter(
                ax, SequentialScheme(vmin=p_sorted[0], vmax=p_sorted[-1], cmap=PLASMA)
            )
        else:
            mlp = MultiLinePlotter(ax, LogAlphaRamp(CategoricalScheme(), p_sorted))

        for fi, d in enumerate(files):
            y_mat = self._vis_ps_y_matrix(d, y_key)
            x_arr = self._vis_ps_x_array(d, x_key)
            s     = np.argsort(x_arr)
            for p_W in p_sorted:
                closest = int(np.argmin(np.abs(d["powers_W"] - p_W)))
                lbl = f"{d['label']} — {self._vis_ps_legend_label(d, closest)}"
                mlp.plot(x_arr[s], y_mat[s, closest], index=fi, value=p_W,
                         label=lbl, width=0.9)

        ax.setLabel("bottom", _VIS_X_LABEL[x_key])
        ax.setLabel("left", "Counts")
        ax.showGrid(x=True, y=True, alpha=0.3)
        ax.setTitle(f"{y_key} spectra")
        self._vis_canvas.draw_idle()

    def _vis_ps_plot_scalar(self, y_key, x_key):
        files = [d for d in self._vis_ps_data
                 if y_key in d["y_avail"] and x_key in d["x_avail"]]
        if not files:
            self._vis_canvas.set_welcome(f"No loaded file has both {y_key} and {x_key}.")
            return

        ax = self._vis_canvas.reset_axes()
        ax.addLegend(labelTextSize="7pt")
        color_idx = 0
        plotted = False

        for d in files:
            y_mat = self._vis_ps_y_matrix(d, y_key)      # (n_entries, n_powers)
            x_arr = self._vis_ps_x_array(d, x_key)        # (n_powers,)
            n_entries = y_mat.shape[0]
            for k in range(n_entries):
                vals  = y_mat[k, :]
                n     = min(len(vals), len(x_arr))
                valid = ~np.isnan(vals[:n]) & ~np.isnan(x_arr[:n])
                if not valid.any():
                    continue
                order = np.argsort(x_arr[:n][valid])
                color = TAB10[color_idx % 10]
                label = f"{d['label']} — peak {k+1}" if n_entries > 1 else d["label"]
                ax.plot(x_arr[:n][valid][order], vals[:n][valid][order],
                        pen=pg.mkPen(color, width=1.2),
                        symbol="o", symbolSize=6, symbolBrush=color, name=label)
                color_idx += 1
                plotted = True

        if not plotted:
            vb = ax.getViewBox()
            vb.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
            vb.setMouseEnabled(x=False, y=False)
            text = pg.TextItem(f"No valid {y_key} vs. {x_key} data.",
                                color="#888888", anchor=(0.5, 0.5))
            text.setPos(0.5, 0.5)
            vb.addItem(text)
        else:
            ax.setLabel("bottom", _VIS_X_LABEL[x_key])
            ax.setLabel("left", y_key)
            ax.showGrid(x=True, y=True, alpha=0.3)
            ax.setTitle(f"{y_key} vs. {x_key}")
        self._vis_canvas.draw_idle()

    # ── PS: Inspect peaks ────────────────────────────────────────

    def _vis_ps_on_inspect_start(self):
        if not _NWA_AVAILABLE:
            QMessageBox.critical(self, "nw_analysis unavailable",
                                 f"nw_analysis could not be imported:\n{_NWA_ERROR}")
            return
        row = self._vis_ps_file_list.currentRow()
        if row < 0 or row >= len(self._vis_ps_data):
            QMessageBox.information(self, "No selection",
                                    "Select a file in the list first.")
            return
        d = self._vis_ps_data[row]
        path = d["path"]
        try:
            with h5py.File(path, "r") as f:
                if "analysis" not in f:
                    QMessageBox.critical(
                        self, "No analysis data",
                        "This file has no 'analysis' subgroup — run and save "
                        "fits in the Analysis tab first."
                    )
                    return
                agrp = f["analysis"]
                required = ("FitParameters", "FitWindowX", "FitWindowRawY", "BackgroundData")
                missing = [k for k in required if k not in agrp]
                if missing:
                    QMessageBox.critical(
                        self, "Incomplete analysis data",
                        "This file's 'analysis' subgroup is missing: "
                        + ", ".join(missing) + ".\nRe-save from the Analysis "
                        "tab (needs the 2026-08+ version that stores "
                        "per-step fit windows)."
                    )
                    return
                fit_params  = agrp["FitParameters"][:]
                win_x       = agrp["FitWindowX"][:]
                win_rawy    = agrp["FitWindowRawY"][:]
                bg          = agrp["BackgroundData"][:]
                # Optional: added alongside the 'linear_peakwidth' mode
                # (2026-08) — absent in older saved files, in which case
                # the Inspect view just skips the reference-point markers.
                bg_ref_x    = agrp["BackgroundRefX"][:] if "BackgroundRefX" in agrp else None
                fit_fn_name = agrp.attrs.get("FitFunction", "gauss1")
                # Optional: added alongside per-spectrum fit-setting
                # overrides (2026-08) — the group's "FitFunction"
                # attribute is only the default/fallback; if overrides
                # made different steps use different fit functions
                # (different n_sub, or gauss vs lorentz), this per-step
                # dataset is needed to interpret each step's
                # FitParameters row correctly. Absent in older files, in
                # which case every step falls back to the single
                # group-level default (correct as long as no per-step
                # overrides were ever used on that file).
                fit_fn_per_step = (agrp["FitFunctionPerStep"].asstr()[:]
                                   if "FitFunctionPerStep" in agrp else None)
        except Exception as exc:
            QMessageBox.critical(self, "Read error", f"Could not read:\n{path}\n\n{exc}")
            return

        m = re.match(r"(gauss|lorentz)(\d+)", str(fit_fn_name))
        self._vis_ps_insp_is_lorentz = bool(m) and m.group(1) == "lorentz"
        self._vis_ps_insp_n_sub      = int(m.group(2)) if m else 1
        self._vis_ps_insp_fit_fn_per_step = fit_fn_per_step

        self._vis_ps_insp_file_idx   = row
        self._vis_ps_insp_fit_params = fit_params
        self._vis_ps_insp_win_x      = win_x
        self._vis_ps_insp_win_rawy   = win_rawy
        self._vis_ps_insp_bg         = bg
        self._vis_ps_insp_bg_ref_x   = bg_ref_x
        self._vis_ps_insp_peak       = 0
        self._vis_ps_insp_power_idx  = 0

        n_peaks = fit_params.shape[0]
        self._vis_ps_insp_peak_sel.blockSignals(True)
        self._vis_ps_insp_peak_sel.clear()
        self._vis_ps_insp_peak_sel.addItems([f"Peak {k+1}" for k in range(n_peaks)])
        self._vis_ps_insp_peak_sel.blockSignals(False)

        self._vis_ps_mode = "inspect"
        self._vis_ps_action_bar.setVisible(True)
        self._vis_ps_inspect_draw()

    def _vis_ps_inspect_peak_changed(self, idx):
        if self._vis_ps_mode != "inspect" or idx < 0:
            return
        self._vis_ps_insp_peak = idx
        self._vis_ps_insp_power_idx = 0
        self._vis_ps_inspect_draw()

    def _vis_ps_inspect_prev(self):
        if self._vis_ps_mode != "inspect":
            return
        self._vis_ps_insp_power_idx = max(0, self._vis_ps_insp_power_idx - 1)
        self._vis_ps_inspect_draw()

    def _vis_ps_inspect_next(self):
        if self._vis_ps_mode != "inspect":
            return
        d = self._vis_ps_data[self._vis_ps_insp_file_idx]
        n_powers = len(d["powers_W"])
        self._vis_ps_insp_power_idx = min(n_powers - 1, self._vis_ps_insp_power_idx + 1)
        self._vis_ps_inspect_draw()

    def _vis_ps_inspect_close(self):
        if self._vis_ps_mode != "inspect":
            return
        self._vis_ps_insp_span = None
        self._vis_ps_action_bar.setVisible(False)
        self._vis_ps_mode = "idle"
        self._vis_ps_plot()

    def _vis_ps_inspect_draw(self):
        """Replicates the Analysis tab's Inspect-fits overlay from data
        saved in the file alone (FitWindowX/FitWindowRawY/BackgroundData/
        FitParameters) — no live fitting session required."""
        d = self._vis_ps_data[self._vis_ps_insp_file_idx]
        j = self._vis_ps_insp_peak
        i = self._vis_ps_insp_power_idx
        n_powers = len(d["powers_W"])

        ax = self._vis_canvas.reset_axes()
        wl = d["wl"]
        x_full = _HC_EV_NM / wl
        s_full = np.argsort(x_full)
        spectra = d["spectra_diff"] if d["spectra_diff"] is not None else d["spectra_raw"]
        ax.plot(x_full[s_full], spectra[s_full, i], pen=pg.mkPen("steelblue", width=1.0))

        win_x_nm = self._vis_ps_insp_win_x[j, i, :]
        win_raw  = self._vis_ps_insp_win_rawy[j, i, :]
        bg_arr   = self._vis_ps_insp_bg[j, i, :]
        valid = ~np.isnan(win_x_nm)
        converged = False

        if valid.any():
            x_win_nm = win_x_nm[valid]
            y_win    = win_raw[valid]
            bg_win   = bg_arr[valid]
            x_win_ev = _HC_EV_NM / x_win_nm
            order = np.argsort(x_win_ev)
            ax.plot(x_win_ev[order], y_win[order], pen=pg.mkPen("orange", width=2.0),
                    symbol="o", symbolSize=5, symbolBrush="orange")

            # Local background curve + its two reference-point x-positions
            # (see analysis.py's Inspect view — same overlay, replicated
            # here from the saved BackgroundData/BackgroundRefX datasets).
            # BackgroundRefX is optional (added 2026-08 alongside the
            # 'linear_peakwidth' mode); older files simply won't draw the
            # reference-point markers.
            ax.plot(x_win_ev[order], bg_win[order],
                    pen=pg.mkPen("yellow", width=1.2, style=Qt.DashLine))
            if self._vis_ps_insp_bg_ref_x is not None:
                ref_x = self._vis_ps_insp_bg_ref_x[j, i, :]
                if not np.any(np.isnan(ref_x)):
                    for k, x_ref_nm in enumerate(ref_x):
                        x_ref_ev = float(_HC_EV_NM / x_ref_nm)
                        ax.addItem(pg.InfiniteLine(
                            pos=x_ref_ev, angle=90,
                            pen=pg.mkPen("yellow", width=1.0, style=Qt.DotLine),
                            label=f"bg ref {k+1}",
                            labelOpts={"position": 0.05 + 0.08*k, "color": "yellow"},
                        ))

            lo_disp, hi_disp = float(np.min(x_win_ev)), float(np.max(x_win_ev))
            self._vis_ps_insp_span = DraggableSpan(ax, color=(0, 150, 0, 60), movable=False)
            self._vis_ps_insp_span.set_range(lo_disp, hi_disp)

            popt = self._vis_ps_insp_fit_params[j, i, :]
            popt = popt[~np.isnan(popt)]
            converged = popt.size > 0
            if converged:
                # Use THIS step's own fit function if per-spectrum
                # overrides were saved (FitFunctionPerStep) -- the
                # group-level default can be wrong for a step that used
                # a different function, and a wrong n_sub here would
                # misinterpret FitParameters' NaN-padding as real values
                # or vice versa. Falls back to the file-wide default
                # (self._vis_ps_insp_is_lorentz/_n_sub) for older files.
                is_lorentz, n_sub = self._vis_ps_insp_is_lorentz, self._vis_ps_insp_n_sub
                if self._vis_ps_insp_fit_fn_per_step is not None:
                    step_fn_name = str(self._vis_ps_insp_fit_fn_per_step[j, i])
                    m_step = re.match(r"(gauss|lorentz)(\d+)", step_fn_name)
                    if m_step:
                        is_lorentz = m_step.group(1) == "lorentz"
                        n_sub = int(m_step.group(2))
                x_fit_nm = np.linspace(float(x_win_nm.min()), float(x_win_nm.max()), 200)
                model_fn = (nwa._lorentz_n_model(n_sub)
                            if is_lorentz
                            else nwa._gauss_n_model(n_sub))
                y_fit_bgsub = model_fn(x_fit_nm, *popt)
                y_fit_raw   = y_fit_bgsub + np.interp(x_fit_nm, x_win_nm, bg_win)
                x_fit_ev    = _HC_EV_NM / x_fit_nm
                order_fit   = np.argsort(x_fit_ev)
                ax.plot(x_fit_ev[order_fit], y_fit_raw[order_fit],
                        pen=pg.mkPen("red", width=1.5))
        else:
            self._vis_ps_insp_span = None

        ax.setLabel("bottom", "Energy (eV)")
        ax.setLabel("left", "Counts")
        status = "fit converged" if converged else "fit did NOT converge / no window data"
        power_i = d["powers_W"][i] * 1e3
        n_peaks = self._vis_ps_insp_fit_params.shape[0]
        ax.setTitle(
            f"{d['label']} — peak {j+1}/{n_peaks}, power step {i+1}/{n_powers} "
            f"(P = {power_i:.4g} mW)\n{status}"
        )
        ax.showGrid(x=True, y=True, alpha=0.3)
        self._vis_canvas.draw_idle()

        self._vis_ps_insp_lbl.setText(f"Peak {j+1}, step {i+1}/{n_powers} — {status}")

    # ══════════════════════════════════════════════════════════════
    # TRPL page
    # ══════════════════════════════════════════════════════════════

    def _build_trpl_page(self):
        page = QWidget()
        sl = QVBoxLayout(page)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        g_files = QGroupBox("Files")
        fl = QVBoxLayout(g_files)
        self._vis_trpl_file_list = QListWidget()
        self._vis_trpl_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        fl.addWidget(self._vis_trpl_file_list)
        fb = QHBoxLayout()
        self._vis_trpl_btn_add    = QPushButton("Add…")
        self._vis_trpl_btn_remove = QPushButton("Remove")
        self._vis_trpl_btn_clear  = QPushButton("Clear")
        self._vis_trpl_btn_add.clicked.connect(self._vis_trpl_on_add)
        self._vis_trpl_btn_remove.clicked.connect(self._vis_trpl_on_remove)
        self._vis_trpl_btn_clear.clicked.connect(self._vis_trpl_on_clear)
        for b in (self._vis_trpl_btn_add, self._vis_trpl_btn_remove, self._vis_trpl_btn_clear):
            fb.addWidget(b)
        fl.addLayout(fb)
        g_files.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_files, stretch=1)

        self._vis_trpl_chk_fits = QCheckBox("Show fit intervals && functions")
        self._vis_trpl_chk_fits.setChecked(True)
        self._vis_trpl_chk_fits.toggled.connect(self._vis_trpl_plot)
        sl.addWidget(self._vis_trpl_chk_fits)

        g_life = QGroupBox("Lifetime(s) / Total Decay Time")
        ll = QVBoxLayout(g_life)
        self._vis_trpl_lifetime_lbl = QLabel("No files loaded.")
        self._vis_trpl_lifetime_lbl.setWordWrap(True)
        ll.addWidget(self._vis_trpl_lifetime_lbl)
        sl.addWidget(g_life)

        return page

    # ── TRPL: file management ────────────────────────────────────

    def _vis_trpl_on_add(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select TRPL HDF5 files", "",
            "HDF5 files (*.h5);;All files (*.*)"
        )
        for p in paths:
            if p not in self._vis_trpl_files:
                d = self._vis_trpl_load_file(p)
                if d is not None:
                    self._vis_trpl_files.append(p)
                    self._vis_trpl_data.append(d)
                    item = QListWidgetItem(d["label"])
                    item.setToolTip(p)
                    self._vis_trpl_file_list.addItem(item)
        self._vis_trpl_plot()

    def _vis_trpl_load_file(self, path):
        try:
            data = _read_trpl_h5_file(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load error", f"Could not load:\n{path}\n\n{exc}")
            return None

        fit_params = fit_range = lifetimes = total_decay_time = None
        data_time_offset = 0.0
        try:
            with h5py.File(path, "r") as f:
                if "analysis" in f:
                    agrp = f["analysis"]
                    # (2026-08) the TRPL tab switched from N independently
                    # fit single exponentials (one range each, "FitRanges"
                    # (n,2)) to a single range holding a sum of N
                    # exponentials ("FitRange" (2,)) — older files saved
                    # under the previous schema simply won't have
                    # "FitRange" and are skipped (their "FitRanges" no
                    # longer matches this overlay's one-range assumption).
                    if "FitParameters" in agrp and "FitRange" in agrp:
                        fit_params = agrp["FitParameters"][:]
                        fit_range = agrp["FitRange"][:]
                    if "Lifetime" in agrp:
                        lifetimes = agrp["Lifetime"][:]
                    # Added alongside trpl.py's Total_Decay_Time (2026-09);
                    # absent in older files, in which case it's just omitted
                    # from the label text below.
                    if "Total_Decay_Time" in agrp:
                        total_decay_time = float(agrp["Total_Decay_Time"][()])
                    # Added alongside trpl.py's "Data time offset" (2026-09):
                    # FitRange (and FitParameters, via the model evaluated
                    # below) was fit against the TRPL tab's own working
                    # Times = this file's raw Times + DataTimeOffset, not
                    # the raw Times this method just read via
                    # _read_trpl_h5_file() into data["times"] above — absent
                    # in older files (or a file where it was never applied),
                    # in which case it's just 0.0 and every use below is a
                    # no-op, exactly reproducing the pre-2026-09 behavior.
                    if "DataTimeOffset" in agrp.attrs:
                        data_time_offset = float(agrp.attrs["DataTimeOffset"])
        except Exception:
            pass   # fit overlay/lifetime text are optional extras

        return {
            "label":            os.path.splitext(os.path.basename(path))[0],
            "path":             path,
            "times":            data["times"],
            "counts":           data["counts"],
            "fit_params":       fit_params,
            "fit_range":        fit_range,
            "lifetimes":        lifetimes,
            "total_decay_time": total_decay_time,
            "data_time_offset": data_time_offset,
        }

    def _vis_trpl_on_remove(self):
        rows = sorted(
            [self._vis_trpl_file_list.row(i)
             for i in self._vis_trpl_file_list.selectedItems()],
            reverse=True,
        )
        for row in rows:
            self._vis_trpl_file_list.takeItem(row)
            self._vis_trpl_files.pop(row)
            self._vis_trpl_data.pop(row)
        self._vis_trpl_plot()

    def _vis_trpl_on_clear(self):
        self._vis_trpl_files.clear()
        self._vis_trpl_data.clear()
        self._vis_trpl_file_list.clear()
        self._vis_trpl_plot()

    # ── TRPL: plot ────────────────────────────────────────────────

    def _vis_trpl_plot(self, *_args):
        if self._type_combo.currentData() != "trpl":
            return
        if not self._vis_trpl_data:
            self._vis_canvas.set_welcome("Add TRPL HDF5 files to begin.")
            self._vis_trpl_lifetime_lbl.setText("No files loaded.")
            return

        ax = self._vis_canvas.reset_axes()
        ax.setLogMode(y=True)
        # TRPL histograms are ~65536 points — clipToView + auto-downsampling
        # keep panning/zooming fast at any zoom level (same technique as the
        # dedicated TRPL tab / the Stitch/Convert tab's TRPL-conversion
        # preview); antialias=False avoids the app-wide antialiased-render
        # default, which is what makes curves this large laggy otherwise.
        ax.setClipToView(True)
        ax.setDownsampling(auto=True, mode="peak")
        ax.addLegend(labelTextSize="7pt")
        mlp = MultiLinePlotter(ax, CategoricalScheme())

        show_fits = self._vis_trpl_chk_fits.isChecked()
        lifetime_lines = []
        for fi, d in enumerate(self._vis_trpl_data):
            mlp.plot(d["times"], d["counts"], index=fi, width=1.0,
                     antialias=False, label=d["label"])

            if show_fits and d["fit_params"] is not None and d["fit_range"] is not None:
                color = pg.mkColor(TAB10[fi % 10])
                region_color = pg.mkColor(color)
                region_color.setAlpha(40)
                # FitRange is in the TRPL tab's own working-time axis
                # (raw Times + DataTimeOffset — see trpl.py's "Data time
                # offset" and its _on_save()), but d["times"]/d["counts"]
                # above are the file's unmodified raw Times — so the
                # shaded region has to be shifted back by that same
                # offset to land on the actual samples it was picked
                # over. offset defaults to 0.0 for files saved before
                # this attribute existed, making this a no-op then.
                offset = d.get("data_time_offset", 0.0)
                xmin, xmax = (float(v) - offset for v in d["fit_range"])
                region = pg.LinearRegionItem(
                    values=(xmin, xmax), orientation="vertical",
                    brush=pg.mkBrush(region_color), movable=False,
                )
                region.setZValue(5)
                ax.addItem(region)
                x_curve = np.linspace(xmin, xmax, 200)
                # Sum of exponentials: one (a, b) row per component, all
                # sharing this one fit range (see trpl.py's save routine).
                # Evaluated at x_curve + offset — i.e. back on the
                # working-time axis FitParameters was actually fit
                # against — then plotted at x_curve's raw-time position.
                y_curve = np.zeros_like(x_curve)
                for a, b in d["fit_params"]:
                    y_curve = y_curve + a * np.exp(-b * (x_curve + offset))
                ax.plot(x_curve, y_curve, pen=pg.mkPen(color, width=1.5))

            if d["lifetimes"] is not None and len(d["lifetimes"]):
                vals = ", ".join(f"{v:.4g} ns" for v in d["lifetimes"])
                line = f"{d['label']}: {vals}"
            else:
                line = f"{d['label']}: —"
            if d["total_decay_time"] is not None:
                line += f"  (total decay time: {d['total_decay_time']:.4g} ns)"
            lifetime_lines.append(line)

        ax.setLabel("bottom", "Time (ns)")
        ax.setLabel("left", "Counts")
        ax.showGrid(x=True, y=True, alpha=0.3)
        ax.setTitle("TRPL")
        self._vis_canvas.draw_idle()

        self._vis_trpl_lifetime_lbl.setText("\n".join(lifetime_lines))
