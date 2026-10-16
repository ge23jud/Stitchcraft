"""Reusable "series → HDF5" converter page for the Stitch / Convert tab.

One `SeriesConverterPage` is one sidebar page plus its own state: load one
or more .origin files whose spectra are columns of a (n_wl, n_cols) matrix,
preview, pick per-pair overlap transition spans (stitching), dark-spectrum
subtraction, power calibration, spot size / rep. rate, per-column baseline
subtraction, then save as HDF5 (or .origin, power series only).

The column axis depends on `kind`:
  "ps"  — power series: one column per excitation power (W, from the file's
          "Excitation power (W)" row).
  "pol" — polarization series: one column per polarization angle (degrees,
          from the "Polarization angle" row); the single "Excitation power"
          header value applies to every column, so Power_uncalibrated/Power/
          Power_density/Pump_fluence are saved as single values (1-element
          arrays), computed exactly the same way as for a power series.

The page owns its pair-navigation and baseline Confirm/Cancel bars
(`nav_bar`, `baseline_bar`); the host tab places them under the shared
canvas and calls `activate()`/`deactivate()` when its "File type" combo
switches to/away from this page.
"""
import os
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QRadioButton, QListWidget, QListWidgetItem, QLineEdit,
    QFileDialog, QMessageBox, QSizePolicy, QAbstractItemView, QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

from plotting import (
    _COMPACT_BTN_STYLE, MultiLinePlotter, CategoricalScheme,
    SequentialScheme, PLASMA, GradientLegend, DraggableSpan,
)
from io_utils import (
    _parse_header_center_disp,
    _write_origin_file,
    _write_h5_file,
    _parse_power_calibration,
    _parse_pol_origin,
)
from pl import (
    _parse_origin_power_series,
    _parse_origin_header,
    _stitch_counts,
    _HC_EV_NM,
)


def _parse_any_series(path):
    """Parse a .origin file as either a power series or a polarization
    series, returning (wl_nm, counts (n_wl, n_cols), header dict). Used for
    dark spectra, which are averaged over their columns anyway — so a dark
    of either measurement type works for either kind of page."""
    try:
        wl, counts, _ = _parse_origin_power_series(path)
        return wl, counts, _parse_origin_header(path)
    except ValueError:
        p = _parse_pol_origin(path)
        return p["wl"], p["counts"], p["hdr"]


class SeriesConverterPage(QWidget):

    def __init__(self, canvas, status_cb, kind="ps", parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self._status = status_cb
        self._kind   = kind

        # What one spectrum column is (see the module docstring).
        if kind == "pol":
            self._step_word = "angle"
            self._col_label = "Angle (°)"
        else:
            self._step_word = "power step"
            self._col_label = "Power (mW)"

        self._files: list   = []
        self._datasets: list = []   # {label, wl, counts, cols, path}
        self._hdrs: list    = []
        self._pairs: list   = []     # (i_a, i_b, ov_lo_wl, ov_hi_wl)
        self._spans: dict   = {}     # {(i_a,i_b): (lo_wl, hi_wl)}
        self._spans_display: dict = {}    # {(i_a,i_b): (xmin, xmax)} display units
        self._pair_idx: int = 0
        self._x_axis: str   = "energy"
        self._span_selector = None
        self._mode: str     = "idle"

        # dark_map: {dataset_label: dark_info_dict | None}
        # dark_info_dict = {label, path, wl, mean (shape n_wl_dark)}
        self._dark_map: dict = {}

        # Power calibration state
        self._cal_atbs_data     = None   # (hwp_arr, powers_W) or None
        self._cal_atsample_data = None   # (hwp_arr, powers_W) or None
        self._cal_atbs_path     = None   # str or None
        self._cal_atsample_path = None   # str or None

        # Baseline subtraction state — applied at the very end, after
        # stitching and dark subtraction (see _compute_stitch()). One
        # constant per column (power step / angle: mean counts over a
        # user-picked range, fit independently from each column's own data
        # and subtracted only from that column) — all columns stay visible
        # while the window is picked, and every column gets its own fit.
        self._baseline_range   = None   # (lo_wl, hi_wl) nm, or None until confirmed
        self._baseline_value   = None   # (n_cols,) fitted constants, or None
        self._baseline_span    = None   # DraggableSpan, active during selection
        self._baseline_pending_range = None   # display-domain (xmin,xmax), pending confirm
        self._baseline_pending_value = None
        self._baseline_preview_item  = None   # dashed constant-level line during selection
        self._baseline_stitch_cache  = None   # (wl_out, counts_diff, cols) — pre-baseline,
                                                   # captured when selection starts
        self._baseline_prev_mode     = None   # _mode to restore on cancel

        # ── Canvas-side action bars (placed by the host tab) ─────────
        # Pair-navigation bar (only shown while selecting transition spans)
        self._nav_bar = QFrame()
        self._nav_bar.setFrameShape(QFrame.StyledPanel)
        nav = QHBoxLayout(self._nav_bar)
        nav.setContentsMargins(8, 4, 8, 4)
        self._btn_prev  = QPushButton("◀  Prev")
        self._lbl_pair  = QLabel("Pair 1 of 1")
        self._lbl_pair.setAlignment(Qt.AlignCenter)
        self._btn_next  = QPushButton("Next  ▶")
        self._btn_done  = QPushButton("✓  Done")
        self._btn_done.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; }"
        )
        self._btn_prev.clicked.connect(self._on_prev_pair)
        self._btn_next.clicked.connect(self._on_next_pair)
        self._btn_done.clicked.connect(self._on_done_spans)
        for w in (self._btn_prev, self._lbl_pair, self._btn_next, self._btn_done):
            nav.addWidget(w)
        self._nav_bar.setVisible(False)

        # Baseline-selection action bar (Confirm/Cancel), shared pattern
        # with the TRPL tab's baseline subtraction — shown only while
        # self._mode == "baseline_selecting".
        self._baseline_bar = QFrame()
        self._baseline_bar.setFrameShape(QFrame.StyledPanel)
        bb = QHBoxLayout(self._baseline_bar)
        bb.setContentsMargins(8, 4, 8, 4)
        self._baseline_action_lbl = QLabel("")
        self._btn_baseline_confirm = QPushButton("Confirm baseline")
        self._btn_baseline_cancel  = QPushButton("Cancel")
        self._btn_baseline_confirm.clicked.connect(self._on_baseline_confirm)
        self._btn_baseline_cancel.clicked.connect(self._on_baseline_cancel)
        bb.addWidget(self._baseline_action_lbl, stretch=1)
        bb.addWidget(self._btn_baseline_confirm)
        bb.addWidget(self._btn_baseline_cancel)
        self._baseline_bar.setVisible(False)

        self._build_page()
        self._refresh_buttons()

    def _build_page(self):
        sl = QVBoxLayout(self)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # Files group
        g_files = QGroupBox("Input files")
        fl = QVBoxLayout(g_files)
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list.setToolTip("Loaded .origin files (sorted by wavelength range)")
        fl.addWidget(self._file_list)
        fb = QHBoxLayout()
        self._btn_add    = QPushButton("Add…")
        self._btn_remove = QPushButton("Remove")
        self._btn_clear  = QPushButton("Clear")
        self._btn_add.clicked.connect(self._on_add_files)
        self._btn_remove.clicked.connect(self._on_remove_files)
        self._btn_clear.clicked.connect(self._on_clear_files)
        for b in (self._btn_add, self._btn_remove, self._btn_clear):
            fb.addWidget(b)
        fl.addLayout(fb)
        g_files.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_files)

        # X-axis group
        g_xaxis = QGroupBox("X axis")
        xl = QHBoxLayout(g_xaxis)
        self._rb_energy = QRadioButton("Energy (eV)")
        self._rb_wl     = QRadioButton("Wavelength (nm)")
        self._rb_energy.setChecked(True)
        self._rb_energy.toggled.connect(self._on_xaxis_changed)
        xl.addWidget(self._rb_energy)
        xl.addWidget(self._rb_wl)
        sl.addWidget(g_xaxis)

        # Transition spans group
        g_spans = QGroupBox("Transition spans")
        spl = QVBoxLayout(g_spans)
        self._span_list = QListWidget()
        self._span_list.setToolTip(
            "One span per overlapping pair.\nClick a row to jump to that pair."
        )
        self._span_list.itemClicked.connect(self._on_span_item_clicked)
        spl.addWidget(self._span_list)
        sl.addWidget(g_spans, stretch=1)

        # Dark spectra group
        g_dark = QGroupBox("Dark spectra")
        dl = QVBoxLayout(g_dark)
        self._dark_list = QListWidget()
        self._dark_list.setToolTip(
            "Dark spectrum matched to each input file.\n"
            "Select a row and click \"Load…\" to set manually."
        )
        self._dark_list.setMaximumHeight(110)
        dl.addWidget(self._dark_list)
        db = QHBoxLayout()
        self._btn_autosearch = QPushButton("Autosearch")
        self._btn_load_dark  = QPushButton("Load…")
        self._btn_clear_dark = QPushButton("Clear")
        self._btn_autosearch.setToolTip(
            "Search for files with 'dark' in the name in each\n"
            "input directory or its 'dark' subfolder.\n"
            "Matches by integration time and center wavelength."
        )
        self._btn_autosearch.clicked.connect(self._on_autosearch_dark)
        self._btn_load_dark.clicked.connect(self._on_load_dark)
        self._btn_clear_dark.clicked.connect(self._on_clear_dark)
        for b in (self._btn_autosearch, self._btn_load_dark, self._btn_clear_dark):
            db.addWidget(b)
        dl.addLayout(db)
        g_dark.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_dark)

        # Metadata group
        g_meta = QGroupBox("Metadata (saved to HDF5)")
        ml = QVBoxLayout(g_meta)
        r_spot = QHBoxLayout()
        r_spot.addWidget(QLabel("Spot diameter (µm):"))
        self._meta_spot_diam = QLineEdit()
        self._meta_spot_diam.setPlaceholderText("optional")
        r_spot.addWidget(self._meta_spot_diam)
        ml.addLayout(r_spot)
        r_rep = QHBoxLayout()
        r_rep.addWidget(QLabel("Rep. rate (MHz):"))
        self._meta_rep_rate = QLineEdit()
        self._meta_rep_rate.setPlaceholderText("optional")
        r_rep.addWidget(self._meta_rep_rate)
        ml.addLayout(r_rep)
        sl.addWidget(g_meta)

        # Power calibration group
        g_cal = QGroupBox("Power calibration")
        cl = QVBoxLayout(g_cal)

        r_atbs = QHBoxLayout()
        r_atbs.addWidget(QLabel("atBS:"))
        self._cal_atbs_lbl = QLabel("—")
        self._cal_atbs_lbl.setWordWrap(False)
        self._cal_atbs_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        r_atbs.addWidget(self._cal_atbs_lbl, stretch=1)
        btn_load_atbs = QPushButton("Load…")
        btn_load_atbs.setFixedWidth(64)
        btn_load_atbs.clicked.connect(self._on_load_cal_atbs)
        r_atbs.addWidget(btn_load_atbs)
        cl.addLayout(r_atbs)

        r_ats = QHBoxLayout()
        r_ats.addWidget(QLabel("atSample:"))
        self._cal_atsample_lbl = QLabel("—")
        self._cal_atsample_lbl.setWordWrap(False)
        self._cal_atsample_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        r_ats.addWidget(self._cal_atsample_lbl, stretch=1)
        btn_load_ats = QPushButton("Load…")
        btn_load_ats.setFixedWidth(64)
        btn_load_ats.clicked.connect(self._on_load_cal_atsample)
        r_ats.addWidget(btn_load_ats)
        cl.addLayout(r_ats)

        btn_cal_auto = QPushButton("Autosearch")
        btn_cal_auto.clicked.connect(self._on_autosearch_cal)
        cl.addWidget(btn_cal_auto)
        g_cal.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_cal)

        # Baseline subtraction group — applied at the very end, after
        # stitching and dark subtraction. Same UX as the TRPL tab's
        # baseline subtraction: drag a range, a constant (mean counts) is
        # fit over it and subtracted from every count.
        g_baseline = QGroupBox("Baseline subtraction")
        bll = QVBoxLayout(g_baseline)
        self._btn_baseline_select = QPushButton("Select baseline range…")
        self._btn_baseline_select.setToolTip(
            "Drag a range with no signal, on the stitched/dark-subtracted\n"
            f"result (all {self._step_word}s shown). A separate constant (mean\n"
            f"counts over the range) is fit per {self._step_word}, from its\n"
            "own data, and subtracted from just that spectrum."
        )
        self._btn_baseline_select.setEnabled(False)
        self._btn_baseline_select.clicked.connect(self._on_start_baseline_select)
        bll.addWidget(self._btn_baseline_select)
        self._btn_baseline_reset = QPushButton("Reset baseline")
        self._btn_baseline_reset.setEnabled(False)
        self._btn_baseline_reset.clicked.connect(self._on_reset_baseline)
        bll.addWidget(self._btn_baseline_reset)
        self._baseline_status_lbl = QLabel("No baseline applied.")
        self._baseline_status_lbl.setWordWrap(True)
        bll.addWidget(self._baseline_status_lbl)
        g_baseline.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_baseline)

        # Action buttons
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        self._btn_preview     = QPushButton("1  Preview spectra")
        self._btn_spans       = QPushButton("2  Select spans")
        self._btn_prev_result = QPushButton("3  Preview result")
        self._btn_stitch      = QPushButton("4  Save as HDF5…")
        self._btn_save_origin = QPushButton("5  Save as .origin…")
        self._btn_preview.clicked.connect(self._on_preview)
        self._btn_spans.clicked.connect(self._on_start_spans)
        self._btn_prev_result.clicked.connect(self._on_preview_result)
        self._btn_stitch.clicked.connect(self._on_stitch_save)
        self._btn_save_origin.clicked.connect(self._on_save_origin)

        bold = QFont(); bold.setBold(True)
        self._btn_stitch.setFont(bold)
        self._btn_stitch.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; }"
            "QPushButton:disabled { background-color: #bbbbbb; color: #666666; }"
        )
        for b in (self._btn_preview, self._btn_spans, self._btn_prev_result,
                  self._btn_stitch, self._btn_save_origin):
            b.setMinimumHeight(32)
            sl.addWidget(b)
        self._btn_save_origin.setVisible(self._kind == "ps")


    # ── file management ─────────────────────────────────────

    def _on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select .origin files", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        added = 0
        for p in paths:
            if p not in self._files:
                self._files.append(p)
                item = QListWidgetItem(os.path.basename(p))
                item.setToolTip(p)
                self._file_list.addItem(item)
                added += 1
        if added:
            self._reset_parsed()
            self._refresh_buttons()
            self._status(
                f"{len(self._files)} file(s) loaded. Click \"Preview\" to continue."
            )

    def _on_remove_files(self):
        rows = sorted(
            [self._file_list.row(i) for i in self._file_list.selectedItems()],
            reverse=True,
        )
        for row in rows:
            self._file_list.takeItem(row)
            if row < len(self._files):
                self._files.pop(row)
        self._reset_parsed()
        self._refresh_buttons()

    def _on_clear_files(self):
        self._files.clear()
        self._file_list.clear()
        self._reset_parsed()
        self._clear_canvas()
        self._refresh_buttons()
        self._status("Files cleared.")

    def _reset_parsed(self):
        self._datasets.clear()
        self._hdrs.clear()
        self._pairs.clear()
        self._spans.clear()
        self._spans_display.clear()
        self._span_list.clear()
        self._dark_map.clear()
        self._dark_list.clear()
        self._span_selector = None
        self._mode = "idle"
        self._nav_bar.setVisible(False)
        self._reset_baseline_state()

    def _reset_baseline_state(self):
        """Discard any baseline selection (confirmed or in progress) — the
        input files/stitch changed, so a previously-fit constant no longer
        applies. Called whenever the input file set changes."""
        self._baseline_range = None
        self._baseline_value = None
        self._baseline_pending_range = None
        self._baseline_pending_value = None
        self._baseline_stitch_cache = None
        if self._baseline_span is not None:
            self._baseline_span.deactivate()
            self._baseline_span = None
        self._remove_baseline_preview_item()
        self._baseline_bar.setVisible(False)
        self._btn_baseline_reset.setEnabled(False)
        self._baseline_status_lbl.setText("No baseline applied.")

    # ── X-axis toggle ────────────────────────────────────────

    def _on_xaxis_changed(self):
        new = "energy" if self._rb_energy.isChecked() else "wavelength"
        if new == self._x_axis:
            return
        converted = {}
        for key, (x0, x1) in self._spans_display.items():
            a, b = _HC_EV_NM / x0, _HC_EV_NM / x1
            converted[key] = (min(a, b), max(a, b))
        self._x_axis = new
        self._spans_display = converted
        if self._mode == "preview":
            self._draw_preview()
        elif self._mode == "spans":
            self._draw_pair(self._pair_idx)
        self._refresh_span_list()

    # ── parsing ──────────────────────────────────────────────

    def _parse_all(self) -> bool:
        """Parse all files; return True if at least 1 parsed successfully."""
        self._datasets.clear()
        self._hdrs.clear()
        errors = []
        for path in self._files:
            label = os.path.splitext(os.path.basename(path))[0]
            try:
                wl, counts, cols, hdr = self._parse_series(path)
                self._datasets.append({
                    "label": label, "wl": wl, "counts": counts,
                    "cols": cols, "path": path,
                })
                self._hdrs.append(hdr)
            except Exception as exc:
                errors.append(f"  {label}: {exc}")
        if errors:
            QMessageBox.warning(self, "Parse errors",
                                "Could not parse:\n" + "\n".join(errors))
        if not self._datasets:
            QMessageBox.warning(self, "No valid files",
                                "None of the selected files could be parsed.")
            return False
        order = sorted(range(len(self._datasets)),
                       key=lambda i: self._datasets[i]["wl"].min())
        self._datasets = [self._datasets[i] for i in order]
        self._hdrs     = [self._hdrs[i]     for i in order]
        self._refresh_dark_list()
        return True

    def _parse_series(self, path):
        """Return (wl_nm, counts (n_wl, n_cols), column values, header dict)
        for one input file — column values are excitation powers (W) for
        "ps", polarization angles (degrees) for "pol"."""
        if self._kind == "pol":
            p = _parse_pol_origin(path)
            return p["wl"], p["counts"], p["angles"], p["hdr"]
        wl, counts, powers = _parse_origin_power_series(path)
        return wl, counts, powers, _parse_origin_header(path)

    def _find_pairs(self) -> bool:
        self._pairs.clear()
        for i in range(len(self._datasets) - 1):
            a, b = self._datasets[i], self._datasets[i + 1]
            ov_lo = max(a["wl"].min(), b["wl"].min())
            ov_hi = min(a["wl"].max(), b["wl"].max())
            if ov_hi > ov_lo:
                self._pairs.append((i, i + 1, ov_lo, ov_hi))
        self._refresh_span_list()
        return bool(self._pairs)

    def _refresh_span_list(self):
        self._span_list.clear()
        for k, (i_a, i_b, _, _) in enumerate(self._pairs):
            a, b = self._datasets[i_a], self._datasets[i_b]
            key = (i_a, i_b)
            if key in self._spans:
                lo_wl, hi_wl = self._spans[key]
                xmin = self._wl_to_x(hi_wl if self._x_axis == "energy" else lo_wl)
                xmax = self._wl_to_x(lo_wl if self._x_axis == "energy" else hi_wl)
                if xmin > xmax:
                    xmin, xmax = xmax, xmin
                text  = f"✅  {a['label']} ↔ {b['label']}\n    [{xmin:.4g} … {xmax:.4g}]"
                color = QColor("#c8e6c9")
            else:
                text  = f"⬜  {a['label']} ↔ {b['label']}"
                color = QColor("#ffffff")
            item = QListWidgetItem(text)
            item.setBackground(color)
            item.setData(Qt.UserRole, k)
            self._span_list.addItem(item)

    # ── dark spectra ─────────────────────────────────────────

    def _parse_dark_file(self, path: str):
        """Parse a .origin file as a dark spectrum. Returns info dict or None."""
        try:
            wl, counts, hdr = _parse_any_series(path)
            center_nm, disp_nm = _parse_header_center_disp(hdr)
            return {
                "path":       path,
                "label":      os.path.splitext(os.path.basename(path))[0],
                "wl":         wl,
                "mean":       counts.mean(axis=1),   # (n_wl,) — mean over columns
                "int_time":   hdr.get("int_time_str", ""),
                "center_nm":  center_nm,
                "disp_nm":    disp_nm,
            }
        except Exception:
            return None

    def _darks_match(self, sig_hdr: dict, dark_info: dict) -> bool:
        """Return True if dark_info is compatible with the signal header."""
        sig_int   = sig_hdr.get("int_time_str", "")
        sig_c, sig_d = _parse_header_center_disp(sig_hdr)
        if sig_int != dark_info["int_time"]:
            return False
        if sig_c is None or dark_info["center_nm"] is None:
            return False
        if abs(sig_c - dark_info["center_nm"]) > 5.0:
            return False
        if sig_d is not None and dark_info["disp_nm"] is not None:
            if abs(sig_d - dark_info["disp_nm"]) > 5.0:
                return False
        return True

    def _autosearch_darks(self, force: bool = False):
        """Search for dark files in input directories and match them.

        force=False (default): skip datasets that already have a dark assigned
                               so manual assignments are never overwritten.
        force=True:            overwrite all entries (explicit Autosearch button).
        """
        # Collect unique directories
        dirs = list(dict.fromkeys(os.path.dirname(p) for p in self._files))

        # Find candidate .origin files with "dark" in name, or in dark/ subfolder
        candidates = []
        seen = set()
        for d in dirs:
            try:
                for fname in os.listdir(d):
                    if "dark" in fname.lower() and fname.lower().endswith(".origin"):
                        fp = os.path.join(d, fname)
                        if fp not in seen:
                            candidates.append(fp)
                            seen.add(fp)
            except OSError:
                pass
            dark_sub = os.path.join(d, "dark")
            if os.path.isdir(dark_sub):
                try:
                    for fname in os.listdir(dark_sub):
                        if "dark" in fname.lower() and fname.lower().endswith(".origin"):
                            fp = os.path.join(dark_sub, fname)
                            if fp not in seen:
                                candidates.append(fp)
                                seen.add(fp)
                except OSError:
                    pass

        # Parse candidates
        parsed = [self._parse_dark_file(p) for p in candidates]
        parsed = [d for d in parsed if d is not None]

        # Match to each dataset
        n_matched = 0
        for dataset, hdr in zip(self._datasets, self._hdrs):
            label = dataset["label"]
            if not force and self._dark_map.get(label) is not None:
                # Preserve existing assignment (manual or previous auto-match)
                n_matched += 1
                continue
            best = next((d for d in parsed if self._darks_match(hdr, d)), None)
            self._dark_map[label] = best
            if best is not None:
                n_matched += 1

        self._refresh_dark_list()
        self._refresh_buttons()
        if not candidates:
            self._status(
                "Autosearch: no dark files found (looking for 'dark' in name "
                "or files in a 'dark' subfolder)."
            )
        else:
            self._status(
                f"Autosearch: {len(candidates)} candidate(s) found, "
                f"{n_matched}/{len(self._datasets)} file(s) matched."
            )

    def _on_autosearch_dark(self):
        if not self._datasets:
            if not self._parse_all():
                return
            self._find_pairs()
        self._autosearch_darks(force=True)

    def _on_load_dark(self):
        """Manually assign a dark file to the selected entry in the dark list."""
        row = self._dark_list.currentRow()
        if row < 0 or row >= len(self._datasets):
            QMessageBox.information(self, "No selection",
                                    "Select a file in the dark list first.")
            return
        dataset = self._datasets[row]
        sig_hdr = self._hdrs[row]

        path, _ = QFileDialog.getOpenFileName(
            self, f"Dark file for '{dataset['label']}'", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if not path:
            return

        dark_info = self._parse_dark_file(path)
        if dark_info is None:
            QMessageBox.critical(self, "Parse error",
                                 f"Could not parse:\n{path}")
            return

        if not self._darks_match(sig_hdr, dark_info):
            sig_int       = sig_hdr.get("int_time_str", "?")
            sig_c, sig_d  = _parse_header_center_disp(sig_hdr)
            reply = QMessageBox.question(
                self, "Metadata mismatch",
                f"The selected dark file does not match on:\n"
                f"  integration time: signal={sig_int}  dark={dark_info['int_time']}\n"
                f"  center wavelength: signal={sig_c}  dark={dark_info['center_nm']}\n"
                f"  dispersion: signal={sig_d}  dark={dark_info['disp_nm']}\n\n"
                "Assign it anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self._dark_map[dataset["label"]] = dark_info
        self._refresh_dark_list()
        self._refresh_buttons()
        self._status(
            f"Dark assigned: {dataset['label']} ← {dark_info['label']}"
        )

    def _on_clear_dark(self):
        self._dark_map.clear()
        self._refresh_dark_list()
        self._refresh_buttons()
        self._status("Dark spectra cleared.")

    # ── power calibration ─────────────────────────────────────

    def _cal_load(self, path, role):
        """Parse a power-calibration file and store it under *role* ('atBS'/'atSample')."""
        try:
            hwp, pows = _parse_power_calibration(path)
        except Exception as exc:
            QMessageBox.critical(self, "Calibration load error",
                                 f"Could not parse:\n{path}\n\n{exc}")
            return False
        short = os.path.basename(path)
        if len(short) > 30:
            short = short[:27] + "…"
        if role == "atBS":
            self._cal_atbs_data  = (hwp, pows)
            self._cal_atbs_path  = path
            self._cal_atbs_lbl.setText(short)
        else:
            self._cal_atsample_data  = (hwp, pows)
            self._cal_atsample_path  = path
            self._cal_atsample_lbl.setText(short)
        self._status(
            f"Calibration {role}: {len(hwp)} HWP steps loaded from {short}"
        )
        return True

    def _on_load_cal_atbs(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select atBS calibration file", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if path:
            self._cal_load(path, "atBS")

    def _on_load_cal_atsample(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select atSample calibration file", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if path:
            self._cal_load(path, "atSample")

    def _on_autosearch_cal(self):
        """Search input-file directories (and a 'Calibration' subdirectory) for
        calibration files whose names contain 'atbs' or 'atsample' (case-insensitive).
        """
        if not self._files:
            QMessageBox.information(self, "No files loaded",
                                    "Add .origin files first.")
            return

        dirs = list(dict.fromkeys(os.path.dirname(p) for p in self._files))

        search_dirs = []
        seen_dirs   = set()
        def _add_with_cal_subfolder(d, include_self=True):
            if include_self and d not in seen_dirs:
                search_dirs.append(d)
                seen_dirs.add(d)
            # Find a 'Calibration' subfolder (case-insensitive)
            try:
                for entry in os.listdir(d):
                    if entry.lower() == "calibration":
                        sub = os.path.join(d, entry)
                        if os.path.isdir(sub) and sub not in seen_dirs:
                            search_dirs.append(sub)
                            seen_dirs.add(sub)
            except OSError:
                pass

        for d in dirs:
            _add_with_cal_subfolder(d)
        # Searched last, so a same-level match always wins: a 'Calibration'
        # folder next to the measurement folder (e.g. <day>/Calibration
        # beside <day>/<sample>_pol/*.origin).
        for d in dirs:
            _add_with_cal_subfolder(os.path.dirname(d), include_self=False)

        atbs_path = atsample_path = None
        for d in search_dirs:
            try:
                for fname in os.listdir(d):
                    if not fname.lower().endswith(".origin"):
                        continue
                    fl = fname.lower()
                    fp = os.path.join(d, fname)
                    if atbs_path is None and "atbs" in fl:
                        atbs_path = fp
                    if atsample_path is None and "atsample" in fl:
                        atsample_path = fp
            except OSError:
                pass
            if atbs_path and atsample_path:
                break

        found = []
        if atbs_path:
            if self._cal_load(atbs_path, "atBS"):
                found.append(f"atBS: {os.path.basename(atbs_path)}")
        if atsample_path:
            if self._cal_load(atsample_path, "atSample"):
                found.append(f"atSample: {os.path.basename(atsample_path)}")

        if found:
            self._status(
                "Calibration autosearch found: " + ", ".join(found)
            )
        else:
            self._status(
                "Calibration autosearch: no matching files found "
                "(looking for 'atBS'/'atSample' in name in input dirs and a 'Calibration/' "
                "subfolder of them or of their parent)."
            )

    def _refresh_dark_list(self):
        self._dark_list.clear()
        for dataset in self._datasets:
            dark_info = self._dark_map.get(dataset["label"])
            if dark_info is not None:
                text  = f"✅  {dataset['label']}\n    ← {dark_info['label']}"
                color = QColor("#c8e6c9")
            else:
                text  = f"⬜  {dataset['label']}  ← not found"
                color = QColor("#ffffff")
            item = QListWidgetItem(text)
            item.setBackground(color)
            self._dark_list.addItem(item)

    def _apply_dark(self, datasets: list) -> tuple:
        """Subtract matched dark from each dataset's counts index-by-index.

        Returns (ds_sub, dark_by_label) where:
          ds_sub         — list of dataset dicts with counts replaced by dark-subtracted values
          dark_by_label  — {label: dark array aligned to file pixel grid | None}
        """
        ds_sub        = []
        dark_by_label = {}
        for d in datasets:
            dark_info = self._dark_map.get(d["label"])
            if dark_info is not None:
                n_sig  = len(d["wl"])
                n_dark = len(dark_info["mean"])
                if n_dark >= n_sig:
                    dark_arr = dark_info["mean"][:n_sig]
                else:
                    # Dark is shorter: pad with zeros (no subtraction for missing pixels)
                    dark_arr = np.zeros(n_sig)
                    dark_arr[:n_dark] = dark_info["mean"]
                counts_sub = d["counts"] - dark_arr[:, np.newaxis]
                ds_sub.append({**d, "counts": counts_sub})
                dark_by_label[d["label"]] = dark_arr
            else:
                ds_sub.append(d)
                dark_by_label[d["label"]] = None
        return ds_sub, dark_by_label

    # ── step 1 — Preview ─────────────────────────────────────

    def _on_preview(self):
        if not self._parse_all():
            return
        self._find_pairs()
        self._autosearch_darks()
        self._mode = "preview"
        self._nav_bar.setVisible(False)
        self._draw_preview()
        self._refresh_buttons()
        n = len(self._datasets)
        n_dark = sum(1 for v in self._dark_map.values() if v is not None)
        dark_note = f", {n_dark}/{n} dark(s) matched" if n_dark else ""
        if n == 1:
            self._status(
                f"1 file loaded{dark_note}. Click \"Save as HDF5\" to convert."
            )
        else:
            self._status(
                f"{n} file(s), {len(self._pairs)} overlapping pair(s){dark_note}."
            )

    def _draw_preview(self):
        ds_plot, dark_by_label = self._apply_dark(self._datasets)
        n_dark  = sum(1 for v in dark_by_label.values() if v is not None)
        subtitle = f"dark-subtracted: {n_dark}/{len(ds_plot)}" if n_dark else "raw"

        ax = self._canvas.reset_axes()
        ax.addLegend(labelTextSize="7pt")
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        for k, d in enumerate(ds_plot):
            x   = self._wl_to_x(d["wl"])
            idx = np.argsort(x)
            mlp.plot(x[idx], d["counts"][idx, -1], index=k, width=1.3,
                     label=f"{d['label']}  ({self._fmt_col(d['cols'][-1])})")
        ax.setLabel("bottom", "Energy (eV)" if self._x_axis == "energy" else "Wavelength (nm)")
        ax.setLabel("left", "Counts")
        ax.showGrid(x=True, y=True, alpha=0.3)
        ax.setTitle(f"Last-{self._step_word} spectra (preview — {subtitle})")
        self._canvas.draw_idle()

    # ── step 2 — Select spans ─────────────────────────────────

    def _on_start_spans(self):
        if not self._datasets:
            if not self._parse_all():
                return
        if not self._pairs and not self._find_pairs():
            QMessageBox.information(self, "No overlaps",
                                    "No overlapping wavelength ranges found.")
            return
        self._mode = "spans"
        self._pair_idx = 0
        self._nav_bar.setVisible(True)
        self._draw_pair(0)
        self._refresh_buttons()
        self._status(
            "Drag to select a transition span for each pair. "
            "Use ◀/▶ to navigate, ✓ Done when finished."
        )

    def _draw_pair(self, idx: int):
        if not (0 <= idx < len(self._pairs)):
            return
        i_a, i_b, ov_lo_wl, ov_hi_wl = self._pairs[idx]
        a, b = self._datasets[i_a], self._datasets[i_b]

        if self._span_selector is not None:
            self._span_selector.deactivate()
            self._span_selector = None

        ds_sub, _ = self._apply_dark([a, b])
        a_plot, b_plot = ds_sub

        ax = self._canvas.reset_axes()
        ax.addLegend(labelTextSize="8pt")
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        for k, d in enumerate((a_plot, b_plot)):
            mask = (d["wl"] >= ov_lo_wl) & (d["wl"] <= ov_hi_wl)
            x_ov = self._wl_to_x(d["wl"][mask])
            c_ov = d["counts"][mask, -1]
            s    = np.argsort(x_ov)
            mlp.plot(x_ov[s], c_ov[s], index=k, width=1.4, label=d["label"])

        x_lo = self._wl_to_x(ov_hi_wl if self._x_axis == "energy" else ov_lo_wl)
        x_hi = self._wl_to_x(ov_lo_wl if self._x_axis == "energy" else ov_hi_wl)
        if x_lo > x_hi:
            x_lo, x_hi = x_hi, x_lo
        ax.setXRange(x_lo, x_hi, padding=0)
        ax.setLabel("bottom", "Energy (eV)" if self._x_axis == "energy" else "Wavelength (nm)")
        ax.setLabel("left", "Counts")
        ax.showGrid(x=True, y=True, alpha=0.3)
        ax.setTitle(
            f"Pair {idx + 1}/{len(self._pairs)}: {a['label']}  ↔  {b['label']}\n"
            "Drag the shaded region's edges to select the transition span"
        )
        self._canvas.draw_idle()

        key = (i_a, i_b)
        stored = self._spans_display.get(key)
        mid = (x_lo + x_hi) / 2.0
        width = (x_hi - x_lo) * 0.1
        initial = stored if stored is not None else (mid - width / 2, mid + width / 2)

        self._span_selector = DraggableSpan(ax, color=(0, 150, 0, 60), movable=True)
        self._span_selector.activate(initial_range=initial, bounds=(x_lo, x_hi))
        self._span_selector.sigRegionSelected.connect(
            lambda xmin, xmax, _idx=idx: self._on_span_selected(xmin, xmax, _idx)
        )
        self._lbl_pair.setText(f"Pair {idx + 1} of {len(self._pairs)}")
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < len(self._pairs) - 1)
        self._span_list.setCurrentRow(idx)

    def _on_span_selected(self, xmin: float, xmax: float, pair_idx: int):
        if pair_idx != self._pair_idx or abs(xmax - xmin) < 1e-12:
            return
        i_a, i_b, _, _ = self._pairs[pair_idx]
        key = (i_a, i_b)
        self._spans_display[key] = (min(xmin, xmax), max(xmin, xmax))
        if self._x_axis == "energy":
            lo_wl = min(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
            hi_wl = max(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
        else:
            lo_wl, hi_wl = min(xmin, xmax), max(xmin, xmax)
        self._spans[key] = (lo_wl, hi_wl)
        self._refresh_span_list()
        self._status(
            f"Pair {pair_idx + 1}: span [{xmin:.5g}, {xmax:.5g}] "
            f"({'eV' if self._x_axis == 'energy' else 'nm'})."
        )

    def _on_span_item_clicked(self, item):
        if self._mode != "spans":
            return
        idx = item.data(Qt.UserRole)
        if idx is not None and 0 <= idx < len(self._pairs):
            self._pair_idx = idx
            self._draw_pair(idx)

    def _on_prev_pair(self):
        if self._pair_idx > 0:
            self._pair_idx -= 1
            self._draw_pair(self._pair_idx)

    def _on_next_pair(self):
        if self._pair_idx < len(self._pairs) - 1:
            self._pair_idx += 1
            self._draw_pair(self._pair_idx)

    def _on_done_spans(self):
        if self._span_selector is not None:
            self._span_selector.deactivate()
            self._span_selector = None
        self._nav_bar.setVisible(False)
        self._mode = "done"
        self._refresh_buttons()
        n_sel, n_req = len(self._spans), len(self._pairs)
        if n_sel < n_req:
            self._status(
                f"Span selection done: {n_sel}/{n_req} spans defined. "
                f"{n_req - n_sel} missing pair(s) will use range midpoint."
            )
        else:
            self._status(f"All {n_sel} span(s) defined.")
        self._draw_preview()

    # ── steps 3–5 — Preview / Save ────────────────────────────

    def _compute_stitch(self):
        """_compute_stitch_core() plus baseline subtraction — the very
        last step, applied after stitching and dark subtraction, if a
        baseline has been confirmed (see _on_baseline_confirm()). This is
        the version everything downstream (previews, both Save actions)
        should call; the baseline selector itself calls
        _compute_stitch_core() directly so it always starts from the
        pre-baseline data, never compounding an already-applied baseline.

        self._baseline_value is a (n_powers,) array, one constant per
        power step — each subtracted only from its own column, not one
        shared value applied to every step (relies on numpy broadcasting a
        (n_powers,) array against counts_diff's (n_wl, n_powers) shape).
        """
        result = self._compute_stitch_core()
        if result is None:
            return None
        wl_out, counts_diff, counts_raw, ds_raw, best_hdr, dark_by_label = result
        if self._baseline_value is not None:
            counts_diff = counts_diff - self._baseline_value
        return wl_out, counts_diff, counts_raw, ds_raw, best_hdr, dark_by_label

    def _compute_stitch_core(self):
        """Build output data. Dark subtraction (if assigned) always precedes stitching.

        Returns (wl_out, counts_diff, counts_raw, ds_raw, best_hdr, dark_by_label)
        or None.
          counts_diff   — dark-subtracted, min shifted to 1 (no baseline subtraction —
                          see _compute_stitch(), the wrapper everything else calls)
          counts_raw    — stitched without dark subtraction, no min-shift
          ds_raw        — original (un-subtracted) source datasets
          dark_by_label — {label: dark_mean_on_file_wl | None}
        """
        if not self._datasets:
            if not self._parse_all():
                return None
            self._find_pairs()

        # ── Single file ──────────────────────────────────────────
        if len(self._datasets) == 1:
            d = self._datasets[0]
            ds_raw                = [d]
            ds_sub, dark_by_label = self._apply_dark(ds_raw)
            wl_out                = ds_sub[0]["wl"]
            counts_diff           = ds_sub[0]["counts"].copy()
            counts_diff           = counts_diff - counts_diff.min() + 1.0
            counts_raw            = d["counts"].copy()
            return wl_out, counts_diff, counts_raw, ds_raw, self._hdrs[0].copy(), dark_by_label

        # ── Multiple files ────────────────────────────────────────
        n_p_list = [d["counts"].shape[1] for d in self._datasets]
        n_p      = min(n_p_list)
        if len(set(n_p_list)) > 1:
            QMessageBox.information(
                self, f"{self._step_word.capitalize()} mismatch",
                f"Files have different {self._step_word} counts: {n_p_list}.\n"
                f"All will be truncated to {n_p} steps.",
            )
        ds_raw = [
            {"label": d["label"], "wl": d["wl"],
             "counts": d["counts"][:, :n_p], "cols": d["cols"][:n_p],
             "path": d["path"]}
            for d in self._datasets
        ]
        if self._kind == "pol":
            # Stitching combines column k of every file, which only makes
            # sense if column k is the same polarization angle in each.
            ref = ds_raw[0]["cols"]
            bad = [d["label"] for d in ds_raw[1:] if not np.allclose(d["cols"], ref)]
            if bad:
                QMessageBox.critical(
                    self, "Angles differ",
                    "Cannot stitch: these files were measured at different "
                    "polarization angles than "
                    f"{ds_raw[0]['label']} ({ref.min():g}–{ref.max():g}°):\n\n"
                    + "\n".join(bad)
                )
                return None

        # Build spans_wl
        spans_wl = []
        for k, (i_a, i_b, ov_lo_wl, ov_hi_wl) in enumerate(self._pairs):
            key = (i_a, i_b)
            if key in self._spans:
                spans_wl.append(self._spans[key])
            else:
                mid = (ov_lo_wl + ov_hi_wl) / 2.0
                spans_wl.append((mid, mid))

        # Stitch dark-subtracted spectrum
        ds_for_stitch, dark_by_label = self._apply_dark(ds_raw)
        try:
            wl_out, counts_diff = _stitch_counts(ds_for_stitch, spans_wl)
        except Exception as exc:
            QMessageBox.critical(self, "Stitch error", str(exc))
            return None

        # Stitch raw spectrum (no dark subtraction)
        try:
            _, counts_raw = _stitch_counts(ds_raw, spans_wl)
        except Exception:
            counts_raw = counts_diff.copy()

        # Best header (earliest date)
        best_hdr  = self._hdrs[0].copy()
        best_date = best_hdr.get("date")
        for h in self._hdrs[1:]:
            d = h.get("date")
            if d is not None and (best_date is None or d < best_date):
                best_date = d
                best_hdr["date_str"] = h.get("date_str", best_hdr.get("date_str", ""))

        counts_diff = counts_diff - counts_diff.min() + 1.0
        return wl_out, counts_diff, counts_raw, ds_raw, best_hdr, dark_by_label

    def _on_preview_result(self):
        result = self._compute_stitch()
        if result is None:
            return
        wl_out, counts_out, _counts_raw, ds, _, dark_by_label = result
        n_dark      = sum(1 for v in dark_by_label.values() if v is not None)
        is_stitched = len(ds) > 1
        dark_note   = f" (dark-subtracted: {n_dark}/{len(ds)})" if n_dark else ""
        base_title  = "Stitched spectrum" if is_stitched else "Spectrum"
        self._draw_stitched(wl_out, counts_out, ds[0]["cols"],
                               title=base_title + dark_note)
        self._status(
            f"Preview: {wl_out.size} wavelengths, "
            f"{counts_out.shape[1]} {self._step_word}s{dark_note}."
        )

    def _on_stitch_save(self):
        result = self._compute_stitch()
        if result is None:
            return
        wl_out, counts_diff, counts_raw, ds, best_hdr, dark_by_label = result
        n_p          = counts_diff.shape[1]
        is_stitched  = len(ds) > 1
        n_dark       = sum(1 for v in dark_by_label.values() if v is not None)

        first_name = os.path.splitext(os.path.basename(ds[0]["path"]))[0]
        default_path = os.path.join(os.path.dirname(ds[0]["path"]), first_name + ".h5")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save as HDF5", default_path,
            "HDF5 files (*.h5);;All files (*.*)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".h5"):
            out_path += ".h5"

        spot_diam = rep_rate = None
        try:
            t = self._meta_spot_diam.text().strip()
            if t:
                spot_diam = float(t)
        except ValueError:
            pass
        try:
            t = self._meta_rep_rate.text().strip()
            if t:
                rep_rate = float(t)
        except ValueError:
            pass

        power_cal = None
        if self._cal_atbs_data is not None or self._cal_atsample_data is not None:
            power_cal = {}
            if self._cal_atbs_data is not None:
                power_cal["atBS"] = self._cal_atbs_data
            if self._cal_atsample_data is not None:
                power_cal["atSample"] = self._cal_atsample_data

        if self._kind == "pol":
            # One excitation power for the whole angle series — passed as a
            # 1-element array, so every power-derived dataset (Power,
            # Power_density, Pump_fluence) is a single number too (same
            # convention as the TRPL converter's per-file power).
            p_exc_W = self._pol_excitation_power_W()
            if p_exc_W is None:
                return
            powers_W = np.array([p_exc_W])
            col_kw   = {"angles_deg": ds[0]["cols"]}
        else:
            powers_W, col_kw = ds[0]["cols"], {}

        try:
            _write_h5_file(
                out_path, wl_out, counts_diff, counts_raw, best_hdr, powers_W,
                stitched=is_stitched,
                source_datasets=ds if is_stitched else None,
                dark_by_label=dark_by_label if n_dark > 0 else None,
                spot_diameter_um=spot_diam,
                rep_rate_mhz=rep_rate,
                power_cal=power_cal,
                baseline_value=self._baseline_value,
                baseline_range_nm=self._baseline_range,
                **col_kw,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))
            return

        action = "Stitched" if is_stitched else "Converted"
        dark_note = f", {n_dark}/{len(ds)} dark(s) subtracted" if n_dark > 0 else ""
        baseline_note = (f", baseline subtracted per {self._step_word}"
                         if self._baseline_value is not None else "")
        self._status(f"Saved: {out_path}")
        QMessageBox.information(
            self, "Saved",
            f"{action} spectrum ({wl_out.size} wavelengths, "
            f"{n_p} {self._step_word}s{dark_note}{baseline_note}) saved to:\n\n{out_path}",
        )
        title = ("Stitched spectrum (dark-subtracted)"
                 if (is_stitched and n_dark) else
                 "Stitched spectrum" if is_stitched else
                 "Spectrum (dark-subtracted)" if n_dark else "Spectrum")
        self._draw_stitched(wl_out, counts_diff, ds[0]["cols"], title=title)

    def _on_save_origin(self):
        result = self._compute_stitch()
        if result is None:
            return
        wl_out, counts_out, _counts_raw, ds, best_hdr, dark_by_label = result
        n_p         = counts_out.shape[1]
        is_stitched = len(ds) > 1
        n_dark      = sum(1 for v in dark_by_label.values() if v is not None)

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save as .origin", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".origin"):
            out_path += ".origin"

        try:
            _write_origin_file(
                out_path, wl_out, counts_out, best_hdr, ds[0]["cols"]
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))
            return

        action    = "Stitched" if is_stitched else "Converted"
        dark_note = f", {n_dark}/{len(ds)} dark(s) subtracted" if n_dark > 0 else ""
        self._status(f"Saved: {out_path}")
        QMessageBox.information(
            self, "Saved",
            f"{action} spectrum ({wl_out.size} wavelengths, "
            f"{n_p} {self._step_word}s{dark_note}) saved to:\n\n{out_path}",
        )
        title = ("Stitched spectrum (dark-subtracted)"
                 if (is_stitched and n_dark) else
                 "Stitched spectrum" if is_stitched else
                 "Spectrum (dark-subtracted)" if n_dark else "Spectrum")
        self._draw_stitched(wl_out, counts_out, ds[0]["cols"], title=title)

    def _draw_stitched(self, wl_out, counts_out, cols, title="Spectrum"):
        ax = self._canvas.reset_axes()
        n_p        = counts_out.shape[1]
        x          = self._wl_to_x(wl_out)
        idx        = np.argsort(x)
        vals       = self._col_display(cols)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        mlp = MultiLinePlotter(ax, SequentialScheme(vmin=vmin, vmax=vmax, cmap=PLASMA))
        for p in range(n_p):
            item = mlp.plot(x[idx], counts_out[idx, p], value=vals[p], width=0.8)
            item.setOpacity(0.85)
        self._canvas.add_colorbar_legend(
            GradientLegend(cmap=PLASMA, vmin=vmin, vmax=vmax, label=self._col_label)
        )
        ax.setLabel("bottom", "Energy (eV)" if self._x_axis == "energy" else "Wavelength (nm)")
        ax.setLabel("left", "Counts")
        ax.setTitle(title)
        ax.showGrid(x=True, y=True, alpha=0.3)
        if self._baseline_range is not None:
            self._draw_baseline_overlay(ax)
        self._canvas.draw_idle()
        return ax

    def _draw_baseline_overlay(self, ax):
        """Shade the wavelength range a confirmed baseline was fit over —
        same convention as the TRPL tab's confirmed-baseline overlay."""
        lo_wl, hi_wl = self._baseline_range
        x_lo = self._wl_to_x(hi_wl if self._x_axis == "energy" else lo_wl)
        x_hi = self._wl_to_x(lo_wl if self._x_axis == "energy" else hi_wl)
        if x_lo > x_hi:
            x_lo, x_hi = x_hi, x_lo
        region = pg.LinearRegionItem(
            values=(x_lo, x_hi), orientation="vertical",
            brush=pg.mkBrush(30, 100, 220, 30), movable=False,
        )
        region.setZValue(5)
        ax.addItem(region)

    # ── baseline subtraction ─────────────────────────────────
    # Applied at the very end, after stitching and dark subtraction — see
    # _compute_stitch(). Same "drag a range, fit a constant (mean),
    # subtract from everything" UX as the TRPL tab's baseline subtraction,
    # routed through the shared _baseline_bar's Confirm/Cancel buttons.
    # Unlike the lifetime-fit-vs-baseline mutual exclusion in the TRPL tab,
    # here it's baseline-selection vs. transition-span-selection that are
    # mutually exclusive (self._mode == "spans").

    def _remove_baseline_preview_item(self):
        if self._baseline_preview_item is not None:
            try:
                self._canvas.plot_item.removeItem(self._baseline_preview_item)
            except Exception:
                pass
            self._baseline_preview_item = None

    def _on_start_baseline_select(self):
        if self._mode == "spans":
            QMessageBox.information(
                self, "Finish span selection first",
                "Click \"✓ Done\" to finish selecting transition spans "
                "before selecting a baseline range."
            )
            return
        if self._mode == "baseline_selecting":
            return

        # Always compute from the pre-baseline stitch (never the already-
        # baseline-subtracted result), so re-selecting after a Reset — or
        # even without one — can't compound an already-applied baseline.
        result = self._compute_stitch_core()
        if result is None:
            return
        wl_out, counts_diff, _counts_raw, ds, _hdr, _dark = result
        self._baseline_stitch_cache = (wl_out, counts_diff, ds[0]["cols"])

        self._baseline_prev_mode = self._mode
        self._mode = "baseline_selecting"
        self._nav_bar.setVisible(False)
        self._refresh_buttons()

        ax = self._draw_stitched(wl_out, counts_diff, ds[0]["cols"],
                                    title="Select baseline range (no signal)")
        x = self._wl_to_x(wl_out)
        x_lo, x_hi = float(np.min(x)), float(np.max(x))
        width = (x_hi - x_lo) * 0.1
        initial = (x_lo, x_lo + width)

        self._baseline_span = DraggableSpan(ax, color=(30, 100, 220, 60), movable=True)
        self._baseline_span.activate(initial_range=initial, bounds=(x_lo, x_hi))
        self._baseline_span.sigRegionSelected.connect(self._on_baseline_span_changed)

        self._btn_baseline_confirm.setEnabled(False)
        self._baseline_bar.setVisible(True)
        self._on_baseline_span_changed(*initial)

        self._status(
            "Drag a range with no signal, then Confirm baseline."
        )

    def _on_baseline_span_changed(self, xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        self._baseline_pending_range = (xmin, xmax)

        wl_out, counts_diff, _cols = self._baseline_stitch_cache
        x = self._wl_to_x(wl_out)
        mask = (x >= xmin) & (x <= xmax)
        n_pts = int(np.sum(mask))
        if n_pts < 1:
            self._baseline_pending_value = None
            self._remove_baseline_preview_item()
            self._btn_baseline_confirm.setEnabled(False)
            self._baseline_action_lbl.setText("Baseline range: no points in range — widen it.")
            return

        # One constant per column (power step / angle): the mean over the
        # selected range of that column's own counts — not one shared value
        # averaged across columns. Every column's curve is visible while the
        # window is picked (_draw_stitched), and each column is fit
        # independently from its own data in that same wavelength range and
        # later subtracted only from itself (see _compute_stitch()).
        baseline_arr = counts_diff[mask, :].mean(axis=0)   # (n_cols,)
        self._baseline_pending_value = baseline_arr

        self._remove_baseline_preview_item()
        x_curve = np.array([float(np.min(x)), float(np.max(x))])
        y_curve = np.full(2, baseline_arr[0])   # reference line at the first column's own constant
        self._baseline_preview_item = self._canvas.plot_item.plot(
            x_curve, y_curve, pen=pg.mkPen("cyan", width=1.5, style=Qt.DashLine)
        )
        self._canvas.draw_idle()

        n_p = counts_diff.shape[1]
        msg = (f"Baseline range: {n_pts} wavelength pt(s) — one constant per "
               f"{self._step_word} (first = {baseline_arr[0]:.4g} counts, "
               f"{n_p} {self._step_word}(s) total)")
        self._baseline_action_lbl.setText(msg)
        self._btn_baseline_confirm.setEnabled(True)

    def _on_baseline_confirm(self):
        if self._mode != "baseline_selecting" or self._baseline_pending_value is None:
            return
        xmin, xmax = self._baseline_pending_range
        wl_out, _counts_diff, _cols = self._baseline_stitch_cache
        if self._x_axis == "energy":
            lo_wl = min(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
            hi_wl = max(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
        else:
            lo_wl, hi_wl = min(xmin, xmax), max(xmin, xmax)
        self._baseline_range = (lo_wl, hi_wl)
        self._baseline_value = self._baseline_pending_value

        if self._baseline_span is not None:
            self._baseline_span.deactivate()
            self._baseline_span = None
        self._remove_baseline_preview_item()
        self._baseline_pending_range = None
        self._baseline_pending_value = None
        self._baseline_stitch_cache = None

        self._mode = "done"
        self._baseline_bar.setVisible(False)
        self._btn_baseline_reset.setEnabled(True)
        self._refresh_buttons()
        n_p = len(self._baseline_value)
        self._baseline_status_lbl.setText(
            f"Baseline fit per {self._step_word} over {lo_wl:.5g}–{hi_wl:.5g} nm "
            f"({n_p} {self._step_word}(s); first = {self._baseline_value[0]:.4g} counts). "
            "Each spectrum's own constant subtracted from just that spectrum."
        )
        self._refresh_result_view()
        self._status(
            f"Baseline subtracted per {self._step_word} ({n_p} {self._step_word}(s))."
        )

    def _on_baseline_cancel(self):
        if self._baseline_span is not None:
            self._baseline_span.deactivate()
            self._baseline_span = None
        self._remove_baseline_preview_item()
        self._baseline_pending_range = None
        self._baseline_pending_value = None
        self._baseline_stitch_cache = None

        prev_mode = self._baseline_prev_mode if self._baseline_prev_mode is not None else "done"
        self._mode = prev_mode
        self._baseline_bar.setVisible(False)
        self._refresh_buttons()
        # Restore whichever view was showing before baseline selection
        # started, not always the stitched-result view — e.g. canceling
        # right after step 1 (before ever stitching) should go back to the
        # simple last-power preview, not jump ahead to the stitched one.
        if prev_mode == "preview":
            self._draw_preview()
        else:
            self._refresh_result_view()
        self._status("Baseline selection canceled.")

    def _on_reset_baseline(self):
        if self._baseline_value is None:
            return
        self._baseline_range = None
        self._baseline_value = None
        self._btn_baseline_reset.setEnabled(False)
        self._baseline_status_lbl.setText("No baseline applied.")
        self._refresh_result_view()
        self._status("Baseline subtraction reset.")

    def _refresh_result_view(self):
        """Redraw the stitched-result preview reflecting the current
        baseline state — used after confirming/canceling/resetting a
        baseline selection, mirroring _on_preview_result()'s own view."""
        result = self._compute_stitch()
        if result is None:
            return
        wl_out, counts_out, _counts_raw, ds, _hdr, dark_by_label = result
        n_dark      = sum(1 for v in dark_by_label.values() if v is not None)
        is_stitched = len(ds) > 1
        dark_note   = f" (dark-subtracted: {n_dark}/{len(ds)})" if n_dark else ""
        base_title  = "Stitched spectrum" if is_stitched else "Spectrum"
        self._draw_stitched(wl_out, counts_out, ds[0]["cols"], title=base_title + dark_note)

    # ── helpers ──────────────────────────────────────────────

    def _wl_to_x(self, wl):
        wl = np.asarray(wl, dtype=float)
        return _HC_EV_NM / wl if self._x_axis == "energy" else wl

    def _col_display(self, cols):
        """Column values in display units: mW for powers (stored in W),
        degrees for angles."""
        cols = np.asarray(cols, dtype=float)
        return cols if self._kind == "pol" else cols * 1e3

    def _fmt_col(self, value):
        return f"{value:g}°" if self._kind == "pol" else f"{value * 1e3:.3g} mW"

    def _pol_excitation_power_W(self):
        """The single excitation power (W) of a polarization series, from the
        "Excitation power" header line of every input file. Returns None
        (after telling the user) if it's missing, or if the files disagree
        and the user declines to use their mean."""
        vals = [h.get("excitation_power_W") for h in self._hdrs]
        missing = [d["label"] for d, v in zip(self._datasets, vals) if v is None]
        if missing:
            QMessageBox.critical(
                self, "No excitation power",
                "Could not read an \"Excitation power\" header line from:\n\n"
                + "\n".join(missing)
            )
            return None
        vals = np.asarray(vals, dtype=float)
        mean = float(vals.mean())
        if len(vals) > 1 and np.ptp(vals) > 0.01 * abs(mean):
            reply = QMessageBox.question(
                self, "Excitation powers differ",
                "The input files were measured at different excitation powers:\n\n"
                + "\n".join(f"{d['label']}: {v * 1e3:.5g} mW"
                             for d, v in zip(self._datasets, vals))
                + f"\n\nUse their mean ({mean * 1e3:.5g} mW) for the saved file?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return None
        return mean

    def _clear_canvas(self):
        self._canvas.set_welcome(self._welcome_text())

    def _welcome_text(self):
        what = "polarization .origin" if self._kind == "pol" else ".origin"
        return f"Add {what} files and click \"Preview\" to begin."

    @property
    def nav_bar(self):
        return self._nav_bar

    @property
    def baseline_bar(self):
        return self._baseline_bar

    def activate(self):
        """Called by the host tab when this page becomes the visible one."""
        if self._span_selector is None and self._mode == "spans":
            # Resume where the user left off — re-arm the span selector
            # for the pair being worked on (torn down by deactivate()).
            self._nav_bar.setVisible(True)
            self._draw_pair(self._pair_idx)
        elif self._mode in ("preview", "done"):
            self._nav_bar.setVisible(False)
            self._draw_preview()
        else:
            self._nav_bar.setVisible(False)
            self._canvas.set_welcome(self._welcome_text())

    def deactivate(self):
        """Called by the host tab when switching away from this page: tear
        down anything drawn on the shared canvas, keep all other state."""
        self._nav_bar.setVisible(False)
        if self._span_selector is not None:
            self._span_selector.deactivate()
            self._span_selector = None
        if self._mode == "baseline_selecting":
            self._on_baseline_cancel()
        self._baseline_bar.setVisible(False)

    def _refresh_buttons(self):
        has1 = len(self._files) >= 1
        has2 = len(self._files) >= 2
        self._btn_preview.setEnabled(has1)
        self._btn_spans.setEnabled(has2)
        self._btn_prev_result.setEnabled(has1)
        self._btn_stitch.setEnabled(has1)
        self._btn_save_origin.setEnabled(has1)
        self._btn_remove.setEnabled(has1)
        self._btn_clear.setEnabled(has1)
        self._btn_load_dark.setEnabled(bool(self._datasets))
        self._btn_baseline_select.setEnabled(has1 and self._mode != "baseline_selecting")
        self._btn_clear_dark.setEnabled(bool(self._dark_map))
