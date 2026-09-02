import os
import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QRadioButton, QComboBox, QStackedWidget,
    QListWidget, QListWidgetItem, QLineEdit, QFileDialog, QMessageBox,
    QSizePolicy, QAbstractItemView, QFrame, QScrollArea, QStyle,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plotting import (
    PGCanvas, _COMPACT_BTN_STYLE, MultiLinePlotter, CategoricalScheme,
    SequentialScheme, PLASMA, GradientLegend, DraggableSpan, make_pg_toolbar,
)
from io_utils import (
    _parse_header_center_disp,
    _write_origin_file,
    _write_h5_file,
    _parse_power_calibration,
    _parse_trpl_dat,
    _write_trpl_h5_file,
)

from pl import (
    _parse_origin_power_series,
    _parse_origin_header,
    _stitch_counts,
    _HC_EV_NM,
)


class StitchTab(QWidget):
    """"Stitch / Convert": one tab, two independent converters selected by
    the "File type" dropdown — "PS (.origin) → HDF5" (power-series stitching,
    the original Stitch/Convert flow: preview → select transition spans →
    save as HDF5/.origin) and "TRPL (.dat) → HDF5" (batch TRPL histogram
    conversion, the original standalone Convert tab). Adding a third file
    type later is just another combo entry + another sidebar page.

    The two flows share nothing except the combo itself and the canvas/
    toolbar on the right (only one is ever visible, so there's no state
    conflict) — every other widget/state variable is namespaced `_ps_*` or
    `_trpl_*` to keep the two independent, exactly as they were as separate
    classes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ══════════════════════════════════════════════════════════
        # PS (.origin) → HDF5 state  (formerly StitchTab)
        # ══════════════════════════════════════════════════════════
        self._ps_files: list   = []
        self._ps_datasets: list = []   # {label, wl, counts, powers, path}
        self._ps_hdrs: list    = []
        self._ps_pairs: list   = []     # (i_a, i_b, ov_lo_wl, ov_hi_wl)
        self._ps_spans: dict   = {}     # {(i_a,i_b): (lo_wl, hi_wl)}
        self._ps_spans_display: dict = {}    # {(i_a,i_b): (xmin, xmax)} display units
        self._ps_pair_idx: int = 0
        self._ps_x_axis: str   = "energy"
        self._ps_span_selector = None
        self._ps_mode: str     = "idle"

        # dark_map: {dataset_label: dark_info_dict | None}
        # dark_info_dict = {label, path, wl, mean (shape n_wl_dark)}
        self._ps_dark_map: dict = {}

        # Power calibration state
        self._ps_cal_atbs_data     = None   # (hwp_arr, powers_W) or None
        self._ps_cal_atsample_data = None   # (hwp_arr, powers_W) or None
        self._ps_cal_atbs_path     = None   # str or None
        self._ps_cal_atsample_path = None   # str or None

        # Baseline subtraction state — applied at the very end, after
        # stitching and dark subtraction (see _ps_compute_stitch()). One
        # constant per power step (mean counts over a user-picked range,
        # fit independently from each step's own data and subtracted only
        # from that step) — all power steps stay visible while the window
        # is picked, and every step gets its own fit.
        self._ps_baseline_range   = None   # (lo_wl, hi_wl) nm, or None until confirmed
        self._ps_baseline_value   = None   # (n_powers,) fitted constants, or None
        self._ps_baseline_span    = None   # DraggableSpan, active during selection
        self._ps_baseline_pending_range = None   # display-domain (xmin,xmax), pending confirm
        self._ps_baseline_pending_value = None
        self._ps_baseline_preview_item  = None   # dashed constant-level line during selection
        self._ps_baseline_stitch_cache  = None   # (wl_out, counts_diff, powers) — pre-baseline,
                                                   # captured when selection starts
        self._ps_baseline_prev_mode     = None   # _ps_mode to restore on cancel

        # ══════════════════════════════════════════════════════════
        # TRPL (.dat) → HDF5 state  (formerly ConvertTab)
        # ══════════════════════════════════════════════════════════
        self._trpl_files: list        = []
        self._trpl_power_map: dict    = {}   # {path: power_mW}
        self._trpl_parsed_cache: dict = {}   # {path: {date_str, ns_per_channel, counts}}

        self._trpl_cal_atbs_data     = None   # (hwp_arr, powers_W) or None
        self._trpl_cal_atsample_data = None
        self._trpl_cal_atbs_path     = None
        self._trpl_cal_atsample_path = None

        # ── Build UI ────────────────────────────────────────────────
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Left sidebar (scrollable so content is never squeezed or
        #    overlapping if it doesn't fit the window height) ────────
        scroll = QScrollArea()
        scrollbar_w = scroll.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        scroll.setFixedWidth(340 + scrollbar_w)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar = QWidget()
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # File type selector — shared, sits above whichever page is active
        g_type = QGroupBox("File type")
        tl = QHBoxLayout(g_type)
        tl.addWidget(QLabel("Convert:"))
        self._type_combo = QComboBox()
        self._type_combo.addItem("PS (.origin) → HDF5", "ps")
        self._type_combo.addItem("TRPL (.dat) → HDF5", "trpl")
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        tl.addWidget(self._type_combo, stretch=1)
        sl.addWidget(g_type)

        self._stack = QStackedWidget()
        self._page_ps   = self._build_ps_page()
        self._page_trpl = self._build_trpl_page()
        self._stack.addWidget(self._page_ps)
        self._stack.addWidget(self._page_trpl)
        sl.addWidget(self._stack, stretch=1)

        scroll.setWidget(sidebar)
        layout.addWidget(scroll)

        # ── Right: canvas + toolbar, shared by both file types ────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        self._canvas  = PGCanvas(right)
        self._toolbar = make_pg_toolbar(self._canvas, right)
        rl.addWidget(self._toolbar)
        rl.addWidget(self._canvas, stretch=1)

        # PS pair-navigation bar (only shown while selecting transition spans)
        self._ps_nav_bar = QFrame()
        self._ps_nav_bar.setFrameShape(QFrame.StyledPanel)
        nav = QHBoxLayout(self._ps_nav_bar)
        nav.setContentsMargins(8, 4, 8, 4)
        self._ps_btn_prev  = QPushButton("◀  Prev")
        self._ps_lbl_pair  = QLabel("Pair 1 of 1")
        self._ps_lbl_pair.setAlignment(Qt.AlignCenter)
        self._ps_btn_next  = QPushButton("Next  ▶")
        self._ps_btn_done  = QPushButton("✓  Done")
        self._ps_btn_done.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; }"
        )
        self._ps_btn_prev.clicked.connect(self._ps_on_prev_pair)
        self._ps_btn_next.clicked.connect(self._ps_on_next_pair)
        self._ps_btn_done.clicked.connect(self._ps_on_done_spans)
        for w in (self._ps_btn_prev, self._ps_lbl_pair, self._ps_btn_next, self._ps_btn_done):
            nav.addWidget(w)
        self._ps_nav_bar.setVisible(False)
        rl.addWidget(self._ps_nav_bar)

        # PS baseline-selection action bar (Confirm/Cancel), shared pattern
        # with the TRPL tab's baseline subtraction — shown only while
        # self._ps_mode == "baseline_selecting".
        self._ps_baseline_bar = QFrame()
        self._ps_baseline_bar.setFrameShape(QFrame.StyledPanel)
        bb = QHBoxLayout(self._ps_baseline_bar)
        bb.setContentsMargins(8, 4, 8, 4)
        self._ps_baseline_action_lbl = QLabel("")
        self._ps_btn_baseline_confirm = QPushButton("Confirm baseline")
        self._ps_btn_baseline_cancel  = QPushButton("Cancel")
        self._ps_btn_baseline_confirm.clicked.connect(self._ps_on_baseline_confirm)
        self._ps_btn_baseline_cancel.clicked.connect(self._ps_on_baseline_cancel)
        bb.addWidget(self._ps_baseline_action_lbl, stretch=1)
        bb.addWidget(self._ps_btn_baseline_confirm)
        bb.addWidget(self._ps_btn_baseline_cancel)
        self._ps_baseline_bar.setVisible(False)
        rl.addWidget(self._ps_baseline_bar)

        layout.addWidget(right, stretch=1)

        self._ps_refresh_buttons()
        self._trpl_refresh_buttons()
        self._on_type_changed()

    # ── File-type switch ─────────────────────────────────────────

    def _on_type_changed(self, _index=None):
        kind = self._type_combo.currentData()
        parent = self.parent()

        if kind == "ps":
            if self._ps_span_selector is None and self._ps_mode == "spans":
                # Resume where the user left off — re-arm the span selector
                # for the pair being worked on (torn down when we left "ps").
                self._ps_nav_bar.setVisible(True)
                self._ps_draw_pair(self._ps_pair_idx)
            elif self._ps_mode in ("preview", "done"):
                self._ps_nav_bar.setVisible(False)
                self._ps_draw_preview()
            else:
                self._ps_nav_bar.setVisible(False)
                self._canvas.set_welcome("Add .origin files and click \"Preview\" to begin.")
            self._stack.setCurrentWidget(self._page_ps)
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage("Stitch / Convert: PS (.origin) → HDF5.")
        else:
            self._ps_nav_bar.setVisible(False)
            if self._ps_span_selector is not None:
                self._ps_span_selector.deactivate()
                self._ps_span_selector = None
            if self._ps_mode == "baseline_selecting":
                self._ps_on_baseline_cancel()
            self._ps_baseline_bar.setVisible(False)
            self._stack.setCurrentWidget(self._page_trpl)
            item = self._trpl_file_list.currentItem()
            path = item.data(Qt.UserRole) if item is not None else None
            if path is not None:
                self._trpl_draw_preview(path)
            else:
                self._canvas.set_welcome("Add .dat files to begin.")
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage("Stitch / Convert: TRPL (.dat) → HDF5.")

    # ══════════════════════════════════════════════════════════════
    # PS (.origin) → HDF5 page
    # ══════════════════════════════════════════════════════════════

    def _build_ps_page(self):
        page = QWidget()
        sl = QVBoxLayout(page)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # Files group
        g_files = QGroupBox("Input files")
        fl = QVBoxLayout(g_files)
        self._ps_file_list = QListWidget()
        self._ps_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._ps_file_list.setToolTip("Loaded .origin files (sorted by wavelength range)")
        fl.addWidget(self._ps_file_list)
        fb = QHBoxLayout()
        self._ps_btn_add    = QPushButton("Add…")
        self._ps_btn_remove = QPushButton("Remove")
        self._ps_btn_clear  = QPushButton("Clear")
        self._ps_btn_add.clicked.connect(self._ps_on_add_files)
        self._ps_btn_remove.clicked.connect(self._ps_on_remove_files)
        self._ps_btn_clear.clicked.connect(self._ps_on_clear_files)
        for b in (self._ps_btn_add, self._ps_btn_remove, self._ps_btn_clear):
            fb.addWidget(b)
        fl.addLayout(fb)
        g_files.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_files)

        # X-axis group
        g_xaxis = QGroupBox("X axis")
        xl = QHBoxLayout(g_xaxis)
        self._ps_rb_energy = QRadioButton("Energy (eV)")
        self._ps_rb_wl     = QRadioButton("Wavelength (nm)")
        self._ps_rb_energy.setChecked(True)
        self._ps_rb_energy.toggled.connect(self._ps_on_xaxis_changed)
        xl.addWidget(self._ps_rb_energy)
        xl.addWidget(self._ps_rb_wl)
        sl.addWidget(g_xaxis)

        # Transition spans group
        g_spans = QGroupBox("Transition spans")
        spl = QVBoxLayout(g_spans)
        self._ps_span_list = QListWidget()
        self._ps_span_list.setToolTip(
            "One span per overlapping pair.\nClick a row to jump to that pair."
        )
        self._ps_span_list.itemClicked.connect(self._ps_on_span_item_clicked)
        spl.addWidget(self._ps_span_list)
        sl.addWidget(g_spans, stretch=1)

        # Dark spectra group
        g_dark = QGroupBox("Dark spectra")
        dl = QVBoxLayout(g_dark)
        self._ps_dark_list = QListWidget()
        self._ps_dark_list.setToolTip(
            "Dark spectrum matched to each input file.\n"
            "Select a row and click \"Load…\" to set manually."
        )
        self._ps_dark_list.setMaximumHeight(110)
        dl.addWidget(self._ps_dark_list)
        db = QHBoxLayout()
        self._ps_btn_autosearch = QPushButton("Autosearch")
        self._ps_btn_load_dark  = QPushButton("Load…")
        self._ps_btn_clear_dark = QPushButton("Clear")
        self._ps_btn_autosearch.setToolTip(
            "Search for files with 'dark' in the name in each\n"
            "input directory or its 'dark' subfolder.\n"
            "Matches by integration time and center wavelength."
        )
        self._ps_btn_autosearch.clicked.connect(self._ps_on_autosearch_dark)
        self._ps_btn_load_dark.clicked.connect(self._ps_on_load_dark)
        self._ps_btn_clear_dark.clicked.connect(self._ps_on_clear_dark)
        for b in (self._ps_btn_autosearch, self._ps_btn_load_dark, self._ps_btn_clear_dark):
            db.addWidget(b)
        dl.addLayout(db)
        g_dark.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_dark)

        # Metadata group
        g_meta = QGroupBox("Metadata (saved to HDF5)")
        ml = QVBoxLayout(g_meta)
        r_spot = QHBoxLayout()
        r_spot.addWidget(QLabel("Spot diameter (µm):"))
        self._ps_meta_spot_diam = QLineEdit()
        self._ps_meta_spot_diam.setPlaceholderText("optional")
        r_spot.addWidget(self._ps_meta_spot_diam)
        ml.addLayout(r_spot)
        r_rep = QHBoxLayout()
        r_rep.addWidget(QLabel("Rep. rate (MHz):"))
        self._ps_meta_rep_rate = QLineEdit()
        self._ps_meta_rep_rate.setPlaceholderText("optional")
        r_rep.addWidget(self._ps_meta_rep_rate)
        ml.addLayout(r_rep)
        sl.addWidget(g_meta)

        # Power calibration group
        g_cal = QGroupBox("Power calibration")
        cl = QVBoxLayout(g_cal)

        r_atbs = QHBoxLayout()
        r_atbs.addWidget(QLabel("atBS:"))
        self._ps_cal_atbs_lbl = QLabel("—")
        self._ps_cal_atbs_lbl.setWordWrap(False)
        self._ps_cal_atbs_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        r_atbs.addWidget(self._ps_cal_atbs_lbl, stretch=1)
        btn_load_atbs = QPushButton("Load…")
        btn_load_atbs.setFixedWidth(64)
        btn_load_atbs.clicked.connect(self._ps_on_load_cal_atbs)
        r_atbs.addWidget(btn_load_atbs)
        cl.addLayout(r_atbs)

        r_ats = QHBoxLayout()
        r_ats.addWidget(QLabel("atSample:"))
        self._ps_cal_atsample_lbl = QLabel("—")
        self._ps_cal_atsample_lbl.setWordWrap(False)
        self._ps_cal_atsample_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        r_ats.addWidget(self._ps_cal_atsample_lbl, stretch=1)
        btn_load_ats = QPushButton("Load…")
        btn_load_ats.setFixedWidth(64)
        btn_load_ats.clicked.connect(self._ps_on_load_cal_atsample)
        r_ats.addWidget(btn_load_ats)
        cl.addLayout(r_ats)

        btn_cal_auto = QPushButton("Autosearch")
        btn_cal_auto.clicked.connect(self._ps_on_autosearch_cal)
        cl.addWidget(btn_cal_auto)
        g_cal.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_cal)

        # Baseline subtraction group — applied at the very end, after
        # stitching and dark subtraction. Same UX as the TRPL tab's
        # baseline subtraction: drag a range, a constant (mean counts) is
        # fit over it and subtracted from every count.
        g_baseline = QGroupBox("Baseline subtraction")
        bll = QVBoxLayout(g_baseline)
        self._ps_btn_baseline_select = QPushButton("Select baseline range…")
        self._ps_btn_baseline_select.setToolTip(
            "Drag a range with no signal, on the stitched/dark-subtracted\n"
            "result (all power steps shown). A separate constant (mean\n"
            "counts over the range) is fit per power step, from that\n"
            "step's own data, and subtracted from just that step."
        )
        self._ps_btn_baseline_select.setEnabled(False)
        self._ps_btn_baseline_select.clicked.connect(self._ps_on_start_baseline_select)
        bll.addWidget(self._ps_btn_baseline_select)
        self._ps_btn_baseline_reset = QPushButton("Reset baseline")
        self._ps_btn_baseline_reset.setEnabled(False)
        self._ps_btn_baseline_reset.clicked.connect(self._ps_on_reset_baseline)
        bll.addWidget(self._ps_btn_baseline_reset)
        self._ps_baseline_status_lbl = QLabel("No baseline applied.")
        self._ps_baseline_status_lbl.setWordWrap(True)
        bll.addWidget(self._ps_baseline_status_lbl)
        g_baseline.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_baseline)

        # Action buttons
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        self._ps_btn_preview     = QPushButton("1  Preview spectra")
        self._ps_btn_spans       = QPushButton("2  Select spans")
        self._ps_btn_prev_result = QPushButton("3  Preview result")
        self._ps_btn_stitch      = QPushButton("4  Save as HDF5…")
        self._ps_btn_save_origin = QPushButton("5  Save as .origin…")
        self._ps_btn_preview.clicked.connect(self._ps_on_preview)
        self._ps_btn_spans.clicked.connect(self._ps_on_start_spans)
        self._ps_btn_prev_result.clicked.connect(self._ps_on_preview_result)
        self._ps_btn_stitch.clicked.connect(self._ps_on_stitch_save)
        self._ps_btn_save_origin.clicked.connect(self._ps_on_save_origin)

        bold = QFont(); bold.setBold(True)
        self._ps_btn_stitch.setFont(bold)
        self._ps_btn_stitch.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; }"
            "QPushButton:disabled { background-color: #bbbbbb; color: #666666; }"
        )
        for b in (self._ps_btn_preview, self._ps_btn_spans, self._ps_btn_prev_result,
                  self._ps_btn_stitch, self._ps_btn_save_origin):
            b.setMinimumHeight(32)
            sl.addWidget(b)

        return page

    # ── PS: file management ─────────────────────────────────────

    def _ps_on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select .origin files", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        added = 0
        for p in paths:
            if p not in self._ps_files:
                self._ps_files.append(p)
                item = QListWidgetItem(os.path.basename(p))
                item.setToolTip(p)
                self._ps_file_list.addItem(item)
                added += 1
        if added:
            self._ps_reset_parsed()
            self._ps_refresh_buttons()
            parent = self.parent()
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    f"{len(self._ps_files)} file(s) loaded. Click \"Preview\" to continue."
                )

    def _ps_on_remove_files(self):
        rows = sorted(
            [self._ps_file_list.row(i) for i in self._ps_file_list.selectedItems()],
            reverse=True,
        )
        for row in rows:
            self._ps_file_list.takeItem(row)
            if row < len(self._ps_files):
                self._ps_files.pop(row)
        self._ps_reset_parsed()
        self._ps_refresh_buttons()

    def _ps_on_clear_files(self):
        self._ps_files.clear()
        self._ps_file_list.clear()
        self._ps_reset_parsed()
        self._ps_clear_canvas()
        self._ps_refresh_buttons()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("Files cleared.")

    def _ps_reset_parsed(self):
        self._ps_datasets.clear()
        self._ps_hdrs.clear()
        self._ps_pairs.clear()
        self._ps_spans.clear()
        self._ps_spans_display.clear()
        self._ps_span_list.clear()
        self._ps_dark_map.clear()
        self._ps_dark_list.clear()
        self._ps_span_selector = None
        self._ps_mode = "idle"
        self._ps_nav_bar.setVisible(False)
        self._ps_reset_baseline_state()

    def _ps_reset_baseline_state(self):
        """Discard any baseline selection (confirmed or in progress) — the
        input files/stitch changed, so a previously-fit constant no longer
        applies. Called whenever the input file set changes."""
        self._ps_baseline_range = None
        self._ps_baseline_value = None
        self._ps_baseline_pending_range = None
        self._ps_baseline_pending_value = None
        self._ps_baseline_stitch_cache = None
        if self._ps_baseline_span is not None:
            self._ps_baseline_span.deactivate()
            self._ps_baseline_span = None
        self._ps_remove_baseline_preview_item()
        self._ps_baseline_bar.setVisible(False)
        self._ps_btn_baseline_reset.setEnabled(False)
        self._ps_baseline_status_lbl.setText("No baseline applied.")

    # ── PS: X-axis toggle ────────────────────────────────────────

    def _ps_on_xaxis_changed(self):
        new = "energy" if self._ps_rb_energy.isChecked() else "wavelength"
        if new == self._ps_x_axis:
            return
        converted = {}
        for key, (x0, x1) in self._ps_spans_display.items():
            a, b = _HC_EV_NM / x0, _HC_EV_NM / x1
            converted[key] = (min(a, b), max(a, b))
        self._ps_x_axis = new
        self._ps_spans_display = converted
        if self._ps_mode == "preview":
            self._ps_draw_preview()
        elif self._ps_mode == "spans":
            self._ps_draw_pair(self._ps_pair_idx)
        self._ps_refresh_span_list()

    # ── PS: parsing ──────────────────────────────────────────────

    def _ps_parse_all(self) -> bool:
        """Parse all files; return True if at least 1 parsed successfully."""
        self._ps_datasets.clear()
        self._ps_hdrs.clear()
        errors = []
        for path in self._ps_files:
            label = os.path.splitext(os.path.basename(path))[0]
            try:
                wl, counts, powers = _parse_origin_power_series(path)
                hdr = _parse_origin_header(path)
                self._ps_datasets.append({
                    "label": label, "wl": wl, "counts": counts,
                    "powers": powers, "path": path,
                })
                self._ps_hdrs.append(hdr)
            except Exception as exc:
                errors.append(f"  {label}: {exc}")
        if errors:
            QMessageBox.warning(self, "Parse errors",
                                "Could not parse:\n" + "\n".join(errors))
        if not self._ps_datasets:
            QMessageBox.warning(self, "No valid files",
                                "None of the selected files could be parsed.")
            return False
        order = sorted(range(len(self._ps_datasets)),
                       key=lambda i: self._ps_datasets[i]["wl"].min())
        self._ps_datasets = [self._ps_datasets[i] for i in order]
        self._ps_hdrs     = [self._ps_hdrs[i]     for i in order]
        self._ps_refresh_dark_list()
        return True

    def _ps_find_pairs(self) -> bool:
        self._ps_pairs.clear()
        for i in range(len(self._ps_datasets) - 1):
            a, b = self._ps_datasets[i], self._ps_datasets[i + 1]
            ov_lo = max(a["wl"].min(), b["wl"].min())
            ov_hi = min(a["wl"].max(), b["wl"].max())
            if ov_hi > ov_lo:
                self._ps_pairs.append((i, i + 1, ov_lo, ov_hi))
        self._ps_refresh_span_list()
        return bool(self._ps_pairs)

    def _ps_refresh_span_list(self):
        self._ps_span_list.clear()
        for k, (i_a, i_b, _, _) in enumerate(self._ps_pairs):
            a, b = self._ps_datasets[i_a], self._ps_datasets[i_b]
            key = (i_a, i_b)
            if key in self._ps_spans:
                lo_wl, hi_wl = self._ps_spans[key]
                xmin = self._ps_wl_to_x(hi_wl if self._ps_x_axis == "energy" else lo_wl)
                xmax = self._ps_wl_to_x(lo_wl if self._ps_x_axis == "energy" else hi_wl)
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
            self._ps_span_list.addItem(item)

    # ── PS: dark spectra ─────────────────────────────────────────

    def _ps_parse_dark_file(self, path: str):
        """Parse a .origin file as a dark spectrum. Returns info dict or None."""
        try:
            wl, counts, _ = _parse_origin_power_series(path)
            hdr = _parse_origin_header(path)
            center_nm, disp_nm = _parse_header_center_disp(hdr)
            return {
                "path":       path,
                "label":      os.path.splitext(os.path.basename(path))[0],
                "wl":         wl,
                "mean":       counts.mean(axis=1),   # (n_wl,) — mean over power steps
                "int_time":   hdr.get("int_time_str", ""),
                "center_nm":  center_nm,
                "disp_nm":    disp_nm,
            }
        except Exception:
            return None

    def _ps_darks_match(self, sig_hdr: dict, dark_info: dict) -> bool:
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

    def _ps_autosearch_darks(self, force: bool = False):
        """Search for dark files in input directories and match them.

        force=False (default): skip datasets that already have a dark assigned
                               so manual assignments are never overwritten.
        force=True:            overwrite all entries (explicit Autosearch button).
        """
        # Collect unique directories
        dirs = list(dict.fromkeys(os.path.dirname(p) for p in self._ps_files))

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
        parsed = [self._ps_parse_dark_file(p) for p in candidates]
        parsed = [d for d in parsed if d is not None]

        # Match to each dataset
        n_matched = 0
        for dataset, hdr in zip(self._ps_datasets, self._ps_hdrs):
            label = dataset["label"]
            if not force and self._ps_dark_map.get(label) is not None:
                # Preserve existing assignment (manual or previous auto-match)
                n_matched += 1
                continue
            best = next((d for d in parsed if self._ps_darks_match(hdr, d)), None)
            self._ps_dark_map[label] = best
            if best is not None:
                n_matched += 1

        self._ps_refresh_dark_list()
        self._ps_refresh_buttons()
        parent = self.parent()
        if not candidates:
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    "Autosearch: no dark files found (looking for 'dark' in name "
                    "or files in a 'dark' subfolder)."
                )
        else:
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    f"Autosearch: {len(candidates)} candidate(s) found, "
                    f"{n_matched}/{len(self._ps_datasets)} file(s) matched."
                )

    def _ps_on_autosearch_dark(self):
        if not self._ps_datasets:
            if not self._ps_parse_all():
                return
            self._ps_find_pairs()
        self._ps_autosearch_darks(force=True)

    def _ps_on_load_dark(self):
        """Manually assign a dark file to the selected entry in the dark list."""
        row = self._ps_dark_list.currentRow()
        if row < 0 or row >= len(self._ps_datasets):
            QMessageBox.information(self, "No selection",
                                    "Select a file in the dark list first.")
            return
        dataset = self._ps_datasets[row]
        sig_hdr = self._ps_hdrs[row]

        path, _ = QFileDialog.getOpenFileName(
            self, f"Dark file for '{dataset['label']}'", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if not path:
            return

        dark_info = self._ps_parse_dark_file(path)
        if dark_info is None:
            QMessageBox.critical(self, "Parse error",
                                 f"Could not parse:\n{path}")
            return

        if not self._ps_darks_match(sig_hdr, dark_info):
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

        self._ps_dark_map[dataset["label"]] = dark_info
        self._ps_refresh_dark_list()
        self._ps_refresh_buttons()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Dark assigned: {dataset['label']} ← {dark_info['label']}"
            )

    def _ps_on_clear_dark(self):
        self._ps_dark_map.clear()
        self._ps_refresh_dark_list()
        self._ps_refresh_buttons()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("Dark spectra cleared.")

    # ── PS: power calibration ─────────────────────────────────────

    def _ps_cal_load(self, path, role):
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
            self._ps_cal_atbs_data  = (hwp, pows)
            self._ps_cal_atbs_path  = path
            self._ps_cal_atbs_lbl.setText(short)
        else:
            self._ps_cal_atsample_data  = (hwp, pows)
            self._ps_cal_atsample_path  = path
            self._ps_cal_atsample_lbl.setText(short)
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Calibration {role}: {len(hwp)} HWP steps loaded from {short}"
            )
        return True

    def _ps_on_load_cal_atbs(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select atBS calibration file", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if path:
            self._ps_cal_load(path, "atBS")

    def _ps_on_load_cal_atsample(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select atSample calibration file", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if path:
            self._ps_cal_load(path, "atSample")

    def _ps_on_autosearch_cal(self):
        """Search input-file directories (and a 'Calibration' subdirectory) for
        calibration files whose names contain 'atbs' or 'atsample' (case-insensitive).
        """
        if not self._ps_files:
            QMessageBox.information(self, "No files loaded",
                                    "Add .origin files first.")
            return

        dirs = list(dict.fromkeys(os.path.dirname(p) for p in self._ps_files))

        search_dirs = []
        seen_dirs   = set()
        for d in dirs:
            if d not in seen_dirs:
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
            if self._ps_cal_load(atbs_path, "atBS"):
                found.append(f"atBS: {os.path.basename(atbs_path)}")
        if atsample_path:
            if self._ps_cal_load(atsample_path, "atSample"):
                found.append(f"atSample: {os.path.basename(atsample_path)}")

        parent = self.parent()
        if found:
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    "Calibration autosearch found: " + ", ".join(found)
                )
        else:
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    "Calibration autosearch: no matching files found "
                    "(looking for 'atBS'/'atSample' in name in input dirs and 'Calibration/' subfolder)."
                )

    def _ps_refresh_dark_list(self):
        self._ps_dark_list.clear()
        for dataset in self._ps_datasets:
            dark_info = self._ps_dark_map.get(dataset["label"])
            if dark_info is not None:
                text  = f"✅  {dataset['label']}\n    ← {dark_info['label']}"
                color = QColor("#c8e6c9")
            else:
                text  = f"⬜  {dataset['label']}  ← not found"
                color = QColor("#ffffff")
            item = QListWidgetItem(text)
            item.setBackground(color)
            self._ps_dark_list.addItem(item)

    def _ps_apply_dark(self, datasets: list) -> tuple:
        """Subtract matched dark from each dataset's counts index-by-index.

        Returns (ds_sub, dark_by_label) where:
          ds_sub         — list of dataset dicts with counts replaced by dark-subtracted values
          dark_by_label  — {label: dark array aligned to file pixel grid | None}
        """
        ds_sub        = []
        dark_by_label = {}
        for d in datasets:
            dark_info = self._ps_dark_map.get(d["label"])
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

    # ── PS: step 1 — Preview ─────────────────────────────────────

    def _ps_on_preview(self):
        if not self._ps_parse_all():
            return
        self._ps_find_pairs()
        self._ps_autosearch_darks()
        self._ps_mode = "preview"
        self._ps_nav_bar.setVisible(False)
        self._ps_draw_preview()
        self._ps_refresh_buttons()
        n = len(self._ps_datasets)
        n_dark = sum(1 for v in self._ps_dark_map.values() if v is not None)
        dark_note = f", {n_dark}/{n} dark(s) matched" if n_dark else ""
        parent = self.parent()
        if n == 1:
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    f"1 file loaded{dark_note}. Click \"Save as HDF5\" to convert."
                )
        else:
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    f"{n} file(s), {len(self._ps_pairs)} overlapping pair(s){dark_note}."
                )

    def _ps_draw_preview(self):
        ds_plot, dark_by_label = self._ps_apply_dark(self._ps_datasets)
        n_dark  = sum(1 for v in dark_by_label.values() if v is not None)
        subtitle = f"dark-subtracted: {n_dark}/{len(ds_plot)}" if n_dark else "raw"

        ax = self._canvas.reset_axes()
        ax.addLegend(labelTextSize="7pt")
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        for k, d in enumerate(ds_plot):
            x   = self._ps_wl_to_x(d["wl"])
            idx = np.argsort(x)
            mlp.plot(x[idx], d["counts"][idx, -1], index=k, width=1.3,
                     label=f"{d['label']}  ({d['powers'][-1] * 1e3:.3g} mW)")
        ax.setLabel("bottom", "Energy (eV)" if self._ps_x_axis == "energy" else "Wavelength (nm)")
        ax.setLabel("left", "Counts")
        ax.showGrid(x=True, y=True, alpha=0.3)
        ax.setTitle(f"Last-power spectra (preview — {subtitle})")
        self._canvas.draw_idle()

    # ── PS: step 2 — Select spans ─────────────────────────────────

    def _ps_on_start_spans(self):
        if not self._ps_datasets:
            if not self._ps_parse_all():
                return
        if not self._ps_pairs and not self._ps_find_pairs():
            QMessageBox.information(self, "No overlaps",
                                    "No overlapping wavelength ranges found.")
            return
        self._ps_mode = "spans"
        self._ps_pair_idx = 0
        self._ps_nav_bar.setVisible(True)
        self._ps_draw_pair(0)
        self._ps_refresh_buttons()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                "Drag to select a transition span for each pair. "
                "Use ◀/▶ to navigate, ✓ Done when finished."
            )

    def _ps_draw_pair(self, idx: int):
        if not (0 <= idx < len(self._ps_pairs)):
            return
        i_a, i_b, ov_lo_wl, ov_hi_wl = self._ps_pairs[idx]
        a, b = self._ps_datasets[i_a], self._ps_datasets[i_b]

        if self._ps_span_selector is not None:
            self._ps_span_selector.deactivate()
            self._ps_span_selector = None

        ds_sub, _ = self._ps_apply_dark([a, b])
        a_plot, b_plot = ds_sub

        ax = self._canvas.reset_axes()
        ax.addLegend(labelTextSize="8pt")
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        for k, d in enumerate((a_plot, b_plot)):
            mask = (d["wl"] >= ov_lo_wl) & (d["wl"] <= ov_hi_wl)
            x_ov = self._ps_wl_to_x(d["wl"][mask])
            c_ov = d["counts"][mask, -1]
            s    = np.argsort(x_ov)
            mlp.plot(x_ov[s], c_ov[s], index=k, width=1.4, label=d["label"])

        x_lo = self._ps_wl_to_x(ov_hi_wl if self._ps_x_axis == "energy" else ov_lo_wl)
        x_hi = self._ps_wl_to_x(ov_lo_wl if self._ps_x_axis == "energy" else ov_hi_wl)
        if x_lo > x_hi:
            x_lo, x_hi = x_hi, x_lo
        ax.setXRange(x_lo, x_hi, padding=0)
        ax.setLabel("bottom", "Energy (eV)" if self._ps_x_axis == "energy" else "Wavelength (nm)")
        ax.setLabel("left", "Counts")
        ax.showGrid(x=True, y=True, alpha=0.3)
        ax.setTitle(
            f"Pair {idx + 1}/{len(self._ps_pairs)}: {a['label']}  ↔  {b['label']}\n"
            "Drag the shaded region's edges to select the transition span"
        )
        self._canvas.draw_idle()

        key = (i_a, i_b)
        stored = self._ps_spans_display.get(key)
        mid = (x_lo + x_hi) / 2.0
        width = (x_hi - x_lo) * 0.1
        initial = stored if stored is not None else (mid - width / 2, mid + width / 2)

        self._ps_span_selector = DraggableSpan(ax, color=(0, 150, 0, 60), movable=True)
        self._ps_span_selector.activate(initial_range=initial, bounds=(x_lo, x_hi))
        self._ps_span_selector.sigRegionSelected.connect(
            lambda xmin, xmax, _idx=idx: self._ps_on_span_selected(xmin, xmax, _idx)
        )
        self._ps_lbl_pair.setText(f"Pair {idx + 1} of {len(self._ps_pairs)}")
        self._ps_btn_prev.setEnabled(idx > 0)
        self._ps_btn_next.setEnabled(idx < len(self._ps_pairs) - 1)
        self._ps_span_list.setCurrentRow(idx)

    def _ps_on_span_selected(self, xmin: float, xmax: float, pair_idx: int):
        if pair_idx != self._ps_pair_idx or abs(xmax - xmin) < 1e-12:
            return
        i_a, i_b, _, _ = self._ps_pairs[pair_idx]
        key = (i_a, i_b)
        self._ps_spans_display[key] = (min(xmin, xmax), max(xmin, xmax))
        if self._ps_x_axis == "energy":
            lo_wl = min(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
            hi_wl = max(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
        else:
            lo_wl, hi_wl = min(xmin, xmax), max(xmin, xmax)
        self._ps_spans[key] = (lo_wl, hi_wl)
        self._ps_refresh_span_list()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Pair {pair_idx + 1}: span [{xmin:.5g}, {xmax:.5g}] "
                f"({'eV' if self._ps_x_axis == 'energy' else 'nm'})."
            )

    def _ps_on_span_item_clicked(self, item):
        if self._ps_mode != "spans":
            return
        idx = item.data(Qt.UserRole)
        if idx is not None and 0 <= idx < len(self._ps_pairs):
            self._ps_pair_idx = idx
            self._ps_draw_pair(idx)

    def _ps_on_prev_pair(self):
        if self._ps_pair_idx > 0:
            self._ps_pair_idx -= 1
            self._ps_draw_pair(self._ps_pair_idx)

    def _ps_on_next_pair(self):
        if self._ps_pair_idx < len(self._ps_pairs) - 1:
            self._ps_pair_idx += 1
            self._ps_draw_pair(self._ps_pair_idx)

    def _ps_on_done_spans(self):
        if self._ps_span_selector is not None:
            self._ps_span_selector.deactivate()
            self._ps_span_selector = None
        self._ps_nav_bar.setVisible(False)
        self._ps_mode = "done"
        self._ps_refresh_buttons()
        n_sel, n_req = len(self._ps_spans), len(self._ps_pairs)
        parent = self.parent()
        if n_sel < n_req:
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    f"Span selection done: {n_sel}/{n_req} spans defined. "
                    f"{n_req - n_sel} missing pair(s) will use range midpoint."
                )
        else:
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(f"All {n_sel} span(s) defined.")
        self._ps_draw_preview()

    # ── PS: steps 3–5 — Preview / Save ────────────────────────────

    def _ps_compute_stitch(self):
        """_ps_compute_stitch_core() plus baseline subtraction — the very
        last step, applied after stitching and dark subtraction, if a
        baseline has been confirmed (see _ps_on_baseline_confirm()). This is
        the version everything downstream (previews, both Save actions)
        should call; the baseline selector itself calls
        _ps_compute_stitch_core() directly so it always starts from the
        pre-baseline data, never compounding an already-applied baseline.

        self._ps_baseline_value is a (n_powers,) array, one constant per
        power step — each subtracted only from its own column, not one
        shared value applied to every step (relies on numpy broadcasting a
        (n_powers,) array against counts_diff's (n_wl, n_powers) shape).
        """
        result = self._ps_compute_stitch_core()
        if result is None:
            return None
        wl_out, counts_diff, counts_raw, ds_raw, best_hdr, dark_by_label = result
        if self._ps_baseline_value is not None:
            counts_diff = counts_diff - self._ps_baseline_value
        return wl_out, counts_diff, counts_raw, ds_raw, best_hdr, dark_by_label

    def _ps_compute_stitch_core(self):
        """Build output data. Dark subtraction (if assigned) always precedes stitching.

        Returns (wl_out, counts_diff, counts_raw, ds_raw, best_hdr, dark_by_label)
        or None.
          counts_diff   — dark-subtracted, min shifted to 1 (no baseline subtraction —
                          see _ps_compute_stitch(), the wrapper everything else calls)
          counts_raw    — stitched without dark subtraction, no min-shift
          ds_raw        — original (un-subtracted) source datasets
          dark_by_label — {label: dark_mean_on_file_wl | None}
        """
        if not self._ps_datasets:
            if not self._ps_parse_all():
                return None
            self._ps_find_pairs()

        # ── Single file ──────────────────────────────────────────
        if len(self._ps_datasets) == 1:
            d = self._ps_datasets[0]
            ds_raw                = [d]
            ds_sub, dark_by_label = self._ps_apply_dark(ds_raw)
            wl_out                = ds_sub[0]["wl"]
            counts_diff           = ds_sub[0]["counts"].copy()
            counts_diff           = counts_diff - counts_diff.min() + 1.0
            counts_raw            = d["counts"].copy()
            return wl_out, counts_diff, counts_raw, ds_raw, self._ps_hdrs[0].copy(), dark_by_label

        # ── Multiple files ────────────────────────────────────────
        n_p_list = [d["counts"].shape[1] for d in self._ps_datasets]
        n_p      = min(n_p_list)
        if len(set(n_p_list)) > 1:
            QMessageBox.information(
                self, "Power step mismatch",
                f"Files have different power-step counts: {n_p_list}.\n"
                f"All will be truncated to {n_p} steps.",
            )
        ds_raw = [
            {"label": d["label"], "wl": d["wl"],
             "counts": d["counts"][:, :n_p], "powers": d["powers"][:n_p],
             "path": d["path"]}
            for d in self._ps_datasets
        ]

        # Build spans_wl
        spans_wl = []
        for k, (i_a, i_b, ov_lo_wl, ov_hi_wl) in enumerate(self._ps_pairs):
            key = (i_a, i_b)
            if key in self._ps_spans:
                spans_wl.append(self._ps_spans[key])
            else:
                mid = (ov_lo_wl + ov_hi_wl) / 2.0
                spans_wl.append((mid, mid))

        # Stitch dark-subtracted spectrum
        ds_for_stitch, dark_by_label = self._ps_apply_dark(ds_raw)
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
        best_hdr  = self._ps_hdrs[0].copy()
        best_date = best_hdr.get("date")
        for h in self._ps_hdrs[1:]:
            d = h.get("date")
            if d is not None and (best_date is None or d < best_date):
                best_date = d
                best_hdr["date_str"] = h.get("date_str", best_hdr.get("date_str", ""))

        counts_diff = counts_diff - counts_diff.min() + 1.0
        return wl_out, counts_diff, counts_raw, ds_raw, best_hdr, dark_by_label

    def _ps_on_preview_result(self):
        result = self._ps_compute_stitch()
        if result is None:
            return
        wl_out, counts_out, _counts_raw, ds, _, dark_by_label = result
        n_dark      = sum(1 for v in dark_by_label.values() if v is not None)
        is_stitched = len(ds) > 1
        dark_note   = f" (dark-subtracted: {n_dark}/{len(ds)})" if n_dark else ""
        base_title  = "Stitched spectrum" if is_stitched else "Spectrum"
        self._ps_draw_stitched(wl_out, counts_out, ds[0]["powers"],
                               title=base_title + dark_note)
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Preview: {wl_out.size} wavelengths, "
                f"{counts_out.shape[1]} power steps{dark_note}."
            )

    def _ps_on_stitch_save(self):
        result = self._ps_compute_stitch()
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
            t = self._ps_meta_spot_diam.text().strip()
            if t:
                spot_diam = float(t)
        except ValueError:
            pass
        try:
            t = self._ps_meta_rep_rate.text().strip()
            if t:
                rep_rate = float(t)
        except ValueError:
            pass

        power_cal = None
        if self._ps_cal_atbs_data is not None or self._ps_cal_atsample_data is not None:
            power_cal = {}
            if self._ps_cal_atbs_data is not None:
                power_cal["atBS"] = self._ps_cal_atbs_data
            if self._ps_cal_atsample_data is not None:
                power_cal["atSample"] = self._ps_cal_atsample_data

        try:
            _write_h5_file(
                out_path, wl_out, counts_diff, counts_raw, best_hdr, ds[0]["powers"],
                stitched=is_stitched,
                source_datasets=ds if is_stitched else None,
                dark_by_label=dark_by_label if n_dark > 0 else None,
                spot_diameter_um=spot_diam,
                rep_rate_mhz=rep_rate,
                power_cal=power_cal,
                baseline_value=self._ps_baseline_value,
                baseline_range_nm=self._ps_baseline_range,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))
            return

        action = "Stitched" if is_stitched else "Converted"
        dark_note = f", {n_dark}/{len(ds)} dark(s) subtracted" if n_dark > 0 else ""
        baseline_note = (", per-power-step baseline subtracted"
                         if self._ps_baseline_value is not None else "")
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(f"Saved: {out_path}")
        QMessageBox.information(
            self, "Saved",
            f"{action} spectrum ({wl_out.size} wavelengths, "
            f"{n_p} power steps{dark_note}{baseline_note}) saved to:\n\n{out_path}",
        )
        title = ("Stitched spectrum (dark-subtracted)"
                 if (is_stitched and n_dark) else
                 "Stitched spectrum" if is_stitched else
                 "Spectrum (dark-subtracted)" if n_dark else "Spectrum")
        self._ps_draw_stitched(wl_out, counts_diff, ds[0]["powers"], title=title)

    def _ps_on_save_origin(self):
        result = self._ps_compute_stitch()
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
                out_path, wl_out, counts_out, best_hdr, ds[0]["powers"]
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))
            return

        action    = "Stitched" if is_stitched else "Converted"
        dark_note = f", {n_dark}/{len(ds)} dark(s) subtracted" if n_dark > 0 else ""
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(f"Saved: {out_path}")
        QMessageBox.information(
            self, "Saved",
            f"{action} spectrum ({wl_out.size} wavelengths, "
            f"{n_p} power steps{dark_note}) saved to:\n\n{out_path}",
        )
        title = ("Stitched spectrum (dark-subtracted)"
                 if (is_stitched and n_dark) else
                 "Stitched spectrum" if is_stitched else
                 "Spectrum (dark-subtracted)" if n_dark else "Spectrum")
        self._ps_draw_stitched(wl_out, counts_out, ds[0]["powers"], title=title)

    def _ps_draw_stitched(self, wl_out, counts_out, powers, title="Spectrum"):
        ax = self._canvas.reset_axes()
        n_p        = counts_out.shape[1]
        x          = self._ps_wl_to_x(wl_out)
        idx        = np.argsort(x)
        vmin, vmax = powers[0] * 1e3, powers[-1] * 1e3
        mlp = MultiLinePlotter(ax, SequentialScheme(vmin=vmin, vmax=vmax, cmap=PLASMA))
        for p in range(n_p):
            item = mlp.plot(x[idx], counts_out[idx, p], value=powers[p] * 1e3, width=0.8)
            item.setOpacity(0.85)
        self._canvas.add_colorbar_legend(
            GradientLegend(cmap=PLASMA, vmin=vmin, vmax=vmax, label="Power (mW)")
        )
        ax.setLabel("bottom", "Energy (eV)" if self._ps_x_axis == "energy" else "Wavelength (nm)")
        ax.setLabel("left", "Counts")
        ax.setTitle(title)
        ax.showGrid(x=True, y=True, alpha=0.3)
        if self._ps_baseline_range is not None:
            self._ps_draw_baseline_overlay(ax)
        self._canvas.draw_idle()
        return ax

    def _ps_draw_baseline_overlay(self, ax):
        """Shade the wavelength range a confirmed baseline was fit over —
        same convention as the TRPL tab's confirmed-baseline overlay."""
        lo_wl, hi_wl = self._ps_baseline_range
        x_lo = self._ps_wl_to_x(hi_wl if self._ps_x_axis == "energy" else lo_wl)
        x_hi = self._ps_wl_to_x(lo_wl if self._ps_x_axis == "energy" else hi_wl)
        if x_lo > x_hi:
            x_lo, x_hi = x_hi, x_lo
        region = pg.LinearRegionItem(
            values=(x_lo, x_hi), orientation="vertical",
            brush=pg.mkBrush(30, 100, 220, 30), movable=False,
        )
        region.setZValue(5)
        ax.addItem(region)

    # ── PS: baseline subtraction ─────────────────────────────────
    # Applied at the very end, after stitching and dark subtraction — see
    # _ps_compute_stitch(). Same "drag a range, fit a constant (mean),
    # subtract from everything" UX as the TRPL tab's baseline subtraction,
    # routed through the shared _ps_baseline_bar's Confirm/Cancel buttons.
    # Unlike the lifetime-fit-vs-baseline mutual exclusion in the TRPL tab,
    # here it's baseline-selection vs. transition-span-selection that are
    # mutually exclusive (self._ps_mode == "spans").

    def _ps_remove_baseline_preview_item(self):
        if self._ps_baseline_preview_item is not None:
            try:
                self._canvas.plot_item.removeItem(self._ps_baseline_preview_item)
            except Exception:
                pass
            self._ps_baseline_preview_item = None

    def _ps_on_start_baseline_select(self):
        if self._ps_mode == "spans":
            QMessageBox.information(
                self, "Finish span selection first",
                "Click \"✓ Done\" to finish selecting transition spans "
                "before selecting a baseline range."
            )
            return
        if self._ps_mode == "baseline_selecting":
            return

        # Always compute from the pre-baseline stitch (never the already-
        # baseline-subtracted result), so re-selecting after a Reset — or
        # even without one — can't compound an already-applied baseline.
        result = self._ps_compute_stitch_core()
        if result is None:
            return
        wl_out, counts_diff, _counts_raw, ds, _hdr, _dark = result
        self._ps_baseline_stitch_cache = (wl_out, counts_diff, ds[0]["powers"])

        self._ps_baseline_prev_mode = self._ps_mode
        self._ps_mode = "baseline_selecting"
        self._ps_nav_bar.setVisible(False)
        self._ps_refresh_buttons()

        ax = self._ps_draw_stitched(wl_out, counts_diff, ds[0]["powers"],
                                    title="Select baseline range (no signal)")
        x = self._ps_wl_to_x(wl_out)
        x_lo, x_hi = float(np.min(x)), float(np.max(x))
        width = (x_hi - x_lo) * 0.1
        initial = (x_lo, x_lo + width)

        self._ps_baseline_span = DraggableSpan(ax, color=(30, 100, 220, 60), movable=True)
        self._ps_baseline_span.activate(initial_range=initial, bounds=(x_lo, x_hi))
        self._ps_baseline_span.sigRegionSelected.connect(self._ps_on_baseline_span_changed)

        self._ps_btn_baseline_confirm.setEnabled(False)
        self._ps_baseline_bar.setVisible(True)
        self._ps_on_baseline_span_changed(*initial)

        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                "Drag a range with no signal, then Confirm baseline."
            )

    def _ps_on_baseline_span_changed(self, xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        self._ps_baseline_pending_range = (xmin, xmax)

        wl_out, counts_diff, _powers = self._ps_baseline_stitch_cache
        x = self._ps_wl_to_x(wl_out)
        mask = (x >= xmin) & (x <= xmax)
        n_pts = int(np.sum(mask))
        if n_pts < 1:
            self._ps_baseline_pending_value = None
            self._ps_remove_baseline_preview_item()
            self._ps_btn_baseline_confirm.setEnabled(False)
            self._ps_baseline_action_lbl.setText("Baseline range: no points in range — widen it.")
            return

        # One constant per power step: the mean over the selected range of
        # that step's own counts — not one shared value averaged across
        # steps. Every power step's curve is visible while the window is
        # picked (_ps_draw_stitched), and each step is fit independently
        # from its own data in that same wavelength range and later
        # subtracted only from itself (see _ps_compute_stitch()).
        baseline_arr = counts_diff[mask, :].mean(axis=0)   # (n_powers,)
        self._ps_baseline_pending_value = baseline_arr

        self._ps_remove_baseline_preview_item()
        x_curve = np.array([float(np.min(x)), float(np.max(x))])
        y_curve = np.full(2, baseline_arr[0])   # reference line at the lowest power's own constant
        self._ps_baseline_preview_item = self._canvas.plot_item.plot(
            x_curve, y_curve, pen=pg.mkPen("cyan", width=1.5, style=Qt.DashLine)
        )
        self._canvas.draw_idle()

        n_p = counts_diff.shape[1]
        msg = (f"Baseline range: {n_pts} wavelength pt(s) — one constant per "
               f"power step (lowest power = {baseline_arr[0]:.4g} counts, "
               f"{n_p} step(s) total)")
        self._ps_baseline_action_lbl.setText(msg)
        self._ps_btn_baseline_confirm.setEnabled(True)

    def _ps_on_baseline_confirm(self):
        if self._ps_mode != "baseline_selecting" or self._ps_baseline_pending_value is None:
            return
        xmin, xmax = self._ps_baseline_pending_range
        wl_out, _counts_diff, _powers = self._ps_baseline_stitch_cache
        if self._ps_x_axis == "energy":
            lo_wl = min(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
            hi_wl = max(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
        else:
            lo_wl, hi_wl = min(xmin, xmax), max(xmin, xmax)
        self._ps_baseline_range = (lo_wl, hi_wl)
        self._ps_baseline_value = self._ps_baseline_pending_value

        if self._ps_baseline_span is not None:
            self._ps_baseline_span.deactivate()
            self._ps_baseline_span = None
        self._ps_remove_baseline_preview_item()
        self._ps_baseline_pending_range = None
        self._ps_baseline_pending_value = None
        self._ps_baseline_stitch_cache = None

        self._ps_mode = "done"
        self._ps_baseline_bar.setVisible(False)
        self._ps_btn_baseline_reset.setEnabled(True)
        self._ps_refresh_buttons()
        n_p = len(self._ps_baseline_value)
        self._ps_baseline_status_lbl.setText(
            f"Baseline fit per power step over {lo_wl:.5g}–{hi_wl:.5g} nm "
            f"({n_p} step(s); lowest power = {self._ps_baseline_value[0]:.4g} counts). "
            "Each step's own constant subtracted from just that step."
        )
        self._ps_refresh_result_view()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Baseline subtracted per power step ({n_p} step(s))."
            )

    def _ps_on_baseline_cancel(self):
        if self._ps_baseline_span is not None:
            self._ps_baseline_span.deactivate()
            self._ps_baseline_span = None
        self._ps_remove_baseline_preview_item()
        self._ps_baseline_pending_range = None
        self._ps_baseline_pending_value = None
        self._ps_baseline_stitch_cache = None

        prev_mode = self._ps_baseline_prev_mode if self._ps_baseline_prev_mode is not None else "done"
        self._ps_mode = prev_mode
        self._ps_baseline_bar.setVisible(False)
        self._ps_refresh_buttons()
        # Restore whichever view was showing before baseline selection
        # started, not always the stitched-result view — e.g. canceling
        # right after step 1 (before ever stitching) should go back to the
        # simple last-power preview, not jump ahead to the stitched one.
        if prev_mode == "preview":
            self._ps_draw_preview()
        else:
            self._ps_refresh_result_view()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("Baseline selection canceled.")

    def _ps_on_reset_baseline(self):
        if self._ps_baseline_value is None:
            return
        self._ps_baseline_range = None
        self._ps_baseline_value = None
        self._ps_btn_baseline_reset.setEnabled(False)
        self._ps_baseline_status_lbl.setText("No baseline applied.")
        self._ps_refresh_result_view()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("Baseline subtraction reset.")

    def _ps_refresh_result_view(self):
        """Redraw the stitched-result preview reflecting the current
        baseline state — used after confirming/canceling/resetting a
        baseline selection, mirroring _ps_on_preview_result()'s own view."""
        result = self._ps_compute_stitch()
        if result is None:
            return
        wl_out, counts_out, _counts_raw, ds, _hdr, dark_by_label = result
        n_dark      = sum(1 for v in dark_by_label.values() if v is not None)
        is_stitched = len(ds) > 1
        dark_note   = f" (dark-subtracted: {n_dark}/{len(ds)})" if n_dark else ""
        base_title  = "Stitched spectrum" if is_stitched else "Spectrum"
        self._ps_draw_stitched(wl_out, counts_out, ds[0]["powers"], title=base_title + dark_note)

    # ── PS: helpers ──────────────────────────────────────────────

    def _ps_wl_to_x(self, wl):
        wl = np.asarray(wl, dtype=float)
        return _HC_EV_NM / wl if self._ps_x_axis == "energy" else wl

    def _ps_clear_canvas(self):
        self._canvas.set_welcome("Add .origin files and click \"Preview\" to begin.")

    def _ps_refresh_buttons(self):
        has1 = len(self._ps_files) >= 1
        has2 = len(self._ps_files) >= 2
        self._ps_btn_preview.setEnabled(has1)
        self._ps_btn_spans.setEnabled(has2)
        self._ps_btn_prev_result.setEnabled(has1)
        self._ps_btn_stitch.setEnabled(has1)
        self._ps_btn_save_origin.setEnabled(has1)
        self._ps_btn_remove.setEnabled(has1)
        self._ps_btn_clear.setEnabled(has1)
        self._ps_btn_load_dark.setEnabled(bool(self._ps_datasets))
        self._ps_btn_baseline_select.setEnabled(has1 and self._ps_mode != "baseline_selecting")
        self._ps_btn_clear_dark.setEnabled(bool(self._ps_dark_map))

    # ══════════════════════════════════════════════════════════════
    # TRPL (.dat) → HDF5 page
    # ══════════════════════════════════════════════════════════════

    def _build_trpl_page(self):
        page = QWidget()
        sl = QVBoxLayout(page)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # Files group
        g_files = QGroupBox("Input files")
        fl = QVBoxLayout(g_files)
        self._trpl_file_list = QListWidget()
        self._trpl_file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._trpl_file_list.setToolTip("Loaded .dat files, one measurement each.")
        self._trpl_file_list.currentItemChanged.connect(self._trpl_on_file_selected)
        fl.addWidget(self._trpl_file_list)
        fb = QHBoxLayout()
        self._trpl_btn_add    = QPushButton("Add…")
        self._trpl_btn_remove = QPushButton("Remove")
        self._trpl_btn_clear  = QPushButton("Clear")
        self._trpl_btn_add.clicked.connect(self._trpl_on_add_files)
        self._trpl_btn_remove.clicked.connect(self._trpl_on_remove_files)
        self._trpl_btn_clear.clicked.connect(self._trpl_on_clear_files)
        for b in (self._trpl_btn_add, self._trpl_btn_remove, self._trpl_btn_clear):
            fb.addWidget(b)
        fl.addLayout(fb)
        g_files.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_files, stretch=1)

        # Per-file power group
        g_power = QGroupBox("Power (selected file)")
        pl = QVBoxLayout(g_power)
        pl.addWidget(QLabel(
            "Select a file above, enter the power at which it was\n"
            "measured, then click \"Set\"."
        ))
        r_pow = QHBoxLayout()
        r_pow.addWidget(QLabel("Power (mW):"))
        self._trpl_power_input = QLineEdit()
        self._trpl_power_input.setPlaceholderText("required")
        r_pow.addWidget(self._trpl_power_input)
        self._trpl_btn_set_power = QPushButton("Set")
        self._trpl_btn_set_power.clicked.connect(self._trpl_on_set_power)
        r_pow.addWidget(self._trpl_btn_set_power)
        pl.addLayout(r_pow)
        g_power.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_power)

        # Metadata group
        g_meta = QGroupBox("Metadata (saved to HDF5)")
        ml = QVBoxLayout(g_meta)
        r_spot = QHBoxLayout()
        r_spot.addWidget(QLabel("Spot diameter (µm):"))
        self._trpl_meta_spot_diam = QLineEdit()
        self._trpl_meta_spot_diam.setPlaceholderText("required")
        r_spot.addWidget(self._trpl_meta_spot_diam)
        ml.addLayout(r_spot)
        r_rep = QHBoxLayout()
        r_rep.addWidget(QLabel("Rep. rate (MHz):"))
        self._trpl_meta_rep_rate = QLineEdit()
        self._trpl_meta_rep_rate.setPlaceholderText("required")
        r_rep.addWidget(self._trpl_meta_rep_rate)
        ml.addLayout(r_rep)
        sl.addWidget(g_meta)

        # Power calibration group
        g_cal = QGroupBox("Power calibration")
        cl = QVBoxLayout(g_cal)

        r_atbs = QHBoxLayout()
        r_atbs.addWidget(QLabel("atBS:"))
        self._trpl_cal_atbs_lbl = QLabel("—")
        self._trpl_cal_atbs_lbl.setWordWrap(False)
        self._trpl_cal_atbs_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        r_atbs.addWidget(self._trpl_cal_atbs_lbl, stretch=1)
        btn_load_atbs = QPushButton("Load…")
        btn_load_atbs.setFixedWidth(64)
        btn_load_atbs.clicked.connect(self._trpl_on_load_cal_atbs)
        r_atbs.addWidget(btn_load_atbs)
        cl.addLayout(r_atbs)

        r_ats = QHBoxLayout()
        r_ats.addWidget(QLabel("atSample:"))
        self._trpl_cal_atsample_lbl = QLabel("—")
        self._trpl_cal_atsample_lbl.setWordWrap(False)
        self._trpl_cal_atsample_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        r_ats.addWidget(self._trpl_cal_atsample_lbl, stretch=1)
        btn_load_ats = QPushButton("Load…")
        btn_load_ats.setFixedWidth(64)
        btn_load_ats.clicked.connect(self._trpl_on_load_cal_atsample)
        r_ats.addWidget(btn_load_ats)
        cl.addLayout(r_ats)

        btn_cal_auto = QPushButton("Autosearch")
        btn_cal_auto.setToolTip(
            "Search each input file's directory (and its parent) for a\n"
            "'Calibration' folder, and match files with 'atBS'/'atSample'\n"
            "in the name (recursing into per-laser subfolders if present)."
        )
        btn_cal_auto.clicked.connect(self._trpl_on_autosearch_cal)
        cl.addWidget(btn_cal_auto)
        g_cal.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_cal)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        self._trpl_btn_convert = QPushButton("Convert to HDF5")
        bold = QFont(); bold.setBold(True)
        self._trpl_btn_convert.setFont(bold)
        self._trpl_btn_convert.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; }"
            "QPushButton:disabled { background-color: #bbbbbb; color: #666666; }"
        )
        self._trpl_btn_convert.setMinimumHeight(32)
        self._trpl_btn_convert.clicked.connect(self._trpl_on_convert)
        sl.addWidget(self._trpl_btn_convert)

        return page

    # ── TRPL: file management ────────────────────────────────────

    def _trpl_on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select TRPL .dat files", "",
            "TRPL files (*.dat);;All files (*.*)"
        )
        added = 0
        for p in paths:
            if p not in self._trpl_files:
                self._trpl_files.append(p)
                item = QListWidgetItem(self._trpl_item_text(p))
                item.setData(Qt.UserRole, p)
                item.setBackground(QColor("#ffffff"))
                self._trpl_file_list.addItem(item)
                added += 1
        if added:
            self._trpl_refresh_buttons()
            parent = self.parent()
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    f"{len(self._trpl_files)} file(s) loaded. Select a file and "
                    "enter its power, then \"Convert to HDF5\"."
                )

    def _trpl_on_remove_files(self):
        rows = sorted(
            [self._trpl_file_list.row(i) for i in self._trpl_file_list.selectedItems()],
            reverse=True,
        )
        for row in rows:
            item = self._trpl_file_list.takeItem(row)
            path = item.data(Qt.UserRole) if item is not None else None
            if path in self._trpl_files:
                self._trpl_files.remove(path)
            self._trpl_power_map.pop(path, None)
            self._trpl_parsed_cache.pop(path, None)
        self._trpl_refresh_buttons()

    def _trpl_on_clear_files(self):
        self._trpl_files.clear()
        self._trpl_power_map.clear()
        self._trpl_parsed_cache.clear()
        self._trpl_file_list.clear()
        self._canvas.set_welcome("Add .dat files to begin.")
        self._trpl_power_input.clear()
        self._trpl_refresh_buttons()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("Files cleared.")

    def _trpl_item_text(self, path):
        label = os.path.basename(path)
        power = self._trpl_power_map.get(path)
        if power is not None:
            return f"✅  {label}\n    {power:.4g} mW"
        return f"⬜  {label}  — no power set"

    def _trpl_refresh_list_item(self, path):
        for row in range(self._trpl_file_list.count()):
            item = self._trpl_file_list.item(row)
            if item.data(Qt.UserRole) == path:
                item.setText(self._trpl_item_text(path))
                item.setBackground(
                    QColor("#c8e6c9") if path in self._trpl_power_map else QColor("#ffffff")
                )
                break

    # ── TRPL: per-file power ─────────────────────────────────────

    def _trpl_on_file_selected(self, current, _previous):
        path = current.data(Qt.UserRole) if current is not None else None
        if path is None:
            return
        power = self._trpl_power_map.get(path)
        self._trpl_power_input.setText(f"{power:.6g}" if power is not None else "")
        self._trpl_draw_preview(path)

    def _trpl_on_set_power(self):
        item = self._trpl_file_list.currentItem()
        if item is None:
            QMessageBox.information(self, "No selection",
                                    "Select a file in the list first.")
            return
        path = item.data(Qt.UserRole)
        text = self._trpl_power_input.text().strip()
        if not text:
            self._trpl_power_map.pop(path, None)
            self._trpl_refresh_list_item(path)
            return
        try:
            power = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid power",
                                "Enter a numeric power in mW.")
            return
        self._trpl_power_map[path] = power
        self._trpl_refresh_list_item(path)
        self._trpl_refresh_buttons()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Power set: {os.path.basename(path)} ← {power:.4g} mW"
            )

    # ── TRPL: preview ─────────────────────────────────────────────

    def _trpl_parse_cached(self, path):
        parsed = self._trpl_parsed_cache.get(path)
        if parsed is None:
            parsed = _parse_trpl_dat(path)
            self._trpl_parsed_cache[path] = parsed
        return parsed

    def _trpl_draw_preview(self, path):
        try:
            parsed = self._trpl_parse_cached(path)
        except Exception as exc:
            QMessageBox.warning(self, "Parse error",
                                f"Could not parse:\n{path}\n\n{exc}")
            return
        counts = parsed["counts"]
        times  = np.arange(len(counts), dtype=float) * parsed["ns_per_channel"]

        ax = self._canvas.reset_axes()
        ax.setLogMode(y=True)
        # TRPL histograms are ~65536 points — by far the largest single curve
        # in the app. clipToView + auto-downsampling keep panning/zooming
        # fast at any zoom level (only ~1 point/pixel is ever drawn), and
        # antialias=False avoids the app-wide antialiased-rendering default,
        # which combined with tens of thousands of points is what caused the
        # heavy lag here (see plotting/multiline.py's MultiLinePlotter.plot).
        ax.setClipToView(True)
        ax.setDownsampling(auto=True, mode="peak")
        mlp = MultiLinePlotter(ax, CategoricalScheme())
        mlp.plot(times, counts, index=0, width=1.0, antialias=False)
        ax.setLabel("bottom", "Time (ns)")
        ax.setLabel("left", "Counts")
        ax.setTitle(os.path.basename(path))
        ax.showGrid(x=True, y=True, alpha=0.3)
        self._canvas.draw_idle()

    # ── TRPL: power calibration ──────────────────────────────────

    def _trpl_cal_load(self, path, role):
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
            self._trpl_cal_atbs_data = (hwp, pows)
            self._trpl_cal_atbs_path = path
            self._trpl_cal_atbs_lbl.setText(short)
        else:
            self._trpl_cal_atsample_data = (hwp, pows)
            self._trpl_cal_atsample_path = path
            self._trpl_cal_atsample_lbl.setText(short)
        self._trpl_refresh_buttons()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Calibration {role}: {len(hwp)} HWP steps loaded from {short}"
            )
        return True

    def _trpl_on_load_cal_atbs(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select atBS calibration file", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if path:
            self._trpl_cal_load(path, "atBS")

    def _trpl_on_load_cal_atsample(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select atSample calibration file", "",
            "Origin files (*.origin);;All files (*.*)"
        )
        if path:
            self._trpl_cal_load(path, "atSample")

    def _trpl_on_autosearch_cal(self):
        """Search input-file directories, their parents, and any 'Calibration'
        folder found in either (recursing into per-laser subfolders such as
        Calibration/TiSa or Calibration/HeNe) for files whose names contain
        'atbs' or 'atsample' (case-insensitive).
        """
        if not self._trpl_files:
            QMessageBox.information(self, "No files loaded",
                                    "Add .dat files first.")
            return

        file_dirs = list(dict.fromkeys(os.path.dirname(p) for p in self._trpl_files))

        search_dirs, seen_dirs = [], set()

        def _add(d):
            if d and d not in seen_dirs and os.path.isdir(d):
                search_dirs.append(d)
                seen_dirs.add(d)

        for d in file_dirs:
            _add(d)
            _add(os.path.dirname(d))

        cal_roots = []
        for d in search_dirs:
            try:
                for entry in os.listdir(d):
                    if entry.lower() == "calibration":
                        sub = os.path.join(d, entry)
                        if os.path.isdir(sub):
                            cal_roots.append(sub)
            except OSError:
                pass

        atbs_path = atsample_path = None

        # Same-level check first (flat layout, as in the PS Stitch/Convert flow).
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

        # Recurse into any 'Calibration' folder found (handles per-laser
        # subfolders such as Calibration/TiSa, Calibration/HeNe). Unlike the
        # same-level check above, this collects *every* candidate rather than
        # stopping at the first match: a Calibration folder with more than one
        # laser subfolder is genuinely ambiguous (e.g. HeNe vs. TiSa), and
        # silently picking one could apply the wrong calibration to the data.
        atbs_candidates, atsample_candidates = [], []
        if not (atbs_path and atsample_path):
            for root_dir in cal_roots:
                for dirpath, _dirnames, filenames in os.walk(root_dir):
                    for fname in filenames:
                        if not fname.lower().endswith(".origin"):
                            continue
                        fl = fname.lower()
                        fp = os.path.join(dirpath, fname)
                        if "atbs" in fl:
                            atbs_candidates.append(fp)
                        if "atsample" in fl:
                            atsample_candidates.append(fp)

        ambiguous = len(atbs_candidates) > 1 or len(atsample_candidates) > 1
        if not ambiguous:
            if atbs_path is None and len(atbs_candidates) == 1:
                atbs_path = atbs_candidates[0]
            if atsample_path is None and len(atsample_candidates) == 1:
                atsample_path = atsample_candidates[0]

        found = []
        if atbs_path and self._trpl_cal_load(atbs_path, "atBS"):
            found.append(f"atBS: {os.path.basename(atbs_path)}")
        if atsample_path and self._trpl_cal_load(atsample_path, "atSample"):
            found.append(f"atSample: {os.path.basename(atsample_path)}")

        parent = self.parent()
        if ambiguous:
            names = "\n".join(
                os.path.relpath(p, os.path.commonpath(cal_roots)) if cal_roots else p
                for p in (atbs_candidates + atsample_candidates)
            )
            QMessageBox.information(
                self, "Multiple calibration files found",
                "More than one candidate atBS/atSample calibration file was "
                "found (e.g. separate HeNe/TiSa subfolders) — pick the "
                "correct pair manually with \"Load…\":\n\n" + names
            )
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    "Calibration autosearch: multiple candidates found, "
                    "select manually."
                )
        elif found:
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    "Calibration autosearch found: " + ", ".join(found)
                )
        else:
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    "Calibration autosearch: no matching files found "
                    "(looking for 'atBS'/'atSample' in name in input dirs, "
                    "their parent, and any 'Calibration' folder)."
                )

    # ── TRPL: convert ────────────────────────────────────────────

    def _trpl_on_convert(self):
        if not self._trpl_files:
            return

        if self._trpl_cal_atbs_data is None or self._trpl_cal_atsample_data is None:
            QMessageBox.warning(
                self, "Calibration required",
                "Load both an atBS and an atSample power calibration file "
                "first (or use \"Autosearch\")."
            )
            return

        try:
            rep_rate = float(self._trpl_meta_rep_rate.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Rep. rate required",
                                "Enter a numeric repetition rate in MHz.")
            return
        try:
            spot_diam = float(self._trpl_meta_spot_diam.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Spot diameter required",
                                "Enter a numeric spot diameter in µm.")
            return

        to_convert = [p for p in self._trpl_files if p in self._trpl_power_map]
        skipped_no_power = [p for p in self._trpl_files if p not in self._trpl_power_map]
        if not to_convert:
            QMessageBox.information(
                self, "No power set",
                "Set a power (mW) for at least one file before converting."
            )
            return

        out_paths = {
            p: os.path.join(os.path.dirname(p),
                            os.path.splitext(os.path.basename(p))[0] + ".h5")
            for p in to_convert
        }
        existing = [op for op in out_paths.values() if os.path.exists(op)]
        if existing:
            reply = QMessageBox.question(
                self, "Overwrite existing files?",
                f"{len(existing)} output file(s) already exist and will be "
                "overwritten:\n\n" + "\n".join(os.path.basename(p) for p in existing[:10])
                + ("\n…" if len(existing) > 10 else "") + "\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        power_cal = {"atBS": self._trpl_cal_atbs_data, "atSample": self._trpl_cal_atsample_data}

        succeeded, failed = [], []
        for path in to_convert:
            try:
                parsed = self._trpl_parse_cached(path)
                _write_trpl_h5_file(
                    out_paths[path], parsed["counts"], parsed["ns_per_channel"],
                    parsed["date_str"], self._trpl_power_map[path],
                    rep_rate, spot_diam, power_cal=power_cal,
                )
                succeeded.append(out_paths[path])
            except Exception as exc:
                failed.append(f"{os.path.basename(path)}: {exc}")

        parent = self.parent()
        msg = f"Converted {len(succeeded)}/{len(to_convert)} file(s) to HDF5."
        if skipped_no_power:
            msg += f"\n{len(skipped_no_power)} file(s) skipped (no power set)."
        if failed:
            msg += "\n\nFailed:\n" + "\n".join(failed)
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Converted {len(succeeded)}/{len(to_convert)} file(s)."
            )
        QMessageBox.information(self, "Conversion complete", msg)

    # ── TRPL: helpers ────────────────────────────────────────────

    def _trpl_refresh_buttons(self):
        has1 = len(self._trpl_files) >= 1
        self._trpl_btn_remove.setEnabled(has1)
        self._trpl_btn_clear.setEnabled(has1)
        self._trpl_btn_convert.setEnabled(has1 and bool(self._trpl_power_map))
