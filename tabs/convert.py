import os
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QListWidget, QListWidgetItem,
    QLineEdit, QFileDialog, QMessageBox, QSizePolicy,
    QAbstractItemView, QFrame, QScrollArea, QStyle,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plotting import PGCanvas, _COMPACT_BTN_STYLE, MultiLinePlotter, CategoricalScheme, make_pg_toolbar
from io_utils import _parse_trpl_dat, _write_trpl_h5_file, _parse_power_calibration


class ConvertTab(QWidget):
    """"Convert": batch-convert non-PL measurement files to HDF5.

    Currently supports one file type — TRPL histograms (PicoHarp .dat) — via
    the "File type" dropdown; more converters can be added later by extending
    that dropdown and branching on its selection.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── State ────────────────────────────────────────────────
        self._files: list        = []
        self._power_map: dict    = {}   # {path: power_mW}
        self._parsed_cache: dict = {}   # {path: {date_str, ns_per_channel, counts}}

        # ── Power calibration state (mirrors Stitch / Convert tab) ─
        self._cal_atbs_data     = None   # (hwp_arr, powers_W) or None
        self._cal_atsample_data = None
        self._cal_atbs_path     = None
        self._cal_atsample_path = None

        # ── Build UI ────────────────────────────────────────────────
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        scroll = QScrollArea()
        scrollbar_w = scroll.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        scroll.setFixedWidth(340 + scrollbar_w)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar = QWidget()
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)

        # File type group
        g_type = QGroupBox("File type")
        tl = QHBoxLayout(g_type)
        tl.addWidget(QLabel("Convert:"))
        self._type_combo = QComboBox()
        self._type_combo.addItem("TRPL (.dat) → HDF5", "trpl")
        tl.addWidget(self._type_combo, stretch=1)
        sl.addWidget(g_type)

        # Files group
        g_files = QGroupBox("Input files")
        fl = QVBoxLayout(g_files)
        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._file_list.setToolTip("Loaded .dat files, one measurement each.")
        self._file_list.currentItemChanged.connect(self._on_file_selected)
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
        self._power_input = QLineEdit()
        self._power_input.setPlaceholderText("required")
        r_pow.addWidget(self._power_input)
        self._btn_set_power = QPushButton("Set")
        self._btn_set_power.clicked.connect(self._on_set_power)
        r_pow.addWidget(self._btn_set_power)
        pl.addLayout(r_pow)
        g_power.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_power)

        # Metadata group
        g_meta = QGroupBox("Metadata (saved to HDF5)")
        ml = QVBoxLayout(g_meta)
        r_spot = QHBoxLayout()
        r_spot.addWidget(QLabel("Spot diameter (µm):"))
        self._meta_spot_diam = QLineEdit()
        self._meta_spot_diam.setPlaceholderText("required")
        r_spot.addWidget(self._meta_spot_diam)
        ml.addLayout(r_spot)
        r_rep = QHBoxLayout()
        r_rep.addWidget(QLabel("Rep. rate (MHz):"))
        self._meta_rep_rate = QLineEdit()
        self._meta_rep_rate.setPlaceholderText("required")
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
        btn_cal_auto.setToolTip(
            "Search each input file's directory (and its parent) for a\n"
            "'Calibration' folder, and match files with 'atBS'/'atSample'\n"
            "in the name (recursing into per-laser subfolders if present)."
        )
        btn_cal_auto.clicked.connect(self._on_autosearch_cal)
        cl.addWidget(btn_cal_auto)
        g_cal.setStyleSheet(_COMPACT_BTN_STYLE)
        sl.addWidget(g_cal)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        sl.addWidget(sep)

        self._btn_convert = QPushButton("Convert to HDF5")
        bold = QFont(); bold.setBold(True)
        self._btn_convert.setFont(bold)
        self._btn_convert.setStyleSheet(
            "QPushButton { background-color: #4caf50; color: white; }"
            "QPushButton:disabled { background-color: #bbbbbb; color: #666666; }"
        )
        self._btn_convert.setMinimumHeight(32)
        self._btn_convert.clicked.connect(self._on_convert)
        sl.addWidget(self._btn_convert)

        scroll.setWidget(sidebar)
        layout.addWidget(scroll)

        # ── Right: canvas (preview of the selected file) ───────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)
        self._canvas  = PGCanvas(right, welcome_msg="Add .dat files to begin.")
        self._toolbar = make_pg_toolbar(self._canvas, right)
        rl.addWidget(self._toolbar)
        rl.addWidget(self._canvas, stretch=1)
        layout.addWidget(right, stretch=1)

        self._refresh_buttons()

    # ── File management ──────────────────────────────────────────

    def _on_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select TRPL .dat files", "",
            "TRPL files (*.dat);;All files (*.*)"
        )
        added = 0
        for p in paths:
            if p not in self._files:
                self._files.append(p)
                item = QListWidgetItem(self._item_text(p))
                item.setData(Qt.UserRole, p)
                item.setBackground(QColor("#ffffff"))
                self._file_list.addItem(item)
                added += 1
        if added:
            self._refresh_buttons()
            parent = self.parent()
            if parent is not None and hasattr(parent, "statusBar"):
                parent.statusBar().showMessage(
                    f"{len(self._files)} file(s) loaded. Select a file and "
                    "enter its power, then \"Convert to HDF5\"."
                )

    def _on_remove_files(self):
        rows = sorted(
            [self._file_list.row(i) for i in self._file_list.selectedItems()],
            reverse=True,
        )
        for row in rows:
            item = self._file_list.takeItem(row)
            path = item.data(Qt.UserRole) if item is not None else None
            if path in self._files:
                self._files.remove(path)
            self._power_map.pop(path, None)
            self._parsed_cache.pop(path, None)
        self._refresh_buttons()

    def _on_clear_files(self):
        self._files.clear()
        self._power_map.clear()
        self._parsed_cache.clear()
        self._file_list.clear()
        self._canvas._welcome()
        self._power_input.clear()
        self._refresh_buttons()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("Files cleared.")

    def _item_text(self, path):
        label = os.path.basename(path)
        power = self._power_map.get(path)
        if power is not None:
            return f"✅  {label}\n    {power:.4g} mW"
        return f"⬜  {label}  — no power set"

    def _refresh_list_item(self, path):
        for row in range(self._file_list.count()):
            item = self._file_list.item(row)
            if item.data(Qt.UserRole) == path:
                item.setText(self._item_text(path))
                item.setBackground(
                    QColor("#c8e6c9") if path in self._power_map else QColor("#ffffff")
                )
                break

    # ── Per-file power ───────────────────────────────────────────

    def _on_file_selected(self, current, _previous):
        path = current.data(Qt.UserRole) if current is not None else None
        if path is None:
            return
        power = self._power_map.get(path)
        self._power_input.setText(f"{power:.6g}" if power is not None else "")
        self._draw_preview(path)

    def _on_set_power(self):
        item = self._file_list.currentItem()
        if item is None:
            QMessageBox.information(self, "No selection",
                                    "Select a file in the list first.")
            return
        path = item.data(Qt.UserRole)
        text = self._power_input.text().strip()
        if not text:
            self._power_map.pop(path, None)
            self._refresh_list_item(path)
            return
        try:
            power = float(text)
        except ValueError:
            QMessageBox.warning(self, "Invalid power",
                                "Enter a numeric power in mW.")
            return
        self._power_map[path] = power
        self._refresh_list_item(path)
        self._refresh_buttons()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
                f"Power set: {os.path.basename(path)} ← {power:.4g} mW"
            )

    # ── Preview ──────────────────────────────────────────────────

    def _parse_cached(self, path):
        parsed = self._parsed_cache.get(path)
        if parsed is None:
            parsed = _parse_trpl_dat(path)
            self._parsed_cache[path] = parsed
        return parsed

    def _draw_preview(self, path):
        try:
            parsed = self._parse_cached(path)
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

    # ── Power calibration (mirrors Stitch / Convert tab) ──────────

    def _cal_load(self, path, role):
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
            self._cal_atbs_data = (hwp, pows)
            self._cal_atbs_path = path
            self._cal_atbs_lbl.setText(short)
        else:
            self._cal_atsample_data = (hwp, pows)
            self._cal_atsample_path = path
            self._cal_atsample_lbl.setText(short)
        self._refresh_buttons()
        parent = self.parent()
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(
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
        """Search input-file directories, their parents, and any 'Calibration'
        folder found in either (recursing into per-laser subfolders such as
        Calibration/TiSa or Calibration/HeNe) for files whose names contain
        'atbs' or 'atsample' (case-insensitive).
        """
        if not self._files:
            QMessageBox.information(self, "No files loaded",
                                    "Add .dat files first.")
            return

        file_dirs = list(dict.fromkeys(os.path.dirname(p) for p in self._files))

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

        # Same-level check first (flat layout, as in the Stitch / Convert tab).
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
        if atbs_path and self._cal_load(atbs_path, "atBS"):
            found.append(f"atBS: {os.path.basename(atbs_path)}")
        if atsample_path and self._cal_load(atsample_path, "atSample"):
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

    # ── Convert ──────────────────────────────────────────────────

    def _on_convert(self):
        if not self._files:
            return

        if self._cal_atbs_data is None or self._cal_atsample_data is None:
            QMessageBox.warning(
                self, "Calibration required",
                "Load both an atBS and an atSample power calibration file "
                "first (or use \"Autosearch\")."
            )
            return

        try:
            rep_rate = float(self._meta_rep_rate.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Rep. rate required",
                                "Enter a numeric repetition rate in MHz.")
            return
        try:
            spot_diam = float(self._meta_spot_diam.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Spot diameter required",
                                "Enter a numeric spot diameter in µm.")
            return

        to_convert = [p for p in self._files if p in self._power_map]
        skipped_no_power = [p for p in self._files if p not in self._power_map]
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

        power_cal = {"atBS": self._cal_atbs_data, "atSample": self._cal_atsample_data}

        succeeded, failed = [], []
        for path in to_convert:
            try:
                parsed = self._parse_cached(path)
                _write_trpl_h5_file(
                    out_paths[path], parsed["counts"], parsed["ns_per_channel"],
                    parsed["date_str"], self._power_map[path],
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

    # ── Helpers ──────────────────────────────────────────────────

    def _refresh_buttons(self):
        has1 = len(self._files) >= 1
        self._btn_remove.setEnabled(has1)
        self._btn_clear.setEnabled(has1)
        self._btn_convert.setEnabled(has1 and bool(self._power_map))
