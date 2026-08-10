from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QSizePolicy
import pyqtgraph as pg

from .theme import CANVAS_BG


class PGCanvas(pg.GraphicsLayoutWidget):
    """Drop-in replacement for the old Matplotlib-backed `_MplCanvas`.

    Call-site compatible with the old API: `reset_axes()` clears everything
    and returns a fresh plot item (in place of a fresh matplotlib Axes);
    `draw_idle()` is a no-op kept so existing call sites don't need to
    change during the port (PyQtGraph repaints automatically).
    """

    sigDataClicked = pyqtSignal(float, float, object)  # x, y, Qt.MouseButton

    def __init__(self, parent=None,
                 welcome_msg="Add .origin files and click\n\"Preview\" to begin."):
        super().__init__(parent=parent)
        self.setBackground(CANVAS_BG)
        self.plot_item = self.addPlot(row=0, col=0)
        self._legend_widget = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        self.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self._welcome_msg = welcome_msg
        self._welcome()

    def reset_axes(self):
        """Wipe curves/ROIs/lines/labels AND any colorbar/legend, and return
        the (now empty) plot item. Colorbar/legend removal is explicit because
        it lives in a sibling GraphicsLayout cell, not inside the plot item's
        own ViewBox, so plot_item.clear() alone would leave it behind."""
        self.plot_item.clear()
        self.plot_item.showAxis("bottom")
        self.plot_item.showAxis("left")
        self.plot_item.setLabel("bottom", "")
        self.plot_item.setLabel("left", "")
        self.plot_item.setTitle(None)
        self.plot_item.setLogMode(x=False, y=False)
        self.plot_item.showGrid(x=False, y=False)
        vb = self.plot_item.getViewBox()
        vb.setMouseEnabled(x=True, y=True)
        # Reset to a known-small range synchronously (not just re-enabling
        # auto-range mode) so a stale, possibly large range from whatever was
        # plotted before this clear() can't leak into a subsequent
        # setLogMode() call before the view has repainted — pyqtgraph computes
        # 10**range eagerly on some axis updates, which overflows on a stale
        # large linear range interpreted as a log-mode one.
        vb.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
        vb.enableAutoRange()
        self._remove_legend_widget()
        return self.plot_item

    def draw_idle(self):
        """Force any axis still in auto-range mode to refit its data right
        now, rather than waiting for pyqtgraph's next paint-triggered
        recompute (`ViewBox.prepareForPaint`) — which can lag or never fire
        before the user looks at the plot (e.g. a redraw on a background
        tab). Axes an explicit setXRange/setYRange already took out of
        auto-range mode are left untouched."""
        self.plot_item.getViewBox().updateAutoRange()

    def add_colorbar_legend(self, gradient_legend):
        self._remove_legend_widget()
        self._legend_widget = gradient_legend
        self.addItem(gradient_legend, row=0, col=1)

    def _remove_legend_widget(self):
        if self._legend_widget is not None:
            try:
                self.removeItem(self._legend_widget)
            except Exception:
                pass
            self._legend_widget = None

    def _on_scene_clicked(self, event):
        vb = self.plot_item.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return
        pt = vb.mapSceneToView(event.scenePos())
        self.sigDataClicked.emit(pt.x(), pt.y(), event.button())

    def set_welcome(self, msg):
        """Show a welcome/placeholder message, replacing whatever the canvas
        was constructed with — used by tabs that share one canvas across
        multiple modes (e.g. a "File type" switch) to give a mode-appropriate
        placeholder instead of the one fixed at __init__ time."""
        self._welcome_msg = msg
        self._welcome()

    def _welcome(self):
        ax = self.reset_axes()
        ax.hideAxis("bottom")
        ax.hideAxis("left")
        vb = ax.getViewBox()
        vb.setRange(xRange=(0, 1), yRange=(0, 1), padding=0)
        vb.setMouseEnabled(x=False, y=False)
        text = pg.TextItem(self._welcome_msg, color="#666666", anchor=(0.5, 0.5))
        text.setPos(0.5, 0.5)
        # Added via plot_item.addItem (not vb.addItem) so the next
        # reset_axes()'s plot_item.clear() actually removes it — added
        # straight to the ViewBox it survives clear() (clear() only walks
        # PlotItem's own tracked item list) and its anchor point then leaks
        # into every future auto-range calculation as a permanent (0.5, 0.5)
        # data bound, which can dwarf real data far smaller than that and
        # squash it flat. ignoreBounds=True keeps it out of that calculation
        # even while it's the only thing showing.
        self.plot_item.addItem(text, ignoreBounds=True)
