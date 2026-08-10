import numpy as np
import pyqtgraph as pg

from .theme import TAB10, PLASMA


class LineColorScheme:
    """Resolves a (QColor, alpha) pair for a line given its index/value."""

    def resolve(self, index=None, value=None):
        raise NotImplementedError


class CategoricalScheme(LineColorScheme):
    """Colors lines by index, cycling through a fixed qualitative palette
    (default: tab10). Replaces cm.get_cmap("tab10")(k % 10) per-file coloring."""

    def __init__(self, palette=None):
        self._palette = palette or TAB10

    def resolve(self, index=None, value=None):
        color = self._palette[(index or 0) % len(self._palette)]
        return pg.mkColor(color), 1.0


class SequentialScheme(LineColorScheme):
    """Colors lines by a continuous value normalized into a colormap (default:
    plasma). Replaces cm.get_cmap("plasma", n)(p) per-power coloring."""

    def __init__(self, vmin, vmax, cmap=None):
        self._cmap = cmap or PLASMA
        self._vmin = vmin
        self._vmax = vmax

    def resolve(self, index=None, value=None):
        span = (self._vmax - self._vmin) or 1.0
        t = (value - self._vmin) / span
        t = min(max(t, 0.0), 1.0)
        return self._cmap.map(t, mode="qcolor"), 1.0


class LogAlphaRamp(LineColorScheme):
    """Wraps a base scheme, log-normalizing `value` against a fixed set of
    known values into an alpha ramp on top of the base scheme's color.
    Replaces visualizer.py's hand-rolled log-scaled `_alpha()` closure."""

    def __init__(self, base_scheme, values, alpha_range=(0.25, 1.0)):
        self._base = base_scheme
        positive = [v for v in values if v > 0]
        log_vals = np.log10(positive) if positive else np.array([0.0])
        self._lo, self._hi = float(log_vals.min()), float(log_vals.max())
        self._alpha_lo, self._alpha_hi = alpha_range

    def resolve(self, index=None, value=None):
        color, _ = self._base.resolve(index=index, value=value)
        if value is None or value <= 0 or self._hi == self._lo:
            return color, self._alpha_hi
        t = (np.log10(value) - self._lo) / (self._hi - self._lo)
        t = min(max(t, 0.0), 1.0)
        alpha = self._alpha_lo + t * (self._alpha_hi - self._alpha_lo)
        return color, alpha


class MultiLinePlotter:
    """Plots one line per call, resolving its (color, alpha) from a
    LineColorScheme. Shared by stitch.py (CategoricalScheme), visualizer.py
    (CategoricalScheme+LogAlphaRamp or SequentialScheme), and analysis.py
    (SequentialScheme)."""

    def __init__(self, plot_item, scheme: LineColorScheme):
        self._plot_item = plot_item
        self._scheme = scheme

    def plot(self, x, y, index=None, value=None, label=None, width=1.0, antialias=None):
        """antialias=None (default) inherits the global pg.setConfigOptions()
        setting, matching prior behavior exactly. Pass antialias=False for
        very large curves (tens of thousands of points+) — the app-wide
        default is antialiased, which combined with a non-default pen width
        forces PyQtGraph's slow per-segment rendering path; fine for the
        few-thousand-point curves every other tab plots, but not for e.g. a
        65536-sample TRPL histogram (see tabs/stitch.py's TRPL-conversion
        mode)."""
        color, alpha = self._scheme.resolve(index=index, value=value)
        pen = pg.mkPen(color=color, width=width)
        kwargs = {} if antialias is None else {"antialias": antialias}
        item = self._plot_item.plot(x, y, pen=pen, name=label, **kwargs)
        item.setOpacity(alpha)
        return item
