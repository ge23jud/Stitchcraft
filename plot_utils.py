"""Pure matplotlib figure-construction logic for the Plot tab — no Qt
imports, mirroring sem_utils.py's UI-free convention.
"""
import math
import os

import numpy as np
from matplotlib import colormaps
from matplotlib.figure import Figure


def _legend_ncols(n_entries, fig_height_in, fontsize=8):
    """Number of legend columns needed so a single-column legend (plus its
    title) wouldn't need more vertical space than the figure has — which
    otherwise makes constrained_layout shrink the axes to fit it, squeezing
    the actual plot."""
    row_in    = fontsize * 1.6 / 72.0   # ~ default matplotlib legend row height
    title_in  = row_in                  # reserve one row's worth for the title
    usable_in = max(fig_height_in - title_in, row_in)
    max_rows  = max(1, int(usable_in / row_in))
    return max(1, math.ceil(n_entries / max_rows))


def build_spectrum_figure(data, figsize=(8, 5)):
    """Build a Figure of counts vs. energy for one HDF5 spectrum (see
    io_utils._read_h5_spectrum()), one line per power step, coloured
    dark-to-light by ascending pump fluence (or power, as a fallback),
    with the legend placed outside the axes.

    Returns (fig, missing_fluence): missing_fluence is True if the file had
    no Pump_fluence dataset, so line labels fall back to power (mW).
    """
    energy   = data["energy"]
    counts   = data["counts"]
    fluence  = data.get("pump_fluence")
    power_mW = data["power_mW"]
    n_powers = counts.shape[1]
    missing_fluence = fluence is None

    values = fluence if fluence is not None else power_mW
    order  = np.argsort(values)
    cmap   = colormaps["plasma"]
    colors = cmap(np.linspace(0.05, 0.95, n_powers))

    fig = Figure(figsize=figsize, constrained_layout=True)
    ax = fig.subplots()
    for rank, i in enumerate(order):
        # Fluence is stored internally in mJ/cm²; the legend shows µJ/cm²
        # (unit given in the legend title, so left off each entry) — power
        # keeps its own "mW" suffix since it's a different, unlabelled quantity.
        label = (f"{fluence[i] * 1e3:.4g}" if not missing_fluence
                 else f"{power_mW[i]:.4g} mW")
        ax.plot(energy, counts[:, i], color=colors[rank], linewidth=1.0, label=label)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("PL Intensity (arb. units)")
    ax.set_title(data["label"])
    ax.grid(True, alpha=0.3)
    legend_title = "Pump Fluence (uJ/cm²)" if not missing_fluence else "Power (no fluence in file)"
    ncols = _legend_ncols(n_powers, figsize[1])
    ax.legend(title=legend_title, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, fontsize=8, title_fontsize=8, ncol=ncols)
    return fig, missing_fluence


def png_path_for(source_path):
    """Return the output .png path: same directory and basename as source_path."""
    base, _ = os.path.splitext(source_path)
    return base + ".png"
