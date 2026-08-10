#!/usr/bin/env python3
"""PL Helper utilities."""

import os
import re
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button as _MplButton
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid


def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _double_gaussian_diff(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
    """G1(x) − G2(x): large Gaussian minus small Gaussian."""
    return (_gaussian(x, amp1, mu1, sigma1)
            - _gaussian(x, amp2, mu2, sigma2))


def _find_symmetric_95_bounds(x, deriv, center):
    """Return (x_lo, x_hi) symmetric about *center* capturing 95.4 % of total ∫dy."""
    total = trapezoid(deriv, x)
    target = 0.954 * total
    d_lo, d_hi = 0.0, float(x[-1] - x[0])
    for _ in range(64):
        d = 0.5 * (d_lo + d_hi)
        mask = (x >= center - d) & (x <= center + d)
        if mask.sum() < 2:
            d_lo = d
            continue
        if trapezoid(deriv[mask], x[mask]) < target:
            d_lo = d
        else:
            d_hi = d
    d = 0.5 * (d_lo + d_hi)
    return center - d, center + d


def spotsize(angle: float, is_derivative: bool = False):
    """Interactive spot-size analysis from a two-column Excel scan.

    Parameters
    ----------
    angle : float
        Angle in degrees. The final result is (x_hi − x_lo) × sin(angle).
    is_derivative : bool, optional
        If False (default) the second column is treated as raw y data and the
        numerical derivative dy/dx is computed before analysis.
        If True the second column is already a derivative and is used directly.

    Workflow
    --------
    1. A file-picker dialog opens – select an Excel file with two columns (x, y).
    2. An interactive plot of dy/dx is displayed.  Click and drag to select two
       x-spans, each containing a Gaussian-shaped peak.
    3. A Gaussian is fitted to each selected span.
    4. For each fit the symmetric integration window [μ−d, μ+d] is determined
       such that ∫[μ−d, μ+d] |dy/dx| dx = 95.4 % of the total ∫ |dy/dx| dx.
    5. A final plot shows the derivative, both Gaussian fits, the 95.4 % regions,
       and the difference Δx between the two Gaussian centres.

    Returns
    -------
    float
        Δx – the absolute difference between the two determined x-positions.
    """
    # --- file selection ---
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select Excel file",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
    )
    root.destroy()
    if not filepath:
        print("No file selected.")
        return None

    df = pd.read_excel(filepath, header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]

    if x.size < 10:
        print(f"Too few data points ({x.size}) after cleaning. Check the file.")
        return None

    deriv = y if is_derivative else np.gradient(y, x)

    # --- interactive span selection ---
    COLORS = ["tab:green", "tab:orange"]
    spans: list[tuple[float, float]] = []

    fig_sel, ax_sel = plt.subplots(figsize=(10, 5))
    ax_sel.plot(x, deriv, color="tab:blue", lw=1.5)
    ax_sel.set_xlabel("x")
    ax_sel.set_ylabel("dy/dx")
    ax_sel.grid(True, alpha=0.4)
    title_text = ax_sel.set_title("Select span 1 – click and drag on the plot")

    def _on_select(xmin: float, xmax: float) -> None:
        if len(spans) >= 2:
            return
        spans.append((xmin, xmax))
        ax_sel.axvspan(xmin, xmax, alpha=0.25, color=COLORS[len(spans) - 1],
                       label=f"Span {len(spans)}  [{xmin:.4g}, {xmax:.4g}]")
        ax_sel.legend(loc="upper right", fontsize=8)
        if len(spans) < 2:
            title_text.set_text("Select span 2 – click and drag on the plot")
        else:
            title_text.set_text("Both spans selected – close this window to continue")
        fig_sel.canvas.draw_idle()

    # Must keep a reference; otherwise Python garbage-collects the widget and
    # no events are delivered.
    _selector = SpanSelector(
        ax_sel, _on_select, "horizontal", useblit=False,
        props=dict(alpha=0.15, facecolor="lightyellow"),
    )
    plt.tight_layout()
    plt.show()

    if len(spans) < 2:
        print("Two spans are required. Aborting.")
        return None

    # --- single Gaussian fit across both spans combined ---
    combined_mask = np.zeros(x.size, dtype=bool)
    for xmin, xmax in spans:
        combined_mask |= (x >= xmin) & (x <= xmax)

    xs, ds = x[combined_mask], deriv[combined_mask]
    if xs.size < 5:
        print(f"Combined spans contain only {xs.size} point(s) – too few to fit.")
        return None

    peak_idx = int(np.argmax(np.abs(ds)))
    span_width = max(s[1] - s[0] for s in spans)
    p0 = [ds[peak_idx], xs[peak_idx], span_width / 4]
    try:
        popt, _ = curve_fit(_gaussian, xs, ds, p0=p0, maxfev=10_000)
    except RuntimeError:
        print("Gaussian fit failed to converge.")
        return None

    amp, mu, sigma = popt[0], popt[1], abs(popt[2])
    x_lo, x_hi = _find_symmetric_95_bounds(x, deriv, mu)
    delta = x_hi - x_lo
    delta_corrected = delta * np.sin(np.radians(angle))

    integral_total = trapezoid(deriv, x)
    bounds_mask = (x >= x_lo) & (x <= x_hi)
    integral_bounds = trapezoid(deriv[bounds_mask], x[bounds_mask])

    print(f"Gaussian fit:  μ = {mu:.6g},  σ = {sigma:.6g}")
    print(f"95.4 % bounds: [{x_lo:.6g}, {x_hi:.6g}]  Δx = {delta:.6g}")
    print(f"Δx × sin({angle}°) = {delta_corrected:.6g}")
    print(f"Integral total: {integral_total:.6g}  |  Integral [x_lo, x_hi]: {integral_bounds:.6g}")

    # --- result plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, deriv, color="tab:blue", lw=1.5, label="dy/dx", zorder=3)

    ax.fill_between(x[bounds_mask], deriv[bounds_mask], alpha=0.30,
                    color="tab:green", label=f"95.4 % area  [{x_lo:.4g}, {x_hi:.4g}]")

    x_fit = np.linspace(x_lo, x_hi, 500)
    ax.plot(x_fit, _gaussian(x_fit, amp, mu, sigma),
            "--", color="tab:green", lw=2, label=f"Gaussian fit  μ={mu:.5g},  σ={sigma:.5g}")
    ax.axvline(mu, color="tab:green", lw=1.2, ls=":", zorder=4)

    # expand upper y-limit to make room for the arrow and text
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + 0.25 * (y_hi - y_lo))
    y_lo, y_hi = ax.get_ylim()

    span = y_hi - y_lo
    y_arrow = y_hi - 0.08 * span
    ax.annotate("", xy=(x_hi, y_arrow), xytext=(x_lo, y_arrow),
                arrowprops=dict(arrowstyle="<->", color="red", lw=2))
    ax.text(mu, y_arrow + 0.02 * span,
            f"Δx = {delta:.5g}  |  Δx × sin({angle}°) = {delta_corrected:.5g}\n"
            f"∫dy total = {integral_total:.4g}  |  ∫dy window = {integral_bounds:.4g}",
            ha="center", va="bottom", color="red", fontsize=10, fontweight="bold")

    ax.set_xlabel("x")
    ax.set_ylabel("dy/dx")
    ax.set_title(f"Derivative · Gaussian fit · Δx × sin({angle}°) = {delta_corrected:.5g}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    return delta


def _parse_origin_spectrum(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse an Origin .origin spectrum file; return (wavelength, counts) arrays."""
    rows = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            try:
                wl = float(parts[0])
                counts = float(parts[1])
                rows.append((wl, counts))
            except (ValueError, IndexError):
                continue
    if not rows:
        raise ValueError(f"No numeric data found in {filepath}")
    arr = np.array(rows)
    order = np.argsort(arr[:, 0])
    return arr[order, 0], arr[order, 1]


def pxl_corr():
    """Dark-subtract two PL spectra; fit G1−G2 interactively with live preview.

    Workflow
    --------
    1. File-picker dialog – select the spectrum file.
    2. File-picker dialog – select the dark spectrum file.
    3. Dark counts are subtracted from the spectrum.
    4. An interactive window opens with three buttons at the bottom:
       - "Fit span" (default, orange): drag to mark the overall fit region.
         Drag again to adjust.
       - "Secondary G" (green): drag to mark the region of the smaller,
         subtracted Gaussian (G2).  Switches the fit from a single Gaussian
         to G1 − G2.  Drag again to adjust.  Leave unset for single-Gaussian.
       - "Done": accept the current fit and close the window.
       The fit and its parameters update live after every drag.
    5. Final parameters are printed; a clean result plot is displayed.

    Returns
    -------
    tuple | None
        Single Gaussian: (amp, mu [nm], sigma [nm])
        Double Gaussian: (amp1, mu1, sigma1, amp2, mu2, sigma2) all in nm.
    """
    root = tk.Tk()
    root.withdraw()
    spectrum_path = filedialog.askopenfilename(
        title="Select spectrum file",
        filetypes=[("Origin / text files", "*.origin *.txt *.dat *.csv"),
                   ("All files", "*.*")],
    )
    if not spectrum_path:
        root.destroy()
        print("No spectrum file selected.")
        return None

    dark_path = filedialog.askopenfilename(
        title="Select dark spectrum file",
        filetypes=[("Origin / text files", "*.origin *.txt *.dat *.csv"),
                   ("All files", "*.*")],
    )
    root.destroy()
    if not dark_path:
        print("No dark spectrum file selected.")
        return None

    wl, counts = _parse_origin_spectrum(spectrum_path)
    wl_dark, counts_dark = _parse_origin_spectrum(dark_path)

    if not np.array_equal(wl, wl_dark):
        counts_dark = np.interp(wl, wl_dark, counts_dark)
    corrected = counts - counts_dark

    # ------------------------------------------------------------------ #
    #  Interactive window                                                  #
    # ------------------------------------------------------------------ #
    fit_span = [None]   # (xmin, xmax) or None
    sec_span = [None]   # (xmin, xmax) or None  – region of G2
    _fit_artists: list = []
    result_store = [None]   # ("single", popt) or ("double", popt)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(bottom=0.14)

    ax.plot(wl, corrected, color="tab:blue", lw=1.2, label="Dark-corrected spectrum")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")
    title_text = ax.set_title(
        "Mode: Fit span  –  drag to set, drag again to adjust"
    )

    def _update_fit() -> None:
        for art in _fit_artists:
            try:
                art.remove()
            except ValueError:
                pass
        _fit_artists.clear()
        result_store[0] = None

        if fit_span[0] is None:
            fig.canvas.draw_idle()
            return

        xmin, xmax = fit_span[0]
        mask = (wl >= xmin) & (wl <= xmax)
        xs, ys = wl[mask], corrected[mask]
        if xs.size < 5:
            fig.canvas.draw_idle()
            return

        peak_idx = int(np.argmax(ys))
        x_fit = np.linspace(xmin, xmax, 500)

        if sec_span[0] is not None:
            # ---- Double Gaussian: G1 − G2 ----
            sx_lo, sx_hi = sec_span[0]
            # G1 initial guess: dominant peak across the fit span
            amp1_0 = float(ys[peak_idx])
            mu1_0  = float(xs[peak_idx])
            sig1_0 = (xmax - xmin) / 4.0
            # G2 initial guess: centred on secondary span
            mu2_0  = (sx_lo + sx_hi) / 2.0
            sig2_0 = max((sx_hi - sx_lo) / 4.0, (xmax - xmin) / 20.0)
            # Estimate G2 amplitude as the gap between G1's prediction and the data
            g1_at_mu2 = amp1_0 * np.exp(-0.5 * ((mu2_0 - mu1_0) / sig1_0) ** 2)
            data_at_mu2 = float(np.interp(mu2_0, xs, ys))
            amp2_0 = max(g1_at_mu2 - data_at_mu2, 0.1 * amp1_0)

            p0 = [amp1_0, mu1_0, sig1_0, amp2_0, mu2_0, sig2_0]
            lb = [0.0, -np.inf, 0.0, 0.0, -np.inf, 0.0]
            ub = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
            try:
                popt, _ = curve_fit(
                    _double_gaussian_diff, xs, ys,
                    p0=p0, bounds=(lb, ub), maxfev=10_000,
                )
            except RuntimeError:
                fig.canvas.draw_idle()
                return

            a1, m1, s1, a2, m2, s2 = (
                popt[0], popt[1], abs(popt[2]),
                popt[3], popt[4], abs(popt[5]),
            )
            fw1 = 2.0 * np.sqrt(2.0 * np.log(2.0)) * s1
            fw2 = 2.0 * np.sqrt(2.0 * np.log(2.0)) * s2
            result_store[0] = ("double", (a1, m1, s1, a2, m2, s2))

            (l_total,) = ax.plot(
                x_fit, _double_gaussian_diff(x_fit, a1, m1, s1, a2, m2, s2),
                "--", color="tab:red", lw=2, zorder=5,
            )
            (l_g1,) = ax.plot(
                x_fit, _gaussian(x_fit, a1, m1, s1),
                ":", color="tab:orange", lw=1.5, zorder=4, alpha=0.85,
            )
            (l_g2,) = ax.plot(
                x_fit, _gaussian(x_fit, a2, m2, s2),
                ":", color="tab:green", lw=1.5, zorder=4, alpha=0.85,
            )
            vl1 = ax.axvline(m1, color="tab:orange", lw=1, ls=":", zorder=4)
            vl2 = ax.axvline(m2, color="tab:green",  lw=1, ls=":", zorder=4)
            info = ax.text(
                0.98, 0.97,
                f"G1: μ={m1:.5g} nm  σ={s1:.5g} nm  FWHM={fw1:.5g} nm\n"
                f"G2: μ={m2:.5g} nm  σ={s2:.5g} nm  FWHM={fw2:.5g} nm",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), zorder=6,
            )
            _fit_artists.extend([l_total, l_g1, l_g2, vl1, vl2, info])

        else:
            # ---- Single Gaussian fallback ----
            p0 = [float(ys[peak_idx]), float(xs[peak_idx]), (xmax - xmin) / 4.0]
            try:
                popt, _ = curve_fit(_gaussian, xs, ys, p0=p0, maxfev=10_000)
            except RuntimeError:
                fig.canvas.draw_idle()
                return

            amp_f, mu_f, sig_f = popt[0], popt[1], abs(popt[2])
            fwhm_f = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sig_f
            result_store[0] = ("single", (amp_f, mu_f, sig_f))

            (line,) = ax.plot(
                x_fit, _gaussian(x_fit, amp_f, mu_f, sig_f),
                "--", color="tab:red", lw=2, zorder=5,
            )
            vline = ax.axvline(mu_f, color="tab:red", lw=1, ls=":", zorder=4)
            info = ax.text(
                0.98, 0.97,
                f"μ = {mu_f:.5g} nm\nσ = {sig_f:.5g} nm\nFWHM = {fwhm_f:.5g} nm",
                transform=ax.transAxes, va="top", ha="right", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), zorder=6,
            )
            _fit_artists.extend([line, vline, info])

        fig.canvas.draw_idle()

    def _on_fit_select(xmin: float, xmax: float) -> None:
        fit_span[0] = (xmin, xmax)
        _update_fit()

    def _on_sec_select(xmin: float, xmax: float) -> None:
        sec_span[0] = (xmin, xmax)
        _update_fit()

    _fit_sel = SpanSelector(
        ax, _on_fit_select, "horizontal", useblit=False, interactive=True,
        props=dict(facecolor="tab:orange", alpha=0.25),
        handle_props=dict(color="darkorange"),
    )
    _sec_sel = SpanSelector(
        ax, _on_sec_select, "horizontal", useblit=False, interactive=True,
        props=dict(facecolor="tab:green", alpha=0.20),
        handle_props=dict(color="darkgreen"),
    )
    _sec_sel.set_active(False)

    ax_btn_fit  = fig.add_axes([0.12, 0.02, 0.22, 0.07])
    ax_btn_sec  = fig.add_axes([0.38, 0.02, 0.22, 0.07])
    ax_btn_done = fig.add_axes([0.68, 0.02, 0.20, 0.07])

    btn_fit  = _MplButton(ax_btn_fit,  "Fit span",    color="moccasin")
    btn_sec  = _MplButton(ax_btn_sec,  "Secondary G", color="lightgray")
    btn_done = _MplButton(ax_btn_done, "Done",         color="lightgreen")

    def _mode_fit(event=None) -> None:
        _fit_sel.set_active(True)
        _sec_sel.set_active(False)
        btn_fit.ax.set_facecolor("moccasin")
        btn_sec.ax.set_facecolor("lightgray")
        title_text.set_text("Mode: Fit span  –  drag to set, drag again to adjust")
        fig.canvas.draw_idle()

    def _mode_sec(event=None) -> None:
        _fit_sel.set_active(False)
        _sec_sel.set_active(True)
        btn_fit.ax.set_facecolor("lightgray")
        btn_sec.ax.set_facecolor("lightgreen")
        title_text.set_text(
            "Mode: Secondary G (G2)  –  drag to set, drag again to adjust"
        )
        fig.canvas.draw_idle()

    def _done(event=None) -> None:
        plt.close(fig)

    btn_fit.on_clicked(_mode_fit)
    btn_sec.on_clicked(_mode_sec)
    btn_done.on_clicked(_done)
    _mode_fit()

    # SpanSelector handle artists expand autoscale limits — pin them to the data.
    ax.set_xlim(wl.min(), wl.max())

    plt.show()

    # ------------------------------------------------------------------ #
    #  After window closed                                                 #
    # ------------------------------------------------------------------ #
    if result_store[0] is None:
        print("No valid fit obtained. Aborting.")
        return None

    kind, params = result_store[0]
    xmin, xmax = fit_span[0]
    x_fit = np.linspace(xmin, xmax, 500)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(wl, corrected, color="tab:blue", lw=1.2, label="Dark-corrected spectrum")
    ax2.axvspan(xmin, xmax, alpha=0.12, color="tab:orange", label="Fit span")

    if kind == "double":
        a1, m1, s1, a2, m2, s2 = params
        fw1 = 2.0 * np.sqrt(2.0 * np.log(2.0)) * s1
        fw2 = 2.0 * np.sqrt(2.0 * np.log(2.0)) * s2
        print(f"G1: amp={a1:.6g},  μ={m1:.6g} nm,  σ={s1:.6g} nm,  FWHM={fw1:.6g} nm")
        print(f"G2: amp={a2:.6g},  μ={m2:.6g} nm,  σ={s2:.6g} nm,  FWHM={fw2:.6g} nm")

        if sec_span[0]:
            ax2.axvspan(sec_span[0][0], sec_span[0][1], alpha=0.15,
                        color="tab:green", label="Secondary G span")
        ax2.plot(x_fit, _double_gaussian_diff(x_fit, a1, m1, s1, a2, m2, s2),
                 "--", color="tab:red", lw=2, label="G1 − G2 fit")
        ax2.plot(x_fit, _gaussian(x_fit, a1, m1, s1),
                 ":", color="tab:orange", lw=1.5, alpha=0.9,
                 label=f"G1  μ={m1:.5g} nm  σ={s1:.5g} nm  FWHM={fw1:.5g} nm")
        ax2.plot(x_fit, _gaussian(x_fit, a2, m2, s2),
                 ":", color="tab:green", lw=1.5, alpha=0.9,
                 label=f"G2  μ={m2:.5g} nm  σ={s2:.5g} nm  FWHM={fw2:.5g} nm")
        ax2.axvline(m1, color="tab:orange", lw=1, ls=":")
        ax2.axvline(m2, color="tab:green",  lw=1, ls=":")
        ax2.set_title("Dark-corrected PL spectrum – G1 − G2 fit")
        ret = (a1, m1, s1, a2, m2, s2)

    else:
        amp, mu, sigma = params
        fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
        print(f"Gaussian:  amp={amp:.6g},  μ={mu:.6g} nm,  σ={sigma:.6g} nm,  FWHM={fwhm:.6g} nm")
        ax2.plot(x_fit, _gaussian(x_fit, amp, mu, sigma),
                 "--", color="tab:red", lw=2,
                 label=f"Gaussian  μ={mu:.5g} nm  σ={sigma:.5g} nm  FWHM={fwhm:.5g} nm")
        ax2.axvline(mu, color="tab:red", lw=1, ls=":")
        ax2.set_title("Dark-corrected PL spectrum – Gaussian fit")
        ret = (amp, mu, sigma)

    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Counts")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------ #
    #  Save decomposition txt (double-Gaussian only)                       #
    # ------------------------------------------------------------------ #
    if kind == "double":
        a1, m1, s1, a2, m2, s2 = params
        root3 = tk.Tk()
        root3.withdraw()
        save_path = filedialog.asksaveasfilename(
            title="Save decomposition file",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        root3.destroy()
        if save_path:
            g1         = _gaussian(wl, a1, m1, s1)
            g2         = _gaussian(wl, a2, m2, s2)
            ratio      = g1 / (g1 + g2)
            wl_r       = wl[::-1]
            ratio_r    = ratio[::-1]
            with open(save_path, "w") as fh:
                for v in ratio_r:
                    fh.write(f"{v:.6f}\n")
            print(f"Decomposition saved to {save_path}")

    return ret


_HC_EV_NM = 1239.84193  # h·c in eV·nm


def _parse_origin_power_series(filepath):
    """Return (wavelength_nm, counts_matrix, powers_W) for all power steps.

    counts_matrix has shape (n_wavelengths, n_powers).
    """
    with open(filepath, encoding="latin-1", errors="replace") as fh:
        lines = fh.readlines()

    powers = None
    for line in lines:
        if line.startswith("Excitation power (W)"):
            parts = line.strip().split("\t")
            powers = np.array([float(p) for p in parts[1:]])
            break
    if powers is None:
        raise ValueError(f"Could not find 'Excitation power (W)' row in {filepath}")

    data_rows = []
    past_hwp = False
    for line in lines:
        if line.strip().startswith("Power HWP"):
            past_hwp = True
            continue
        if not past_hwp:
            continue
        parts = line.strip().split("\t")
        try:
            row = [float(v) for v in parts]
            if len(row) >= 2:
                data_rows.append(row)
        except (ValueError, IndexError):
            continue  # skip HWP continuation lines

    if not data_rows:
        raise ValueError(f"No numeric data found in {filepath}")

    arr = np.array(data_rows)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr[:, 0], arr[:, 1:], powers


def _parse_origin_header(filepath):
    """Return header metadata dict from a .origin file."""
    with open(filepath, encoding="latin-1", errors="replace") as fh:
        raw = fh.readlines()

    result = {"raw_header_lines": raw[:9]}

    if raw:
        date_str = raw[0].split("\t", 1)[-1].strip()
        result["date_str"] = date_str
        try:
            result["date"] = datetime.strptime(date_str, "%A, %B %d, %Y, %I:%M %p")
        except ValueError:
            result["date"] = None

    if len(raw) > 3:
        m = re.match(r"([\d.]+)\s*s", raw[3].split("\t", 1)[-1].strip())
        result["int_time_str"] = m.group(1) if m else "0.500"
    else:
        result["int_time_str"] = "0.500"

    hwp_raw = []
    in_hwp = False
    for line in raw:
        if line.strip().startswith("Power HWP"):
            in_hwp = True
            hwp_raw.append(line.rstrip("\r\n"))
        elif in_hwp:
            if line[:1] in (" ", "\t"):
                hwp_raw.append(line.rstrip("\r\n"))
            else:
                break
    result["hwp_raw"] = hwp_raw

    return result


def _stitch_counts(datasets_sorted, spans_wl):
    """Stitch counts matrices from sorted datasets at wavelength span boundaries.

    In each span the left file's wavelength grid is kept; the right file is
    interpolated onto it. The two are combined with a linearly ramped weight
    that goes from 1 (all left) at the low-wavelength edge of the span to 0
    (all right) at the high-wavelength edge, in equal steps across the span's
    measurement points.

    Returns (wl_out, counts_out) where counts_out is (n_wl, n_powers).
    """
    n = len(datasets_sorted)
    n_powers = datasets_sorted[0]["counts"].shape[1]

    wl_parts, counts_parts = [], []

    for i, d in enumerate(datasets_sorted):
        wl_i = d["wl"]
        c_i = d["counts"]

        lo_wl = spans_wl[i - 1][1] if i > 0 else -np.inf
        hi_wl = spans_wl[i][0]     if i < n - 1 else np.inf

        # Pure segment: strictly between the two neighbouring span boundaries
        mask = (wl_i > lo_wl) & (wl_i < hi_wl)
        if np.any(mask):
            wl_parts.append(wl_i[mask])
            counts_parts.append(c_i[mask])

        # Transition span into the next file
        if i < n - 1:
            sp_lo, sp_hi = spans_wl[i]
            d_next = datasets_sorted[i + 1]
            mask_span = (wl_i >= sp_lo) & (wl_i <= sp_hi)
            wl_span = wl_i[mask_span]
            c_left = c_i[mask_span]                              # (n_span, n_powers)
            n_span = len(wl_span)
            if n_span > 0:
                # Order by wavelength so the weight ramp runs from the span's
                # low edge (left file, weight 1) to its high edge (right file,
                # weight 0), regardless of the file's native point order.
                order_span = np.argsort(wl_span)
                wl_span = wl_span[order_span]
                c_left = c_left[order_span]
                c_right = np.stack([                             # interpolate right file
                    np.interp(wl_span, d_next["wl"], d_next["counts"][:, p])
                    for p in range(n_powers)
                ], axis=1)                                       # (n_span, n_powers)
                w_left = np.linspace(1.0, 0.0, n_span)[:, None]  # 1 → 0 across the span
                wl_parts.append(wl_span)
                counts_parts.append(w_left * c_left + (1.0 - w_left) * c_right)

    wl_out = np.concatenate(wl_parts)
    counts_out = np.concatenate(counts_parts, axis=0)
    order = np.argsort(wl_out)
    return wl_out[order], counts_out[order]


def _write_origin_file(output_path, wl_out, counts_out, header_meta, powers_W):
    """Write a stitched power-series spectrum as a .origin file."""
    n_powers = len(powers_W)
    wl_min, wl_max = wl_out.min(), wl_out.max()
    center_nm = (wl_min + wl_max) / 2.0
    center_ev = _HC_EV_NM / center_nm
    disp_nm   = wl_max - wl_min
    disp_ev   = abs(_HC_EV_NM / wl_min - _HC_EV_NM / wl_max)

    raw = header_meta["raw_header_lines"]
    int_time_str = header_meta.get("int_time_str", "0.500")

    def _copy(idx):
        return raw[idx].rstrip("\r\n") + "\n"

    with open(output_path, "w", encoding="latin-1") as fh:
        fh.write(f"Date:\t{header_meta['date_str']}\n")
        for i in range(1, 5):                      # Measurement type … Excitation power
            fh.write(_copy(i))
        fh.write(f"Center wavelength\t{center_nm:.3f} nm / {center_ev:.3f} eV\n")
        fh.write(f"Dispersion window:\t{disp_nm:.3f} nm / {disp_ev:.3f} eV\n")
        for i in range(7, 9):                      # Entrance slit … Exit slit
            fh.write(_copy(i))
        fh.write("\n")
        fh.write("Wavelength\t" + "\t".join(["Powerspectrum"] * n_powers) + "\n")
        fh.write("(nm)\t" + "\t".join([f"(Counts/{int_time_str}s)"] * n_powers) + "\n")
        fh.write("Excitation power (W)\t" +
                 "\t".join(str(float(p)) for p in powers_W) + "\n")
        # Write HWP positions as a single tab-separated line so that:
        #   (a) the first data row lands on line 15 (get_numcols reads line 15), and
        #   (b) LC_get_rows(filename, 14, 14) can parse the values as %f fields.
        # The input stores them as a numpy-style array "[ 70.  71. ... ]" which
        # MATLAB cannot read as numeric — so we re-format as tab-separated integers.
        hwp_raw = header_meta.get("hwp_raw", [])
        if hwp_raw:
            label = hwp_raw[0].partition("\t")[0]          # "Power HWP Position (°)"
            first_part = hwp_raw[0].partition("\t")[2] if "\t" in hwp_raw[0] else ""
            rest_parts = " ".join(l.strip() for l in hwp_raw[1:])
            array_str = (first_part + " " + rest_parts).replace("[", "").replace("]", "")
            hwp_vals = [v for v in re.split(r"\s+", array_str.strip()) if v]
            fh.write(label + "\t" + "\t".join(hwp_vals) + "\n")
        for row_idx in range(len(wl_out)):
            counts_str = "\t".join(
                str(int(round(float(c)))) for c in counts_out[row_idx]
            )
            fh.write(f"{wl_out[row_idx]:.9f}\t{counts_str}\n")


def plot_highest_power(files=None, x_axis="energy"):
    """Plot all power-series spectra from each selected .origin file.

    Each file gets its own subplot. Spectra are coloured from dark (low power)
    to bright (high power) using a sequential colormap. A shared colorbar shows
    the power scale.

    Parameters
    ----------
    files : list of str, optional
        Paths to .origin files. A file dialog opens when None.
    x_axis : {"energy", "wavelength"}
        "energy"     → x-axis in eV (converted from nm).
        "wavelength" → x-axis in nm.

    Returns
    -------
    fig, axes
    """
    if files is None:
        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="Select .origin power-series files",
            filetypes=[("Origin files", "*.origin"), ("All files", "*.*")],
        )
        root.destroy()
        if not paths:
            print("No files selected.")
            return None, None
        files = list(paths)

    xlabel = "Energy (eV)" if x_axis == "energy" else "Wavelength (nm)"

    datasets = []
    for filepath in files:
        label = os.path.splitext(os.path.basename(filepath))[0]
        try:
            wl, counts, powers = _parse_origin_power_series(filepath)
        except Exception as exc:
            print(f"Skipping {label}: {exc}")
            continue
        datasets.append((label, wl, counts, powers))

    if not datasets:
        print("No files could be parsed.")
        return None, None

    fig, ax = plt.subplots(figsize=(10, 5))

    for label, wl, counts, powers in datasets:
        if x_axis == "energy":
            x = _HC_EV_NM / wl
            sort = np.argsort(x)
            x = x[sort]
            counts = counts[sort]
        else:
            x = wl

        ax.plot(x, counts[:, -1], lw=1.2,
                label=f"{label}  ({powers[-1] * 1e3:.3g} mW)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, ax


def stitch_select_spans(files=None, x_axis="energy"):
    """Show each overlapping pair of spectra and let the user select a span.

    Files are sorted by their wavelength range. For every adjacent pair whose
    ranges overlap, a window opens showing both curves clipped to the overlap
    region. The user drags to select a span, then clicks "Next" to proceed.

    Parameters
    ----------
    files : list of str, optional
        Paths to .origin files. A file dialog opens when None.
    x_axis : {"energy", "wavelength"}

    Returns
    -------
    list of dict
        One entry per overlapping pair::

            {"file_a": str, "file_b": str, "xmin": float, "xmax": float}

        xmin/xmax are in the chosen x_axis units.
    """
    if files is None:
        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="Select .origin power-series files",
            filetypes=[("Origin files", "*.origin"), ("All files", "*.*")],
        )
        root.destroy()
        if not paths:
            print("No files selected.")
            return []
        files = list(paths)

    # parse last spectrum from every file
    datasets = []
    for filepath in files:
        label = os.path.splitext(os.path.basename(filepath))[0]
        try:
            wl, counts, powers = _parse_origin_power_series(filepath)
        except Exception as exc:
            print(f"Skipping {label}: {exc}")
            continue
        datasets.append({"label": label, "wl": wl, "counts": counts[:, -1]})

    if len(datasets) < 2:
        print("Need at least two files to find overlaps.")
        return []

    # sort by ascending min wavelength
    datasets.sort(key=lambda d: d["wl"].min())

    # find adjacent pairs that overlap in wavelength
    pairs = []
    for i in range(len(datasets) - 1):
        a, b = datasets[i], datasets[i + 1]
        ov_lo = max(a["wl"].min(), b["wl"].min())
        ov_hi = min(a["wl"].max(), b["wl"].max())
        if ov_hi > ov_lo:
            pairs.append((a, b, ov_lo, ov_hi))

    if not pairs:
        print("No overlapping wavelength regions found.")
        return []

    print(f"Found {len(pairs)} overlapping pair(s). Select a span in each window.")

    results = []

    for pair_idx, (a, b, ov_lo_wl, ov_hi_wl) in enumerate(pairs):
        span_store = [None]
        confirmed = [False]

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.subplots_adjust(bottom=0.18)

        for d, color in zip((a, b), ("tab:blue", "tab:orange")):
            mask = (d["wl"] >= ov_lo_wl) & (d["wl"] <= ov_hi_wl)
            wl_ov = d["wl"][mask]
            c_ov  = d["counts"][mask]
            if x_axis == "energy":
                x_ov = _HC_EV_NM / wl_ov
                sort = np.argsort(x_ov)
                x_ov, c_ov = x_ov[sort], c_ov[sort]
            else:
                x_ov = wl_ov
            ax.plot(x_ov, c_ov, lw=1.2, color=color, label=d["label"])

        if x_axis == "energy":
            x_lo = _HC_EV_NM / ov_hi_wl
            x_hi = _HC_EV_NM / ov_lo_wl
            xlabel = "Energy (eV)"
        else:
            x_lo, x_hi = ov_lo_wl, ov_hi_wl
            xlabel = "Wavelength (nm)"

        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        title = ax.set_title(
            f"Overlap {pair_idx + 1}/{len(pairs)}: {a['label']} & {b['label']}"
            "  —  drag to select span, then click Next"
        )

        span_patch = [None]

        def _on_select(xmin, xmax):
            span_store[0] = (xmin, xmax)
            if span_patch[0] is not None:
                try:
                    span_patch[0].remove()
                except ValueError:
                    pass
            span_patch[0] = ax.axvspan(xmin, xmax, alpha=0.25, color="tab:green")
            title.set_text(
                f"Overlap {pair_idx + 1}/{len(pairs)}: span [{xmin:.5g}, {xmax:.5g}]"
                "  —  drag to adjust, then click Next"
            )
            fig.canvas.draw_idle()

        _sel = SpanSelector(
            ax, _on_select, "horizontal", useblit=False, interactive=True,
            props=dict(facecolor="tab:green", alpha=0.20),
            handle_props=dict(color="darkgreen"),
        )

        ax_btn = fig.add_axes([0.78, 0.02, 0.18, 0.09])
        btn = _MplButton(ax_btn, "Next", color="lightgreen")

        def _next(event=None):
            confirmed[0] = True
            plt.close(fig)

        btn.on_clicked(_next)
        plt.show()

        if span_store[0] is not None:
            xmin, xmax = span_store[0]
            results.append({
                "file_a": a["label"], "file_b": b["label"],
                "xmin": xmin, "xmax": xmax,
            })
            print(f"  Pair {pair_idx + 1}: span [{xmin:.5g}, {xmax:.5g}]")
        else:
            print(f"  Pair {pair_idx + 1}: no span selected, skipping.")

    return results


def stitch_spectra(files=None, spans=None, x_axis="energy", output_path=None):
    """Stitch multiple .origin power-series files into a single .origin file.

    Parameters
    ----------
    files : list of str, optional
        Input .origin files. File dialog if None.
    spans : list of dict, optional
        Output of stitch_select_spans. If None, stitch_select_spans is called
        automatically to collect spans interactively.
    x_axis : {"energy", "wavelength"}
        Must match the x_axis used when spans were selected.
    output_path : str, optional
        Destination path. Save-as dialog if None.

    Returns
    -------
    str or None
        Path to the written file, or None if aborted.
    """
    if files is None:
        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="Select .origin power-series files",
            filetypes=[("Origin files", "*.origin"), ("All files", "*.*")],
        )
        root.destroy()
        if not paths:
            print("No files selected.")
            return None
        files = list(paths)

    if spans is None:
        spans = stitch_select_spans(files=files, x_axis=x_axis)
        if not spans:
            print("No spans selected.")
            return None

    # Parse all files
    datasets, hdrs = [], []
    for filepath in files:
        label = os.path.splitext(os.path.basename(filepath))[0]
        try:
            wl, counts, powers = _parse_origin_power_series(filepath)
            hdr = _parse_origin_header(filepath)
        except Exception as exc:
            print(f"Skipping {label}: {exc}")
            continue
        datasets.append({"label": label, "wl": wl, "counts": counts, "powers": powers})
        hdrs.append(hdr)

    if len(datasets) < 2:
        print("Need at least 2 parseable files.")
        return None

    # Sort by ascending min wavelength
    order = sorted(range(len(datasets)), key=lambda i: datasets[i]["wl"].min())
    datasets = [datasets[i] for i in order]
    hdrs     = [hdrs[i]     for i in order]

    # Build span lookup keyed by (label_a, label_b) in both directions
    span_lookup = {}
    for s in spans:
        span_lookup[(s["file_a"], s["file_b"])] = (s["xmin"], s["xmax"])
        span_lookup[(s["file_b"], s["file_a"])] = (s["xmin"], s["xmax"])

    spans_wl = []
    for i in range(len(datasets) - 1):
        la, lb = datasets[i]["label"], datasets[i + 1]["label"]
        if (la, lb) in span_lookup:
            xmin, xmax = span_lookup[(la, lb)]
            if x_axis == "energy":
                lo_wl = min(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
                hi_wl = max(_HC_EV_NM / xmin, _HC_EV_NM / xmax)
            else:
                lo_wl, hi_wl = min(xmin, xmax), max(xmin, xmax)
        else:
            ov_lo = max(datasets[i]["wl"].min(), datasets[i + 1]["wl"].min())
            ov_hi = min(datasets[i]["wl"].max(), datasets[i + 1]["wl"].max())
            mid = (ov_lo + ov_hi) / 2.0
            print(f"No span for {la} & {lb}, using midpoint {mid:.3f} nm")
            lo_wl = hi_wl = mid
        spans_wl.append((lo_wl, hi_wl))

    # Harmonise power-step count across files
    n_p_list = [d["counts"].shape[1] for d in datasets]
    n_p = min(n_p_list)
    if len(set(n_p_list)) > 1:
        print(f"Warning: power step counts differ {n_p_list}, truncating to {n_p}")
    for d in datasets:
        d["counts"] = d["counts"][:, :n_p]
        d["powers"] = d["powers"][:n_p]

    # Stitch
    wl_out, counts_out = _stitch_counts(datasets, spans_wl)

    # Earliest date for the header
    best_date_str = hdrs[0].get("date_str", "")
    best_date = hdrs[0].get("date")
    for h in hdrs[1:]:
        d = h.get("date")
        if d is not None and (best_date is None or d < best_date):
            best_date = d
            best_date_str = h.get("date_str", best_date_str)
    hdrs[0]["date_str"] = best_date_str

    # Output path
    if output_path is None:
        root = tk.Tk()
        root.withdraw()
        output_path = filedialog.asksaveasfilename(
            title="Save stitched spectrum",
            defaultextension=".origin",
            filetypes=[("Origin files", "*.origin"), ("All files", "*.*")],
        )
        root.destroy()
        if not output_path:
            print("No output path selected.")
            return None

    _write_origin_file(output_path, wl_out, counts_out, hdrs[0], datasets[0]["powers"])
    print(f"Stitched spectrum saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PL Helper")
    sub = parser.add_subparsers(dest="command")

    p_spotsize = sub.add_parser("spotsize", help="Interactive spot-size analysis")
    p_spotsize.add_argument(
        "angle", type=float,
        help="Angle in degrees. Result is (x_hi - x_lo) * sin(angle).",
    )
    p_spotsize.add_argument(
        "--is-derivative", action="store_true",
        help="Treat the second column as an already-computed derivative (skip np.gradient).",
    )

    sub.add_parser("pxl_corr", help="Dark-subtract two PL spectra and fit a Gaussian")

    p_php = sub.add_parser("plot_highest_power",
                           help="Plot last-power spectrum from each selected .origin file")
    p_php.add_argument("files", nargs="*", metavar="FILE",
                       help=".origin files to plot (omit to open a file dialog)")
    p_php.add_argument("--wavelength", action="store_true",
                       help="Use wavelength (nm) as x-axis instead of energy (eV)")

    p_sss = sub.add_parser("stitch_select_spans",
                           help="Interactively select a span in each overlapping pair")
    p_sss.add_argument("files", nargs="*", metavar="FILE",
                       help=".origin files (omit to open a file dialog)")
    p_sss.add_argument("--wavelength", action="store_true",
                       help="Use wavelength (nm) as x-axis instead of energy (eV)")

    p_ss = sub.add_parser("stitch_spectra",
                          help="Stitch .origin files interactively and save a new .origin file")
    p_ss.add_argument("files", nargs="*", metavar="FILE",
                      help=".origin files to stitch (omit to open a file dialog)")
    p_ss.add_argument("--wavelength", action="store_true",
                      help="Use wavelength (nm) as x-axis instead of energy (eV)")
    p_ss.add_argument("-o", "--output", default=None,
                      help="Output path (omit for save dialog)")

    args = parser.parse_args()
    if args.command == "spotsize":
        spotsize(angle=args.angle, is_derivative=args.is_derivative)
    elif args.command == "pxl_corr":
        pxl_corr()
    elif args.command == "plot_highest_power":
        plot_highest_power(
            files=args.files if args.files else None,
            x_axis="wavelength" if args.wavelength else "energy",
        )
    elif args.command == "stitch_select_spans":
        spans = stitch_select_spans(
            files=args.files if args.files else None,
            x_axis="wavelength" if args.wavelength else "energy",
        )
        print(f"\nSelected {len(spans)} span(s):")
        for s in spans:
            print(f"  {s['file_a']} & {s['file_b']}: [{s['xmin']:.5g}, {s['xmax']:.5g}]")
    elif args.command == "stitch_spectra":
        stitch_spectra(
            files=args.files if args.files else None,
            x_axis="wavelength" if args.wavelength else "energy",
            output_path=args.output,
        )
    else:
        parser.print_help()
