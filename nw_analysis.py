"""
nw_analysis.py  —  Python port of the MATLAB nanowire PL analysis pipeline
                   (mirrors main_Benni.m + Analysis.m + Nanowire.m)

Typical workflow
----------------
    import nw_analysis as nwa
    import matplotlib.pyplot as plt

    files = nwa.list_origin_files(r"\\path\\to\\measurement\\folder")
    nwarray = [nwa.load_origin_file(f, spot_radius_um=5.2, rep_rate_hz=82e6)
               for f in files]

    for nw in nwarray:
        nwa.movemean(nw, 2)
        nwa.get_darkspectrum(nw, 'firstspec')
        nwa.subtract_background(nw)
        nwa.set_startconditions(nw, window_width=0.05, x_unit='eV',
                                spectrum_selection='maxpeak', y_scale='log')
        nwa.fit_nw(nw, subtract_fit_background='raw', fitfunction='gauss1',
                   show_progress=True)
        nwa.integrate_spectra(nw, center=1.1, width=0.125*2*1e5,
                              spectrumtype='no_background', method='trapz')
        nwa.thresholds(nw, mode=['area'])

    plt.show()
"""

import os
import re
import glob as _glob
import numpy as np
from types import SimpleNamespace
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Nanowire data structure
# ══════════════════════════════════════════════════════════════════════════════

def _new_nanowire():
    return SimpleNamespace(
        name=None,
        folder_path=None,
        # Spectral axis
        wavelength=None,        # 1-D ndarray, eV or nm
        wavelength_unit='eV',
        # Spectra  (n_wl, n_powers)
        spectra_raw=None,
        spectra_diff=None,      # dark-subtracted / processed
        spectrum_dark=None,     # 1-D ndarray
        # Power
        power=None,             # (n_powers,) ndarray, mW
        hwp=None,               # (n_powers,) ndarray, HWP positions
        power_uncalibrated=None,
        rep_rate=None,          # Hz
        # Spot
        spot_radius_short=None, # µm
        pump_fluence=None,      # µJ/cm²
        power_density=None,     # kW/cm²
        # Metadata
        integration_time=None,  # s
        excitation_wavelength=None,
        # Processing
        movemean_performed=None,
        # Fitting start conditions
        start_conditions=None,  # (3, n_peaks): [peakindex, fitwindow_pts, maxsignal_idx]
        n_sel_peaks=0,
        fit_model=None,
        background_type=None,
        # Fit results  (lists of lists, [peak_idx][power_idx])
        fits=None,
        fit_data=None,          # each entry: ndarray (N, 3) = [X, Y, bg]
        peak_maximum=None,      # (n_peaks, n_powers)
        peak_integral=None,     # (n_peaks, n_powers) — after local bg subtraction
        peak_pos=None,          # list[list[ndarray]]
        peak_pos_err=None,
        peak_area=None,
        peak_area_err=None,
        fwhm=None,
        fwhm_err=None,
        total_peak_area=None,   # (n_powers,)
        findpeaks_fwhm=None,    # (n_peaks, n_powers)
        # Spectral integrals
        specsum=None,           # list of dicts {values, center, width, spectrumtype}
        # Threshold results
        threshold=None,
        threshold_err=None,
        slope=None,
        slope_err=None,
        dominant_peak_index=None,
        threshold_integral=None, threshold_err_integral=None,
        slope_integral=None,     slope_err_integral=None,
        threshold_max=None,      threshold_err_max=None,
        slope_max=None,          slope_err_max=None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Origin file parsing  (LabControl format)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_origin(filepath):
    """
    Parse a LabControl .origin power-series file.

    Returns
    -------
    wl        : (N,) ndarray  — wavelength axis (eV if centre > 10 else nm)
    counts    : (N, P) ndarray — raw counts
    powers    : (P,) ndarray  — excitation powers in W (as stored)
    hwp       : (P,) ndarray  — HWP positions (NaN if not found)
    meta      : dict           — integration_time (s), excitation_wl (nm), raw header
    """
    with open(filepath, encoding='latin-1', errors='replace') as fh:
        lines = fh.readlines()

    powers, hwp = None, None
    meta = {'raw_header': lines[:9], 'integration_time': None, 'excitation_wl': None}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('Excitation power (W)'):
            parts = stripped.split('\t')
            try:
                powers = np.array([float(p) for p in parts[1:] if p.strip()])
            except ValueError:
                pass
        if stripped.startswith('Power HWP') and hwp is None:
            parts = stripped.split('\t')
            try:
                hwp = np.array([float(v) for v in parts[1:] if v.strip()])
            except ValueError:
                pass
        m = re.match(r'Integration time.*?([0-9.]+)\s*s', stripped, re.IGNORECASE)
        if m:
            meta['integration_time'] = float(m.group(1))

    # data rows: lines after the last "Power HWP" header row
    data_rows, past_hwp = [], False
    for line in lines:
        if line.strip().startswith('Power HWP'):
            past_hwp = True
            continue
        if not past_hwp:
            continue
        parts = line.strip().split('\t')
        try:
            row = [float(v) for v in parts]
            if len(row) >= 2:
                data_rows.append(row)
        except (ValueError, IndexError):
            continue

    arr = np.array(data_rows)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    wl = arr[:, 0]
    counts = arr[:, 1:]

    if powers is None:
        powers = np.arange(1, counts.shape[1] + 1, dtype=float)
    if hwp is None:
        hwp = np.full(len(powers), np.nan)

    return wl, counts, powers, hwp, meta


def load_origin_file(filepath, spot_radius_um=None, rep_rate_hz=None,
                     power_unit='W'):
    """
    Load a LabControl .origin file into a Nanowire object.

    Parameters
    ----------
    filepath       : str
    spot_radius_um : float or None — 1/e² spot radius (µm), short axis
    rep_rate_hz    : float or None — laser repetition rate (Hz)
    power_unit     : 'W' (default) or 'mW' — unit of power in the file

    Returns
    -------
    nw : SimpleNamespace (Nanowire)
    """
    wl, counts, powers, hwp, meta = _parse_origin(filepath)

    nw = _new_nanowire()
    nw.name = os.path.splitext(os.path.basename(filepath))[0]
    nw.folder_path = os.path.dirname(filepath)

    # Determine if wavelength axis is in nm or eV
    if wl.mean() > 10:
        nw.wavelength_unit = 'nm'
        nw.wavelength = wl.copy()
    else:
        nw.wavelength_unit = 'eV'
        nw.wavelength = wl.copy()

    nw.spectra_raw = counts.copy()

    # Powers: convert to mW
    if power_unit == 'W':
        nw.power = powers * 1e3   # → mW
    else:
        nw.power = powers.copy()
    nw.power_uncalibrated = nw.power.copy()
    nw.hwp = hwp
    nw.rep_rate = rep_rate_hz
    nw.integration_time = meta.get('integration_time')
    nw.specsum = []

    if spot_radius_um is not None:
        set_spotsize(nw, spot_radius_um)

    return nw


def list_origin_files(folder_path):
    """Return sorted list of .origin file paths in folder_path."""
    pattern = os.path.join(folder_path, '*.origin')
    return sorted(_glob.glob(pattern))


def select_files(folder_path):
    """
    Interactive file selector: print numbered list of .origin files and
    let the user type which ones to load.  Returns list of paths.
    """
    files = list_origin_files(folder_path)
    if not files:
        print('No .origin files found in', folder_path)
        return []
    print('Available files:')
    for i, f in enumerate(files):
        print(f'  [{i}] {os.path.basename(f)}')
    raw = input('Enter indices separated by spaces (blank = all): ').strip()
    if not raw:
        return files
    try:
        indices = [int(x) for x in raw.split()]
        return [files[i] for i in indices]
    except (ValueError, IndexError):
        print('Invalid selection, returning all files.')
        return files


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Core utilities
# ══════════════════════════════════════════════════════════════════════════════

def crop_vector(vector, x1, x2, x_axis=None):
    """
    Crop vector between x1 and x2.
    If x_axis is None, x1/x2 are treated as indices.
    Returns (cropped_vector, cropped_x_axis).
    """
    vector = np.asarray(vector, dtype=float)
    if x_axis is None:
        i1, i2 = int(round(x1)), int(round(x2))
        i1, i2 = max(0, min(i1, i2)), min(len(vector) - 1, max(i1, i2))
        return vector[i1:i2 + 1], np.arange(i1, i2 + 1, dtype=float)

    x_axis = np.asarray(x_axis, dtype=float)
    if x1 == np.inf:
        i1 = np.argmax(x_axis)
    elif x1 == -np.inf:
        i1 = np.argmin(x_axis)
    else:
        i1 = int(np.argmin(np.abs(x_axis - x1)))

    if x2 == np.inf:
        i2 = np.argmax(x_axis)
    elif x2 == -np.inf:
        i2 = np.argmin(x_axis)
    else:
        i2 = int(np.argmin(np.abs(x_axis - x2)))

    a1, a2 = max(0, min(i1, i2)), min(len(vector) - 1, max(i1, i2))
    return vector[a1:a2 + 1], x_axis[a1:a2 + 1]


def subtract_local_background(X, Y, mode):
    """
    Subtract linear or constant baseline from Y over interval X.
    mode: 'linear', 'constant', or anything else (→ no subtraction).
    Returns (X, Y_corrected, local_background).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if mode == 'linear':
        n_pts = 4
        if len(X) > n_pts + 1:
            x1 = np.mean(X[:n_pts]);  y1 = np.mean(Y[:n_pts])
            x2 = np.mean(X[-n_pts:]); y2 = np.mean(Y[-n_pts:])
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
            bg = y1 + slope * (X - x1)
        else:
            bg = np.full_like(Y, np.min(Y))
        return X, Y - bg, bg
    elif mode == 'constant':
        n_smooth = 20
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(Y, size=min(n_smooth, len(Y)), mode='reflect')
        bg = np.full_like(Y, np.min(smoothed))
        return X, Y - bg, bg
    else:
        return X, Y, np.zeros_like(Y)


def _movmean(x, width):
    """Uniform moving average matching MATLAB movmean(x, width)."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x.astype(float), size=width, mode='reflect')


def _get_spectrum(nw, spectrumtype):
    """
    Return (n_wl, n_powers) spectrum matrix.
    spectrumtype: 'raw', 'final', 'no_background'
    """
    if spectrumtype == 'final' and nw.spectra_diff is not None:
        return nw.spectra_diff.copy()
    if spectrumtype != 'raw' and nw.spectrum_dark is not None:
        base = nw.spectra_raw if nw.spectra_raw is not None else nw.spectra_diff
        return base - nw.spectrum_dark[:, np.newaxis]
    raw = nw.spectra_raw
    if raw is None:
        raw = nw.spectra_diff
    return raw.copy() if raw is not None else None


def _get_wavelength(nw, x_unit=None):
    """Return wavelength axis in requested unit (no conversion stored)."""
    wl = nw.wavelength.copy()
    if x_unit is None:
        return wl
    if x_unit == 'eV' and nw.wavelength_unit == 'nm':
        return 1239.84193 / wl
    if x_unit == 'nm' and nw.wavelength_unit == 'eV':
        return 1239.84193 / wl
    return wl


def _x_label(x_unit):
    labels = {'eV': 'Photon energy (eV)', 'nm': 'Wavelength (nm)',
              'px': 'Detector position (px)'}
    return labels.get(x_unit, x_unit)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Lineshape functions  (match MATLAB gaussian.m / lorentzian.m)
# ══════════════════════════════════════════════════════════════════════════════

def gaussian(A, FWHM, center, x):
    """Gaussian with integrated area A, FWHM, center (MATLAB gaussian.m)."""
    return (A / (FWHM * np.sqrt(np.pi / (4 * np.log(2))))
            * np.exp(-4 * np.log(2) / FWHM**2 * (x - center)**2))


def gaussian_N(x, *params):
    """Sum of N Gaussians. params = (A1, FWHM1, c1, A2, FWHM2, c2, ...)."""
    n = len(params) // 3
    y = np.zeros_like(x, dtype=float)
    for i in range(n):
        y += gaussian(params[3*i], params[3*i+1], params[3*i+2], x)
    return y


def lorentzian(A, FWHM, center, x):
    """Lorentzian with integrated area A, FWHM, center (MATLAB lorentzian.m)."""
    return A * FWHM / (2 * np.pi) / ((center - x)**2 + (FWHM / 2)**2)


def lorentzian_N(x, *params):
    """Sum of N Lorentzians. params = (A1, FWHM1, c1, A2, FWHM2, c2, ...)."""
    n = len(params) // 3
    y = np.zeros_like(x, dtype=float)
    for i in range(n):
        y += lorentzian(params[3*i], params[3*i+1], params[3*i+2], x)
    return y


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Basic analysis  (movemean, dark spectrum, background subtraction)
# ══════════════════════════════════════════════════════════════════════════════

def movemean(nw, width):
    """Smooth spectra with a uniform moving average of given width."""
    if nw.movemean_performed is not None:
        print('movemean: already performed, skipping.')
        return
    src = nw.spectra_diff if nw.spectra_diff is not None else nw.spectra_raw
    result = np.stack([_movmean(src[:, i], width)
                       for i in range(src.shape[1])], axis=1)
    nw.spectra_diff = result
    nw.movemean_performed = width


def get_darkspectrum(nw, dark_filename='firstspec'):
    """
    Set the dark spectrum on nw.
    dark_filename: 'firstspec' uses the first (lowest-power) spectrum as dark.
    """
    if dark_filename == 'firstspec':
        nw.spectrum_dark = nw.spectra_raw[:, 0].copy()
        return
    # Load from file
    if not dark_filename.endswith('.origin'):
        dark_filename += '.origin'
    path = os.path.join(nw.folder_path, dark_filename)
    if not os.path.isfile(path):
        print(f'Dark spectrum file not found: {path}, using first spectrum.')
        nw.spectrum_dark = nw.spectra_raw[:, 0].copy()
        return
    wl_dark, counts_dark, *_ = _parse_origin(path)
    nw.spectrum_dark = counts_dark[:, 0]


def subtract_background(nw):
    """Subtract dark spectrum from spectra → stored in nw.spectra_diff."""
    if nw.spectrum_dark is None:
        print('subtract_background: no dark spectrum set.')
        return
    dark = nw.spectrum_dark.copy()
    if nw.movemean_performed is not None:
        dark = _movmean(dark, nw.movemean_performed)
        nw.spectrum_dark = dark
    src = nw.spectra_diff if nw.spectra_diff is not None else nw.spectra_raw
    nw.spectra_diff = src - dark[:, np.newaxis]


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Fitting internals
# ══════════════════════════════════════════════════════════════════════════════

def _gauss1_model(x, a, b, c):
    """a*exp(-((x-b)/c)^2) — same as MATLAB fittype('gauss1')."""
    return a * np.exp(-((x - b) / c) ** 2)


def _gauss_n_model(n):
    """Return a function fitting n Gaussians (MATLAB gauss1..4 convention)."""
    def model(x, *p):
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            a, b, c = p[3*i], p[3*i+1], p[3*i+2]
            y += a * np.exp(-((x - b) / c) ** 2)
        return y
    return model


def _lorentz1_model(x, a1, b1, c1):
    """lorentzian(a1, c1, b1, x) — matches MATLAB fit_lorentz1."""
    return lorentzian(a1, c1, b1, x)


def _lorentz_n_model(n):
    """Return a function fitting n Lorentzians."""
    def model(x, *p):
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            A, FWHM, center = p[3*i], p[3*i+1], p[3*i+2]
            y += lorentzian(A, FWHM, center, x)
        return y
    return model


def _find_peaks_sorted(Y, X, n_smooth=1):
    """Find peaks sorted by prominence (descending). Returns (positions, prominences, widths)."""
    y = _movmean(Y, n_smooth) if n_smooth > 1 else Y
    if X[1] < X[0]:  # decreasing axis
        y = y[::-1]; x = X[::-1]
    else:
        x = X
    peak_idx, props = find_peaks(y, prominence=0)
    if len(peak_idx) == 0:
        return np.array([]), np.array([]), np.array([])
    proms = props['prominences']
    widths_result = peak_widths(y, peak_idx, rel_height=0.5)
    wids = widths_result[0] * np.abs(x[1] - x[0]) if len(x) > 1 else np.ones(len(peak_idx))
    order = np.argsort(proms)[::-1]
    return x[peak_idx[order]], proms[order], wids[order]


def _gauss_params_from_popt(popt, pcov, n):
    """
    Extract peak positions, FWHM, areas (and 2-sigma errors) from gauss_n fit.
    MATLAB convention: FWHM = c*2*sqrt(ln2), area = a*c*sqrt(pi).
    Returns lists of length n: (pos, pos_err, area, area_err, fwhm, fwhm_err).
    """
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(3*n)
    pos, pos_err, area, area_err, fwhm_list, fwhm_err_list = [], [], [], [], [], []
    for i in range(n):
        a, b, c = popt[3*i], popt[3*i+1], popt[3*i+2]
        da, db, dc = perr[3*i], perr[3*i+1], perr[3*i+2]
        # 2-sigma errors (match MATLAB convention divided by 2 for errorbar)
        pos.append(b);           pos_err.append(2*db)
        fwhm_val = abs(c) * 2 * np.sqrt(np.log(2))
        fwhm_e   = 2*dc * 2 * np.sqrt(np.log(2))
        fwhm_list.append(fwhm_val); fwhm_err_list.append(fwhm_e)
        area_val = a * abs(c) * np.sqrt(np.pi)
        area_e   = np.sqrt(np.pi) * np.sqrt((2*dc*a)**2 + (abs(c)*2*da)**2)
        area.append(area_val); area_err.append(area_e)
    return (np.array(pos), np.array(pos_err), np.array(area), np.array(area_err),
            np.array(fwhm_list), np.array(fwhm_err_list))


def _lorentz_params_from_popt(popt, pcov, n):
    """
    Extract peak positions, FWHM, areas for n Lorentzians.
    Model params: (A, FWHM, center) per peak. Area = A (analytical).
    """
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(3*n)
    pos, pos_err, area, area_err, fwhm_list, fwhm_err_list = [], [], [], [], [], []
    for i in range(n):
        A, FWHM, center = popt[3*i], popt[3*i+1], popt[3*i+2]
        dA, dFWHM, dc = perr[3*i], perr[3*i+1], perr[3*i+2]
        pos.append(center);      pos_err.append(2*dc)
        fwhm_list.append(FWHM); fwhm_err_list.append(2*dFWHM)
        area.append(A);          area_err.append(2*dA)
    return (np.array(pos), np.array(pos_err), np.array(area), np.array(area_err),
            np.array(fwhm_list), np.array(fwhm_err_list))


def _fit_gauss(X, Y, n=1):
    """
    Fit n Gaussians to (X, Y).  Returns (popt, pcov, peakindex, findpeaks_fwhm)
    where peakindex is the array index of the strongest fitted peak.
    Returns (None, None, mid, nan) on failure.
    """
    mid = len(X) // 2
    flip = X[0] > X[-1] if len(X) > 1 else False
    if flip:
        X, Y = X[::-1], Y[::-1]

    smooth = max(1, min(30, len(X)//10)) if n > 1 else 1
    pos, proms, wids = _find_peaks_sorted(Y, X, n_smooth=smooth)
    if len(pos) == 0:
        pos = X[np.argmax(Y):np.argmax(Y)+1]
        proms = np.array([max(Y)])
        wids = np.array([(X[-1]-X[0])/4])

    n_found = min(n, len(pos))
    p0, lo, hi = [], [], []
    for i in range(n):
        idx = i % n_found
        a0 = float(proms[idx])
        b0 = float(pos[idx])
        c0 = float(wids[idx]) / (2 * np.sqrt(np.log(2)))
        c0 = max(c0, 1e-6)
        p0 += [a0, b0, c0]
        lo += [0, float(X[0]), 0]
        hi += [a0 * 1.5, float(X[-1]), float(X[-1] - X[0])]

    model = _gauss_n_model(n)
    try:
        popt, pcov = curve_fit(model, X, Y, p0=p0,
                               bounds=(lo, hi), maxfev=6000)
    except (RuntimeError, ValueError):
        if flip:
            X = X[::-1]
        return None, None, mid, np.nan

    amplitudes = [popt[3*i] for i in range(n)]
    best = int(np.argmax(amplitudes))
    b_best = popt[3*best + 1]
    if flip:
        X = X[::-1]
    peakindex = int(np.argmin(np.abs(X - b_best)))
    findpeaks_fwhm = float(wids[0]) if len(wids) > 0 else np.nan
    return popt, pcov, peakindex, findpeaks_fwhm


def _fit_lorentz(X, Y, n=1):
    """
    Fit n Lorentzians to (X, Y).  Returns (popt, pcov, peakindex, findpeaks_fwhm).
    Model per peak: (A, FWHM, center).
    """
    mid = len(X) // 2
    flip = X[0] > X[-1] if len(X) > 1 else False
    if flip:
        X, Y = X[::-1], Y[::-1]

    pos, proms, wids = _find_peaks_sorted(Y, X, n_smooth=max(1, len(X)//20))
    if len(pos) == 0:
        pos = X[np.argmax(Y):np.argmax(Y)+1]
        proms = np.array([max(Y)])
        wids = np.array([(X[-1]-X[0])/4])

    n_found = min(n, len(pos))
    x_range = float(X[-1] - X[0])
    p0, lo, hi = [], [], []
    for i in range(n):
        idx = i % n_found
        A0 = float(proms[idx])
        c0 = float(pos[idx])
        f0 = max(float(wids[idx]), 1e-6)
        p0 += [A0, f0, c0]
        lo += [0, 0, float(X[0])]
        hi += [A0 * 1.5 * x_range, x_range, float(X[-1])]

    model = _lorentz_n_model(n)
    try:
        popt, pcov = curve_fit(model, X, Y, p0=p0,
                               bounds=(lo, hi), maxfev=6000)
    except (RuntimeError, ValueError):
        if flip:
            X = X[::-1]
        return None, None, mid, np.nan

    centers = [popt[3*i+2] for i in range(n)]
    areas   = [popt[3*i]   for i in range(n)]
    best_c = centers[int(np.argmax(areas))]
    if flip:
        X = X[::-1]
    peakindex = int(np.argmin(np.abs(X - best_c)))
    return popt, pcov, peakindex, float(wids[0]) if len(wids) else np.nan


def _run_fit(X, Y, fitfunction):
    """
    Dispatch to the correct fit function.
    Returns (popt, pcov, peakindex, findpeaks_fwhm, n_peaks, is_lorentz).
    """
    fitfunction = fitfunction.lower()
    is_lorentz = False
    n = 1
    if fitfunction in ('gauss1', 'gauss2', 'gauss3', 'gauss4'):
        n = int(fitfunction[-1])
        popt, pcov, pi, fw = _fit_gauss(X, Y, n)
    elif fitfunction == 'lorentz1':
        n = 1
        is_lorentz = True
        popt, pcov, pi, fw = _fit_lorentz(X, Y, 1)
    elif fitfunction.startswith('lorentz'):
        m = re.search(r'(\d+)$', fitfunction)
        n = int(m.group(1)) if m else 1
        is_lorentz = True
        popt, pcov, pi, fw = _fit_lorentz(X, Y, n)
    else:
        popt, pcov, pi, fw = _fit_gauss(X, Y, 1)
    return popt, pcov, pi, fw, n, is_lorentz


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Interactive: set_startconditions
# ══════════════════════════════════════════════════════════════════════════════

def set_startconditions(nw, window_width, x_unit='eV',
                        spectrum_selection='maxpeak',
                        spectrum_type='final', y_scale='log'):
    """
    Interactively select peaks to fit.

    1. Shows spectrum with highest peak; user clicks on peak maxima and presses Enter.
    2. For each peak, shows a zoomed window; user drags a SpanSelector to define
       the fit window, then clicks the 'Confirm' button.

    Stores results in nw.start_conditions (shape 3 × n_peaks):
        row 0: peakindex  (index into wavelength array)
        row 1: fitwindow  (number of points in fit window)
        row 2: maxsignal_spectrum  (index of the reference spectrum)

    Parameters
    ----------
    nw               : Nanowire
    window_width     : float  — initial display half-width in x_unit
    x_unit           : 'eV' or 'nm'
    spectrum_selection : 'maxpeak', 'maxsum', or integer index
    spectrum_type    : 'final', 'raw', or 'no_background'
    y_scale          : 'log' or 'linear'
    """
    wl = _get_wavelength(nw, x_unit)
    spectra = _get_spectrum(nw, spectrum_type)
    if spectra is None:
        spectra = nw.spectra_raw

    # Select reference spectrum
    if spectrum_selection == 'maxpeak':
        max_spec_idx = int(np.argmax(np.max(spectra, axis=0)))
    elif spectrum_selection == 'maxsum':
        max_spec_idx = int(np.argmax(np.sum(spectra, axis=0)))
    else:
        max_spec_idx = int(spectrum_selection)

    spectrum = spectra[:, max_spec_idx]

    # ── Step 1: click peak positions ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(wl, spectrum)
    ax.set_yscale(y_scale)
    ax.set_xlabel(_x_label(x_unit))
    ax.set_ylabel('PL intensity (arb.u.)')
    ax.set_title(f'{nw.name}\nLeft-click peaks, press Enter when done',
                 fontsize=9)
    plt.tight_layout()
    pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)

    n_peaks = len(pts)
    if n_peaks == 0:
        print('set_startconditions: no peaks selected.')
        return

    start_conditions = np.full((3, n_peaks), np.nan)

    # ── Step 2: for each peak, select fit window ───────────────────────────────
    for j, (click_x, _) in enumerate(pts):
        peak_idx = int(np.argmin(np.abs(wl - click_x)))

        # Zoom display
        lo = wl[peak_idx] - window_width / 2
        hi = wl[peak_idx] + window_width / 2
        mask = (wl >= lo) & (wl <= hi)
        wl_win = wl[mask]
        sp_win = spectrum[mask]

        selected = [None]

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(wl_win, sp_win, 'x-')
        ax2.set_yscale(y_scale)
        ax2.set_xlabel(_x_label(x_unit))
        ax2.set_ylabel('PL intensity (arb.u.)')
        ax2.set_title(f'{nw.name} — peak {j+1}/{n_peaks}\n'
                      'Drag to select fit window, then click Confirm', fontsize=9)
        plt.tight_layout()

        ax_btn = fig2.add_axes([0.8, 0.01, 0.15, 0.06])
        btn = mwidgets.Button(ax_btn, 'Confirm')

        span_obj = [None]

        def _on_select(xmin, xmax, _j=j):
            selected[0] = (xmin, xmax)

        span_obj[0] = mwidgets.SpanSelector(
            ax2, _on_select, 'horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='steelblue'))

        def _on_click(_event):
            plt.close(fig2)

        btn.on_clicked(_on_click)
        plt.show(block=True)

        if selected[0] is not None:
            xmin, xmax = selected[0]
            i1 = int(np.argmin(np.abs(wl - xmin)))
            i2 = int(np.argmin(np.abs(wl - xmax)))
            fitwindow = abs(i2 - i1)
        else:
            # Fallback: use display window width
            i1 = int(np.argmin(np.abs(wl - lo)))
            i2 = int(np.argmin(np.abs(wl - hi)))
            fitwindow = abs(i2 - i1)

        fitwindow = max(fitwindow, 4)  # at least 4 points
        start_conditions[:, j] = [peak_idx, fitwindow, max_spec_idx]

    nw.start_conditions = start_conditions
    nw.n_sel_peaks = n_peaks
    print(f'set_startconditions: {n_peaks} peak(s) selected.')


# ══════════════════════════════════════════════════════════════════════════════
# 8.  fit_nw  —  peak fitting with power-series tracking
# ══════════════════════════════════════════════════════════════════════════════

def fit_nw(nw, subtract_fit_background='none', fitfunction='gauss1',
           show_progress=False, fixed_window_below_index=None):
    """
    Fit peaks across all power steps.

    Mirrors Analysis.fit_nw:
    - Starts at the manually selected spectrum and works outward in both
      directions (handles lasing burndown).
    - Fit window width is fixed; center tracks the fitted peak position.

    Parameters
    ----------
    nw                        : Nanowire
    subtract_fit_background   : 'linear', 'constant', 'none', or 'raw'
    fitfunction               : 'gauss1'..'gauss4', 'lorentz1', 'lorentz<N>'
    show_progress             : bool — show each fit in a figure
    fixed_window_below_index  : int or None — 0-based power-step index
        (spectra are ordered by ascending power, index 0 = lowest power).
        Spectra with index < this value do NOT get their fit window
        re-centered on the neighboring spectrum's fitted peak position;
        instead the window is frozen at whatever position it had reached
        right at this index and held constant for every lower-power step.
        None (default) reproduces the original behavior: the window
        tracks the fitted peak across the entire power series.
    """
    if nw.start_conditions is None or nw.n_sel_peaks == 0:
        print('fit_nw: run set_startconditions first.')
        return

    # Choose spectrum data
    if subtract_fit_background == 'raw':
        spec_data = nw.spectra_raw.copy()
    else:
        spec_data = _get_spectrum(nw, 'final')
        if spec_data is None:
            spec_data = nw.spectra_raw.copy()

    wl = nw.wavelength
    n_wl, n_powers = spec_data.shape
    n_peaks = nw.n_sel_peaks

    fits = [[None]*n_powers for _ in range(n_peaks)]
    fit_data_arr = [[None]*n_powers for _ in range(n_peaks)]
    peak_max = np.full((n_peaks, n_powers), np.nan)
    peak_int = np.full((n_peaks, n_powers), np.nan)
    findpeaks_fwhm = np.full((n_peaks, n_powers), np.nan)

    # Results from integration
    peak_pos_cell   = [[None]*n_powers for _ in range(n_peaks)]
    peak_pos_err_c  = [[None]*n_powers for _ in range(n_peaks)]
    peak_area_cell  = [[None]*n_powers for _ in range(n_peaks)]
    peak_area_err_c = [[None]*n_powers for _ in range(n_peaks)]
    fwhm_cell       = [[None]*n_powers for _ in range(n_peaks)]
    fwhm_err_cell   = [[None]*n_powers for _ in range(n_peaks)]

    if show_progress:
        fig_p, ax_p = plt.subplots(1, 2, figsize=(10, 4))
        plt.tight_layout()
        plt.ion()
        plt.show()

    for j in range(n_peaks):
        sc = nw.start_conditions[:, j]
        fitwindow    = int(round(sc[1]))
        fitwindow    = 2 * round(fitwindow / 2)   # make even
        fitwindow    = max(fitwindow, 4)
        max_spec_idx = int(round(sc[2]))
        peak_idx_arr = np.zeros(n_powers, dtype=int)
        peak_idx_arr[max_spec_idx] = int(round(sc[0]))

        # Iteration order: start at max_spec_idx, go down then up
        order = list(range(max_spec_idx, -1, -1)) + list(range(max_spec_idx+1, n_powers))

        def _seed_for(target_idx, tracked_val, current_val):
            """Peak-index guess to seed `target_idx` with: the tracked
            position from the just-processed neighbor, unless
            fixed_window_below_index applies to `target_idx`, in which
            case the window stays frozen at `current_val` instead of
            following the tracked peak."""
            if (fixed_window_below_index is not None
                    and target_idx < fixed_window_below_index):
                return current_val
            return tracked_val

        for i in order:
            i1 = max(0,        peak_idx_arr[i] - fitwindow // 2)
            i2 = min(n_wl - 1, peak_idx_arr[i] + fitwindow // 2)
            Y_raw = spec_data[i1:i2+1, i]
            X     = wl[i1:i2+1]

            peak_max[j, i] = float(np.nanmax(Y_raw)) if len(Y_raw) else np.nan

            X_bg, Y_bg, bg = subtract_local_background(X, Y_raw,
                                                        subtract_fit_background
                                                        if subtract_fit_background != 'raw'
                                                        else 'none')
            peak_int[j, i] = float(np.trapz(Y_bg, X_bg)) if len(Y_bg) > 1 else 0.0
            fit_data_arr[j][i] = np.column_stack([X_bg, Y_bg, bg])

            popt, pcov, last_pi, fw = None, None, fitwindow // 2, np.nan
            if len(Y_bg) > 3 and np.any(Y_bg != 0) and not np.any(np.isnan(Y_bg)):
                try:
                    popt, pcov, last_pi, fw, n_sub, is_lorentz = _run_fit(
                        X_bg, Y_bg, fitfunction)
                except Exception:
                    pass

            findpeaks_fwhm[j, i] = fw

            # Store fit object as SimpleNamespace with popt/pcov
            if popt is not None:
                fit_ns = SimpleNamespace(popt=popt, pcov=pcov,
                                         n=n_sub, is_lorentz=is_lorentz,
                                         fitfunction=fitfunction)
                fits[j][i] = fit_ns

                # Extract parameters
                if is_lorentz:
                    pp, pe, pa, pae, pf, pfe = _lorentz_params_from_popt(
                        popt, pcov, n_sub)
                else:
                    pp, pe, pa, pae, pf, pfe = _gauss_params_from_popt(
                        popt, pcov, n_sub)

                peak_pos_cell[j][i]   = pp
                peak_pos_err_c[j][i]  = pe
                peak_area_cell[j][i]  = pa
                peak_area_err_c[j][i] = pae
                fwhm_cell[j][i]       = pf
                fwhm_err_cell[j][i]   = pfe

                # Track peak: local index of strongest peak
                if is_lorentz:
                    best_sub = int(np.argmax([popt[3*k] for k in range(n_sub)]))
                    b_best = popt[3*best_sub + 2]  # center
                else:
                    best_sub = int(np.argmax([popt[3*k] for k in range(n_sub)]))
                    b_best = popt[3*best_sub + 1]  # center
                last_pi = int(np.argmin(np.abs(wl[i1:i2+1] - b_best)))

            else:
                peak_pos_cell[j][i]   = np.array([np.nan])
                peak_pos_err_c[j][i]  = np.array([np.nan])
                peak_area_cell[j][i]  = np.array([np.nan])
                peak_area_err_c[j][i] = np.array([np.nan])
                fwhm_cell[j][i]       = np.array([np.nan])
                fwhm_err_cell[j][i]   = np.array([np.nan])

            # Update peak index for adjacent spectrum
            new_pi = last_pi + peak_idx_arr[i] - fitwindow // 2
            new_pi = int(np.clip(new_pi, 0, n_wl - 1))
            if i > 0 and i <= max_spec_idx:
                peak_idx_arr[i - 1] = _seed_for(i - 1, new_pi, peak_idx_arr[i])
            elif i > max_spec_idx and i + 1 < n_powers:
                peak_idx_arr[i + 1] = _seed_for(i + 1, new_pi, peak_idx_arr[i])
            elif i == max_spec_idx and i + 1 < n_powers:
                seed = (peak_idx_arr[max_spec_idx - 1]
                        if max_spec_idx > 0
                        else peak_idx_arr[max_spec_idx])
                peak_idx_arr[i + 1] = _seed_for(i + 1, seed, peak_idx_arr[i])

            if show_progress and popt is not None:
                ax_p[0].cla()
                ax_p[0].semilogy(wl, spec_data[:, i], alpha=0.5)
                ax_p[0].semilogy(X_bg, Y_bg + bg, 'k-')
                ax_p[0].semilogy(X_bg, Y_bg, 'r-')
                ax_p[0].set_title(f'{nw.name}\nP={nw.power[i]:.3f} mW', fontsize=8)
                ax_p[1].cla()
                x_plot = np.linspace(X_bg[0], X_bg[-1], 200)
                model_fn = _lorentz_n_model(n_sub) if is_lorentz else _gauss_n_model(n_sub)
                ax_p[1].plot(X_bg, Y_bg, 'k+')
                ax_p[1].plot(x_plot, model_fn(x_plot, *popt), 'r-')
                ax_p[1].set_title(f'peak {j+1}, power step {i}', fontsize=8)
                plt.pause(0.05)

    if show_progress:
        plt.ioff()

    nw.fits            = fits
    nw.fit_data        = fit_data_arr
    nw.peak_maximum    = peak_max
    nw.peak_integral   = peak_int
    nw.findpeaks_fwhm  = findpeaks_fwhm
    nw.peak_pos        = peak_pos_cell
    nw.peak_pos_err    = peak_pos_err_c
    nw.peak_area       = peak_area_cell
    nw.peak_area_err   = peak_area_err_c
    nw.fwhm            = fwhm_cell
    nw.fwhm_err        = fwhm_err_cell
    nw.fit_model       = fitfunction
    nw.background_type = subtract_fit_background

    # Total peak area per power step
    total = np.zeros(n_powers)
    for i in range(n_powers):
        vals = [np.nansum(peak_area_cell[j][i])
                for j in range(n_peaks)
                if peak_area_cell[j][i] is not None]
        total[i] = np.nansum(vals)
    nw.total_peak_area = total

    print(f'fit_nw: completed ({n_peaks} peak(s), {n_powers} power step(s)).')


# ══════════════════════════════════════════════════════════════════════════════
# 9.  integrate_spectra
# ══════════════════════════════════════════════════════════════════════════════

def integrate_spectra(nw, center, width, spectrumtype='no_background',
                      method='trapz', subtract_bg=False):
    """
    Integrate spectrum over [center - width/2, center + width/2] for every
    power step and append the result to nw.specsum.

    Parameters
    ----------
    nw           : Nanowire
    center       : float  — spectral centre (in nw.wavelength units)
    width        : float  — full integration window width
    spectrumtype : 'raw', 'final', 'no_background'
    method       : 'trapz', 'sum', or 'rect'
    subtract_bg  : False, 'linear', or 'constant'  — local background

    Returns
    -------
    specsum_values : (n_powers,) ndarray
    """
    spectra = _get_spectrum(nw, spectrumtype)
    if spectra is None:
        print('integrate_spectra: no spectra available.')
        return None
    wl = nw.wavelength
    n_powers = spectra.shape[1]
    out = np.full(n_powers, np.nan)

    for i in range(n_powers):
        y, x = crop_vector(spectra[:, i], center - width/2, center + width/2, wl)
        if len(y) == 0:
            continue
        if subtract_bg:
            x, y, _ = subtract_local_background(x, y, subtract_bg)
        if method == 'sum':
            out[i] = float(np.sum(y))
        elif method == 'trapz' and len(y) > 1:
            out[i] = float(np.trapz(y, x))
        elif method == 'rect' and len(y) > 1:
            out[i] = float(np.sum(y * np.gradient(x)))
        elif len(y) == 1:
            out[i] = float(y[0])

    # replace zeros with nan (first spectrum used as dark might be zero)
    out[out == 0] = np.nan
    out = np.abs(out)

    entry = dict(values=out, center=center, width=width,
                 spectrumtype=spectrumtype)
    if nw.specsum is None:
        nw.specsum = []
    nw.specsum.append(entry)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 10.  thresholds  (interactive L-L kink finder)
# ══════════════════════════════════════════════════════════════════════════════

def _threshold_single(power, values, title='', peak_idx=1):
    """
    Interactive threshold finder for one peak.

    Shows the L-L curve; user drags a rectangle to select the sub-threshold
    (ASE) region; a linear fit is computed; pressing Enter confirms or
    Backspace redoes.

    Returns (threshold_mw, threshold_err, slope, slope_err).
    """
    power  = np.asarray(power,  dtype=float)
    values = np.asarray(values, dtype=float)
    label = f'"{title}"  peak-{peak_idx}' if peak_idx else f'"{title}"  total'

    result = dict(threshold=np.nan, threshold_err=np.nan,
                  slope=np.nan, slope_err=np.nan)
    confirmed = [False]

    while not confirmed[0]:
        include = [None]

        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(power, values, zorder=3, picker=True)
        ax.set_xlabel('Power (mW)')
        ax.set_ylabel('Peak area (arb.u.)')
        ax.set_title(f'{label}\nDrag rectangle to select ASE region, then Confirm',
                     fontsize=9)
        plt.tight_layout()

        ax_confirm = fig.add_axes([0.7, 0.01, 0.13, 0.06])
        ax_redo    = fig.add_axes([0.85, 0.01, 0.12, 0.06])
        btn_confirm = mwidgets.Button(ax_confirm, 'Confirm')
        btn_redo    = mwidgets.Button(ax_redo,    'Redo')

        rect_pts = [np.array([], dtype=int)]

        def _on_rect(eclick, erelease):
            x0, x1 = sorted([eclick.xdata, erelease.xdata])
            y0, y1 = sorted([eclick.ydata, erelease.ydata])
            inside = np.where((power >= x0) & (power <= x1) &
                              (values >= y0) & (values <= y1))[0]
            rect_pts[0] = inside
            offsets = sc.get_offsets()
            colors = np.full(len(power), 0.4)
            colors[inside] = 1.0
            sc.set_alpha(colors)
            fig.canvas.draw_idle()

        rs = mwidgets.RectangleSelector(
            ax, _on_rect, useblit=True,
            button=[1], minspanx=0, minspany=0, spancoords='data',
            interactive=True)

        def _do_confirm(_e):
            include[0] = rect_pts[0].tolist()
            confirmed[0] = True
            plt.close(fig)

        def _do_redo(_e):
            include[0] = None
            plt.close(fig)

        btn_confirm.on_clicked(_do_confirm)
        btn_redo.on_clicked(_do_redo)
        plt.show(block=True)

        if not confirmed[0]:
            continue

        sel = include[0]
        if sel is None or len(sel) < 2:
            print('Not enough points selected — using all points.')
            sel = list(range(len(power)))

        excl = [i for i in range(len(power)) if i not in sel]
        p_sel, v_sel = power[sel], values[sel]

        # Filter NaN
        valid = ~(np.isnan(p_sel) | np.isnan(v_sel))
        p_sel, v_sel = p_sel[valid], v_sel[valid]
        if len(p_sel) < 2:
            print('threshold: not enough valid points for fit.')
            result = dict(threshold=np.nan, threshold_err=np.nan,
                          slope=np.nan, slope_err=np.nan)
            confirmed[0] = True
            continue

        coeffs = np.polyfit(p_sel, v_sel, 1)   # [slope, intercept]
        thresh = -coeffs[1] / coeffs[0] if coeffs[0] != 0 else np.nan

        # Error via covariance
        if len(p_sel) >= 3:
            A = np.vstack([p_sel, np.ones(len(p_sel))]).T
            _, residuals, _, _ = np.linalg.lstsq(A, v_sel, rcond=None)
            if len(residuals) == 0:
                residuals = np.array([np.sum((v_sel - np.polyval(coeffs, p_sel))**2)])
            sigma2 = residuals[0] / max(len(p_sel) - 2, 1)
            ATA_inv = np.linalg.inv(A.T @ A) * sigma2
            d_slope = np.sqrt(ATA_inv[0, 0])
            d_inter = np.sqrt(ATA_inv[1, 1])
            thresh_err = np.sqrt((d_inter/coeffs[0])**2 +
                                 (d_slope*coeffs[1]/coeffs[0]**2)**2)
        else:
            d_slope, thresh_err = np.nan, np.nan

        result = dict(threshold=thresh, threshold_err=thresh_err,
                      slope=coeffs[0], slope_err=d_slope)

        # Show result plot
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(power, values, 'bx')
        if len(p_sel) >= 2:
            x_fit = np.linspace(0, max(power)*1.05, 200)
            ax2.plot(x_fit, np.polyval(coeffs, x_fit), 'r-')
        if not np.isnan(thresh):
            ax2.axvline(thresh, color='k', linestyle='--',
                        label=f'Threshold = {thresh:.3f} mW')
        ax2.set_xlabel('Power (mW)')
        ax2.set_ylabel('Peak area (arb.u.)')
        ax2.set_title(f'{label}\n'
                      f'Threshold = {thresh:.3f} mW ± {thresh_err:.4f}\n'
                      f'Slope = {coeffs[0]:.3g}  |  Enter=confirm, Backspace=redo',
                      fontsize=8)
        ax2.legend(fontsize=8)
        plt.tight_layout()
        plt.show(block=False)

        key = [None]

        def _on_key(event, _f=fig2):
            key[0] = event.key
            plt.close(_f)

        fig2.canvas.mpl_connect('key_press_event', _on_key)
        plt.waitforbuttonpress(timeout=0)
        try:
            plt.close(fig2)
        except Exception:
            pass

        if key[0] == 'backspace':
            confirmed[0] = False
        else:
            confirmed[0] = True

    print(f'Threshold: {result["threshold"]:.3f} mW ± {result["threshold_err"]:.4f}  '
          f'| Slope: {result["slope"]:.3g}')
    return (result['threshold'], result['threshold_err'],
            result['slope'],    result['slope_err'])


def thresholds(nw, mode='area'):
    """
    Interactively find lasing thresholds.

    mode : str or list — 'area', 'integral', 'max', 'total', 'all'
    """
    if isinstance(mode, str):
        mode = [mode]

    do_area     = any(m in ('area',     'all') for m in mode)
    do_integral = any(m in ('integral', 'all') for m in mode)
    do_max      = any(m in ('max',      'all') for m in mode)
    do_total    = any(m in ('total',    'all') for m in mode)

    power      = nw.power
    n_peaks    = nw.n_sel_peaks
    name       = nw.name or ''

    # Convert peak_area cell to matrix (max sub-peak per cell)
    def _cell_max_mat(cell, n_pk, n_pw):
        mat = np.full((n_pk, n_pw), np.nan)
        for j in range(n_pk):
            for i in range(n_pw):
                v = cell[j][i] if cell is not None else None
                if v is not None and np.any(~np.isnan(v)):
                    mat[j, i] = float(np.nanmax(v))
        return mat

    n_pw = len(power)

    if do_area and nw.peak_area is not None:
        area_mat = _cell_max_mat(nw.peak_area, n_peaks, n_pw)
        thresh_arr = np.full(n_peaks, np.nan)
        thresh_err_arr = np.full(n_peaks, np.nan)
        slope_arr = np.full(n_peaks, np.nan)
        slope_err_arr = np.full(n_peaks, np.nan)
        for j in range(n_peaks):
            t, te, s, se = _threshold_single(power, area_mat[j], name, j+1)
            thresh_arr[j], thresh_err_arr[j] = t, te
            slope_arr[j], slope_err_arr[j]   = s, se
        nw.threshold     = thresh_arr
        nw.threshold_err = thresh_err_arr
        nw.slope         = slope_arr
        nw.slope_err     = slope_err_arr

    if do_integral and nw.peak_integral is not None:
        thresh_arr = np.full(n_peaks, np.nan)
        thresh_err_arr = np.full(n_peaks, np.nan)
        slope_arr = np.full(n_peaks, np.nan)
        slope_err_arr = np.full(n_peaks, np.nan)
        for j in range(n_peaks):
            t, te, s, se = _threshold_single(power, nw.peak_integral[j], name, j+1)
            thresh_arr[j], thresh_err_arr[j] = t, te
            slope_arr[j], slope_err_arr[j]   = s, se
        nw.threshold_integral     = thresh_arr
        nw.threshold_err_integral = thresh_err_arr
        nw.slope_integral         = slope_arr
        nw.slope_err_integral     = slope_err_arr

    if do_max and nw.peak_maximum is not None:
        thresh_arr = np.full(n_peaks, np.nan)
        thresh_err_arr = np.full(n_peaks, np.nan)
        slope_arr = np.full(n_peaks, np.nan)
        slope_err_arr = np.full(n_peaks, np.nan)
        for j in range(n_peaks):
            t, te, s, se = _threshold_single(power, nw.peak_maximum[j], name, j+1)
            thresh_arr[j], thresh_err_arr[j] = t, te
            slope_arr[j], slope_err_arr[j]   = s, se
        nw.threshold_max     = thresh_arr
        nw.threshold_err_max = thresh_err_arr
        nw.slope_max         = slope_arr
        nw.slope_err_max     = slope_err_arr

    if do_total and nw.total_peak_area is not None:
        t, te, s, se = _threshold_single(power, nw.total_peak_area, name, 0)
        nw.threshold_sum     = np.array([t])
        nw.threshold_err_sum = np.array([te])
        nw.slope_sum         = np.array([s])
        nw.slope_err_sum     = np.array([se])

    # Dominant peak index (peak with max area across all powers)
    if nw.peak_area is not None and n_peaks > 0:
        area_mat = _cell_max_mat(nw.peak_area, n_peaks, n_pw)
        nw.dominant_peak_index = int(np.nanargmax(np.nanmax(area_mat, axis=1)))


# ══════════════════════════════════════════════════════════════════════════════
# 11.  Spot size / density / fluence
# ══════════════════════════════════════════════════════════════════════════════

def set_spotsize(nw, radius_um):
    """Set spot radius (µm) and compute pump fluence / power density."""
    nw.spot_radius_short = float(radius_um)
    _compute_density_fluence(nw)


def _compute_density_fluence(nw):
    """Compute pump_fluence (µJ/cm²) and power_density (kW/cm²) from stored power and spot."""
    if nw.spot_radius_short is None or nw.power is None:
        return
    r_m = nw.spot_radius_short * 1e-6   # µm → m
    area_cm2 = np.pi * (r_m * 1e2)**2   # m² → cm²
    if nw.rep_rate is not None and nw.rep_rate > 0:
        nw.pump_fluence = nw.power / (nw.rep_rate * 1e-3 * area_cm2)  # mW/Hz → µJ/cm²
        # power_density: peak power = fluence * rep_rate / pulse_duration
        # approximate as average power density
    nw.power_density = nw.power * 1e-3 / area_cm2 * 1e-3  # mW → kW/cm²


# ══════════════════════════════════════════════════════════════════════════════
# 12.  Utility
# ══════════════════════════════════════════════════════════════════════════════

def cell_to_mat(cell, operator='max', operator_cell=None):
    """
    Convert a list-of-lists of ndarrays to a 2-D ndarray by applying operator
    element-wise.  operator: 'max', 'min', 'mean', 'sum'.
    """
    if cell is None or len(cell) == 0:
        return np.array([])
    n_rows = len(cell)
    n_cols = len(cell[0]) if hasattr(cell[0], '__len__') else 1
    out = np.full((n_rows, n_cols), np.nan)
    src = operator_cell if operator_cell is not None else cell
    for i in range(n_rows):
        for j in range(n_cols):
            v = src[i][j] if (hasattr(src[i], '__getitem__') and j < len(src[i])) else src[i]
            if v is None:
                continue
            v = np.asarray(v, dtype=float)
            if operator == 'max':
                out[i, j] = float(np.nanmax(v)) if v.size else np.nan
            elif operator == 'min':
                out[i, j] = float(np.nanmin(v)) if v.size else np.nan
            elif operator == 'mean':
                out[i, j] = float(np.nanmean(v)) if v.size else np.nan
            elif operator == 'sum':
                out[i, j] = float(np.nansum(v)) if v.size else np.nan
    return out


def nmev(x):
    """Convert between nm and eV: nmev(nm)→eV, nmev(eV)→nm."""
    return 1239.84193 / np.asarray(x, dtype=float)
