from pathlib import Path

import numpy as np
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit

from ysfitsutilpy import fitsxy2py, trim_ccd, make_summary, LACOSMIC_KEYS

__all__ = ["USEFUL_KEYS", "OBJSECTS", "NICSECTS", "FOURIERSECTS", "FOURIPEAKSECT",
           "PWD", "DARK_PATHS", "FLAT_PATHS", "MASK_PATHS",
           "OBJSLICES", "NICSLICES", "FOURIERSLICES", "FOURIPEAKSLICE",
           "NHAO_LOCATION",
           "NIC_CRREJ_KEYS", "GAIN", "RDNOISE",
           "FIND_KEYS",
           "infer_filter", "split_oe", "split_quad",
           "multisin", "fit_sinusoids", "fft_peak_freq", "summary_nic"
           ]


USEFUL_KEYS = [
    "DATE-OBS", "UT-STR", "EXPTIME", "UT-END", "DATA-TYP", "OBJECT",
    "FILTER", "POL-AGL1", "PA", "INSROT", "IMGROT", "WAVEPLAT",
    "SHUTTER", "AIRMASS", "ZD", "ALTITUDE", "AZIMUTH",
    "DITH_NUM", "DITH_NTH", "DITH_RAD",
    "NAXIS1", "NAXIS2", "BIN-FCT1", "BIN-FCT2", "RA2000", "DEC2000",
    "DOM-HUM", "DOM-TMP", "OUT-HUM", "OUT-TMP", "OUT-WND", "WEATHER",
    "NICTMP1", "NICTMP2", "NICTMP3", "NICTMP4", "NICTMP5", "NICHEAT", "DET-ID"
]


def OBJSECTS(right_half=False):
    if right_half:
        #                 150       440          150      440
        return dict(J=["[28:178, 300:740]", "[213:363, 300:740]"],
                    H=["[53:203, 325:765]", "[233:383, 325:765]"],
                    K=["[48:198, 335:775]", "[218:368, 335:775]"])

    else:
        #                 150       420          150      440
        return dict(J=["[540:690, 300:740]", "[725:875, 300:740]"],
                    H=["[565:715, 325:765]", "[745:895, 325:765]"],
                    K=["[560:710, 335:775]", "[730:880, 335:775]"])


NICSECTS = dict(lower="[:, :512]", upper="[:, 513:]", left="[:512, :]", right="[513:, :]")

VERTICALSECTS = ["[:, 100:250]", "[:, 850:974]"]
FOURIPEAKSECT = "[300:500, :]"
# FOURIERSECTS = dict(
#     J=["[10:285]", "[755:1010]"],
#     H=["[10:310]", "[780:1010]"],
#     K=["[10:320]", "[790:1010]"],
# )

FOURIERSECTS = dict(
    J=["[10:245]", "[795:1010]"],
    H=["[10:270]", "[850:1010]"],
    K=["[10:280]", "[830:1010]"],
)

PWD = Path(__file__).parent
DARK_PATHS = dict(
    J=PWD/"dark"/"v_j191022_DARK_120.fits",
    H=PWD/"dark"/"v_h191022_DARK_120.fits",
    K=PWD/"dark"/"v_k191022_DARK_120.fits",
)
FLAT_PATHS = dict(
    J=dict(o=PWD/"flat"/"v_j180507_FLAT_all_o.fits",
           e=PWD/"flat"/"v_j180507_FLAT_all_e.fits"),
    H=dict(o=PWD/"flat"/"v_h180507_FLAT_all_o.fits",
           e=PWD/"flat"/"v_h180507_FLAT_all_e.fits"),
    K=dict(o=PWD/"flat"/"v_k180507_FLAT_all_o.fits",
           e=PWD/"flat"/"v_k180507_FLAT_all_e.fits")
)
MASK_PATHS = dict(
    J=PWD/"mask"/"j_mask.fits",
    H=PWD/"mask"/"h_mask.fits",
    K=PWD/"mask"/"k_mask.fits",
)

GAIN = dict(J=9.2, H=9.8, K=9.4)
RDNOISE = dict(J=50, H=75, K=83)

NIC_CRREJ_KEYS = LACOSMIC_KEYS.copy()
NIC_CRREJ_KEYS["sepmed"] = True
NIC_CRREJ_KEYS['satlevel'] = np.inf
NIC_CRREJ_KEYS['objlim'] = 5
NIC_CRREJ_KEYS['sigfrac'] = 5
NIC_CRREJ_KEYS['cleantype'] = 'median'


FIND_KEYS = {'o': dict(ratio=1.0,  # 1.0: circular gaussian
                       sigma_radius=1.5,  # default values 1.5
                       sharplo=0.2, sharphi=1.0,  # default values 0.2 and 1.0
                       roundlo=-1.0, roundhi=1.0,  # default values -1 and +1
                       theta=0.0,  # major axis angle from x-axis in DEGREE
                       sky=0.0, exclude_border=True,
                       brightest=None, peakmax=None),
             'e': dict(ratio=1.0,  # 1.0: circular gaussian
                       sigma_radius=1.5,  # default values 1.5
                       sharplo=0.2, sharphi=1.0,  # default values 0.2 and 1.0
                       roundlo=-1.0, roundhi=1.0,  # default values -1 and +1
                       theta=0.0,  # major axis angle from x-axis in DEGREE
                       sky=0.0, exclude_border=True,
                       brightest=None, peakmax=None)}


def _fits2sl(fits_sect):
    pyth_slice = {}
    for k, sects in fits_sect.items():
        pyth_slice[k] = []
        for sect in sects:
            pyth_slice[k].append(fitsxy2py(sect))
    return pyth_slice


def OBJSLICES(right_half=False):
    return _fits2sl(OBJSECTS(right_half))


FOURIERSLICES = _fits2sl(FOURIERSECTS)
NICSLICES = {}
VERTICALSLICES = []

for k, sect in NICSECTS.items():
    NICSLICES[k] = fitsxy2py(sect)

for sect in VERTICALSECTS:
    VERTICALSLICES.append(fitsxy2py(sect))

FOURIPEAKSLICE = fitsxy2py(FOURIPEAKSECT)


NHAO_LOCATION = dict(lon=134.3356, lat=35.0253, elevation=0.449)


def infer_filter(ccd, filt=None, verbose=True):
    if filt is None:
        try:
            filt = ccd.header["FILTER"]
            if verbose:
                print(f"Assuming filter is '{filt}' from header.")
        except (KeyError, AttributeError):
            raise TypeError("Filter cannot be inferred from the given ccd.")
    return filt


def split_oe(ccd, filt=None, right_half=False, verbose=True):
    filt = infer_filter(ccd, filt=filt, verbose=verbose)
    ccd_o = trim_ccd(ccd, fits_section=OBJSECTS(right_half)[filt][0])
    ccd_o.header["OERAY"] = ("o", "O-ray or E-ray. Either 'o' or 'e'.")
    ccd_e = trim_ccd(ccd, fits_section=OBJSECTS(right_half)[filt][1])
    ccd_e.header["OERAY"] = ("e", "O-ray or E-ray. Either 'o' or 'e'.")
    return (ccd_o, ccd_e)


def split_quad(ccd):
    nx = ccd.data.shape[1]
    i_half = nx//2

    quads = [ccd.data[-i_half:, :i_half],
             ccd.data[-i_half:, -i_half:],
             ccd.data[:i_half, :i_half],
             ccd.data[:i_half, -i_half:]]
    return quads


def multisin(x, f, a, p, c):
    res = np.zeros(x.size)
    if a is not None:
        if len(a) > 0:
            if not (len(f) == len(a) == len(p)):
                raise ValueError("f, a, p must have identical length. "
                                 + f"Now they are {len(f)}, {len(a)}, {len(p)}.")
            for _a, _f, _p in zip(a, f, p):
                # res += _a*np.sin(2*np.pi*_f*x + _p)
                # Mathematically, the "integrated" version below should
                # work better, in my opinion, but apparently it does not
                # modify the results much...
                # -- 2019-12-26 11:46:03 (KST: GMT+09:00), YPB
                w = 2*np.pi*_f
                wx_l = w*(x - 0.5)
                wx_r = w*(x + 0.5)
                res += _a/w * (np.cos(wx_l + _p) - np.cos(wx_r + _p))
                # res += 1
    # for _a, _f, _p in zip(a, f, p):
    #     res += _a*np.sin(2*np.pi*_f*x + _p)

    return c + res


def lin_multisin(x, f, a, p, c):
    res = np.zeros(x.size)

    if a is not None:
        if len(a) > 0:
            if not (len(f) == len(a) == len(p)):
                raise ValueError("f, a, p must have identical length. "
                                 + f"Now they are {len(f)}, {len(a)}, {len(p)}.")
            for _a, _f, _p in zip(a, f, p):
                # res += _a*np.sin(2*np.pi*_f*x + _p)
                w = 2*np.pi*_f
                wx_l = w*(x - 0.5)
                wx_r = w*(x + 0.5)
                res += _a/w * (np.cos(wx_l + _p) - np.cos(wx_r + _p))
    return c + res


def fit_sinusoids(xdata, ydata, freqs, p0=None, **kwargs):
    """ Fit const + sin_functions for given frequencies.
    Parameters
    ----------
    xdata, ydata : array-like
        The x and y data to be used for ``scipy.optimize.curve_fit``.
    freqs : array-like
        The frequencies to be used to make sinusoidal curves. The
        resulting function will be ``c + sum(a_i*sin(2*pi*freqs[i]*x +
        p_i)``, where ``a_i`` and ``p_i`` are the amplitude and phase of
        the ``i``-th frequency sine curve.

    Returns
    -------
    popt, pcov: arrays
        The result from ``scipy.optimize.curve_fit``.
    """
    def _sin(x, *pars):
        n = len(pars)
        # if n == 1:
        #     return pars[-1]
        c = pars[-1]
        a = pars[:n//2]
        p = pars[n//2:-1]
        # Resulting popt will be [amp0, ..., phase0, ..., const]
        return multisin(x, freqs, a, p, c)

    n = len(freqs)
    if n == 0 or freqs is None:
        # A constant fitting is identical to weighted mean, which in
        # this case is just a mean as we don't have any weight.
        return (np.mean(ydata), None, None), None

    if p0 is None:
        p0 = np.zeros(2*n + 1)
        p0[:n] += 1  # initial guess of amplitudes
        p0[-1] = np.mean(ydata)  # initial guess of constant value

    popt, pcov = curve_fit(_sin, xdata, ydata, p0=p0, **kwargs)

    return (popt[:n], popt[n:-1], popt[-1]), pcov


def fft_peak_freq(fftamplitude, max_peaks=5, min_freq=0,
                  sigclip_kw={'sigma_lower': 3, 'sigma_upper': 3}):
    """ Select the FFT amplitude peaks for positive frequencies.
    Parameters
    ----------
    fftamplitude : 1d array
        _Absolute_ FFT amplitude in 1d. For example, ``fftamplitude =
        np.abs(np.fft.fft(data1d))``.
    max_peaks : int, optional
        The maximum number of peaks to be found.
    min_freq: float, optional
        The minimum frequency to pick. Giving positive value
        automatically removes negative frequencies. Default 0 means the
        DC is included.
    sigclip_kw :
        The arguments passed to ``astropy.stats.sigma_clip``. It's
        generally not very important to tune this for NIC data as of Dec
        2019.

    Returns
    -------
    freq : 1d array
        The FFT frequencies of the peaks in 1d.

    Example
    -------
    >>> amp = np.abs(np.fft.fft(data))/data.shape[0]  # normalize by ny
    >>> f =  fft_peak_freq(amp, 5)

    Note
    ----
    This is made for extracting frequencies only and use them in the
    next "sinusoidal fitting" procedure. If you still want to use it for
    FFT, you can do, e.g., ``fft_amp_peak = amp[np.where(np.in1d(f,
    np.fft.fftfreq(data.size)))]``.
    This part uses trf and cauchy loss to severly reject outliers for
    linear trend fitting. I guess this will not affect our results as
    the FFT peaks are "super strong". See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares

    """
    def lin_fun(x, a, b):
        return a*x + b

    amp_raw = np.asarray(fftamplitude)
    if amp_raw.ndim > 1:
        raise ValueError("Input array (fft) must be 1-D")

    # i_half = amp_raw.size // 2 + 1
    # amp = amp_raw[:i_half]

    freq_raw = np.fft.fftfreq(amp_raw.size)

    # Robust linear fit, find points with high residuals
    freqmask = (freq_raw < min_freq)
    amp = amp_raw[~freqmask]
    freq = freq_raw[~freqmask]
    popt, _ = curve_fit(lin_fun, freq, amp, method='trf', loss='cauchy')
    resid = amp - lin_fun(freq, *popt)

    # Find points above k-sigma level by sigma-clip
    mask = sigma_clip(resid, **sigclip_kw).mask
    mask_idx = np.where(mask)[0]  # masked = higher values from sigclip.

    # highest ``max_peaks`` (highest to lower peaks):
    top_n_idx = np.argsort(resid)[::-1][:max_peaks]

    # Select those meet BOTH top N peaks & high after sigma clip
    idx = np.intersect1d(top_n_idx, mask_idx)

    try:
        return freq[idx]
    except (TypeError, IndexError):
        return []


def summary_nic(inputs, output=None, keywords=USEFUL_KEYS, pandas=True,
                verbose=True, **kwargs):
    '''Extracts summary from the headers of FITS files.

    Parameters
    ----------
    inputs : glob pattern or list-like of path-like
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or
        list of files (each element must be path-like).

    output: str or path-like, optional
        The directory and file name of the output summary file.

    keywords: list or str(``"*"``), optional
        The list of the keywords to extract (keywords should be in str).

    pandas : bool, optional
        Whether to return pandas. If ``False``, astropy table object is
        returned. It will save csv format regardless of ``format``.
    '''
    return make_summary(inputs=inputs, output=output,
                        keywords=keywords, pandas=pandas, verbose=verbose, **kwargs)
