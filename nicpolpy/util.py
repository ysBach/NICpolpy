import numpy as np
import ccdproc
from scipy.optimize import curve_fit
from astropy.stats import sigma_clip

__all__ = ["USEFUL_KEYS", "OBJSECTS", "NICSECTS", "GAIN", "RDNOISE",
           "FIND_KEYS", "LACOSMIC_KEYS",
           "fitsxy2py", "infer_filter", "split_oe", "split_quad",
           "multisin", "fit_sinusoids", "fft_peak_freq"
           ]
#    "split_load",

USEFUL_KEYS = ["DATE-OBS", "UT-STR", "EXPTIME", "DATA-TYP", "OBJECT",
               "FILTER", "POL-AGL1", "WAVEPLAT", "SHUTTER",
               "AIRMASS", "ZD", "ALTITUDE", "AZIMUTH",
               "DITH_NUM", "DITH_NTH", "DITH_RAD",
               "NAXIS1", "NAXIS2", "BIN-FCT1", "BIN-FCT2",
               "RA2000", "DEC2000", "DOM-HUM", "DOM-TMP",
               "OUT-HUM", "OUT-TMP", "OUT-WND", "WEATHER",
               "NICTMP1", "NICTMP2", "NICTMP3", "NICTMP4", "NICTMP5",
               "NICHEAT", "DET-ID"]

OBJSECTS = dict(J=["[530:690, 295:745]", "[710:870, 295:745]"],
                H=["[555:715, 320:770]", "[735:895, 320:770]"],
                K=["[550:710, 330:780]", "[720:880, 330:780]"])

NICSECTS = dict(lower="[:, :512]", upper="[:, 513:]",
                left="[:512, :]", right="[513:, :]")

GAIN = dict(J=9.2, H=9.8, K=9.4)
RDNOISE = dict(J=50, H=75, K=83)

LACOSMIC_KEYS = {'sigclip': 4.5,
                 'sigfrac': 0.5,
                 'objlim': 1.0,
                 'satlevel': np.inf,
                 'pssl': 0.0,
                 'niter': 4,
                 'sepmed': False,
                 'cleantype': 'medmask',
                 'fsmode': 'median',
                 'psfmodel': 'gauss',
                 'psffwhm': 2.5,
                 'psfsize': 7,
                 'psfk': None,
                 'psfbeta': 4.765}

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


def fitsxy2py(fits_section):
    ''' Given FITS section in str, returns the slices in python convention.
    Parameters
    ----------
    fits_section : str
        The section specified by FITS convention, i.e., bracket embraced,
        comma separated, XY order, 1-indexing, and including the end index.
    Note
    ----
    >>> np.eye(5)[fitsxy2py('[1:2,:]')]
    # array([[1., 0.],
    #       [0., 1.],
    #       [0., 0.],
    #       [0., 0.],
    #       [0., 0.]])
    '''
    slicer = ccdproc.utils.slices.slice_from_string
    sl = slicer(fits_section, fits_convention=True)
    return sl


def infer_filter(ccd, filt=None, verbose=True):
    if filt is None:
        filt = ccd.header["FILTER"]
        if verbose:
            print(f"Assuming filter is '{filt}' from header.")
    return filt


def split_oe(ccd, filt=None, verbose=True):
    filt = infer_filter(ccd, filt=filt, verbose=verbose)
    ccd_o = ccdproc.trim_image(ccd, fits_section=OBJSECTS[filt][0])
    ccd_e = ccdproc.trim_image(ccd, fits_section=OBJSECTS[filt][1])
    return (ccd_o, ccd_e)

# # FIXME: Drop the dependency on yfu...
# def split_load(fpath, outputs=None, dtype='float32'):
#     import ysfitsutilpy as yfu
#     hdr = fits.getheader(fpath)
#     filt = hdr["FILTER"]
#     # ccd = yfu.CCDData_astype(yfu.load_ccd(fpath), dtype=dtype)
#     cuts = yfu.imcopy(fpath, fits_sections=PCRSECTS[filt],
#                       outputs=outputs, dtype=dtype)
#     return cuts


def split_quad(ccd):
    nx = ccd.data.shape[1]
    i_half = nx//2

    quads = [ccd.data[-i_half:, :i_half],
             ccd.data[-i_half:, -i_half:],
             ccd.data[:i_half, :i_half],
             ccd.data[:i_half, -i_half:]]
    return quads


def multisin(x, f, c, a, p):
    res = np.zeros(x.size)
    if len(a) > 0 or a is not None:
        if not (len(f) == len(a) == len(p)):
            raise ValueError("f, a, p must have identical length. "
                             + f"Now they are {len(f)}, {len(a)}, {len(p)}.")
        for _a, _f, _p in zip(a, f, p):
            res += _a*np.sin(2*np.pi*_f*x + _p)
    # for _a, _f, _p in zip(a, f, p):
    #     res += _a*np.sin(2*np.pi*_f*x + _p)

    return c + res


def fit_sinusoids(xdata, ydata, freqs, **kwargs):
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
        return multisin(x, freqs, c, a, p)

    n = len(freqs)
    if n == 0 or freqs is None:
        # A constant fitting is identical to weighted mean, which in
        # this case is just a mean as we don't have any weight.
        return (np.mean(ydata), None, None), None

    p0 = np.zeros(2*n + 1)
    popt, pcov = curve_fit(_sin, xdata, ydata, p0=p0, **kwargs)

    return (popt[-1], popt[:n], popt[n:-1]), pcov


def fft_peak_freq(fftamplitude, max_peak=5,
                  sigclip_kw={'sigma_lower': np.inf, 'sigma_upper': 3}):
    """ Select the FFT amplitude peaks for positive frequencies.
    Parameters
    ----------
    fftamplitude : 1d array
        _Absolute_ FFT amplitude in 1d. For example, ``fftamplitude =
        np.abs(np.fft.fft(data1d))``.
    max_peak : int, optional
        The maximum number of peaks to be found.
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

    amp = np.asarray(fftamplitude)
    freq = np.fft.fftfreq(amp.size)
    if amp.ndim > 1:
        raise ValueError("Input array (fft) must be 1-D")

    i_half = amp.size // 2 + 1
    amp = amp[1:i_half]  # exclude 0-th (DC level)

    # Robust linear fit, find points with high residuals
    xx = np.linspace(1, i_half, amp.size)  # actually just a dummy...
    popt, _ = curve_fit(lin_fun, xx, amp, method='trf', loss='cauchy')
    resid = amp - lin_fun(xx, *popt)

    # Find points above k-sigma level by sigma-clip
    mask = sigma_clip(resid, **sigclip_kw).mask
    mask_idx = np.where(mask)[0]  # masked = higher values from sigclip.

    # highest ``max_peak`` (highest to lower peaks):
    top_n_idx = np.argsort(resid)[::-1][:max_peak]

    # Select those meet BOTH top N peaks & high after sigma clip
    idx = np.intersect1d(top_n_idx, mask_idx)

    return freq[idx + 1]
