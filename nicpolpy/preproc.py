from multiprocessing import Pool

import astropy
import numpy as np
from astropy.nddata import CCDData
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.time import Time

from ysfitsutilpy import (LACOSMIC_KEYS, add_to_header, crrej, datahdr_parse,
                          fitsxy2py)

from .util import (FOURIERSECTS, GAIN, RDNOISE, VERTICALSECTS, fft_peak_freq,
                   fit_sinusoids, infer_filter, multisin)

__all__ = ["cr_reject_nic", "vertical_correct", "lrsubtract", "fit_fourier",
           "find_fourier_peaks"]


def cr_reject_nic(ccd, mask=None, filt=None, update_header=True, crrej_kw=None,
                  verbose=True, full=False):
    """
    Parameters
    ----------
    ccd: CCDData
        The ccd to be processed.
    filt: str, None, optional.
        The filter name in one-character string (case insensitive). If
        ``None``(default), it will be inferred from the header (key
        ``"FILTER"``).
    update_header: bool, optional.
        Whether to update the header if there is any.
    crrej_kw: dict, optional.
        The keyword arguments for the ``astroscrappy.detect_cosmics``.
        If ``None`` (default), the parameters for IRAF-version of L.A.
        Cosmic, except for the ``sepmed = True`` and ``gain`` and
        ``readnoise`` are replaced to the values of NIC detectors.
    verbose: bool, optional
        The verbose paramter to ``astroscrappy.detect_cosmics``.
    Returns
    -------
    nccd: CCDData
        The cosmic-ray removed object.

    Note
    ----
    astroscrappy automatically correct gain (i.e., output frame in the
    unit of electrons, not ADU). In this function, this is undone, i.e.,
    I divide it with the gain to restore ADU unit. Also ``nccd.mask``
    will contain the mask from cosmic-ray detection.
    """

    if crrej_kw is None:
        try:
            gain = ccd.gain.value
            rdnoise = ccd.rdnoise.value
        except AttributeError:
            filt = infer_filter(ccd, filt=filt, verbose=verbose)
            gain = GAIN[filt]
            rdnoise = RDNOISE[filt]
        crrej_kw = LACOSMIC_KEYS.copy()
        crrej_kw["sepmed"] = True
        crrej_kw['gain'] = gain
        crrej_kw['rdnoise'] = rdnoise
        crrej_kw['satlevel'] = 65535.
        crrej_kw['objlim'] = 5
        crrej_kw['sigfrac'] = 5

    nccd, crmask = crrej(
        ccd,
        mask=mask,
        **crrej_kw,
        propagate_crmask=False,
        verbose=verbose
    )

    if full:
        return nccd, crmask, crrej_kw
    else:
        return nccd


def vertical_correct(ccd, fitting_sections=None,
                     method='median', sigclip_kw=dict(sigma=2, maxiters=5),
                     dtype='float32', return_pattern=False,
                     update_header=True, verbose=False):
    ''' Correct vertical strip patterns.
    Paramters
    ---------
    ccd : CCDData, HDU object, HDUList, or ndarray.
        The CCD to subtract the vertical pattern.
    fitting_sections : list of two str, optional.
        The sections to be used for the vertical pattern estimation.
        This must be identical to the usual FITS section (i.e., that
        used in SAO ds9 or IRAF, 1-indexing and last-index-inclusive),
        not in python. **Give it in the order of ``[<upper>, <lower>]``
        in FITS y-coordinate.**
    method : str, optional.
        One of ``['med', 'avg', 'median', 'average', 'mean']``.
    sigma, maxiters : float and int, optional
        A sigma-clipping will be done to remove hot pixels and cosmic
        rays for estimating the vertical pattern. To turn sigma-clipping
        off, set ``maxiters=0``.
    sigclip_kw : dict, optional
        The keyword arguments for the sigma clipping.
    dtype : str, dtype, optional
        The data type to be returned.
    return_pattern : bool, optional.
        If ``True``, the subtracted pattern will also be returned.
        Default is ``False``.
    update_header : bool, optional.
        Whether to update the header if there is any.
    Return
    ------

    '''
    _t = Time.now()
    data, hdr = datahdr_parse(ccd)

    if fitting_sections is None:
        fitting_sections = VERTICALSECTS
    elif len(fitting_sections) != 2:
        raise ValueError("fitting_sections must have two elements.")

    if method in ['median', 'med']:
        methodstr = 'taking median'
        fun = np.median
        idx = 1
    elif method in ['average', 'avg', 'mean']:
        methodstr = 'taking average'
        fun = np.mean
        idx = 0
    else:
        raise ValueError("method not understood; it must be one of "
                         + "['med', 'avg', 'median', 'average', 'mean']")

    parts = []   # The two horizontal box areas from raw data
    strips = []  # The estimated patterns (1-D)
    for sect in fitting_sections:
        parts.append(data[fitsxy2py(sect)])

    try:
        if sigclip_kw["maxiters"] == 0:
            clipstr = "no clipping"
            for part in parts:
                strips.append(fun(part, axis=0))
    except KeyError:
        pass  # The user wants to use default value of maxiters in astropy.

    clipstr = f"sigma-clipping in astropy (v {astropy.__version__})"
    if sigclip_kw:
        clipstr += f", given {sigclip_kw}."
    else:
        clipstr += "."

    for part in parts:
        clip = sigma_clipped_stats(part, axis=0, **sigclip_kw)
        strips.append(clip[idx])

    ny, nx = data.shape
    vpattern = np.repeat(strips, ny/2, axis=0)
    vsub = data - vpattern

    if update_header and hdr is not None:
        s = (f"Vertical pattern subtracted using {fitting_sections} "
             + f"by {methodstr} with {clipstr}")
        # add as history
        add_to_header(hdr, 'h', s, verbose=verbose, t_ref=_t)

    try:
        nccd = CCDData(data=vsub, header=hdr)
    except ValueError:
        nccd = CCDData(data=vsub, header=hdr, unit='adu')

    nccd.data = nccd.data.astype(dtype)
    if return_pattern:
        return nccd, vpattern
    return nccd


def lrsubtract(ccd, fitting_sections=["[:, 50:100]", "[:, 924:974]"],
               method='median', sigma=3, maxiters=5, sigclip_kw={},
               dtype='float32', update_header=True, verbose=False):
    """Subtract left from right quadrants w/ vertical pattern removal."""
    _t = Time.now()

    nccd = vertical_correct(ccd=ccd,
                            fitting_sections=fitting_sections,
                            method=method, sigma=sigma, maxiters=maxiters,
                            sigclip_kw=sigclip_kw, dtype=dtype,
                            return_pattern=False,
                            update_header=update_header)
    nx = nccd.data.shape[1]
    i_half = nx//2
    nccd.data[:, i_half:] -= nccd.data[:, :i_half]
    if update_header:
        s = f"Subtracted left half ({i_half} columns) from the right half."
        # add as history
        add_to_header(nccd.header, 'h', s, t_ref=_t, verbose=verbose)
    # nccd = trim_image(nccd, fits_section=f"[{i_half + 1}:, :]")
    return nccd


def find_fourier_peaks(data, axis=0, max_peaks=3, min_freq=0.01,
                       sigclip_kw={'sigma_lower': np.inf, 'sigma_upper': 3}):
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Unly 2-D data is supported.")

    # FFT along axis
    npix = data.shape[axis]
    amp = np.abs(np.fft.fft(data, axis=axis)) / npix
    amp = amp - amp.mean(axis=0, keepdims=True)
    # ^ axis 0 will be frequency index
    # _, amp_median, _ = sigma_clipped_stats(amp, axis=axis - 1, std_ddof=1)
    amp_median = np.median(amp, axis=axis - 1)

    # Select high amplitude frequencies
    freq2fit = fft_peak_freq(amp_median,
                             max_peaks=max_peaks,
                             min_freq=min_freq,
                             sigclip_kw=sigclip_kw)
    return freq2fit


def _fitter(x, y, mask, freqs, idx):
    n = len(freqs)
    p0 = np.zeros(2*n + 1)
    p0[:n] += 1  # initial guess of amplitudes
    p0[-1] = 0  # initial guess of constant value, after vertical sub.
    if np.count_nonzero(mask) == x.size == y.size:
        return idx, [np.nan]*p0.size, np.zeros(x.size)
    try:
        popt, _ = fit_sinusoids(x[~mask], y[~mask], freqs, p0=p0)
    except RuntimeError:
        popt = (np.zeros(n), np.zeros(n), np.median(y))

    pattern_idx = multisin(x, freqs, *popt)

    # std_orig = np.std(y[~mask], ddof=1)
    # std_subt = np.std(pattern_idx[~mask], ddof=1)
    # if std_subt >= std_orig/10:
    #     const = np.median(y)
    #     pattern_idx[:] = const
    #     nans = np.array([np.nan]*n)
    #     popt = (nans, nans, const)

    return idx, popt, pattern_idx


def fit_fourier(data, freqs, mask=None, filt=None,
                apply_crrej_mask=False, apply_sigclip_mask=True,
                fitting_y_sections=None,
                subtract_x_sections=["[520:900]"],
                npool=5):
    """Fit Fourier series along column."""
    if mask is None:
        _mask = np.zeros(data.shape).astype(bool)
    else:
        _mask = mask.copy()

    if apply_crrej_mask:
        if filt is None:
            raise ValueError("filt must be given if apply_crrej_mask is True.")
        mask_cr = cr_reject_nic(data, filt=filt, verbose=False).mask
        _mask = _mask | mask_cr

    if fitting_y_sections is None:
        try:
            fitting_y_sections = FOURIERSECTS[filt]
        except KeyError:
            fitting_y_sections = ["[10:245]", "[850:1010]"]
    elif isinstance(fitting_y_sections, str):
        fitting_y_sections = [fitting_y_sections]

    if isinstance(subtract_x_sections, str):
        subtract_x_sections = [subtract_x_sections]

    noslice = slice(None, None, None)
    subsls = [(noslice, fitsxy2py(s)[0]) for s in subtract_x_sections]
    fitsls = [(fitsxy2py(s)[0], noslice) for s in fitting_y_sections]

    fitmask = np.ones(data.shape).astype(bool)  # initialize with masking True
    submask = np.ones(data.shape).astype(bool)  # initialize with masking True
    for fitsl in fitsls:
        fitmask[fitsl] = False

    for subsl in subsls:
        submask[subsl] = False

    _mask = _mask | fitmask | submask

    if apply_sigclip_mask:
        _data = np.ma.array(data, mask=mask)
        mask_sc = sigma_clip(_data, axis=0, sigma=3, maxiters=5).mask
        _mask = _mask | mask_sc

    ny, nx = data.shape
    yy, xx = np.mgrid[:ny, :nx]

    pool = Pool(npool)
    args = [
        list(yy.T),            # x to eval (= y_index of data, 0~1024)
        list(data.T),          # y for fit (pixel value)
        list(_mask.T),         # mask for fit
        [freqs]*nx,
        np.arange(nx)          # x_index of data
    ]
    res = np.array(pool.starmap(_fitter, np.array(args).T))
    pool.close()
    res = res[np.argsort(res[:, 0])]  # sort by index
    popts = np.array(res[:, 1])
    pattern = np.stack(res[:, 2], axis=1)

    return pattern, popts, _mask

    # for fitting_section in fitting_sections:
    #     try:
    #         fitsl = fitsxy2py(fitting_section)
    #     except IndexError:  # if only 1-D FITS index is given
    #         fitsl = [slice(None, None, None), slice(None, None, None)]
    #         fitsl[axis] = fitsxy2py(fitting_section)
    #         fitsl = tuple(fitsl)

    #     fitmask[fitsl] = False
    # # mask all pixels (1) outside the fitting_sections OR (2) mask from
    # # the input.
    # fitmask = mask | fitmask

    # pattern = np.zeros(data.shape)

    # pool = Pool(npool)
    # if axis == 1:
    #     d2fit = d2fit.T
    #     m2fit = m2fit.T
    # args = [
    #     [idx2fit] * n_fit,  # the ``x``-value for fit (n_fit, 512)
    #     list(d2fit.T),      # the ``y``-value for fit (475, nx)
    #     list(m2fit.T),                      # the mask for fit (475, nx)
    #     [idx2sub] * n_fit,  # the ``x`` to eval. func (0~1024)
    #     [i for i in idx2sub_iter]           # index along ``axis``
    # ]
    # res = np.array(pool.starmap(_fitter, np.array(args).T))
    # res = [np.argsort(res[:, 0])]  # sort by index
    # popts = np.array(res[:, 1])
    # pattern[subsl] += np.stack(res[:, 2], axis=axis - 1)

    # d2fit = data.copy()
    # d2fit[]

    # fitsls = []
    # idx2fit = []
    # d2fit = []
    # m2fit = []
    # for fitting_section in fitting_sections:
    #     try:
    #         fitsl = fitsxy2py(fitting_section)[axis - 1]
    #     except IndexError:  # if only 1-D FITS index is given
    #         fitsl = fitsxy2py(fitting_section)[0]

    #     fitsls.append(fitsl)
    #     idx2fit.append(np.arange(*fitsl[axis].indices(data.shape[axis])))
    #     d2fit.append(data[fitsl])
    #     m2fit.append(mask[fitsl])

    # idx2fit = np.concatenate(idx2fit)
    # d2fit = np.concatenate(d2fit, axis=axis)
    # m2fit = np.concatenate(m2fit, axis=axis)

    # idx2sub = np.arange(*subsl[axis].indices(data.shape[axis]))
    # idx2sub_iter = np.arange(*subsl[axis - 1].indices(data.shape[axis - 1]))
    # n_fit = idx2sub_iter.size

    # pattern = np.zeros(data.shape)

    # pool = Pool(npool)
    # if axis == 1:
    #     d2fit = d2fit.T
    #     m2fit = m2fit.T
    # args = [
    #     [idx2fit] * n_fit,  # the ``x``-value for fit (n_fit, 512)
    #     list(d2fit.T),      # the ``y``-value for fit (475, nx)
    #     list(m2fit.T),                      # the mask for fit (475, nx)
    #     [idx2sub] * n_fit,  # the ``x`` to eval. func (0~1024)
    #     [i for i in idx2sub_iter]           # index along ``axis``
    # ]
    # res = np.array(pool.starmap(_fitter, np.array(args).T))
    # res = [np.argsort(res[:, 0])]  # sort by index
    # popts = np.array(res[:, 1])
    # pattern[subsl] += np.stack(res[:, 2], axis=axis - 1)

    # return pattern, popts, mask


'''
def fouriersub(ccd, mask=None, dtype='float32',
               peak_infer_section="[900:930, :]",
               max_peaks=3, min_freq=0.01,
               subtract_section="[513:,]",
               fitting_sections=["[:, :250]", "[:, 800:]"],
               sigclip_kw={'sigma_lower': np.inf, 'sigma_upper': 3},
               full=True, **kwargs):
    pattern = []
    popt = []
    freq = []
    idxmap, _ = np.mgrid[:ccd.data.shape[0], :ccd.data.shape[1]]
    if mask is not None:
        try:
            ccd.mask = ccd.mask | mask
        except TypeError:
            ccd.mask = mask

    # Note l_or_u should also be changed if the order in this zip is changed.
    for s1, s2, s3 in zip([NICSECTS['lower'], NICSECTS['upper']],
                          [fitting_section_lower, fitting_section_upper],
                          [subtract_section_lower, subtract_section_upper]):
        half_ccd = trim_image(ccd, fits_section=s1)  # upper or lower
        half_ccd_infer = trim_image(half_ccd, fits_section=peak_infer_section)
        half_ccd_infer_cr = cr_reject_nic(half_ccd_infer, verbose=False,
                                          update_header=False)
        freq2fit = find_fourier_peaks(half_ccd_infer_cr.data,
                                      max_peaks=max_peaks,
                                      min_freq=min_freq,
                                      sigclip_kw=sigclip_kw)
        _pattern, _popt = extrapolate_fourier(ccd.data,
                                              freqs=freq2fit,
                                              fitting_section=s2,
                                              subtract_section=s3,
                                              mask=ccd.mask, axis=0, **kwargs)
        pattern.append(_pattern[fitsxy2py(s1)])
        # _pattern_data = np.zeros_like(ccd.data)
        # _pattern_data[fitsxy2py(s3)] = _pattern
        # pattern.append(_pattern_data[fitsxy2py(s1)])
        popt.append(_popt)
        freq.append(freq2fit)

    pattern = np.vstack(pattern)
    data_fc = ccd.data - pattern
    try:
        ccd_fc = CCDData(data=data_fc)
    except ValueError:
        ccd_fc = CCDData(data=data_fc, unit='adu')

    hdr = ccd.header.copy()
    hdr.add_history(
        f"""
Strongest Fourier series found from section {peak_infer_section}. Total
({len(freq[0])}, {len(freq[1])}) frequencies from the lower and upper
quadrant are selected, respectively (see FIT-FXXX keys). Then each
column is fitted with <const + multiple sine> functions, and
extrapolated from ({fitting_section_lower}, {fitting_section_upper})
for lower and upper quadrants, respectively. The pattern is estimated
only for the sections of {subtract_section_lower} and
{subtract_section_upper}.
        """.replace("\n", " ")
    )
    hdr["FIT-AXIS"] = ("COL",
                       "The direction to which Fourier series is fitted.")
    for k, freqlist in enumerate(freq):
        L_or_U = "L" if k == 0 else "U"
        l_or_u = "lower" if k == 0 else "upper"
        hdr[f"FIT-NF{L_or_U}"] = (
            len(freqlist),
            f"No. of freq. used for fit in {l_or_u} quadrant."
        )
        for i, f in enumerate(freqlist):
            hdr[f"FIT-F{L_or_U}{i+1:02d}"] = (
                f,
                f"[1/pix] {i+1:02d}-th frequency for {l_or_u} quadrant"
            )

    ccd_fc.data = ccd_fc.data.astype(dtype)
    ccd_fc.header = hdr

    if full:
        return ccd_fc, pattern, np.vstack(popt), freq
    return ccd_fc
'''

"""
def fouriersub(ccd, mask=None, dtype='float32',
            peak_infer_section="[900:, :]", max_peaks=3, min_freq=0.01,
            subtract_section_upper="[:, 513:]",
            subtract_section_lower="[:, :512]",
            fitting_section_lower="[:, :250]",
            fitting_section_upper="[:, 800:]",
            sigclip_kw={'sigma_lower': np.inf, 'sigma_upper': 3},
            full=True, **kwargs):
'''

for fouriersub
Find Fourier component using columns, subtract by fitting.
Parameters
----------
ccd: CCDData
    The CCD to be used. It's recommended to include mask, e.g.,
    by cosmic ray detection.
mask: ndarray, optional
    The mask you want add in addition to the one from cosmic-ray
    removal.
peak_infer_section : str, optional
    The section in FITS format to be used for inferring the
    dominent fourier series. See Note.
max_peaks : int, optional
    The maximum number of peaks to be found. See Note.
subtract_section_upper, subtract_section_lower: str, optional
    The section in FITS format for the upper and lower quadrants to
    be subtracted. Currently, the x-axis one must be full range
    (``:``).
fitting_section_lower, fitting_section_upper: str, optional
    The section in FITS format to be used for fitting the
    dominent fourier series. See Note.
full: bool, optional.
    If ``False``, only the resulting fourier pattern subtracted CCD
    will be returned. Otherwise (default), it will return four
    things: pattern removed CCD, pattern in ndarray (same shape as
    ``ccd.data``), ``popt`` which is the fitted parameters (can be
    fed directly into ``multisin(x, freqs, *popt)``), and the
    frequencies used in the upper/lower quadrants.
kwargs:
    The keyword arguments for ``scipy.optimize.curve_fit``
    during fitting the sinusoidal functions to
    ``fitting_section_lower`` and ``fitting_section_upper``.
Note
----
From the section specified by ``peak_infer_section``, first run FFT
along columns. Then do a robust linear fitting to the index VS
FFT absolute amplitude (index differ from the FFT frequencies by
upto addition and/or multiplication by constant(s)). Subtracting
this linear feature and find frequencies where the residuals
(amplitude - linear fit) meet both criteria: (1) above
sigma-clip upper bound and (2) top ``max_peaks`` (default 3).
This way, only upto ``max_peaks`` peak frequencies will be found.
Then the sum of ``max_peaks`` sine functions, which have
amplitude and phase as free parameters, with fixed frequencies,
will be fitted (at most ``2*max_peaks`` free parameters) based on
the ``fitting_sections``. Note that I didn't intend to include
outlier removal in the sinusoidal fitting process. If you wanted
to reject some pixels, they should have been masked a priori
(``ccd.mask``) by clear reasons. I believe sigma-clipping in
_fitting_ is mathematically baseless and must be avoided in any
science.
'''
pattern = []
popt = []
freq = []
idxmap, _ = np.mgrid[:ccd.data.shape[0], :ccd.data.shape[1]]
if mask is not None:
    try:
        ccd.mask = ccd.mask | mask
    except TypeError:
        ccd.mask = mask

# Note l_or_u should also be changed if the order in this zip is changed.
for s1, s2, s3 in zip([NICSECTS['lower'], NICSECTS['upper']],
                        [fitting_section_lower, fitting_section_upper],
                        [subtract_section_lower, subtract_section_upper]):
    half_ccd = trim_image(ccd, fits_section=s1)  # upper or lower
    half_ccd_infer = trim_image(half_ccd, fits_section=peak_infer_section)
    half_ccd_infer_cr = cr_reject_nic(half_ccd_infer, verbose=False,
                                        update_header=False)
    freq2fit = find_fourier_peaks(half_ccd_infer_cr.data,
                                    max_peaks=max_peaks,
                                    min_freq=min_freq,
                                    sigclip_kw=sigclip_kw)
    _pattern, _popt = extrapolate_fourier(ccd.data,
                                            freqs=freq2fit,
                                            fitting_section=s2,
                                            subtract_section=s3,
                                            mask=ccd.mask, axis=0, **kwargs)
    pattern.append(_pattern[fitsxy2py(s1)])
    # _pattern_data = np.zeros_like(ccd.data)
    # _pattern_data[fitsxy2py(s3)] = _pattern
    # pattern.append(_pattern_data[fitsxy2py(s1)])
    popt.append(_popt)
    freq.append(freq2fit)

pattern = np.vstack(pattern)
data_fc = ccd.data - pattern
try:
    ccd_fc = CCDData(data=data_fc)
except ValueError:
    ccd_fc = CCDData(data=data_fc, unit='adu')

hdr = ccd.header.copy()
hdr.add_history(
    f'''
Strongest Fourier series found from section {peak_infer_section}. Total
({len(freq[0])}, {len(freq[1])}) frequencies from the lower and upper
quadrant are selected, respectively (see FIT-FXXX keys). Then each
column is fitted with <const + multiple sine> functions, and
extrapolated from ({fitting_section_lower}, {fitting_section_upper})
for lower and upper quadrants, respectively. The pattern is estimated
only for the sections of {subtract_section_lower} and
{subtract_section_upper}.
    '''.replace("\n", " ")
)
hdr["FIT-AXIS"] = ("COL",
                    "The direction to which Fourier series is fitted.")
for k, freqlist in enumerate(freq):
    L_or_U = "L" if k == 0 else "U"
    l_or_u = "lower" if k == 0 else "upper"
    hdr[f"FIT-NF{L_or_U}"] = (
        len(freqlist),
        f"No. of freq. used for fit in {l_or_u} quadrant."
    )
    for i, f in enumerate(freqlist):
        hdr[f"FIT-F{L_or_U}{i+1:02d}"] = (
            f,
            f"[1/pix] {i+1:02d}-th frequency for {l_or_u} quadrant"
        )

ccd_fc.data = ccd_fc.data.astype(dtype)
ccd_fc.header = hdr

if full:
    return ccd_fc, pattern, np.vstack(popt), freq
return ccd_fc
"""

'''
def extrapolate_fourier(data, fitting_section,
                        subtract_section, freqs, axis=0, mask=None, **kwargs):
    """Fit Fourier series along column (axis 0) or row (axis 1)."""
    fitsl = fitsxy2py(fitting_section)
    subsl = fitsxy2py(subtract_section)
    d2fit = data[fitsl]
    if mask is None:
        mask = np.zeros_like(data).astype(bool)
    m2fit = mask[fitsl]
    # patternshape = data[subsl].shape

    # idxall = np.arange[data.shape[axis] - 0.1]
    idx2fit = np.arange(*fitsl[axis].indices(data.shape[axis]))
    idx2sub = np.arange(*subsl[axis].indices(data.shape[axis]))
    idx2sub_iter = np.arange(*subsl[axis - 1].indices(data.shape[axis - 1]))
    print(idx2fit)
    print(idx2sub)
    print(idx2sub_iter)
    # sigclip = sigma_clip(d2fit, sigma=3, maxiters=5, axis=axis)

    popts = []
    pattern = np.zeros(data.shape)
    for i in idx2sub_iter:  # range(patternshape[axis - 1]):
        if axis == 0:
            d_i = d2fit[:, i]
            m_i = m2fit[:, i]  # | sigclip.mask[:, i]
            popt, _ = fit_sinusoids(idx2fit[~m_i], d_i[~m_i],
                                    freqs=freqs, **kwargs)
            pattern[subsl[0], i] = multisin(idx2sub, freqs, *popt)
        else:
            d_i = d2fit[i, :]
            m_i = m2fit[i, :]  # | sigclip.mask[i, :]

            popt, _ = fit_sinusoids(idx2fit[~m_i], d_i[~m_i],
                                    freqs=freqs, **kwargs)
            pattern[i, :] = multisin(idx2sub, freqs, *popt)
        popts.append(popt)

    return pattern, popts
'''
