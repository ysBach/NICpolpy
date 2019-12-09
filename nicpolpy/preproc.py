from pathlib import Path

import astroscrappy
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from astroscrappy import detect_cosmics
from ccdproc import trim_image

from .util import (GAIN, LACOSMIC_KEYS, NICSECTS, RDNOISE, fft_peak_freq,
                   fit_sinusoids, fitsxy2py, infer_filter, multisin)

__all__ = ["cr_reject_nic", "vertical_correct", "lrsubtract",
           "find_fourier_peaks", "extrapolate_fourier", "NICCCD"]


def cr_reject_nic(ccd, filt=None, update_header=True, crrej_kw=None,
                  verbose=True):
    """
    Parameters
    ----------
    ccd: CCDData
        The ccd to be processed.
    filt: str, optional.
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
    # asdfasfa asdfsdsafasdf dfa sf sad fsad fsa df awe fa f sev aweffv awe
    # fvawfv awe fvawef avwev fA
    if not isinstance(ccd, CCDData):
        ccd = CCDData(ccd, unit='adu')
        hdr = None
    else:
        hdr = ccd.header

    filt = infer_filter(ccd, filt=filt, verbose=verbose)
    if crrej_kw is None:
        crrej_kw = LACOSMIC_KEYS
        crrej_kw["sepmed"] = True
        crrej_kw.update({"gain": GAIN[filt.upper()],
                         "readnoise": RDNOISE[filt.upper()]})

    str_cr = ("Cosmic-Ray rejected by astroscrappy (v {}), "
              + "with parameters: {}")

    nccd = ccd.copy()
    m, d = detect_cosmics(nccd.data, **crrej_kw, verbose=verbose)

    # astroscrappy automatically does the gain correction, so return
    # back to avoid confusion.
    nccd.data = d / crrej_kw["gain"]
    nccd.mask = m

    if update_header and hdr is not None:
        try:
            nccd.header.add_history(str_cr.format(astroscrappy.__version__,
                                                  crrej_kw))
        except AttributeError:
            nccd.header["HISTORY"] = str_cr.format(astroscrappy.__version__,
                                                   crrej_kw)

    return nccd


def vertical_correct(ccd, fitting_sections=["[:, 50:100]", "[:, 924:974]"],
                     method='median', sigma=3, maxiters=5, sigclip_kw={},
                     dtype='float32', return_pattern=False,
                     update_header=True):
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
    if isinstance(ccd, (CCDData, fits.PrimaryHDU, fits.ImageHDU)):
        data = ccd.data.copy()
        hdr = ccd.header.copy()
    elif isinstance(ccd, fits.HDUList):
        data = ccd[0].data.copy()
        hdr = ccd[0].header.copy()
    else:
        data = ccd.copy()
        hdr = None

    if len(fitting_sections) != 2:
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

    if maxiters == 0:
        clipstr = "no clipping"
        for part in parts:
            strips.append(fun(part, axis=0))

    else:
        clipstr = f"{sigma}-sigma {maxiters}-iterations clipping"
        if sigclip_kw:
            clipstr += f" with {sigclip_kw}"
        for part in parts:
            clip = sigma_clipped_stats(part, sigma=sigma,
                                       maxiters=maxiters,
                                       axis=0,
                                       **sigclip_kw)
            strips.append(clip[idx])

    ny, nx = data.shape
    vpattern = np.repeat(strips, ny/2, axis=0)
    vsub = data - vpattern

    if update_header and hdr is not None:
        hdr.add_history(f"Vertical pattern subtracted using {fitting_sections}"
                        + f" by {methodstr} with {clipstr}")

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
               dtype='float32', update_header=True):
    """Subtract left from right quadrants w/ vertical pattern removal."""
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
        nccd.header.add_history(f"Subtracted left half ({i_half} columns)"
                                + " from the right half.")
    # nccd = trim_image(nccd, fits_section=f"[{i_half + 1}:, :]")
    return nccd


def find_fourier_peaks(data, axis=0, max_peak=5,
                       sigclip_kw={'sigma_lower': np.inf, 'sigma_upper': 3}):
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Unly 2-D data is supported.")

    # FFT along axis
    npix = data.shape[axis]
    amp = np.abs(np.fft.fft(data, axis=axis)) / npix
    # ^ axis 0 will be frequency index
    amp_median = np.median(amp, axis=axis - 1)

    # Select high amplitude frequencies
    freq2fit = fft_peak_freq(amp_median,
                             max_peak=max_peak,
                             sigclip_kw=sigclip_kw)
    return freq2fit


def extrapolate_fourier(data, fitting_section,
                        subtract_section, freqs, axis=0, mask=None, **kwargs):
    """Fit Fourier series along column (axis 0) or row (axis 1)."""
    fitsl = fitsxy2py(fitting_section)
    subsl = fitsxy2py(subtract_section)
    d2fit = data[fitsl](subtract_section)
    d2fit = data[fitsl]
    m2fit = mask[fitsl]
    # patternshape = data[subsl].shape

    # idxall = np.arange[data.shape[axis] - 0.1]
    idx2fit = np.arange(*fitsl[axis].indices(data.shape[axis]))
    idx2sub = np.arange(*subsl[axis].indices(data.shape[axis]))
    idx2sub_iter = np.arange(*subsl[axis - 1].indices(data.shape[axis - 1]))

    popts = []
    pattern = np.zeros(data.shape)
    for i in idx2sub_iter:  # range(patternshape[axis - 1]):
        if axis == 0:
            d_i = d2fit[:, i]
            m_i = m2fit[:, i]
            popt, _ = fit_sinusoids(idx2fit[~m_i], d_i[~m_i],
                                    freqs=freqs, **kwargs)
            pattern[subsl[0], i] = multisin(idx2sub, freqs, *popt)
        else:
            d_i = d2fit[i, subsl[1]]
            m_i = m2fit[i, subsl[1]]
            popt, _ = fit_sinusoids(idx2fit[~m_i], d_i[~m_i],
                                    freqs=freqs, **kwargs)
            pattern[i, :] = multisin(idx2sub, freqs, *popt)
        popts.append(popt)

    return pattern, popts


def fouriersub(ccd, mask=None, dtype='float32',
               peak_infer_section="[900:, :]", max_peak=5,
               subtract_section_upper="[:, 513:]",
               subtract_section_lower="[:, :512]",
               fitting_section_lower="[:, :250]",
               fitting_section_upper="[:, 800:]",
               sigclip_kw={'sigma_lower': np.inf, 'sigma_upper': 3},
               full=True, **kwargs):
    """ Find Fourier component using columns, subtract by fitting.
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
    max_peak : int, optional
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
    sigma-clip upper bound and (2) top ``max_peak`` (default 5).
    This way, only upto ``max_peak`` peak frequencies will be found.
    Then the sum of ``max_peak`` sine functions, which have
    amplitude and phase as free parameters, with fixed frequencies,
    will be fitted (at most ``2*max_peak`` free parameters) based on
    the ``fitting_sections``. Note that I didn't intend to include
    outlier removal in the sinusoidal fitting process. If you wanted
    to reject some pixels, they should have been masked a priori
    (``ccd.mask``) by clear reasons. I believe sigma-clipping in
    _fitting_ is mathematically baseless and must be avoided in any
    science.
    """
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
        half_ccd = trim_image(ccd, fits_section=s1)
        half_ccd_infer = trim_image(half_ccd, fits_section=peak_infer_section)
        freq2fit = find_fourier_peaks(half_ccd_infer.data,
                                      max_peak=max_peak,
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


class NICCCD:
    def __init__(self, fpath, filter_name=None):
        self.fpath = fpath
        self.ccd = CCDData.read(fpath)
        # for future calculation, set float32 dtype:
        self.ccd.data = self.ccd.data.astype('float32')
        self.filt = infer_filter(self.ccd, filt=filter_name)
        self.crrej = False

    # def cr_reject(self, update_header=True, crrej_kw=None,
    #               verbose=True, output=None, overwrite=False):
    #     self.ccd_cr = cr_reject_nic(self.ccd, filt=self.filt,
    #                                 update_header=update_header,
    #                                 crrej_kw=crrej_kw,
    #                                 verbose=verbose)
    #     self.crrej = True
    #     if output is not None:
    #         self.ccd_cr.write(Path(output), overwrite=overwrite)

    def correct_fourier(self, mask=None, crrej_kw=None,
                        crrej_verbose=True, output=None, overwrite=False,
                        peak_infer_section="[900:, :]", max_peak=5,
                        subtract_section_upper="[513:, 513:]",
                        subtract_section_lower="[513:, :512]",
                        fitting_section_lower="[:, :250]",
                        fitting_section_upper="[:, 800:]",
                        sigclip_kw={'sigma_lower': np.inf, 'sigma_upper': 3},
                        **kwargs):
        # This crrej is merely to make mask to pixels for Fourier fitting.
        self.ccd_cr_raw = cr_reject_nic(self.ccd, filt=self.filt,
                                        update_header=False,
                                        crrej_kw=crrej_kw,
                                        verbose=crrej_verbose)
        try:
            self.ccd.mask = self.ccd.mask | self.ccd_cr_raw.mask
        except TypeError:
            self.ccd.mask = self.ccd_cr_raw.mask

        ccd = self.ccd

        self.additional_mask = mask
        self.peak_infer_section = peak_infer_section
        self.fitting_section_lower = fitting_section_lower
        self.fitting_section_upper = fitting_section_upper
        self.fourier_peak_sigclip_kw = sigclip_kw
        self.fourier_extrapolation_kw = kwargs
        res = fouriersub(ccd=ccd, mask=self.additional_mask,
                         peak_infer_section=self.peak_infer_section,
                         max_peak=max_peak,
                         subtract_section_upper=subtract_section_upper,
                         subtract_section_lower=subtract_section_lower,
                         fitting_section_lower=self.fitting_section_lower,
                         fitting_section_upper=self.fitting_section_upper,
                         sigclip_kw=self.fourier_peak_sigclip_kw,
                         full=True, **self.fourier_extrapolation_kw)
        self.ccd_fc = res[0]
        self.pattern = res[1]
        self.popt = res[2]
        self.freq = res[3]

        if output is not None:
            self.ccd_fc.write(Path(output), overwrite=overwrite)
