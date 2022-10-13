"""
Tools used to "prepare" the NIC images, even prior to the standard
preprocessing. This includes techniques such as:
(1) vertical pattern subtraction
(2) Fourier pattern subtraction
and minor modifications to the header (including DATE-OBS, COUNTER, etc).
"""

import time
from pathlib import Path

import astropy
import numpy as np
import pandas as pd
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from .ysfitsutilpy4nicpolpy import (CCDData_astype, _parse_data_header, cmt2hdr, crrej,
                                    slicefy, fixpix, load_ccd, medfilt_bpm, imslice,
                                    update_process)

from .util import (BPM_KW, GAIN, HDR_KEYS, NIC_CRREJ_KEYS, OBJSLICES, RDNOISE,
                   VERTICALSECTS, _load_as_dict, _sanitize_fits, _save,
                   _save_or_load_summary, _select_summary_rows, _set_dir_iol,
                   _summary_path_parse, add_maxsat, infer_filter, iterator, _set_fstem_proc)

__all__ = [
    "fixpix_leftonly_verti_strip", "fixpix_oe",
    "cr_reject_nic", "vertical_correct", "lrsubtract",
    "fourier_lrsub", "proc_16_vertical", "proc_16_vfv",
    "make_darkmask"
]


def fixpix_leftonly_verti_strip(ccd, mask=None, filt=None, verbose=0):
    """ Run fixpix on vertical strip (oe regeions only) of the given CCD

    Parameters
    ----------
    ccd : CCDData
        The input CCDData object.

    mask : CCDData, ndarray, optional
        The mask frame.
        Default: `None`.

    filt : str, optional
        The filter name.
        Default: `None`.

    verbose : bool, optional
        Verbose?
        Default: `True`.

    Notes
    -----
    Assumed that the given CCD is the full frame, not right_half.
    """
    filt = infer_filter(ccd, filt=filt, verbose=verbose >= 1)
    sl_o = (slice(None, None, None), OBJSLICES(True)[filt][0][1])
    sl_e = (slice(None, None, None), OBJSLICES(True)[filt][1][1])
    # ^ Do as if we're processing the right half of the image.
    ccd_l_o = ccd.data[sl_o]
    ccd_l_e = ccd.data[sl_e]
    if mask is None:
        msk_l_o, msk_l_e = None, None
    else:  # Convert to CCDData first, cuz mask can be both ndarray or CCDData.
        msk_l_o = CCDData(mask).data[sl_o]
        msk_l_e = CCDData(mask).data[sl_e]

    ccd_l_o = fixpix(ccd_l_o, msk_l_o, priority=(1, 0), update_header=False)
    ccd_l_e = fixpix(ccd_l_e, msk_l_e, priority=(1, 0), update_header=False)

    nccd = ccd.copy()
    nccd.data[sl_o] = ccd_l_o.data
    nccd.data[sl_e] = ccd_l_e.data

    return nccd


# FIXME: DEPRECATED
def fixpix_oe(ccd, mask, maskpath=None, filt=None, verbose=0):
    _ccd = ccd.copy()
    sls = OBJSLICES()[infer_filter(ccd, filt=filt, verbose=0)]
    for sl_oe in sls:
        _ccd.data[sl_oe] = fixpix(
            ccd.data[sl_oe],
            mask=mask.data[sl_oe],
            maskpath=maskpath,
            priority=(1, 0),
            verbose=verbose >= 1
        ).data
    return _ccd


def lrsubtract(
        ccd,
        fitting_sections=["[:, 50:100]", "[:, 924:974]"],
        method="median",
        sigclip_kw=dict(sigma=2, maxiters=5),
        dtype="float32",
        update_header=True,
        verbose=False
):
    """Subtract left from right quadrants w/ vertical pattern removal."""
    _t = Time.now()

    nccd = vertical_correct(ccd=ccd,
                            fitting_sections=fitting_sections,
                            method=method,
                            sigclip_kw=sigclip_kw,
                            dtype=dtype,
                            return_pattern=False,
                            update_header=update_header)
    nx = nccd.data.shape[1]
    i_half = nx//2
    nccd.data[:, i_half:] -= nccd.data[:, :i_half]
    if update_header:
        # add as history
        cmt2hdr(nccd.header, "h", t_ref=_t, verbose=verbose,
                s=f"Subtracted left half ({i_half} columns) from the right half.")
    # nccd = trim_image(nccd, trimsec=f"[{i_half + 1}:, :]")
    return nccd


def fourier_lrsub(
        ccd,
        cut_wavelength=100,
        verbose_bpm=False,
        verbose=False,
        do_mbpm=True,
        bpm_kw=BPM_KW,
):
    """ Subtract the Fourier pattern
    Parameters
    ----------
    ccd : CCDData
        The ccd to be processed.

    cut_wavelength : int, optional.
        The minimum wavelength of the Fourier transform to regard as noise. If
        ``200``, noise pattern of wavelength >= 200 will be removed. Will be
        added to ``FFTCUTWL`` in the header.

    copy : bool, optional.
        Whether to copy the original ccd.

    verbose_bpm : bool, optional.
        Verbose option for the median bad-pixel mask algorithm
        (`~ysfitsutilpy.medfilt_bpm`).

    bpm_kw : dict, optional.
        The keyword arguments for the median bad-pixel mask algorithm
        (`~ysfitsutilpy.medfilt_bpm`).

    Returns
    -------
    _ccd : CCDData
        The processed ccd.

    Note
    ----
    First, **smooth** the left half of the NIC image using median filtered
    bad-pixel masking algrithm (`~ysfitsutilpy.medfilt_bpm`). Then perform
    real-valued FFT along the column in the left half of the NIC image. The
    amplitude of this FFT is set to 0 for wavelength >= `cut_wavelength`
    [pixel]. This is to remove high-frequency artificial noise. The pattern map
    is reconstructed by the inverse FFT using the remaining amplitudes only.
    Finally, this pattern is duplicated to the right half of the NIC image and
    subtracted from the original frame. The left half of the resulting frame is
    kept to visually inspect if the Fourier pattern is appropriately subtracted
    (e.g., hot/CR-hit pixels are properly removed in the bad-pixel masking so
    that it did not result in artificial sinusoidal noise in the `pattern`).

    """
    nccd = ccd.copy()

    if do_mbpm:
        ccd_l = imslice(nccd, trimsec="[:512, :]", update_header=False)
        ccd_l = medfilt_bpm(ccd_l, verbose=verbose_bpm, **bpm_kw)
        nccd.data[:, :ccd_l.shape[1]] = ccd_l.data
        nccd.header = ccd_l.header  # to add MBPM logs

    filt = infer_filter(ccd, verbose=verbose >= 1)
    # sl_l_o = (slice(None, None, None), OBJSLICES(True)[filt][0][1])
    # sl_l_e = (slice(None, None, None), OBJSLICES(True)[filt][1][1])
    # sl_r_o = (slice(None, None, None), OBJSLICES(False)[filt][0][1])
    # sl_r_e = (slice(None, None, None), OBJSLICES(False)[filt][1][1])

    _t = Time.now()
    for i in range(2):
        sl_l_oe = (slice(None, None, None), OBJSLICES(True)[filt][i][1])
        # ^ Do as if we're processing the right half of the image.
        sl_r_oe = (slice(None, None, None), OBJSLICES(False)[filt][i][1])
        # ^ The true right part of the frames.
        amp_comp = np.fft.rfft(nccd.data[sl_l_oe], axis=0)
        amp_comp[cut_wavelength:, :] = 0
        pattern_pure = np.fft.irfft(amp_comp, axis=0)
        nccd.data[sl_l_oe] -= pattern_pure
        nccd.data[sl_r_oe] -= pattern_pure

    _t = Time.now()
    amp_comp = np.fft.rfft(ccd_l.data, axis=0)
    amp_comp[cut_wavelength:, :] = 0
    # pattern_pure = np.fft.irfft(amp_comp, axis=0)
    # pattern = np.tile(pattern_pure, 2)
    # nccd.data = nccd.data - pattern
    nccd.header.set(
        "FFTCUTWL",
        cut_wavelength,
        "FFT cut wavelength (amplitude[this:, :] = 0)"
    )
    cmt2hdr(
        nccd.header, "h", verbose=verbose, t_ref=_t,
        s=("FFT(left half) to get pattern map (see FFTCUTWL for the cut wavelength); "
           + "subtracted from both left/right.")
    )

    return nccd


def sep_extract_nic():
    """
    """

    return


def cr_reject_nic(
        ccd,
        mask=None,
        filt=None,
        update_header=True,
        add_process=True,
        crrej_kw=None,
        verbose=True,
        full=False
):
    """Do LACosmic-like cosmic-ray rejection on NIC images.

    Parameters
    ----------
    ccd: CCDData
        The ccd to be processed.

    filt: str, None, optional.
        The filter name in one-character string (case insensitive). If
        `None`(default), it will be inferred from the header (key
        ``"FILTER"``).

    update_header: bool, optional.
        Whether to update the header if there is any.

    add_process : bool, optional.
        Whether to add ``PROCESS`` key to the header.

    crrej_kw: dict, optional.
        The keyword arguments for the ``astroscrappy.detect_cosmics``. If
        `None` (default), the parameters for IRAF-version of L.A. Cosmic,
        except for the ``sepmed = True`` and ``gain`` and ``readnoise`` are
        replaced to the values of NIC detectors.
        If ``gain`` or ``rdnoise`` are given, they will be used instead of the
        nic default.

    verbose: bool, optional
        The verbose paramter to ``astroscrappy.detect_cosmics``.

    Returns
    -------
    nccd: CCDData
        The cosmic-ray removed object.

    Note
    ----
    astroscrappy automatically correct gain (i.e., output frame in the unit of
    electrons, not ADU). In this function, this is undone, i.e., I divide it
    with the gain to restore ADU unit. Also ``nccd.mask`` will contain the mask
    from cosmic-ray detection.
    """
    crkw = NIC_CRREJ_KEYS.copy()

    if "gain" not in crkw or "rdnoise" not in crkw:
        filt = infer_filter(ccd, filt=filt, verbose=verbose)
        crkw.setdefault("gain", GAIN[filt])
        crkw.setdefault("rdnoise", RDNOISE[filt])

    if crrej_kw is not None:
        crkw.update(crrej_kw)

    nccd, crmask = crrej(ccd,
                         mask=mask,
                         **crkw,
                         propagate_crmask=False,
                         update_header=update_header,
                         add_process=add_process,
                         verbose=verbose)

    if full:
        return nccd, crmask, crrej_kw
    else:
        return nccd


def _multi2_int16(
        ccd,
        maxval=15000,
        minval=-15000,
        blankval=-32768,
        update_header=True
):
    """Multiply the image by 2 but keep the data within the range of ``int16``.

    Parameters
    ----------
    ccd: CCDData
        The ccd to be processed.

    maxval, minval: int, optional.
        The maximum/minimum value of the image to be kept, **BEFORE**
        multiplying by 2.

    blankval: int, optional.
        The value to be used for blank pixels (``BLANK`` keyword in the
        header).

    Returns
    -------
    _ccd: CCDData
        The processed ccd.
    """
    _t = Time.now()
    nccd = ccd.copy()
    nccd.data *= 2
    nccd.data[(nccd.data < 2*minval) | (nccd.data > 2*maxval)] = blankval
    # assign dtype of int16 for just in case.
    CCDData_astype(nccd, np.int16, copy=False)

    if update_header:
        nccd.header["BLANK"] = (blankval, "Blank value")
        cmt2hdr(nccd.header, "h", verbose=False, t_ref=_t,
                s=("Multiply by 2 and convert to (signed) int16; Pixels outside "
                   + f"({minval=}, {maxval=}) are replaced by `BLANK` BEFORE multi by 2."))

    return nccd


def vertical_correct(
        ccd,
        mask=None,
        maskpath=None,
        fitting_sections=None,
        method="median",
        sigclip_kw=dict(sigma=2, maxiters=5),
        dtype="float32",
        return_pattern=False,
        update_header=True,
        verbose=False
):
    """ Correct vertical strip patterns.

    Paramters
    ---------
    ccd : CCDData, HDU object, HDUList, or ndarray.
        The CCD to subtract the vertical pattern.

    fitting_sections : list of two str, optional.
        The sections to be used for the vertical pattern estimation. This must
        be identical to the usual FITS section (i.e., that used in SAO ds9 or
        IRAF, 1-indexing and last-index-inclusive), not in python. **Give it in
        the order of ``[<upper>, <lower>]`` in FITS y-coordinate.**

    method : str, optional.
        One of ``["med", "avg", "median", "average", "mean"]``.

    sigma, maxiters : float and int, optional
        A sigma-clipping will be done to remove hot pixels and cosmic rays for
        estimating the vertical pattern. To turn sigma-clipping off, set
        ``maxiters=0``.

    sigclip_kw : dict, optional
        The keyword arguments for the sigma clipping.

    dtype : str, dtype, optional
        The data type to be returned.

    return_pattern : bool, optional.
        If `True`, the subtracted pattern will also be returned.
        Default is `False`.

    update_header : bool, optional.
        Whether to update the header if there is any.

    Return
    ------

    """
    _t = Time.now()
    data, hdr = _parse_data_header(ccd)
    mask, _ = _parse_data_header(mask)

    if fitting_sections is None:
        fitting_sections = VERTICALSECTS
    elif len(fitting_sections) != 2:
        raise ValueError("fitting_sections must have two elements.")

    if method in ["median", "med"]:
        methodstr = "taking median"
        fun = np.nanmedian
        idx = 1
    elif method in ["average", "avg", "mean"]:
        methodstr = "taking average"
        fun = np.nanmean
        idx = 0
    else:
        raise ValueError("method not understood; "
                         + "it must be one of [med, avg, median, average, mean].")

    # The two horizontal box areas from raw data:
    parts = [data[slicefy(sect)] for sect in fitting_sections]
    if mask is not None:
        # Corresponding mask sections:
        mparts = [mask[slicefy(sect)] for sect in fitting_sections]
    else:
        mparts = [None]*len(fitting_sections)

    strips = []  # The estimated patterns (1-D)
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

    maskstr = "" if mask is None else " (using pixel mask) "
    maskstr += "" if maskpath is None else f"{maskpath=}"

    for part, mpart in zip(parts, mparts):
        clip = sigma_clipped_stats(part, mask=mpart, axis=0, **sigclip_kw)
        strips.append(clip[idx])

    ny, nx = data.shape
    vpattern = np.repeat(strips, ny/2, axis=0)
    vsub = data - vpattern.astype(dtype)

    if update_header and hdr is not None:
        # add as history
        cmt2hdr(
            hdr, "h", verbose=verbose, t_ref=_t,
            s=(f"Vertical pattern subtracted using {fitting_sections} by "
               + f"{methodstr} with {clipstr + maskstr}")
        )

    try:
        nccd = CCDData(data=vsub, header=hdr)
    except ValueError:
        nccd = CCDData(data=vsub, header=hdr, unit="adu")

    CCDData_astype(nccd, dtype, copy=False)
    # if dtype.startswith("i"):
    #     nccd.data = np.around(nccd.data).astype(dtype)
    # else:
    #     nccd.data = nccd.data.astype(dtype)

    if return_pattern:
        return nccd, vpattern
    return nccd


def _do_16bit_vertical(
        fpath,
        dir_out,
        mask=None,
        maskpath=None,
        npixs=(5, 5),
        bezels=((20, 20), (20, 20)),
        process_title="Basic preprocessing",
        skip_if_exists=True,
        method="median",
        sigclip_kw=dict(sigma=2, maxiters=5),
        fitting_sections=None,
        dtype="int16",
        multi2for_int16=False,
        maxval=15000,
        minval=-15000,
        blankval=-32768,
        setid=None,
        full=False,
        save=True,
        verbose=1
):
    """ Changes to 16-bit and correct the vertical pattern.

    Parameters
    ----------
    dir_out : path-like
        The directory for the resulting FITS files to be saved, RELATIVE
        to the cwd.

    skip_if_exists : bool, optional.
        Whether to skip all process if the file exists in `outdir`.
        Default: `True`.

    npixs : length-2 tuple of int, optional
        The numbers of extrema to find, in the form of ``[small, large]``, so
        that ``small`` number of smallest and ``large`` number of largest pixel
        values will be found. If `None`, no extrema is found (`None` is
        returned for that extremum).
        Deafult: ``(5, 5)``

    bezels : list of list of int, optional.
        If given, must be a list of list of int. Each list of int is in the
        form of ``[lower, upper]``, i.e., the first ``lower`` and last
        ``upper`` rows/columns are ignored.

    multi2for_int16 : bool, optional.
        Whether to multiply 2 to the data, when `dtype` is ``"int16"``. It is
        particularly useful when the vertical correct has value of half integer
        (e.g., 0.5, 1.5, 2.5, ...). Note that NIC has saturation well below
        10,000 ADU and extraordinary bad pixels have only ~ 10,000 ADU, so a
        multiplication by 2 does not result in loss of data. For just in case,
        use `maxval`, `minval`, and `blankval`.

    maxval, minval: int, optional.
        The maximum/minimum value of the image to be kept, **BEFORE**
        multiplying by 2. Used only if `multi2for_int16` is `True`.

    blankval: int, optional.
        The value to be used for blank pixels (``BLANK`` keyword in the
        header). Used only if `multi2for_int16` is `True`.

    setid : int, optional.
        The SETID value. Default: 0.

    verbose : int
        Larger number means it becomes more verbose::
            * 0: print nothing
            * 1: Only very essential things
            * 2: + verbose for summary CSV file
            * 3: + the HISTORY in each FITS file"s header
    """
    ccd_nbit, outstem, skipit = _sanitize_fits(
        fpath,
        dir_out,
        proc2add="v",
        setid=setid,
        process_title=process_title,
        skip_if_exists=skip_if_exists,
        verbose=verbose - 1,
        assert_almost_equal=True
    )
    if skipit:
        return (None, None) if full else None

    ccd_nbit = add_maxsat(
        ccd_nbit,
        mmaskpath=maskpath,
        npixs=npixs,
        bezels=bezels,
        verbose=verbose
    )

    # == vertical pattern subtraction ==================================================== #
    if dtype == "int16" and multi2for_int16:
        ccd_nbit_v = vertical_correct(
            ccd_nbit,
            mask=mask,
            maskpath=maskpath,
            sigclip_kw=sigclip_kw,
            fitting_sections=fitting_sections,
            method=method,
            dtype="float32",
            return_pattern=False
        )
        ccd_nbit_v = _multi2_int16(
            ccd_nbit_v,
            maxval=maxval,
            minval=minval,
            blankval=blankval
        )

    else:
        ccd_nbit_v = vertical_correct(
            ccd_nbit,
            mask=mask,
            maskpath=maskpath,
            sigclip_kw=sigclip_kw,
            fitting_sections=fitting_sections,
            method=method,
            dtype=dtype,
            return_pattern=False
        )

    ccd_nbit_v.header["SETID"] = (setid, "Pol mode set number of OBJECT on the night")
    ccd_nbit_v.header["LV0FRM"] = str(fpath)
    ccd_nbit_v.header["LVNOW"] = (1, "The current level; see LViFRM for history.")
    update_process(ccd_nbit_v.header, "v", additional_comment=dict(v="vertical pattern"))
    cmt2hdr(ccd_nbit_v.header, "h", "-"*72, time_fmt=None, verbose=verbose >= 2)

    if save:
        _save(ccd_nbit_v, dir_out, outstem)
    return (ccd_nbit_v, Path(dir_out)/f"{outstem}.fits") if full else None


def _do_vfv(
        fpath,
        dir_out,
        dir_out_v=None,
        mask=None,
        maskpath=None,
        process_title="Basic preprocessing",
        vertical_already=False,
        vc_kw=dict(method="median",
                   sigclip_kw=dict(sigma=2, maxiters=5),
                   fitting_sections=None),
        multi2for_int16=False,
        multi2_kw=dict(maxval=15000, minval=-15000, blankval=-32768),
        do_fourier=True,
        do_fixpix_before_fourier=True,
        do_mbpm_before_fourier=True,
        cut_wavelength=100,
        bpm_kw=BPM_KW,
        vertical_again=True,
        skip_if_exists=True,
        setid=None,
        verbose=True
):
    """ Subtracts vertical pattern, FFT on the left half (internally does
    FIXPIX), and vertical correction again.

    Parameters
    ----------
    fpath : path-like
        The path to the FITS file to be processed.

    dir_out : path-like
        The directory the FITS file is meant to be saved.

    dir_out_v : path-like, optional.
        The directory for the intermediate FITS files (vertical pattern
        corrected, **multiplied by 2** if `multi2for_int16` is `True`, in
        signed int16) to be saved, RELATIVE to the cwd. If `None` (default),
        nothing will be saved.

    vertical_already : bool, optional.
        Whether vertical pattern correction has already been done.
        Default: `False`.

    multi2for_int16 : bool, optional.
        Whether to multiply 2 to the data, when `dtype` is ``"int16"``. It is
        particularly useful when the vertical correct has value of half integer
        (e.g., 0.5, 1.5, 2.5, ...). Note that NIC has saturation well below
        10,000 ADU and extraordinary bad pixels have only ~ 10,000 ADU, so a
        multiplication by 2 does not result in loss of data. For just in case,
        use `maxval`, `minval`, and `blankval`.

    maxval, minval: int, optional.
        The maximum/minimum value of the image to be kept, **BEFORE**
        multiplying by 2. Used only if `multi2for_int16` is `True`.

    blankval: int, optional.
        The value to be used for blank pixels (``BLANK`` keyword in the
        header). Used only if `multi2for_int16` is `True`.

    cut_wavelength : int, optional.
        The minimum wavelength of the Fourier transform to regard as noise. If
        ``200``, only the noise patterns of wavelength >= 200 will be used for
        fourier pattern analysis. If it is too small, artificial high-frequency
        pattern will appear in the fourier pattern... Will be added to
        ``FFTCUTWL`` in the header.

    do_fourier : bool, optional.
        Whether to do Fourier transform on the left half of the image.

    do_fixpix_before_fourier : bool, optional.
        Whether to do FIXPIX before Fourier transform.

    do_mbpm_before_fourier : bool, optional.
        Whether to do MBPM before Fourier transform.

    bpm_kw : dict, optional.
        Importantly includes med_sub_clip, med_rat_clip, std_rat_clip (list of
        float). See `~ysfitsutilpy.medfilt_bpm`.
        Used only if `do_fourier` is `True`.

    vertical_again : bool, optional.
        Whether to do vertical pattern subtraction once again after the Fourier
        pattern removal. This is useful because the initial vertical pattern
        subtraction might have been affected by Fourier pattern of wavelength
        >~30 pixels.
        Default: `True`.

    fitting_sections : list of two str, optional.
        The sections to be used for the vertical pattern estimation. This must
        be identical to the usual FITS section (i.e., that used in SAO ds9 or
        IRAF, 1-indexing and last-index-inclusive), not in python. **Give it in
        the order of ``[<upper>, <lower>]`` in FITS y-coordinate.**
        For `vertical_again`.

    method : str, optional.
        One of ``["med", "avg", "median", "average", "mean"]``.

    sigclip_kw : dict, optional
        The keyword arguments for the sigma clipping for estimating the
        vertical pattern. To turn sigma-clipping off, set ``maxiters=0``.

    skip_if_exists : bool, optional.
        Whether to skip all process if the file exists in `outdir`.
        Default: `True`.

    setid : int, optional.
        The SETID value. Default: None.

    Note
    ----
    The algorithm is as follows:

      1. Subtract the vertical pattern.
      2. if do_fourier and do_fixpix_before_fourier: Fix the pixels on the left
        half of the frame based on `~ysfitsutilpy.fixpix`.
      3. if do_fourier and do_mbpm_before_fourier: Mask(fix) bad pixels on the
        left half of the frame based on `~ysfitsutilpy.medfilt_bpm` algorithm.
      4. if do_fourier: FFT on the left half of the image, subtract the pattern
        from the right half (using wavelengths larger than `cut_wavelength`).
      5. if vertical_again: Subtract the vertical pattern again.
    """
    proc2add = "v" if vertical_already else ""
    proc2add += ("fv" if vertical_again else "f") if do_fourier else ""

    ccd, outstem, skipit = _sanitize_fits(
        fpath,
        dir_out,
        proc2add=proc2add,
        setid=setid,
        process_title=process_title,
        skip_if_exists=skip_if_exists,
        verbose=verbose,
        assert_almost_equal=False  # no need to check
    )
    if skipit:
        return None

    if "BLANK" in ccd.header:
        del ccd.header["BLANK"]

    # == vertical pattern subtraction ==================================================== #
    if not vertical_already:
        ccd = vertical_correct(
            ccd,
            mask=mask,
            maskpath=maskpath,
            dtype="float32",  # hard-coded
            return_pattern=False,  # hard-coded
            update_header=True,  # hard-coded
            verbose=verbose,
            **vc_kw
        )

        update_process(ccd.header, "v", additional_comment=dict(v="vertical pattern"))

        if dir_out_v is not None:
            if multi2for_int16:
                _nccd2save = _multi2_int16(ccd, **multi2_kw)
            else:
                _nccd2save = CCDData_astype(ccd, "int16", copy=False)

            _os_v = outstem.replace("-PROC-vfv", "-PROC-v").replace("-PROC-vf", "-PROC-v")
            _save(_nccd2save, dir_out_v, _os_v)

    # == Fourier pattern removal ========================================================= #
    if do_fourier:
        if do_fixpix_before_fourier:
            # -- FIXPIX but with vertical direction (y-axis) is prioritized
            # Do only for the left part -- no need to do it on the right part.
            _t = Time.now()
            ccd = fixpix_leftonly_verti_strip(ccd, mask, filt=None)
            cmt2hdr(ccd.header, "h", verbose=verbose >= 1, t_ref=_t,
                    s="FIXPIX on the left part of the image")

        # -- Fourier pattern removal
        ccd = fourier_lrsub(
            ccd,
            cut_wavelength=cut_wavelength,
            verbose_bpm=verbose >= 1,
            verbose=verbose >= 1,
            do_mbpm=do_mbpm_before_fourier,
            bpm_kw=bpm_kw
        )
        CCDData_astype(ccd, "float32", copy=False)
        update_process(ccd.header, "f", add_comment=False,
                       additional_comment={"f": "fourier pattern"})

        if "BLANK" in ccd.header:
            del ccd.header["BLANK"]

        if vertical_again:
            ccd = vertical_correct(
                ccd,
                mask=mask,
                maskpath=maskpath,
                dtype="float32",  # hard-coded
                return_pattern=False,  # hard-coded
                update_header=True,  # hard-coded
                verbose=verbose >= 1,
                **vc_kw
            )
            update_process(ccd.header, "v", add_comment=False)

    ccd.header["LV1FRM"] = str(fpath)
    ccd.header["LVNOW"] = (2, "The current level; see LViFRM for history.")

    cmt2hdr(ccd.header, "h", "-"*72, time_fmt=None, verbose=verbose >= 2)
    _save(ccd, dir_out, outstem)

    return ccd


# def _do_fourier(
#         fpath,
#         dir_out,
#         cut_wavelength=200,
#         med_sub_clip=[-5, 5],
#         med_rat_clip=[0.5, 2],
#         std_rat_clip=[-5, 5],
#         vertical_again=True,
#         fitting_sections=None,
#         method="median",
#         sigclip_kw=dict(sigma=2, maxiters=5),
#         skip_if_exists=True,
#         verbose=True
# ):
#     """
#     Parameters
#     ----------
#     fpath : path-like
#         The path to the FITS file to be processed.

#     dir_out : path-like
#         The directory the FITS file is meant to be saved.

#     cut_wavelength : int, optional.
#         The minimum wavelength of the Fourier transform to regard as noise. If
#         ``200``, noise pattern of wavelength >= 200 will be removed. Will be
#         added to ``FFTCUTWL`` in the header.

#     med_sub_clip, med_rat_clip, std_rat_clip : list of float, optional.
#         See `~ysfitsutilpy.medfilt_bpm`.

#     vertical_again : bool, optional.
#         Whether to do vertical pattern subtraction once again after the Fourier
#         pattern removal. This is useful because the initial vertical pattern
#         subtraction might have been affected by Fourier pattern of wavelength
#         >~30 pixels.
#         Default: `True`.

#     fitting_sections : list of two str, optional.
#         The sections to be used for the vertical pattern estimation. This must
#         be identical to the usual FITS section (i.e., that used in SAO ds9 or
#         IRAF, 1-indexing and last-index-inclusive), not in python. **Give it in
#         the order of ``[<upper>, <lower>]`` in FITS y-coordinate.** For
#         `vertical_again`.

#     method : str, optional.
#         One of ``["med", "avg", "median", "average", "mean"]``.
#         For `vertical_again`.

#     sigclip_kw : dict, optional
#         The keyword arguments for the sigma clipping for estimating the
#         vertical pattern. To turn sigma-clipping off, set ``maxiters=0``.

#     skip_if_exists : bool, optional.
#         Whether to skip all process if the file exists in `outdir`.
#         Default: `True`.

#     """
#     fpath = Path(fpath)
#     outstem = fpath.stem + "f"  # hard-coded
#     # == Skip if conditions meet ========================================================= #
#     if skip_if_exists and (dir_out/f"{outstem}.fits").exists():
#         return

#     ccd_v = load_ccd(fpath)
#     ccd_vf = fourier_lrsub(
#         ccd_v,
#         cut_wavelength=cut_wavelength,
#         med_sub_clip=med_sub_clip,
#         med_rat_clip=med_rat_clip,
#         std_rat_clip=std_rat_clip,
#         verbose_bpm=verbose,
#         verbose=verbose
#     )
#     ccd_vf = CCDData_astype(ccd_vf, "float32")
#     update_process(ccd_vf.header, "f", add_comment=False,
#                    additional_comment={"f": "fourier pattern"})
#     if vertical_again:
#         # actually, ccd_vfv is made, but for simplicity, just call it ccd_vf.
#         ccd_vf = vertical_correct(
#             ccd_vf,
#             sigclip_kw=sigclip_kw,
#             fitting_sections=fitting_sections,
#             method=method,
#             dtype="float32",  # hard-coded
#             return_pattern=False
#         )
#         update_process(ccd_vf.header, "v")

#     _save(ccd_vf, dir_out, outstem)
#     return ccd_vf


# FIXME: Deprecated in favor of proc_16_vfv
def proc_16_vertical(
        dir_in,
        dir_out,
        dir_log=None,
        dir_mask=None,
        objects=None,
        objects_exclude=False,
        csv_in=None,
        csv_out=None,
        sigclip_kw=dict(sigma=2, maxiters=5),
        fitting_sections=None,
        method="median",
        skip_if_exists=True,
        rm_nonpol=True,
        rm_test=True,
        verbose=0,
        show_progress=True
):
    """Convert to 16-bit and save after the vertical pattern removal

    verbose : int
        Larger number means it becomes more verbose::
            * 0: print nothing
            * 1: Only very essential things
            * 2: + verbose for summary CSV file
            * 3: + the HISTORY in each FITS file"s header
    """
    dir_in, dir_out, dir_log = _set_dir_iol(dir_in, dir_out, dir_log)

    icpath = _summary_path_parse(dir_log, csv_in, f"summary_{dir_in.name}.csv")
    ocpath = _summary_path_parse(dir_log, csv_out, f"summary_{dir_out.name}.csv")
    if ocpath.exists() and skip_if_exists:
        print(f"Loading the existing summary CSV file from {ocpath}; \n\t"
              + "SKIPPING all 16-bit & vertical pattern subtraction.")
        return pd.read_csv(ocpath)

    # == Load and filter the summary CSV file ============================================ #
    # Use the long-version of the keywords for the RAW summary file.
    summ_raw = _save_or_load_summary(
        dir_in,
        summary_path=icpath,
        keywords=HDR_KEYS["all"],
        skip_if_exists=skip_if_exists,
        rm_nonpol=rm_nonpol,
        rm_test=rm_test,
        verbose=verbose
    )
    summ_raw = _select_summary_rows(
        summ_raw,
        include_dark=True,
        include_flat=True,
        objects=objects,
        objects_exclude=objects_exclude
    )
    # ------------------------------------------------------------------------------------ #

    if verbose > 0:
        print("32-bit -> 16-bit && Vertical pattern subtraction; saving to\n"
              + f"  * FITS: {dir_out} \n  * CSV : {ocpath}")
    time.sleep(0.5)  # For tqdm to show progress bar properly on Jupyter notebook

    _, masks, paths = _load_as_dict(dir_mask, ["FILTER"], verbose=verbose > 2)
    for _, row in iterator(summ_raw.iterrows(), show_progress=show_progress):
        filt = row["FILTER"]
        fpath = row["file"]
        setid = row["SETID"]
        mask = masks.get(filt.upper(), None)
        mpath = paths.get(filt.upper(), None)

        _ = _do_16bit_vertical(
            fpath,
            dir_out,
            mask=mask,
            maskpath=mpath,
            skip_if_exists=skip_if_exists,
            sigclip_kw=sigclip_kw,
            fitting_sections=fitting_sections,
            method=method,
            setid=setid,
            verbose=verbose >= 3
        )

    # Use the short-version of the keywords for the PROCESSED summary file.
    summ_raw_v = _save_or_load_summary(
        dir_out,
        ocpath,
        rm_nonpol=rm_nonpol,
        keywords=HDR_KEYS["simple"],
        skip_if_exists=skip_if_exists,
        verbose=verbose >= 2
    )
    if verbose > 0:
        print("DONE.")

    return summ_raw_v


def proc_16_vfv(
        dir_in,
        dir_out,
        dir_out_v=None,
        dir_log=None,
        dir_mask=None,
        objects=None,
        objects_exclude=False,
        csv_in=None,
        csv_out=None,
        do_vertical=True,
        vc_kw=dict(method="median",
                   sigclip_kw=dict(sigma=2, maxiters=5),
                   fitting_sections=None),
        multi2for_int16=False,
        multi2_kw=dict(maxval=15000, minval=-15000, blankval=-32768),
        do_fourier=True,
        cut_wavelength=100,
        vertical_again=True,
        bpm_kw=BPM_KW,
        skip_if_exists=True,
        rm_nonpol=True,
        rm_test=True,
        verbose=0,
        show_progress=True
):
    """
    FFT cannot accept ``NaN`` value. We must either ignore mask or use FIXPIX
    before FFT. Also we need to remove the vertical pattern before FFT
    (float32) - not int16, that's why it doesn't use the defualt output of
    do_16bit_vertical (which is in int16).

    skip_if_exists : bool, optional.
        Whether to skip all process if the file exists in `outdir`.
        Default: `True`.

    verbose : int
        Larger number means it becomes more verbose::
            * 0: print nothing
            * 1: Only very essential things
            * 2: + verbose for summary CSV file
            * 3: + the HISTORY in each FITS file's header
    """
    dir_in, dir_out, dir_log = _set_dir_iol(dir_in, dir_out, dir_log)
    icpath = _summary_path_parse(dir_log, csv_in, f"summary_{dir_in.name}.csv")
    ocpath = _summary_path_parse(dir_log, csv_out, f"summary_{dir_out.name}.csv")
    if ocpath.exists() and skip_if_exists:
        print(f"Loading the existing summary CSV file from {ocpath};"
              + "\n\tSKIPPING the v-f-v process.")
        return pd.read_csv(ocpath)

    # == Load and filter the summary CSV file ============================================ #
    # Use the long-version of the keywords for the RAW summary file.
    summ_raw = _save_or_load_summary(
        dir_in,
        summary_path=icpath,
        keywords=HDR_KEYS["all"],
        skip_if_exists=skip_if_exists,
        rm_nonpol=rm_nonpol,
        rm_test=rm_test,
        add_setid=True,
        verbose=verbose
    )
    summ_raw = _select_summary_rows(
        summ_raw,
        include_dark=True,
        include_flat=True,
        objects=objects,
        objects_exclude=objects_exclude
    )
    # ------------------------------------------------------------------------------------ #

    if verbose >= 1:
        print("vfv (vertical-FIXPIX-Fourier-vertical) process; saving to\n"
              + f"  * FITS: {dir_out} \n  * CSV : {ocpath}")
    time.sleep(0.5)  # For tqdm to show progress bar properly on Jupyter notebook

    _, masks, paths = _load_as_dict(dir_mask, ["FILTER"], verbose=verbose >= 2)

    for _, row in iterator(summ_raw.iterrows(), show_progress=show_progress):
        filt = row["FILTER"]
        fpath = row["file"]
        setid = row["SETID"]

        mask = masks.get(filt.upper(), None)
        mpath = paths.get(filt.upper(), None)
        _ = _do_vfv(
            fpath,
            dir_out,
            dir_out_v=dir_out_v,
            mask=mask,
            maskpath=mpath,
            do_vertical=do_vertical,
            vc_kw=vc_kw,
            multi2for_int16=multi2for_int16,
            multi2_kw=multi2_kw,
            do_fourier=do_fourier,
            cut_wavelength=cut_wavelength,
            vertical_again=vertical_again,
            bpm_kw=bpm_kw,
            skip_if_exists=skip_if_exists,
            setid=setid,
            verbose=verbose >= 3
        )

    # Use the short-version of the keywords for the PROCESSED summary file.
    summ_raw_vfv = _save_or_load_summary(
        dir_out,
        ocpath,
        rm_nonpol=rm_nonpol,
        keywords=HDR_KEYS["simple"],
        skip_if_exists=skip_if_exists,
        verbose=verbose >= 2
    )
    if verbose > 0:
        print("DONE.")

    return summ_raw_vfv


# # !FIXME: deprecate in favor of proc_vfv
# def proc_fourier(
#         dir_in,
#         dir_out,
#         dir_log=None,
#         summary_out=None,
#         cut_wavelength=200,
#         med_sub_clip=[-5, 5],
#         med_rat_clip=[0.5, 2],
#         std_rat_clip=[-5, 5],
#         vertical_again=True,
#         sigclip_kw=dict(sigma=2, maxiters=5),
#         fitting_sections=None,
#         method="median",
#         skip_if_exists=True,
#         rm_nonpol=True,
#         verbose=0,
#         show_progress=True
# ):
#     """
#     FFT cannot accept ``NaN`` value. We must either ignore mask or use FIXPIX
#     before FFT. Also we need to remove the vertical pattern before FFT
#     (float32) - not int16, that's why we cannot use do_16bit_vertical.

#     skip_if_exists : bool, optional.
#         Whether to skip all process if the file exists in `outdir`.
#         Default: `True`.

#     verbose : int
#         Larger number means it becomes more verbose::
#             * 0: print nothing
#             * 1: Only very essential things
#             * 2: + verbose for summary CSV file
#             * 3: + the HISTORY in each FITS file's header
#     """
#     dir_in, dir_out, dir_log = _set_dir_iol(dir_in, dir_out, dir_log)
#     fpaths = dir_in.glob("*.fits")
#     summary_out = _summary_path_parse(dir_log, summary_out, f"summary_{dir_out.name}.csv")

#     if summary_out.exists() and skip_if_exists:
#         print(f"Loading the existing summary CSV file from {summary_out}; \n\t"
#               + "SKIPPING all 16-bit & vertical pattern subtraction.")
#         return pd.read_csv(summary_out)

#     if verbose >= 1:
#         print("Fourier pattern subtraction; saving to\n"
#               + f"  * FITS: {dir_out} \n  * CSV : {summary_out}")
#     time.sleep(0.5)  # For tqdm to show progress bar properly on Jupyter notebook

#     for fpath in iterator(fpaths, show_progress=show_progress):
#         _ = _do_fourier(
#             fpath,
#             dir_out,
#             cut_wavelength=cut_wavelength,
#             skip_if_exists=skip_if_exists,
#             med_sub_clip=med_sub_clip,
#             med_rat_clip=med_rat_clip,
#             std_rat_clip=std_rat_clip,
#             vertical_again=vertical_again,
#             sigclip_kw=sigclip_kw,
#             fitting_sections=fitting_sections,
#             method=method,
#             verbose=verbose >= 3
#         )

#     summ_raw_vf = _save_or_load_summary(
#         dir_out,
#         summary_out,
#         skip_if_exists=skip_if_exists,
#         keywords=HDR_KEYS["all"],
#         rm_nonpol=rm_nonpol,
#         verbose=verbose >= 2
#     )
#     return summ_raw_vf


def make_darkmask(fpath, output=None, thresh=(-10, 50), percentile=(0.01, 99.99)):
    """
    Parameters
    ----------
    fpath : path-like
        The path to the master DARK frame to be used to make DARKMASK file.
    dark_thresh : tuple, optional
        [description], by default (-10, 100)
    dark_percentile : tuple, optional
        [description], by default (0.01, 99.99)

    Notes
    -----
    00000001 = 1: original mask
    00000010 = 2: dark above upper threshold
    00000100 = 4: dark above upper percentile
    00001000 = 8: dark below lower threshold
    00010000 = 16: dark below lower percentile
    """
    _darkmask = load_ccd(fpath)
    thmin = np.min(thresh)
    thmax = np.max(thresh)
    ptmin = np.min(percentile)
    ptmax = np.max(percentile)
    _pt = np.percentile(_darkmask.data, [ptmin, ptmax])
    _darkmask.data = ((_darkmask.data > thmax)*(2**1) + (_darkmask.data > _pt[1])*(2**2)
                      + (_darkmask.data < thmin)*(2**3) + (_darkmask.data < _pt[0])*(2**4))
    _darkmask = CCDData_astype(_darkmask, "uint8")
    _darkmask.header["OBJECT"] = "DARKMASK"
    _darkmask.header.set("DKTHMIN", thmin, "dark threshold min (code 2**1)")
    _darkmask.header.set("DKTHMAX", thmax, "dark threshold max (code 2**2)")
    _darkmask.header.set("DKPTMIN", ptmin, "dark percentile min (code 2**3)")
    _darkmask.header.set("DKPTMAX", ptmax, "dark percentile max (code 2**4)")
    cmt2hdr(
        _darkmask.header, "c",
        ("Mask values mean: 00000010 = 2**1: dark above upper threshold; "
            + "00000100 = 2**2: dark above upper percentile; "
            + "00001000 = 2**3: dark below lower threshold; "
            + "00010000 = 2**4: dark below lower percentile; "
            + "Refer to DKTHMIN, DKTHMAX, DKPTMIN, DKPTMAX.")
    )
    if output is not None:
        output = Path(output)
        output.parent.mkdir(exist_ok=True, parents=True)
        _darkmask.write(output, overwrite=True)

    return _darkmask
