import time
from pathlib import Path

import astropy
import numpy as np
import pandas as pd
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from ysfitsutilpy import (CCDData_astype, _parse_data_header, add_to_header,
                          crrej, fitsxy2py, group_combine, load_ccd,
                          medfilt_bpm, trim_ccd, update_process)
from ysfitsutilpy.preproc import bdf_process

from .util import (GAIN, NIC_CRREJ_KEYS, RDNOISE, USEFUL_KEYS, VERTICALSECTS,
                   _find_calframe, _load_as_dict, _sanitize_objects,
                   _save_or_load_summary, _set_dir_iol, _set_fstem,
                   infer_filter, iterator, split_oe)

__all__ = ["prepare",
           "cr_reject_nic", "vertical_correct", "lrsubtract",
           "fourier_lrsub", "proc_16_vertical", "proc_fourier",
           "make_dark", "make_fringe", "preproc_nic"
           ]


def _save(ccd, savedir, fstem, return_path=False):
    if savedir is None:
        return

    outpath = Path(savedir)/f"{fstem}.fits"
    try:
        ccd.write(outpath, overwrite=True, output_verify='fix')
    except FileNotFoundError:
        outpath.parent.mkdir(parents=True)
        ccd.write(outpath, overwrite=True, output_verify='fix')
    if return_path:
        return outpath


def _do_16bit_vertical(fpath, dir_out, skip_if_exists=True, sigclip_kw=dict(sigma=2, maxiters=5),
                       fitting_sections=None, method='median', verbose=True, show_progress=True):
    ''' Changes to 16-bit and correct the vertical pattern.
    dir_out : path-like
        The directory for the resulting FITS files to be saved, RELATIVE to ``self.dir_work``.

    verbose : int
        Larger number means it becomes more verbose::
            * 0: print nothing
            * 1: Only very essential things
            * 2: + verbose for summary CSV file
            * 3: + the HISTORY in each FITS file's header
    '''
    fpath = Path(fpath)
    ccd_orig = load_ccd(fpath)
    _t = Time.now()

    # == Set output stem =================================================================================== #
    outstem, _ = _set_fstem(ccd_orig.header)
    outstem += "-PROC-v"

    # == Skip if conditions meet =========================================================================== #
    if skip_if_exists and (dir_out/f"{outstem}.fits").exists():
        return

    # == First, change the bit ============================================================================= #
    ccd_nbit = ccd_orig.copy()
    ccd_nbit = CCDData_astype(ccd_nbit, dtype='int16')

    add_to_header(ccd_nbit.header, 'h', verbose=verbose,
                  s="{:=^72s}".format(' Basic preprocessing start '))

    if ccd_orig.dtype != ccd_nbit.dtype:
        add_to_header(ccd_nbit.header, 'h', t_ref=_t, verbose=verbose,
                      s=f"Changed dtype (BITPIX): {ccd_orig.dtype} to {ccd_nbit.dtype}")

    # == Then check if identical =========================================================================== #
    # It takes < ~20 ms on MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB (2400MHz
    # DDR4), Radeon Pro 560X (4GB)]
    #   ysBach 2020-05-15 16:06:08 (KST: GMT+09:00)
    np.testing.assert_almost_equal(
        ccd_orig.data - ccd_nbit.data,
        np.zeros(ccd_nbit.data.shape)
    )

    # == Set counter ======================================================================================= #
    try:
        counter = ccd_nbit.header["COUNTER"]
    except KeyError:
        try:
            counter = fpath.stem.split('_')[1]
            counter = counter.split('.')[0]
            # e.g., hYYMMDD_dddd.object.pcr.fits
        except IndexError:  # e.g., test images (``h.fits```)
            counter = 9999
        ccd_nbit.header['COUNTER'] = (counter, "Image counter of the day, 1-indexing; 9999=TEST")

    # == Update warning-invoking parts ===================================================================== #
    try:
        ccd_nbit.header["MJD-STR"] = float(ccd_nbit.header["MJD-STR"])
        ccd_nbit.header["MJD-END"] = float(ccd_nbit.header["MJD-END"])
    except KeyError:
        pass

    # == vertical pattern subtraction ====================================================================== #
    ccd_nbit_v = vertical_correct(
        ccd_nbit,
        sigclip_kw=sigclip_kw,
        fitting_sections=fitting_sections,
        method=method,
        dtype='int16',
        return_pattern=False
    )
    update_process(ccd_nbit_v.header, "v", additional_comment=dict(v="vertical pattern"))

    _save(ccd_nbit_v, dir_out, outstem)
    return ccd_nbit_v


def _do_fourier(fpath, dir_out, cut_wavelength=200, med_sub_clip=[-5, 5], med_rat_clip=[0.5, 2],
                std_rat_clip=[-5, 5], skip_if_exists=True, verbose=True):
    fpath = Path(fpath)
    outstem = fpath.stem + "f"  # hard-coded
    # == Skip if conditions meet =================================================================== #
    if skip_if_exists and (dir_out/f"{outstem}.fits").exists():
        return

    ccd_v = load_ccd(fpath)
    ccd_vf = fourier_lrsub(
        ccd_v,
        cut_wavelength=cut_wavelength,
        med_sub_clip=med_sub_clip,
        med_rat_clip=med_rat_clip,
        std_rat_clip=std_rat_clip,
        verbose_bpm=verbose,
        verbose=verbose
    )
    ccd_vf = CCDData_astype(ccd_vf, 'float32')
    update_process(ccd_vf.header, "f", add_comment=False, additional_comment={'f': "fourier pattern"})
    _save(ccd_vf, dir_out, outstem)
    return ccd_vf


def prepare(fpath, outdir=Path('.'),
            kw_vertical=dict(sigclip_kw=dict(sigma=2, maxiters=5), fitting_sections=None, method='median',
                             update_header=True, verbose=False),
            dir_vc=None, dir_vcfs=None,
            kw_fourier={'med_sub_clip': [-5, 5], 'med_rat_clip': [0.5, 2], 'std_rat_clip': [-5, 5]},
            save_nonpol=False, split=True, verbose=False):
    ''' Rename the original NHAO NIC image and convert to certain dtype.

    Note
    ----
    Original NHAO NIC image is in 32-bit integer, and is twice the size it should be. To save the
    storage, it is desirable to convert those to 16-bit. As the bias is not added to the FITS frame
    from NHAO NIC, the pixel value in the raw FITS file can be negative. Fortunately, however, the
    maximum pixel value when saturation occurs is only about 20k, and therefore using ``int16`` rather
    than ``uint16`` is enough.

    Here, not only reducing the size, the file names are updated using the original file name and
    header information:
        ``<FILTER (j, h, k)><System YYMMDD>_<COUNTER:04d>.fits``
    It is then updated to
        ``<FILTER (j, h, k)>_<System YYYYMMDD>_<COUNTER:04d>_<OBJECT>_<EXPTIME:.1f>_<POL-AGL1:04.1f>
          _<INSROT:+04.0f>_<IMGROT:+04.0f>_<PA:+06.1f>.fits``
    '''
    ccd_orig = load_ccd(fpath)
    _t = Time.now()
    ccd_nbit = ccd_orig.copy()
    ccd_nbit = CCDData_astype(ccd_nbit, dtype='int16')

    add_to_header(ccd_nbit.header, 'h', verbose=verbose,
                  s="{:=^72s}".format(' Basic preprocessing start '))

    if ccd_orig.dtype != ccd_nbit.dtype:
        add_to_header(ccd_nbit.header, 'h', t_ref=_t, verbose=verbose,
                      s=f"Changed dtype (BITPIX): {ccd_orig.dtype} to {ccd_nbit.dtype}")

    # == First, check if identical ===================================================================== #
    # It takes < ~20 ms on MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB (2400MHz
    # DDR4), Radeon Pro 560X (4GB)]
    #   ysBach 2020-05-15 16:06:08 (KST: GMT+09:00)
    np.testing.assert_almost_equal(
        ccd_orig.data - ccd_nbit.data,
        np.zeros(ccd_nbit.data.shape)
    )

    # == Set counter =================================================================================== #
    try:
        counter = ccd_nbit.header["COUNTER"]
    except KeyError:
        try:
            counter = fpath.stem.split('_')[1]
            counter = counter.split('.')[0]
            # e.g., hYYMMDD_dddd.object.pcr.fits
        except IndexError:  # e.g., test images (``h.fits```)
            counter = 9999
        ccd_nbit.header['COUNTER'] = (counter, "Image counter of the day, 1-indexing; 9999=TEST")

    # == Update warning-invoking parts ================================================================= #
    try:
        ccd_nbit.header["MJD-STR"] = float(ccd_nbit.header["MJD-STR"])
        ccd_nbit.header["MJD-END"] = float(ccd_nbit.header["MJD-END"])
    except KeyError:
        pass

    # == Set output stem =============================================================================== #
    outstem, polmode = _set_fstem(ccd_nbit.header)

    # == vertical pattern subtraction ================================================================== #
    ccd_nbit_vc = vertical_correct(ccd_nbit, **kw_vertical, dtype='int16', return_pattern=False)

    if dir_vc is not None:
        _save(ccd_nbit_vc, dir_vc, outstem + "_vc")

    if polmode:
        # == Do Fourier pattern subtraction ============================================================ #
        ccd_nbit_vcfs = fourier_lrsub(ccd_nbit_vc, cut_wavelength=200, **kw_fourier)
        ccd_nbit_vcfs = CCDData_astype(ccd_nbit_vcfs, dtype='int16')

        if dir_vcfs is not None:
            _save(ccd_nbit_vcfs, dir_vcfs, outstem + "_vc_fs")

        if split:
            ccds = split_oe(ccd_nbit_vcfs, verbose=verbose)
            ccd_out = []
            for _ccd, oe in zip(ccds, 'oe'):
                _save(_ccd, outdir, outstem + f"_{oe:s}")
                ccd_out.append(_ccd)

        else:
            ccd_out = ccd_nbit_vcfs
            _save(ccd_out, outdir, outstem)

    else:
        ccd_out = ccd_nbit_vc
        if save_nonpol:
            _save(ccd_out, outdir, outstem)

    return ccd_out


def cr_reject_nic(ccd, mask=None, filt=None, update_header=True, add_process=True, crrej_kw=None,
                  verbose=True, full=False):
    """

    Parameters
    ----------

    ccd: CCDData
        The ccd to be processed.

    filt: str, None, optional.
        The filter name in one-character string (case insensitive). If `None`(default), it will be
        inferred from the header (key ``"FILTER"``).

    update_header: bool, optional.
        Whether to update the header if there is any.

    add_process : bool, optional.
        Whether to add ``PROCESS`` key to the header.

    crrej_kw: dict, optional.
        The keyword arguments for the ``astroscrappy.detect_cosmics``. If `None` (default), the
        parameters for IRAF-version of L.A. Cosmic, except for the ``sepmed = True`` and ``gain`` and
        ``readnoise`` are replaced to the values of NIC detectors.

    verbose: bool, optional
        The verbose paramter to ``astroscrappy.detect_cosmics``.

    Returns
    -------
    nccd: CCDData
        The cosmic-ray removed object.

    Note
    ----
    astroscrappy automatically correct gain (i.e., output frame in the unit of electrons, not ADU). In
    this function, this is undone, i.e., I divide it with the gain to restore ADU unit. Also
    ``nccd.mask`` will contain the mask from cosmic-ray detection.
    """
    crkw = NIC_CRREJ_KEYS.copy()

    try:
        gain = ccd.gain.value
        rdnoise = ccd.rdnoise.value
    except AttributeError:
        filt = infer_filter(ccd, filt=filt, verbose=verbose)
        gain = GAIN[filt]
        rdnoise = RDNOISE[filt]

    crkw['gain'] = gain
    crkw['rdnoise'] = rdnoise

    if crrej_kw is not None:
        for k, v in crrej_kw.items():
            crkw[k] = v

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


def vertical_correct(ccd, fitting_sections=None, method='median', sigclip_kw=dict(sigma=2, maxiters=5),
                     dtype='float32', return_pattern=False, update_header=True, verbose=False):
    ''' Correct vertical strip patterns.

    Paramters
    ---------

    ccd : CCDData, HDU object, HDUList, or ndarray.
        The CCD to subtract the vertical pattern.

    fitting_sections : list of two str, optional.
        The sections to be used for the vertical pattern estimation. This must be identical to the
        usual FITS section (i.e., that used in SAO ds9 or IRAF, 1-indexing and last-index-inclusive),
        not in python. **Give it in the order of ``[<upper>, <lower>]`` in FITS y-coordinate.**

    method : str, optional.
        One of ``['med', 'avg', 'median', 'average', 'mean']``.

    sigma, maxiters : float and int, optional
        A sigma-clipping will be done to remove hot pixels and cosmic rays for estimating the vertical
        pattern. To turn sigma-clipping off, set ``maxiters=0``.

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

    '''
    _t = Time.now()
    data, hdr = _parse_data_header(ccd)

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
        raise ValueError("method not understood; it must be one of [med, avg, median, average, mean].")

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
    vsub = data - vpattern.astype(dtype)

    if update_header and hdr is not None:
        # add as history
        add_to_header(hdr, 'h', verbose=verbose, t_ref=_t,
                      s=f"Vertical pattern subtracted using {fitting_sections} by {methodstr} with {clipstr}")

    try:
        nccd = CCDData(data=vsub, header=hdr)
    except ValueError:
        nccd = CCDData(data=vsub, header=hdr, unit='adu')

    nccd.data = nccd.data.astype(dtype)
    if return_pattern:
        return nccd, vpattern
    return nccd


def lrsubtract(ccd, fitting_sections=["[:, 50:100]", "[:, 924:974]"], method='median',
               sigclip_kw=dict(sigma=2, maxiters=5), dtype='float32', update_header=True, verbose=False):
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
        add_to_header(nccd.header, 'h', t_ref=_t, verbose=verbose,
                      s=f"Subtracted left half ({i_half} columns) from the right half.")
    # nccd = trim_image(nccd, fits_section=f"[{i_half + 1}:, :]")
    return nccd


def fourier_lrsub(ccd, cut_wavelength=200, copy=True, verbose_bpm=False, verbose=False, **kwargs):
    if copy:
        _ccd = ccd.copy()
    else:
        _ccd = ccd
    ccd_l = trim_ccd(_ccd, fits_section="[:512, :]", update_header=False)
    ccd_l = medfilt_bpm(ccd_l, verbose=verbose_bpm, **kwargs)
    ccd.header = ccd_l.header  # to add MBPM logs

    amp_comp = np.fft.rfft(ccd_l.data, axis=0)
    amp_comp[cut_wavelength:, :] = 0
    pattern_pure = np.fft.irfft(amp_comp, axis=0)
    pattern = np.tile(pattern_pure, 2)
    _ccd.data = _ccd.data - pattern
    add_to_header(
        _ccd.header, 'h', verbose=verbose,
        s=("FFT(left half) to get pattern map (see FFTCUTWL for the cut wavelength); "
           + "subtracted from both left/right.")
    )
    _ccd.header["FFTCUTWL"] = (cut_wavelength, "FFT cut wavelength (amplitude[this:, :] = 0)")
    return _ccd


def proc_16_vertical(dir_in, dir_out, dir_log=None,  sigclip_kw=dict(sigma=2, maxiters=5),
                     fitting_sections=None, method='median', skip_if_exists=True, ignore_nonpol=True,
                     verbose=0, show_progress=True):
    '''
    dir_in, dir_out : path-like
        The directories where the raw (original) FITS are in and that for the resulting FITS files to
        be saved.

    verbose : int
        Larger number means it becomes more verbose::
            * 0: print nothing
            * 1: Only very essential things
            * 2: + verbose for summary CSV file
            * 3: + the HISTORY in each FITS file's header
    '''
    dir_in, dir_out, dir_log = _set_dir_iol(dir_in, dir_out, dir_log)

    path_summ_in = dir_log/"summary_raw.csv"
    path_summ_out = dir_log/f"summary_{dir_out.name}.csv"

    summ_raw = _save_or_load_summary(dir_in, path_summ_in, skip_if_exists=skip_if_exists,
                                     ignore_nonpol=ignore_nonpol, verbose=verbose >= 2)
    if verbose > 0:
        print("32-bit -> 16-bit && Vertical pattern subtraction; saving to\n"
              + f"  * FITS: {dir_out} \n  * CSV : {path_summ_out}")
    time.sleep(0.5)  # For tqdm to show progress bar properly on Jupyter notebook

    for fpath in iterator(summ_raw['file'], show_progress=show_progress):
        _ = _do_16bit_vertical(fpath, dir_out, skip_if_exists=skip_if_exists,
                               sigclip_kw=sigclip_kw, fitting_sections=fitting_sections, method=method,
                               show_progress=show_progress, verbose=verbose >= 3)

    summ_raw_v = _save_or_load_summary(dir_out, path_summ_out, ignore_nonpol=ignore_nonpol,
                                       skip_if_exists=skip_if_exists, verbose=verbose >= 2)
    return summ_raw_v


def proc_fourier(dir_in, dir_out, dir_log=None, cut_wavelength=200, med_sub_clip=[-5, 5],
                 med_rat_clip=[0.5, 2], std_rat_clip=[-5, 5], skip_if_exists=True, ignore_nonpol=True,
                 verbose=0, show_progress=True):
    '''
    dir_out : path-like
        The directory for the resulting FITS files to be saved, RELATIVE to ``self.dir_work``.

    verbose : int
        Larger number means it becomes more verbose::
            * 0: print nothing
            * 1: Only very essential things
            * 2: + verbose for summary CSV file
            * 3: + the HISTORY in each FITS file's header
    '''
    dir_in, dir_out, dir_log = _set_dir_iol(dir_in, dir_out, dir_log)
    fpaths = dir_in.glob("*.fits")
    path_summ_out = dir_log/f"summary_{dir_out.name}.csv"

    if verbose >= 1:
        print("Fourier pattern subtraction; saving to"
              + f"  * FITS: {dir_out} \n  * CSV : {path_summ_out}")
    time.sleep(0.5)  # For tqdm to show progress bar properly on Jupyter notebook

    for fpath in iterator(fpaths, show_progress=show_progress):
        _ = _do_fourier(fpath, dir_out, cut_wavelength=cut_wavelength, skip_if_exists=skip_if_exists,
                        med_sub_clip=med_sub_clip, med_rat_clip=med_rat_clip, std_rat_clip=std_rat_clip,
                        verbose=verbose >= 3)

    summ_raw_vf = _save_or_load_summary(dir_out, path_summ_out, skip_if_exists=skip_if_exists,
                                        ignore_nonpol=ignore_nonpol, verbose=verbose >= 2)
    return summ_raw_vf


def make_dark(dir_in, dir_out, dir_log=None, dark_object="DARK", combine='med', reject='sc', sigma=[3., 3.],
              dark_min=3, skip_if_exists=True, verbose=0, **kwargs):
    '''
    verbose : int
        Larger number means it becomes more verbose::
            * 0: print nothing
            * 1: Only very essential things
            * 2: + progress-like info (printing filter and exptime)
            * 3: + fits stack info (grouping) + header update
            * 4: + imcombine verbose
    '''
    dir_in, dir_out, dir_log = _set_dir_iol(dir_in, dir_out, dir_log)

    try:
        _summary = pd.read_csv(dir_log/f"summary_{dir_in.name}.csv")
    except FileNotFoundError:
        raise FileNotFoundError(f"Summary for {dir_in} is not found in {dir_log}."
                                + " Try changing dir_in or dir_log.")

    if skip_if_exists:
        if verbose >= 1:
            print("Loading existing dark frames...")
        _, darks, darkpaths = _load_as_dict(dir_out, ["FILTER", "EXPTIME"], verbose=verbose >= 2)
        if darks is not None:  # if some darks already exist
            return darks, darkpaths
        # else, run the code below

    if verbose >= 1:
        print("Making master dark of the night (grouping & combining dark frames).")

    darks = group_combine(
        _summary,
        type_key=["OBJECT"],
        type_val=[dark_object],
        group_key=["FILTER", "EXPTIME"],
        combine=combine,
        reject=reject,
        sigma=sigma,
        verbose=verbose - 1,
        **kwargs
    )
    darkpaths = {}
    for k, dark_k in darks.items():
        filt, exptime = k
        darks[k].data[dark_k.data < dark_min] = 0
        add_to_header(darks[k].header, 'h', verbose=verbose >= 3,
                      s=f"Pixels with value < {dark_min:.2f} in combined dark are replaced with 0.")
        fstem = f"{filt.lower()}_mdark_{exptime:.1f}"
        darkpaths[k] = _save(darks[k], dir_out, fstem, return_path=True)

    return darks, darkpaths


def make_fringe(dir_in, dir_out, dir_log=None, dir_dark=None, dir_flat=None,
                fringe_object=".*\_[sS][kK][yY]", combine='med', reject='sc', sigma=[3., 3.],
                skip_if_exists=True, verbose=0, **kwargs):
    dir_in, dir_out, dir_log = _set_dir_iol(dir_in, dir_out, dir_log)

    try:
        _summary = pd.read_csv(dir_log/f"summary_{dir_in.name}.csv")
    except FileNotFoundError:
        raise FileNotFoundError(f"Summary for {dir_in} is not found in {dir_log}."
                                + " Try changing dir_in or dir_log.")

    if skip_if_exists:
        _, fringes, fringepaths = _load_as_dict(dir_out, ["FILTER", "OBJECT", "EXPTIME", "POL-AGL1"],
                                                verbose=verbose >= 2)
        if fringes is not None:  # if some darks already exist
            return fringes, fringepaths
        # else, run the code below

    if dir_dark is not None:
        dir_dark, darks, darkpaths = _load_as_dict(dir_dark, ["FILTER", "EXPTIME"], False)
    else:
        darks, darkpaths = None, None

    if dir_flat is not None:
        dir_flat, flats, flatpaths = _load_as_dict(dir_flat, ["FILTER"], False)
    else:
        raise ValueError("Flat is mandatory for fringe frames at this moment. Please specify flat_dir.")

    if verbose >= 1:
        print("Making master fringe of the night (grouping & combining sky fringe frames).")

    # FIXME: How to skip if master fringe already exists?!
    fringes = group_combine(
        _summary,
        type_key=["OBJECT"],
        type_val=[fringe_object],
        group_key=["FILTER", "OBJECT", "EXPTIME", "POL-AGL1"],
        combine=combine,
        reject=reject,
        sigma=sigma,
        verbose=verbose >= 2,
        **kwargs
    )
    fringepaths = {}
    for k, fringe_k in fringes.items():
        filt, objname, exptime, polagl = k
        fstem = f"{filt.lower()}_fringe_{objname}_{exptime:.1f}_{polagl:04.1f}"
        if dir_dark is not None or dir_flat is not None:
            mdark, mdarkpath = _find_calframe(darks, darkpaths, (filt.upper(), exptime), "Dark", verbose >= 2)
            mflat, mflatpath = _find_calframe(flats, flatpaths, filt.upper(), "Flat", verbose >= 2)
            fringes[k] = bdf_process(fringe_k, verbose_bdf=verbose >= 3,
                                     mdark=mdark, mdarkpath=mdarkpath, mflat=mflat, mflatpath=mflatpath)
        fringepaths[k] = _save(fringes[k], dir_out, fstem, return_path=True)


def preproc_nic(dir_in, dir_out, dir_log=None, objects=None, objects_exclude=False, dir_flat=None,
                dir_dark=None, dir_fringe=None, skip_if_exists=True, fringe_object=".*\_[sS][kK][yY]",
                verbose=0):
    """
    Parameters
    ----------
    objects_include, objects_exclude : str, list-like, None, optional.
        The FITS files with certain ``OBJECT`` values in headers to be used or unused in the flat
        correction, respectively.

    """
    dir_in, dir_out, dir_log = _set_dir_iol(dir_in, dir_out, dir_log)
    path_summ_out = dir_log/f"summary_{dir_out.name}.csv"
    if skip_if_exists and path_summ_out.exists():
        if verbose:
            print(f"Loading the existing summary CSV file from {path_summ_out}; SKIPPING all preprocessing.")
        return pd.read_csv(path_summ_out)

    try:
        _summary = pd.read_csv(dir_log/f"summary_{dir_in.name}.csv")
        _mask_dark = _summary["OBJECT"].str.fullmatch("dark", case=False)
        _mask_flat = _summary["OBJECT"].str.fullmatch("flat", case=False)
        _mask_test = _summary["OBJECT"].str.fullmatch("test", case=False)
        _mask_fringe = _summary["OBJECT"].str.match(fringe_object)
        # ^ Fringe is already flat-corrected, so remove it in the process below _summary_filt
        _summary = _summary[~(_mask_fringe | _mask_flat | _mask_dark | _mask_test)]
        _summary = _summary.reset_index(drop=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Summary for {dir_in} is not found in {dir_log}."
                                + " Try changing dir_in or dir_log.")

    if verbose >= 1:
        print("Loading calibration frames...")

    if verbose >= 2:
        print("  * Flat frames... ", end='')
    dir_flat, flats, flatpaths = _load_as_dict(dir_flat, ['FILTER'], verbose >= 2)

    if verbose >= 2:
        print("\n  * Dark frames... ", end='')
    dir_dark, darks, darkpaths = _load_as_dict(dir_dark, ['FILTER', 'EXPTIME'], verbose >= 2)

    if verbose >= 2:
        print("\n  * Fringe frames... ", end='')
    dir_fringe, fringes, fringepaths = _load_as_dict(dir_fringe, ['FILTER', 'OBJECT', 'POL-AGL1'],
                                                     verbose >= 2)

    if dir_flat is None and dir_dark is None and dir_fringe is None:
        while True:
            to_proceed = input("There is no flat/dark/fringe found. Are you sure to proceed? [y/N] ")
            if to_proceed.lower() in ['', 'N']:
                return
            elif to_proceed.lower() not in ['y']:
                print("Type yes='y', NO='', 'n'. ")

    objects, objects_exclude = _sanitize_objects(objects, objects_exclude)
    objnames = np.unique(_summary["OBJECT"])

    if verbose >= 1:
        print(f"Flat correction (&& dark and fringe if given); saving to {dir_out}")

    for filt in 'jhk':
        if verbose >= 1:
            print(f"  * {filt.upper()}: ", end=' ')
        _summary_filt = _summary[_summary["FILTER"] == filt.upper()]
        mflat, mflatpath = _find_calframe(flats, flatpaths, filt.upper(), "Flat", verbose >= 2)

        for objname in objnames:
            if ((objects_exclude and (objname in objects))
                    or (not objects_exclude and (objname not in objects))):
                continue

            if verbose >= 1:
                print(objname, end='... ')

            _summary_obj = _summary_filt[_summary_filt["OBJECT"].str.match(objname)]
            _summary_obj = _summary_obj.reset_index(drop=True)
            for i, row in _summary_obj.iterrows():
                setid = 1 + i//4
                fpath = Path(row['file'])
                ccd = load_ccd(fpath)
                hdr = ccd.header
                val_dark = (filt.upper(), hdr["EXPTIME"])
                val_frin = (filt.upper(), hdr["POL-AGL1"], hdr["OBJECT"] + "_sky")
                if verbose >= 2:
                    print(fpath)
                mdark, mdarkpath = _find_calframe(darks, darkpaths, val_dark, "Dark", verbose >= 2)
                mfringe, mfringepath = _find_calframe(fringes, fringepaths, val_frin, "Fringe", verbose >= 2)

                suffix = ''
                suffix += "D" if mdark is not None else ''
                suffix += "F" if mflat is not None else ''
                suffix += "Fr" if mfringe is not None else ''

                if verbose >= 3:
                    print("\n", fpath)

                nccd = bdf_process(ccd,
                                   mflatpath=mflatpath, mflat=mflat,
                                   mdarkpath=mdarkpath, mdark=mdark,
                                   mfringepath=mfringepath, mfringe=mfringe,
                                   verbose_bdf=verbose >= 3)
                nccd.header["SETID"] = (setid, "Pol mode set number of OBJECT on the night")
                nccds = split_oe(nccd, verbose=verbose >= 3)
                for oe, nccd_oe in zip('oe', nccds):
                    _save(nccd_oe, dir_out, fpath.stem + suffix + f"_{oe}")

            if verbose >= 1:
                print()

    summ_reduced = _save_or_load_summary(dir_out, path_summ_out, keywords=USEFUL_KEYS+["OERAY", "SETID"],
                                         skip_if_exists=skip_if_exists, verbose=verbose >= 2)

    return summ_reduced
