from pathlib import Path

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter

from .ysfitsutilpy4nicpolpy import (cmt2hdr, df_selector, errormap,
                                    group_combine, hedit, listify, ndfy,
                                    run_reduc_plan)
from .ysphotutilpy4nicpolpy import sep_back, sep_extract

from .util import (FLATERR, GAIN, RDNOISE, SATLEVEL, _load_as_dict, _save,
                   infer_filter)

__all__ = [
    "make_cal", "fringe_scale_mask", "make_dark", "make_fringe"
]


def make_cal(
        framename,
        plan_in,
        dir_out,
        group_key,
        fmts,
        removeit="REMOVEIT",
        hedit_keys=None,
        hedit_values=None,
        combine_kw=dict(
            combine="med",
            reject="sc",
            sigma=[3., 3.]
        ),
        pixel_min=None,
        skip_if_exists=True,
        verbose=1
):
    df = plan_in.loc[plan_in[removeit] == 0].copy()
    if len(df) == 0:
        if verbose >= 1:
            print(f"No frames for {framename} found.")
        return None, None

    ccds = group_combine(
        df,
        group_key=group_key,
        **combine_kw,
        verbose=verbose,
    )

    paths = {}
    if dir_out is not None:
        dir_out = Path(dir_out)
        dir_out.mkdir(exist_ok=True, parents=True)

    for k, ccd_k in ccds.items():
        if pixel_min is not None:
            ccd_k.data[ccd_k.data < pixel_min] = 0
            cmt2hdr(ccd_k.header, "h", verbose=verbose >= 2,
                    s=f"Pixels < {pixel_min:.2f} are replaced with 0.")
        if isinstance(k, tuple):
            fstem = f"{framename}_" + fmts.format(*k)
        else:
            fstem = f"{framename}_" + fmts.format(k)
        try:
            del ccd_k.header["BLANK"]
        except (KeyError, IndexError):
            pass
        if hedit_keys is not None:
            ccd_k = hedit(ccd_k, hedit_keys, hedit_values, verbose=verbose >= 2)
        cmt2hdr(ccd_k.header, "h", "-"*72, time_fmt=None, verbose=verbose >= 2)
        paths[k] = _save(ccd_k, dir_out, fstem, return_path=True)

    return ccds, paths


def fringe_scale_mask(ccd, bezels=20, minarea=np.pi*5**2, **kwargs):
    if ccd is None:
        return None
    filt = ccd.header["FILTER"].upper()
    bezels = ndfy([ndfy(b, 2, default=0) for b in listify(bezels)], ccd.ndim)
    skymask = np.ones(ccd.shape, dtype=bool)
    data = ccd.data[bezels[0][0]:-bezels[0][1], bezels[1][0]:-bezels[1][1]]
    var = errormap(
        data,
        gain_epadu=GAIN[filt],
        rdnoise_electron=RDNOISE[filt],
        flat_err=FLATERR[filt],
        return_variance=True
    )
    _, segm = sep_extract(data, 5, var=var, minarea=minarea, **kwargs)
    skymask[bezels[0][0]:-bezels[0][1], bezels[1][0]:-bezels[1][1]] = segm == 0
    return skymask


def make_dark(
        plan_in,
        dir_out,
        combine_kw=dict(
            combine="med",
            reject="sc",
            sigma=[3., 3.]
        ),
        dark_min=3,
        skip_if_exists=True,
        load=True,
        verbose=0
):
    """ Makes master dark frames. Mostly hard coded for NIC polarimetric mode.
    Except for deveopment, most of the parameters are not need to be changed.
    """
    group_key = ("FILTER", "EXPTIME")

    dir_out = Path(dir_out)
    if skip_if_exists:
        if load:
            _, ccds, paths = _load_as_dict(dir_out, group_key, verbose=verbose >= 2)
            if ccds is not None:  # if some darks already exist
                if verbose >= 1:
                    print("Loading existing dark frames.")
                return ccds, paths
        else:
            if verbose >= 1:
                print("Skipping dark frame creation.")
            return None, None
    # else, run the code below
    df = df_selector(
        plan_in,
        fullmatch={"OBJECT": "DARK"},
        flags=0,
        querystr=None
    )

    if len(df) == 0:
        if verbose:
            print("No dark frames found.")
        return None, None

    if verbose >= 1:
        print("Making master dark of the night (grouping & combining dark frames).")

    ccds = group_combine(
        df,
        group_key=group_key,
        **combine_kw,
        verbose=verbose - 1,
    )

    paths = {}
    for k, ccd_k in ccds.items():
        filt, exptime = k
        ccd_k.data[ccd_k.data < dark_min] = 0
        cmt2hdr(ccd_k.header, "h", verbose=verbose >= 3,
                s=f"Pixels < {dark_min:.2f} in combined dark are replaced with 0.")
        fstem = f"{filt.lower()}_mdark_{exptime:.1f}"
        paths[k] = _save(ccd_k, dir_out, fstem, return_path=True)

    return ccds, paths


def make_fringe(
        plan_in,
        dir_out,
        combine_kw=dict(
            combine="med",
            reject="sc",
            sigma=[3., 3.]
        ),
        col_dark="DARKFRM",
        col_flat="FLATFRM",
        skip_if_exists=True,
        load=True,
        verbose=0
):
    """ Makes master fringe frames. Mostly hard coded for NIC polarimetric mode.
    Except for deveopment, most of the parameters are not need to be changed.

    Returned dict is `dict` of `list`!
    This is because there can be multiple fringe frames for the same
    `group_key`, separated by some amount of time.

    """
    group_key = ("FILTER", "OBJECT", "EXPTIME", "POL-AGL1")
    dir_out = Path(dir_out)

    if skip_if_exists:
        if load:
            _, ccds, paths = _load_as_dict(dir_out, group_key, verbose=verbose >= 2)
            if ccds is not None:  # if some fringe already exist
                if verbose >= 1:
                    print("Loading existing fringe frames.")
                return ccds, paths
        else:
            if verbose >= 1:
                print("Skipping fringe frame creation.")
            return None, None
    # else, run the code below

    plan = df_selector(
        plan_in,
        fullmatch={"OBJECT": ".*\_[sS][kK][yY]"},
        flags=0,
        querystr=None
    )

    if len(plan) == 0:
        if verbose:
            print("No fringe frames found.")
        return None, None

    if verbose >= 1:
        print("Making master fringe of the night (grouping & combining fringe frames).")

    diffidx = np.ediff1d(plan.index)
    _frin_idx = list(np.where(diffidx != 1)[0] + 1)  # at least length 3 (# of FILTER)
    frin_idx_begs = [0] + _frin_idx
    frin_idx_ends = _frin_idx + [len(plan)]
    # ^ Indices to "chop" the `plan`` into fringe frames of consecutive frameIDs.
    # The length from frin_idx_begs[i] to frin_idx_ends[i] is almost always 4*k
    # (mostly k=1), but just in case I generalized it.

    # Below, `paths` and `ccds` are `dict` of `list`.
    # This is because there can be multiple fringe frames for the same
    # `group_key`, separated by some amount of time.
    paths, ccds, idxs = {}, {}, {}
    for beg, end in zip(frin_idx_begs, frin_idx_ends):
        plan_block = plan.iloc[beg:end]  # likely only 4 frames (one for each POL-AGL1)
        ccds_block = run_reduc_plan(
            plan_block,  # DARKFRM, FLATFRM will be used
            return_ccd=True,
            col_dark=col_dark,
            col_flat=col_flat
        )
        ccds_group = group_combine(
            ccds_block,
            group_key=group_key,
            **combine_kw,
            verbose=verbose - 1,
        )

        for k, ccd_k in ccds_group.items():
            filt, objname, exptime, polagl1 = k
            if idxs.get(k) is None:
                idxs[k] = 0
                paths[k] = []
                ccds[k] = []
            else:
                idxs[k] += 1

            fstem = (f"{filt.lower()}_mfringe_{objname}_{exptime:.1f}"
                     + f"_{polagl1:.1f}_{idxs[k]:d}")
            paths[k].append(_save(ccd_k, dir_out, fstem, return_path=True))
            ccds[k].append(ccd_k)

    return ccds, paths


# def make_fringe(
#         path_df_in,
#         dir_out,
#         fringe_object=".*\_[sS][kK][yY]",
#         combine="avg",
#         reject="sc",
#         sigma=[3., 3.],
#         skip_if_exists=True,
#         verbose=0,
#         **kwargs
# ):
#     """ Makes master fringe frames.
#     Combine if >= 2 frames in consecutive counters.

#     skip_if_exists : bool, optional.
#         Whether to skip all process if the file exists in `outdir`.
#         Default: `True`.
#     """
#     df = pd.read_csv(path_df_in)

#     _, dir_out, _ = _set_dir_iol(None, dir_out, None)
#     if skip_if_exists:
#         _, fringes, fringepaths = _load_as_dict(dir_out, ["FILTER", "OBJECT", "EXPTIME"],
#                                                 verbose=verbose >= 2)
#         if fringes is not None:  # if some darks already exist
#             return fringes, fringepaths
#         # else, run the code below

#     if dir_dark is not None:
#         dir_dark, darks, darkpaths = _load_as_dict(dir_dark, ["FILTER", "EXPTIME"], False)
#     else:
#         darks, darkpaths = None, None

#     if dir_flat is not None:
#         dir_flat, flats, flatpaths = _load_as_dict(dir_flat, ["FILTER"], False)
#     else:
#         raise ValueError(
#             "Flat is mandatory for fringe frames at this moment. Please specify flat_dir."
#         )

#     if verbose >= 1:
#         print("Making master fringe of the night (grouping & combining sky fringe frames).")

#     # FIXME: How to skip if master fringe already exists?!
#     fringes = group_combine(
#         _summary,
#         type_key=["OBJECT"],
#         type_val=[fringe_object],
#         group_key=["FILTER", "OBJECT", "EXPTIME"],
#         combine=combine,
#         reject=reject,
#         sigma=sigma,
#         verbose=verbose >= 2,
#         **kwargs
#     )
#     fringepaths = {}
#     for k, fringe_k in fringes.items():
#         filt, objname, exptime = k
#         fstem = f"{filt.lower()}_fringe_{objname}_{exptime:.1f}"
#         if dir_dark is not None or dir_flat is not None:
#             mdark, mdarkpath = _find_calframe(
#                 darks, darkpaths, (filt.upper(), exptime), "Dark", verbose >= 2
#             )
#             mflat, mflatpath = _find_calframe(
#                 flats, flatpaths, filt.upper(), "Flat", verbose >= 2
#             )
#             fringes[k] = ccdred(
#                 fringe_k,
#                 mdark=mdark,
#                 mflat=mflat,
#                 mdarkpath=mdarkpath,
#                 mflatpath=mflatpath,
#                 verbose_bdf=verbose >= 3
#             )
#         fringepaths[k] = _save(fringes[k], dir_out, fstem, return_path=True)


def quick_detect_obj(
        image,
        mask=None,
        thresh_ksig=10,
        scale_maxiters=20,
        gsigma=6,
        box_size=(50, 50),
        filter_size=(5, 5),
        minarea=50,
        verbose=0,
):
    """ Detect objects in an image.

    Parameters
    ----------
    image : ndarray
        Image to detect objects in.

    mask : ndarray, optional
        Mask to apply to the image.

    thresh_ksig : float, optional
        Threshold in units of sigma (errormap) in the convolved image (see
        Notes). Default is 10.

    scale_maxiters : int, optional
        Maximum number of iterations for the scaling algorithm (See Notes).
        Default is 20.

    gsigma : float, optional
        Sigma of the Gaussian kernel used for the detecting algorithm (See
        Notes). Default is 6.

    box_size : int or array_like (int)
        Name in photutils; `bh`, `bw` order in sep. Default is ``(50, 50)``::

          * **sep**: Size of background boxes in pixels. Default is 64.
          * **photutils**: The box size along each axis. If `box_size` is a
            scalar then a square box of size `box_size` will be used. If
            `box_size` has two elements, they should be in ``(ny, nx)`` order.
            For best results, the box shape should be chosen such that the
            `data` are covered by an integer number of boxes in both
            dimensions. When this is not the case, see the `edge_method`
            keyword for more options.

    filter_size : int or array_like (int), optional
        Name in photutils; `fh`, `fw` order in sep. Default is ``(5, 5)``.::

          * **sep**: Filter width and height in boxes. Default is 3.
          * **photutils**: The window size of the 2D median filter to apply to
            the low-resolution background map. If `filter_size` is a scalar
            then a square box of size `filter_size` will be used. If
            `filter_size` has two elements, they should be in ``(ny, nx)``
            order. A filter size of ``1`` (or ``(1, 1)``) means no filtering.

    minarea : int, optional
        Minimum number of pixels required for an object. Default is 50.

    Notes
    -----
    This function does (1) convolve the image with a Gaussian kernel of sigma =
    `gsigma`, (2) threshold the image at `thresh_ksig` multiplied by
    sigma-clipped std.dev. of that convolved image (``std_conv``), and (3)
    detect objects in the convolved image with background = constant =
    sigma-clipped median (``med_conv``). The detection process is repeated for
    `scale_maxiters` times, and the threshold at the i-th iteration (i =
    0..`scale_maxiters`) is adjusted by ``std_conv * thresh_ksig * 0.9**i``,
    until at least one object is detected.

    The reason for the first step (convolution) is to detect very faint
    objects, i.e., when the SNR is only order of 10. We found this convolution
    significantly increase the detectability of faint objects. (especislly, the
    center position is critically wrong for faint objects).

    Anyway the flux from this detection process will not be used (only center
    region is important).

    Note that Gaussian convolved pix value is equal to "best fit Gaussian
    amplitude at the pixel" * "const1" + "const2" (see StetsonPB 1987 PASP 99
    191, eq. (1)).
    """
    conv = gaussian_filter(image, gsigma)  # gsigma = 6 ~ FWHM/2.35 with FWHM ~ 13-15 pix
    conv -= sep_back(conv, mask=mask, box_size=box_size, filter_size=filter_size).back()
    _, med_conv, std_conv = sigma_clipped_stats(conv.ravel())
    for _scale in range(scale_maxiters):
        scale = 0.9**_scale
        thresh = thresh_ksig*std_conv*scale
        obj, segm = sep_extract(
            conv, thresh, bkg=med_conv, minarea=minarea*scale,
            sort_by='flux', sort_ascending=False
        )
        if len(obj) > 0:
            if verbose > 0:
                print(f"{len(obj)} objects found at {scale=}")
            break

    if len(obj) == 0:
        if verbose > 0:
            print(f"No object found")
        obj = None
        segm = None
        conv = None

    return obj, segm, conv


# def vertical_correct_remaining(
#         img,
#         filt=None,
#         mask=None,
#         thresh_ksig=3,
#         scale_maxiters=5,
#         gsigma=6,
#         box_size=(50, 50),
#         filter_size=(5, 5),
#         minarea=50,
#         sigclip_kw=dict(sigma=2, maxiters=5)
# ):
#     """ Removes still remaining vertical pattern in the o/e frame.
#     """
#     filt = infer_filter(filt)
#     gain = GAIN[filt]
#     rdn = RDNOISE[filt]
#     mask_sat = img > SATLEVEL[filt]
#     obj, segm, img0detect = quick_detect_obj(
#         img,
#         thresh_ksig=thresh_ksig,
#         minarea=minarea,
#         gsigma=gsigma,
#         box_size=box_size,
#         filter_size=filter_size
#     )
#     if len(obj) > 0:
#         obj = obj.loc[obj["flux"] == obj["flux"].max()]
#         mask = mask_sat | (segm > 0)
#         _bkg_kw = dict(mask=mask, box_size=(10, 10), filter_size=(5, 3))
#         # -- Subtract 1: img1 = original - sep_back
#         _bkg1 = sep_back(img, **_bkg_kw)
#         bkg1 = _bkg1.back() - _bkg1.globalback  # only the pattern.
#         img1 = img - bkg1
#         # -- Subtract 2: img2 = original - (vertical pattern from (original - sep_back))
#         _bkg2 = sigma_clipped_stats(img1, mask=mask, axis=0)[1]
#         _bkg2med = np.median(_bkg2)
#         _x = np.arange(0, _bkg2.size)
#         bkg2 = csaps.csaps(_x, _bkg2 - _bkg2med, _x, smooth=0.1)
#         img2 = img0 - bkg2  # subtract from the original image
#         err = yfu.errormap(img2, gain_epadu=gain, rdnoise_electron=rdn)

#     _bkg1 = sep_back(img, mask=mask, box_size=box_size, filter_size=filter_size)
#     bkg1 = _bkg1.back() - _bkg1.globalback  # only the pattern.
#     img1 = img - bkg1
#     _bkg2 = sigma_clipped_stats(img1, mask=mask, axis=0, **sigclip_kw)[1]
#     return _bkg2 - np.median(_bkg2)  # only the pattern.
