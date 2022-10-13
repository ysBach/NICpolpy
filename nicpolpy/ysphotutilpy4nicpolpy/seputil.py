"""
A collection of convenience functions used with the package sep.

Why use sep, not photutils?
===========================
Many times the purpose of SExtractor is to mearly **find** objects, not
detailed photometry of them. Thus, the calculational speed is an important
factor.

sep is about 100 times faster in background estimation than photutils from many
tests (see sep/bench.py of the sep repo):

```
# MBP 15" [2018, macOS 11.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)]
$ python bench.py
test image shape: (1024, 1024)
test image dtype: float32
measure background:  12.34 ms
subtract background:   3.38 ms
background array:   6.94 ms
rms array:   5.21 ms
extract:  88.87 ms  [1167 objects]

sep version:       1.10.0
photutils version: 0.7.1

| test                    | sep             | photutils       | ratio  |
|-------------------------|-----------------|-----------------|--------|
| 1024^2 image background |        11.68 ms |       851.76 ms |  72.94 |
| circles  r= 5  subpix=5 |    3.40 us/aper |   45.30 us/aper |  13.31 |
| circles  r= 5  exact    |    3.89 us/aper |   45.76 us/aper |  11.76 |
| ellipses r= 5  subpix=5 |    3.89 us/aper |   75.29 us/aper |  19.37 |
| ellipses r= 5  exact    |   10.88 us/aper |   63.71 us/aper |   5.85 |
```

photutils 1.4 and sep 1.2.0:
```
# MBP 14" [2021, macOS 12.2.1, M1Pro(6P+2E/G16c/N16c/32G)]
# -- ANACONDA INTEL
test image shape: (1024, 1024)
test image dtype: float32
measure background:  10.18 ms
subtract background:   2.10 ms
background array:   3.80 ms
rms array:   2.81 ms
extract:  65.67 ms  [1167 objects]

sep version:       1.2.0
photutils version: 1.4.1.dev19+g3e7460c3

| test                    | sep             | photutils       | ratio  |
|-------------------------|-----------------|-----------------|--------|
| 1024^2 image background |         8.60 ms |        38.81 ms |   4.51 |
| circles  r= 5  subpix=5 |    1.94 us/aper |   40.75 us/aper |  21.01 |
| circles  r= 5  exact    |    2.24 us/aper |   31.40 us/aper |  14.01 |
| ellipses r= 5  subpix=5 |    2.41 us/aper |   34.44 us/aper |  14.31 |
| ellipses r= 5  exact    |    5.42 us/aper |   45.10 us/aper |   8.33 |
```

```
# MBP 14" [2021, macOS 12.2.1, M1Pro(6P+2E/G16c/N16c/32G)]
# -- MINIFORGE ARM64 (default numpy installation)
# Miniforge with ``conda install numpy "blas=*=*accelerate*"`` had almost identical timings
# (+- 10%) whith this.
test image shape: (1024, 1024)
test image dtype: float32
measure background:   7.28 ms
subtract background:   3.48 ms
background array:   4.22 ms
rms array:   3.70 ms
extract:  32.39 ms  [1167 objects]

sep version:       1.2.0
photutils version: 1.4.1.dev19+g3e7460c3

| test                    | sep             | photutils       | ratio  |
|-------------------------|-----------------|-----------------|--------|
| 1024^2 image background |         7.20 ms |        36.58 ms |   5.08 |
| circles  r= 5  subpix=5 |    1.19 us/aper |   21.41 us/aper |  18.05 |
| circles  r= 5  exact    |    1.55 us/aper |   18.39 us/aper |  11.88 |
| ellipses r= 5  subpix=5 |    1.60 us/aper |   21.48 us/aper |  13.45 |
| ellipses r= 5  exact    |    3.90 us/aper |   29.29 us/aper |   7.51 |
```

Maybe more benchmark for extraction is needed (photutils's detect_sources,
source_properties), but it's already tempting to use sep over photutils given
the purpose is only to extract the objects, not photometry.

Pixel convention
================
Note that sep also uses same pixel notation as photutils: pixel 0 covers -0.5
to +0.5. Simple test code:
>>> test = np.zeros((15, 15))
>>> test[7, 8] = 1
>>> test[8, 9] = 2
>>> obj = pd.DataFrame(sep.extract(test, 0))
>>> for c in obj.columns:
>>>     print(f"{c}: {obj[c].values}")
>>> # x: [8.66666667]
>>> # y: [7.66666667]
"""
from warnings import warn

import numpy as np
import pandas as pd

from .util import bezel_mask

try:
    import sep
except ImportError:
    warn("Package sep is not installed. Some functions will not work.")


__all__ = ['sep_back', 'sep_extract', 'sep_flux_auto']

sep_default_kernel = np.array([[1.0, 2.0, 1.0],
                               [2.0, 4.0, 2.0],
                               [1.0, 2.0, 1.0]], dtype=np.float32)


def _sanitize_byteorder(data):
    if data.dtype.byteorder == '>':
        return data.byteswap().newbyteorder()
    else:
        return data


def sep_back(
        data,
        mask=None,
        maskthresh=0.0,
        filter_threshold=0.0,
        box_size=(64, 64),
        filter_size=(3, 3)
):
    """
    Notes
    -----
    This includes `sep`'s `Background`. Equivalent processes in photutils may
    include `Background2D`.

    Parameters
    ----------
    data : CCDData or array-like
        The 2D array from which to estimate the background.

    mask : 2-d `~numpy.ndarray`, optional
        Mask array.

    maskthresh : float, optional
        Only in `sep`. The effective mask will be ``m = (mask.astype(float) >
        maskthresh)``::

          * **sep**: Mask threshold. This is the inclusive upper limit on the
            mask value in order for the corresponding pixel to be unmasked. For
            boolean arrays, False and True are interpreted as 0 and 1, resp.
            Thus, given a threshold of zero, `True` corresponds to masked and
            `False` corresponds to unmasked.
          * **photutils**: In photutils, sigma clipping is used (need check).

    filter_threshold : int, optional
        Name in photutils; `fthresh` in the oritinal sep. Default is 0.0 ::

        * **sep**: Filter threshold. Default is 0.0.
        * **photutils**: The threshold value for used for selective median
          filtering of the low-resolution 2D background map. The median filter
          will be applied to only the background meshes with values larger than
          `filter_threshold`.  Set to `None` to filter all meshes (default).

    box_size : int or array_like (int)
        Name in photutils; `bh`, `bw` order in sep. Default is ``(64, 64)``::

          * **sep**: Size of background boxes in pixels. Default is 64.
          * **photutils**: The box size along each axis. If `box_size` is a
            scalar then a square box of size `box_size` will be used. If
            `box_size` has two elements, they should be in ``(ny, nx)`` order.
            For best results, the box shape should be chosen such that the
            `data` are covered by an integer number of boxes in both
            dimensions. When this is not the case, see the `edge_method`
            keyword for more options.

    filter_size : int or array_like (int), optional
        Name in photutils; `fh`, `fw` order in sep. Default is ``(3, 3)``.::

          * **sep**: Filter width and height in boxes. Default is 3.
          * **photutils**: The window size of the 2D median filter to apply to
            the low-resolution background map. If `filter_size` is a scalar
            then a square box of size `filter_size` will be used. If
            `filter_size` has two elements, they should be in ``(ny, nx)``
            order. A filter size of ``1`` (or ``(1, 1)``) means no filtering.

    Returns
    -------
    bkg : sep.Background
        Use `bkg.back()` and `bkg.rms()` to get the background and rms error.
        All other methods/attributes include `bkg.subfrom()`, `bkg.globalback`,
        and `bkg.globalrms`.
    """
    try:
        data = _sanitize_byteorder(data)
    except AttributeError:  # if data is in CCDData...
        data = _sanitize_byteorder(data.data)

    if mask is not None:
        mask = np.asarray(mask).astype(bool)

    box_size = np.atleast_1d(box_size)
    if len(box_size) == 1:
        box_size = np.repeat(box_size, 2)
    box_size = (min(box_size[0], data.shape[0]), min(box_size[1], data.shape[1]))

    filter_size = np.atleast_1d(filter_size)
    if len(filter_size) == 1:
        filter_size = np.repeat(filter_size, 2)

    kw = dict(mask=mask, bw=box_size[1], bh=box_size[0],
              fw=filter_size[1], fh=filter_size[0],
              maskthresh=maskthresh, fthresh=filter_threshold)
    try:
        bkg = sep.Background(data, **kw)
    except ValueError:  # Non-native byte order
        data = data.byteswap().newbyteorder()
        try:
            bkg = sep.Background(data, **kw)
        except ValueError:  # e.g., int16 not supported
            bkg = sep.Background(data.astype('float32'), **kw)

    return bkg


def sep_extract(
        data,
        thresh,
        bkg=None,
        mask=None,
        maskthresh=0.0,
        err=None,
        var=None,
        pos_ref=None,
        sort_by=None,
        sort_ascending=True,
        bezel_x=[0, 0],
        bezel_y=[0, 0],
        gain=None,
        minarea=5,
        maxarea=None,
        filter_kernel=sep_default_kernel,
        filter_type='matched',
        deblend_nthresh=32,
        deblend_cont=0.005,
        clean=True,
        clean_param=1.0
):
    """
    Notes
    -----
    This includes `sep`'s `extract`. Equivalent processes in photutils may
    include `detect_sources` and `source_properties`. Maybe we can use
    ``extract(data=data, err=err, thresh=3)`` for a snr > 3 extraction.

    Parameters
    ----------
    data : CCDData or array-like
        The 2D array from which to estimate the background.

    thresh : float, optional.
        Only in sep. Threshold pixel value for detection. If an `err` or `var`
        array is **not** given, this is interpreted as an absolute threshold.
        If `err` or `var` is given, this is interpreted as a relative
        threshold: the absolute threshold at pixel (j, i) will be ``thresh *
        err[j, i]`` or ``thresh * sqrt(var[j, i])``. Note: If you want to give
        pixel-wise threshold, make the `err` with such threshold values and set
        ``thresh = 1``.

    bkg : sep.Background object, numeric, ndarray or `None`
        The `sep.Background` object used to extract sky and sky rms.

    mask : `~numpy.ndarray`, optional
        Mask array. `True` values, or numeric values greater than `maskthresh`,
        are considered masked. Masking a pixel is equivalent to setting data to
        zero and noise (if present) to infinity.

    maskthresh : float, optional
        Mask threshold. This is the inclusive upper limit on the mask value in
        order for the corresponding pixel to be unmasked. For boolean arrays,
        `False` and `True` are interpreted as 0 and 1, respectively. Thus,
        given a threshold of zero, True corresponds to masked and `False`
        corresponds to unmasked.
        Default is ``0.0``.

    err, var : float or `~numpy.ndarray`, optional
        Error *or* variance (specify at most one). This can be used to specify
        a pixel-by-pixel detection threshold; see `thresh`.

    pos_ref : `None`, list-like of two floats, optional.
        If not `None`, it must be the (x, y) position of the reference point.
        The returned `obj` will have ``'dist_ref'`` column which is the
        distance of the object's position (``sqrt((obj["x"] - pos_ref[0])**2 +
        (obj["y"] - pos_ref[1])**2)``) and sorted based on this by default (see
        `sort_by`).

    sort_by : str, optional.
        The column name to sort the output. If `pos_ref` is not `None`, a new
        column is added to the `sep` results, called ``"dist_ref"`` and
        `sort_by` is based on this column by default. Otherwise, it should be a
        name of column in `sep` result.

    sort_ascending : bool, optional.
        Sort ascending vs. descending. Specify list for multiple sort orders.
        If this is a list of bools, must match the length of the `sort_by`.

    bezel_x, bezel_y : int, float, 2-array-like, optional
        The bezel (border width) for x and y axes. If array-like, it should be
        ``(lower, upper)``. Mathematically put, only objects with center
        ``(bezel_x[0] + 0.5 < center_x) & (center_x < nx - bezel_x[1] - 0.5)``
        (similar for y) will be selected. If you want to keep some stars
        outside the edges, put negative values (e.g., ``-5``).

    gain : float, optional
        Conversion factor between data array units and poisson counts. This
        does not affect detection; it is used only in calculating Poisson noise
        contribution to uncertainty param(eters such as `errx2`. If not given,
        no Poisson noise will be added.

    minarea : int, optional
        Minimum number of pixels required for an object. Default is 5.

    maxarea : int, optional
        Maximum number of pixels required for an object. It is useful to reject
        any artifact (too low threshold) or extended source. Default is `None`.

    filter_kernel : `~numpy.ndarray` or None, optional
        Filter kernel used for on-the-fly filtering (used to enhance
        detection). Default is a 3x3 array: [[1,2,1], [2,4,2], [1,2,1]]. Set to
        `None` to skip convolution.

    filter_type : {'matched', 'conv'}, optional
        Filter treatment. This affects filtering behavior when a noise array is
        supplied. ``'matched'`` (default) accounts for pixel-to-pixel noise in
        the filter kernel. ``'conv'`` is simple convolution of the data array,
        ignoring pixel-to-pixel noise across the kernel. ``'matched'`` should
        yield better detection of faint sources in areas of rapidly varying
        noise (such as found in coadded images made from semi-overlapping
        exposures). The two options are equivalent when noise is constant.

    deblend_nthresh : int, optional
        Number of thresholds used for object deblending. Default is 32.

    deblend_cont : float, optional
        Minimum contrast ratio used for object deblending. Default is 0.005. To
        entirely disable deblending, set to 1.0.

    clean : bool, optional
        Perform cleaning? Default is True.

    clean_param : float, optional
        Cleaning parameter (see SExtractor manual). Default is 1.0.

    Returns
    -------
    obj, segm

    Example
    -------
    To test:
    >>> def bkg(data, mask, th):
    >>>     return sep.Background(data, mask=mask, maskthresh=th).back()
    >>> np.random.seed(1234)
    >>> data = np.random.normal(scale=100, size=(1000, 1000))
    >>> bs = []
    >>> bs.append(bkg(data, mask=None, th=0))
    >>> bs.append(bkg(data, mask=data, th=-1.e+5))
    >>> bs.append(bkg(data, mask=data, th=0))
    >>> bs.append(bkg(data, mask=data, th=+1.e+5))
    >>> bs.append(bkg(data, mask=data.astype(bool), th=+1.e+5))
    >>> np.testing.assert_array_almost_equal(bs[1], bs[2])
    >>> fig, axs = plt.subplots(2, 4, figsize=(8, 5))
    >>> for idx, b in enumerate(bs):
    >>>     ax = axs[idx%2, idx//2]
    >>>     ax.imshow(b, origin='lower')
    >>>     ax.set(title=idx)
    >>> plt.tight_layout()
    >>> plt.show()
    """
    if err is not None and var is not None:
        raise ValueError("Upto one of `err` and `var` can be given.")

    try:
        data = _sanitize_byteorder(data)
    except AttributeError:  # if data is in CCDData...
        data = _sanitize_byteorder(data.data)

    if mask is not None:
        mask = np.asarray(mask).astype(float)

    if bkg is None:
        data_skysub = data
        # No need to further update `var` or `err`.
    elif isinstance(bkg, (int, float, np.ndarray)):
        data_skysub = data - bkg
        # No need to further update `var` or `err`.
    else:
        data_skysub = data - bkg.back()
        if var is not None:  # Then err is None (see above)
            var = var + bkg.rms()**2
        elif err is not None:  # Then var is None (see above)
            err = np.sqrt(err**2 + bkg.rms()**2)

    obj, segm = sep.extract(
        data_skysub,
        thresh=thresh,
        err=err,
        var=var,
        mask=mask,
        maskthresh=maskthresh,
        minarea=minarea,
        filter_kernel=filter_kernel,
        filter_type=filter_type,
        deblend_nthresh=deblend_nthresh,
        deblend_cont=deblend_cont,
        clean=clean,
        clean_param=clean_param,
        gain=gain,
        segmentation_map=True
    )
    obj = pd.DataFrame(obj)
    n_original = len(obj)

    ny, nx = data.shape
    mask = bezel_mask(obj['x'], obj['y'], nx, ny, bezel_x=bezel_x, bezel_y=bezel_y)
    if maxarea is not None:
        mask = mask & (obj["npix"] <= maxarea)
    obj = obj[~mask]

    # use 1-indexing for the ``segm_label``
    obj = obj.reset_index()
    obj = obj.rename(columns={'index': 'segm_label'})
    obj['segm_label'] += 1
    # log the original input threshold
    obj.insert(loc=1, column='thresh_raw', value=thresh)

    if pos_ref is not None:
        pos_ref = np.array(pos_ref).flatten()
        if pos_ref.size != 2:
            raise ValueError(
                f"pos_ref must have the size of two (now it is {pos_ref.size})."
            )
        dist_ref = np.sqrt((obj["x"] - pos_ref[0])**2 + (obj["y"] - pos_ref[1])**2)
        obj.insert(loc=1, column='dist_ref', value=dist_ref)
        sort_by = "dist_ref"

    if sort_by is not None:
        obj = obj.sort_values(sort_by, ascending=sort_ascending).reset_index(drop=True)

    # Set segm value to 0 (False) if removed by bezel.
    if len(obj) < n_original:
        segm_survived = np.isin(segm, obj['segm_label'].values)
        segm[~segm_survived] = 0

    return obj, segm


def sep_flux_auto(data, sepext, err=None, phot_autoparams=(2.5, 3.5)):
    """ Calculate FLUX_AUTO
    # https://sep.readthedocs.io/en/v1.0.x/apertures.html#equivalent-of-flux-auto-e-g-mag-auto-in-source-extractor
    """

    sepx, sepy = sepext['x'], sepext['y']
    sepa, sepb = sepext['a'], sepext['b']
    septh = sepext['theta']

    r_kron, nrej_k = sep.kron_radius(data, sepx, sepy, sepa, sepb, septh, 6.0)
    fl, dfl, nrej = sep.sum_ellipse(data, sepx, sepy, sepa, sepb, septh,
                                    r=phot_autoparams[0]*r_kron, err=err, subpix=1)
    nrej |= nrej_k  # combine flags into 'flag'

    r_min = phot_autoparams[1]  # R_min = 3.5
    use_circle = r_kron * np.sqrt(sepa*sepb) < r_min
    cfl, cdfl, nrej_c = sep.sum_circle(
        data,
        sepx[use_circle],
        sepy[use_circle],
        r_min,
        err=err,
        subpix=1
    )
    fl[use_circle] = cfl
    dfl[use_circle] = cdfl
    nrej[use_circle] = nrej_c
    return fl, dfl, nrej


# def sep_find_obj(
#         ccd, mask=None, err=None, var=None,
#         thresh_tests=[30, 20, 10, 6, 5, 4, 3],
#         bezel_x=None, bezel_y=None, box_size=(64, 64),
#         filter_size=(12, 12), deblend_cont=1, minarea=100, verbose=True,
#         update_header=True, **extract_kw
# ):
#     """
#     Parameters
#     ----------
#     ccd : CCDData or ndarray.
#         The CCD or ndarray to find object.

#     thresh_tests : list-like of float, optional.
#         The SNR thresholds to be used for finding the object. It is
#         first sorted in descending order, and if more than one object is
#         found, that value is used.

#     bezel_x, bezel_y : int, float, list of such, optional.
#         The x and y bezels, in ``[lower, upper]`` convention.

#     box_size : int or array-like (int) optional.
#         The background smooting box size. Default is ``(64, 64)``
#         for NIC. **Note**: If array-like, order must be ``[height,
#         width]``, i.e., y and x size.

#     filter_size : int or array-like (int) optional.
#         The 2D median filter size. Default is ``(12, 12)`` for NIC.
#         **Note**: If array-like, order must be ``[height, width]``,
#         i.e., y and x size.

#     minarea : int, optional
#         Minimum number of pixels required for an object. Default is
#         100 for NIC.

#     deblend_cont : float, optional
#         Minimum contrast ratio used for object deblending. To
#         entirely disable deblending, set to 1.0 (default). Default of
#         sep was 0.005.

#     # gauss_fbox : int, float, array-like of such, optional.
#     #     The fitting box size to fit a Gaussian2D function to the
#     #     objects found by `sep`. This is done to automatically set
#     #     aperture sizes of the object.

#     Returns
#     -------

#     Note
#     ----
#     This includes `sep`'s `extract` and `background`.
#     Equivalent processes in photutils may include `detect_sources`
#     and `source_properties`, and `Background2D`, respectively.

#     Example
#     -------
#     >>>
#     """

#     if isinstance(ccd, CCDData):
#         _arr = ccd.data.copy()
#         _mask = ccd.mask
#         if update_header:
#             try:
#                 from ysfitsutilpy.hdrutil import add2hdr
#             except ImportError:
#                 raise ImportError("ysfitsutilpy is needed for update_header.")
#     else:
#         _arr = np.array(ccd)
#         _mask = None
#         update_header = False  # override
#         if update_header and verbose:
#             warn("Given array, not CCDData. Header will not be updated.")

#     if _mask is None:
#         _mask = np.zeros_like(_arr, dtype=bool)

#     if mask is not None:
#         _mask = _mask | mask

#     bkg_kw = dict(mask=_mask, maskthresh=0.0, filter_threshold=0.0,
#                   box_size=box_size, filter_size=filter_size)

#     sepv = sep.__version__
#     s_bkg = f"Background estimated from sep (v {sepv}) with {bkg_kw}."

#     _t = Time.now()
#     bkg = sep_back(_arr, **bkg_kw)
#     if update_header:
#         add2hdr(ccd.header, 'h', s_bkg, verbose=verbose, t_ref=_t)

#     thresh_tests = np.sort(np.atleast_1d(thresh_tests))[::-1]
#     for thresh in thresh_tests:
#         ext_kw = dict(bkg=bkg, err=err, mask=_mask, thresh=thresh,
#                       minarea=minarea,
#                       deblend_cont=deblend_cont, bezel_x=bezel_x,
#                       bezel_y=bezel_y, **extract_kw)
#         s_obj = f"Objects found from sep (v {sepv}) with {ext_kw}."

#         _t = Time.now()
#         obj, seg = sep_extract(_arr, **ext_kw)
#         if update_header:
#             add2hdr(ccd.header, 'h', s_obj, verbose=verbose, t_ref=_t)

#         nobj = len(obj)
#         ccd.header["NOBJ-SEP"] = (nobj, "Number of objects found from SEP.")

#     if nobj < 1:
#         warn("No object found!", Warning)
#     elif nobj > 1:
#         # Sort obj such that the 0-th is our target.
#         ny, nx = ccd.data.shape
#         obj['_r'] = np.sqrt((obj['x'] - nx/2)**2 + (obj['y'] - ny/2)**2)
#         obj.sort_values('_r', inplace=True)

#         if update_header:
#             s = ("{} objects found; Only the one closest to the FOV center "
#                  + "(segmentation map label = {}) will be used.")
#             s = s.format(nobj, obj['segm_label'].values[0])
#             add2hdr(ccd.header, 'h', s, verbose=verbose)

#     return bkg, obj, seg
