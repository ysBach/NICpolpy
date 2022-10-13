import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter
from .ysphotutilpy4nicpolpy import sep_back, sep_extract


__all__ = ["quick_detect_obj"]


def quick_detect_obj(
        image,
        mask=None,
        thresh_ksig=3,
        scale_maxiters=5,
        gsigma=6,
        box_size=(50, 50),
        filter_size=(5, 5),
        minarea=50,
        verbose=0,
):
    # Anyway the flux from sep will not be used, so make a Gaussian convolved image
    # to easily find the target. If I don't do this, I see some cases when centroid
    # position is critically wrong for fainter object.
    # NOTE: Gaussian convolved pix value =
    #   best fit Gaussian amplitude at the pixel * const + const
    #   (see StetsonPB 1987 PASP 99 191, eq. (1))
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




# def remove_fringe(
#         ccd,
#         detect_gsigma=6,
#         detect_ksig=3,
#         detect_minarea=50,
#         detect_box_size=(50, 50),
#         detect_filter_size=(5, 5),
#         box_size=(10, 10),
#         filter_size=(5, 5),
#         sigclip_kw=dict(sigma=2, maxiters=5),
#         model="chebyshev2d",
#         verbose=0,
# ):
#     img0 = ccd.data
#     filt = ccd.header['FILTER']
#     detkw = dict(thresh_ksig=detect_ksig, minarea=detect_minarea, gsigma=detect_gsigma,
#                  box_size=detect_box_size, filter_size=detect_filter_size)
#     # Anyway the *flux* from sep will not be used, so make a Gaussian convolved image
#     # to easily find the target. If I don't do this, I see some cases when centroid
#     # position is critically wrong for fainter object.
#     # NOTE: Gaussian convolved pix value = best fit Gaussian amplitude at the pixel*
#     #  const + const ---- see StetsonPB 1987 PASP 99 191, eq. (1)
#     obj0, segm0, _ = quick_detect_obj(img0, **detkw, verbose=verbose)
#     if obj0 is None:
#         # pos_ref_shifted = np.array([0, 0])
#         # img0 = np.empty_like(img0)*np.nan  # To make ``phot`` table with all NaN values.
#         # err = None
#         # pos = pos_ref_shifted
#         # peak = np.nan
#         # mask_sat = None
#         mask = img0 > SATLEVEL[filt]
#     else:
#         obj0 = obj0.loc[obj0["flux"] == obj0["flux"].max()]
#         mask_sat = img0 > SATLEVEL[filt]
#         mask = mask_sat | (segm0 > 0)

#     _bkg_kw = dict(mask=mask, box_size=box_size, filter_size=filter_size)

#     verti = vertical_correct_remaining(img0, mask=mask, box_size=box_size,
#                                        filter_size=filter_size, sigclip_kw=sigclip_kw)
#     imgs = [img0 - verti]  # subtract vertical pattern from the original image
#     grids = gridding(imgs[0])[0]  # this is a "constant" ndarray.
#     _obj, _segm, _imgd = quick_detect_obj(imgs[0], mask=mask, **detkw)
#     # FIXME: if _obj is None: ...

#     objs = [_obj.loc[_obj["flux"] == _obj["flux"].max()]]
#     segms = [_segm]
#     imgds = [_imgd]
#     poss = [np.array((objs[0].iloc[0]['x'], objs[0].iloc[0]['y']))]
#     masks = [mask_sat | (segms[0] > 0)]
#     fitted_2, fitter_2, _, _, _ = fit_model_iter(
#         model, imgs[0], mask=masks[0], maxiters=10, fitter_name="lin",
#         x_degree=1, y_degree=9, full=True
#     )
#     bkgs = [fitted_2(*grids[::-1])]
#     _bkgmed = np.median(bkgs[0])
#     return
