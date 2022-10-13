from warnings import warn

import numpy as np
import bottleneck as bn
from astropy.nddata import CCDData
from astropy.stats import sigma_clip
from astropy.table import Table

__all__ = ['quick_sky_circ', 'sky_fit', "annul2values"]


def quick_sky_circ(ccd, pos, r_in=10, r_out=20):
    """ Estimate sky with crude presets
    """
    from photutils.aperture import CircularAnnulus
    annulus = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
    return sky_fit(ccd, annulus)


def sky_fit(
        ccd,
        annulus,
        mask=None,
        method='mode',
        mode_option='sex',
        std_ddof=1,
        **kwargs
):
    """ Estimate the sky value from image and annulus.

    Parameters
    ----------
    ccd: `~astropy.nddata.CCDData`
        The image data to extract sky at given annulus.

    annulus: annulus object
        The annulus which will be used to estimate sky values.

    method : {"mean", "median", "mode"}, optional.
        The method to estimate sky value. You can give options to "mode" case;
        see `mode_option`. "mode" is analogous to Mode Estimator Background
        of photutils.

    mode_option : {"sex", "IRAF", "MMM"}, optional.
        Three options::

          * sex  == (med_factor, mean_factor) = (2.5, 1.5)
          * IRAF == (med_factor, mean_factor) = (3, 2)
          * MMM  == (med_factor, mean_factor) = (3, 2)

        where ``msky = (med_factor * med) - (mean_factor * mean)``.

    std_ddof : int, optional.
        The "delta-degrees of freedom" for sky standard deviation calculation.

    kwargs : dict, optional.
        The keyword arguments for sigma-clipping.

    Returns
    -------
    skytable: `~astropy.table.Table`
        The table of the followings.

    msky : float
        The estimated sky value within the all_sky data, after sigma clipping.

    ssky : float
        The sample standard deviation of sky value within the all_sky data,
        after sigma clipping.

    nsky : int
        The number of pixels which were used for sky estimation after the
        sigma clipping.

    nrej : int
        The number of pixels which are rejected after sigma clipping.
    """
    def _sstd(arr, ddof=0, axis=None):
        return np.sqrt(arr.size/(arr.size - ddof))*bn.nanstd(arr, axis=axis)

    skydicts = []
    skys = annul2values(ccd, annulus, mask=mask)

    for i, sky in enumerate(skys):
        skydict = {}
        sky_clip = sigma_clip(sky, masked=False, stdfunc=_sstd, **kwargs)
        std = np.std(sky_clip, ddof=std_ddof)
        nrej = sky.size - sky_clip.size
        nsky = sky.size - nrej
        if nrej < 0:
            raise ValueError('nrej < 0: check the code')

        if nrej > nsky:  # rejected > survived
            warn('More than half of the pixels rejected.')

        if method.lower() == 'mode':
            mean = np.mean(sky_clip)
            med = np.median(sky_clip)
            if mode_option.lower() == 'sex':
                skydict["msky"] = med if (mean - med)/std > 0.3 else (2.5*med) - (1.5*mean)
            elif mode_option.lower() == 'iraf':
                skydict["msky"] = mean if (mean < med) else 3*med - 2*mean
            elif mode_option.lower() == 'mmm':
                skydict["msky"] = 3*med - 2*mean
            else:
                raise ValueError('mode_option not understood')
        elif method.lower() == 'mean':
            skydict["msky"] = np.mean(sky_clip)
        elif method.lower() == 'median':
            skydict["msky"] = np.median(sky_clip)


        skydict['ssky'] = std
        skydict['nsky'] = nsky
        skydict['nrej'] = nrej
        skydicts.append(skydict)
    skytable = Table(skydicts)
    # skytable["msky"].unit = u.adu / u.pix
    # skytable["ssky"].unit = u.adu / u.pix
    # skytable["nrej"].unit = u.pix
    return skytable


def annul2values(
        ccd,
        annulus,
        mask=None
):
    ''' Extracts the pixel values from the image with annuli

    Parameters
    ----------
    ccd : CCDData, ndarray
        The image which the annuli in ``annulus`` are to be applied.
    annulus : `~photutils.Aperture` object
        The annuli to be used to extract the pixel values.
    # fill_value: float or nan
    #     The pixels which are masked by ``ccd.mask`` will be replaced with
    #     this value.

    Returns
    -------
    values: list of ndarray
        The list of pixel values. Length is the same as the number of annuli in
        ``annulus``.
    '''
    if isinstance(ccd, CCDData):
        _ccd = ccd.copy()
        _arr = _ccd.data
        _mask = _ccd.mask
    else:  # ndarrayk
        _arr = np.array(ccd)
        _mask = None

    if _mask is None:
        _mask = np.zeros_like(_arr).astype(bool)

    if mask is not None:
        _mask |= mask

    an_masks = annulus.to_mask(method='center')
    try:
        if annulus.isscalar:  # as of photutils 0.7
            an_masks = [an_masks]
    except AttributeError:
        pass

    # FIXME: use the new an_mask.get_values() introduced in 2021 Feb
    # values = []
    # for i, an_mask in enumerate(an_masks):
    #     # result identical to an.data itself, but just for safety...
    #     in_an = (an_mask.data == 1).astype(float)  # float for NaN below
    #     # replace to NaN for in_an=0, because sometimes pixel itself is 0...
    #     in_an[in_an == 0] = np.nan
    #     skys_i = an_mask.multiply(_arr, fill_value=np.nan) * in_an
    #     ccdmask_i = an_mask.multiply(_mask, fill_value=False)
    #     mask_i = (np.isnan(skys_i) + ccdmask_i).astype(bool)
    #     # skys_i = an.multiply(_arr, fill_value=np.nan)
    #     # sky_xy = np.nonzero(an.data)
    #     # sky_all = mask_im[sky_xy]
    #     # sky_values = sky_all[~np.isnan(sky_all)]
    #     # values.append(sky_values)
    #     values.append(np.array(skys_i[~mask_i].ravel(), dtype=_arr.dtype))

    return [am.get_values(_arr, _mask) for am in an_masks]
