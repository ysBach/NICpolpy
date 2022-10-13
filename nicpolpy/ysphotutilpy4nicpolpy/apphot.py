from warnings import warn

import numpy as np
from astropy import units as u
from astropy.nddata import CCDData
from astropy.table import QTable
from photutils import aperture_photometry
from photutils.aperture import Aperture

from .background import sky_fit

__all__ = ["apphot_annulus"]


# TODO: Put centroiding into this apphot_annulus ?
# TODO: use variance instead of error (see photutils 0.7)
# TODO: one_aperture_per_row : bool, optional.
# `photutils.aperture_photometry` produces 1-row result if multiple radii aperture is given with column
# names starting from ``aperture_sum_0`` and ``aperture_sum_err_0``.
def apphot_annulus(
        ccd,
        aperture,
        annulus=None,
        t_exposure=None,
        exposure_key="EXPTIME",
        error=None,
        mask=None,
        sky_keys=None,
        sky_min=None,
        aparea_exact=False,
        npix_mask_ap=2,
        verbose=False,
        pandas=True,
        **kwargs
):
    ''' Do aperture photometry using annulus.

    Parameters
    ----------
    ccd : CCDData
        The data to be photometried. Preferably in ADU.

    aperture, annulus : aperture and annulus object or list of such.
        The aperture and annulus to be used for aperture photometry. For
        multi-position aperture, just use, e.g., ``CircularAperture(positions,
        r=10)``. For multiple radii, use, e.g., ``[CircularAperture(positions,
        r=r) for r in radii]``.

    exposure_key : str
        The key for exposure time. Together with `t_exposure_unit`, the
        function will normalize the signal to exposure time. If `t_exposure`
        is not None, this will be ignored.

    error : array-like or Quantity, optional
        See `~photutils.aperture_photometry` documentation. The pixel-wise
        error map to be propagated to magnitued error.

    sky_keys : dict
        args/kwargs of `sky_fit`. If `None`(default), 3-sigma 5-iters clipping
        with ``ddof=1`` is performed, and then the modal sky value is estimated
        by SExtractor estimator (mode = 2.5median - 1.5mean). To use different
        parameters, give those kwargs to `sky_keys` as dict.

    sky_min : float
        The minimum value of the sky to be used for sky subtraction.

    aparea_exact : bool, optional
        Whether to calculate the aperture area (``'aparea'`` column) exactly or
        not. If `True`, the area outside the image (aperture goes outside the
        CCD) **and** those specified by mask (aperture contains masked pixels)
        are not counted in the ``aparea`` value. It is important to prevent
        *oversubtracting* sky values. Default is `False`.

    npix_mask_ap : int, optional.
        If the number of masked pixels in the aperture is equal to or greater
        than `npix_mask_ap`, the column ``"bad"`` will be marked as ``1``.

        ..note::
            Currently it is not checked for annulus (works only for aperture)

    pandas : bool, optional.
        Whether to convert to `~pandas.DataFrame`.

    **kwargs :
        kwargs for `~photutils.aperture_photometry`.

    Returns
    -------
    phot_f: astropy.table.Table
        The photometry result.

    bad code
      * 1 (2^0) : number of masked pixels ``> npix_mask_ap`` within aperture.
      * 2 (2^1) : number of masked pixels ``> npix_mask_an`` within annulus.
        (not implemented yet)
    '''
    def _propagate_ccdmask(ccd, additional_mask=None):
        ''' Propagate the CCDData's mask and additional mask from ysfitsutilpy.

        Parameters
        ----------
        ccd : CCDData, ndarray
            The ccd to extract mask. If ndarray, it will only return a copy of
            `additional_mask`.

        additional_mask : mask-like, None
            The mask to be propagated.

        Notes
        -----
        The original ``ccd.mask`` is not modified. To do so,
        >>> ccd.mask = propagate_ccdmask(ccd, additional_mask=mask2)
        '''
        from copy import deepcopy
        if additional_mask is None:
            try:
                mask = ccd.mask.copy()
            except AttributeError:  # i.e., if ccd.mask is None
                mask = None
        else:
            if ccd.mask is None:
                mask = deepcopy(additional_mask)
            else:
                mask = ccd.mask | additional_mask
            # except (TypeError, AttributeError):  # i.e., if ccd.mask is None:
        return mask

    _ccd = ccd.copy()

    if isinstance(_ccd, CCDData):
        _arr = _ccd.data
        _mask = _propagate_ccdmask(_ccd, additional_mask=mask)
        if t_exposure is None:
            try:
                t_exposure = _ccd.header[exposure_key]
            except (KeyError, IndexError):
                t_exposure = 1
                if verbose:
                    warn("The exposure time info not given and not found from the header"
                         + f" ({exposure_key}). Setting it to 1 sec.")
    else:  # ndarray
        _arr = np.array(_ccd)
        _mask = mask
        if t_exposure is None:
            t_exposure = 1
            if verbose:
                warn("The exposure time info not given. Setting it to 1 sec.")

    # [multi-position, same radius] case results in ONE Aperture object with
    # multiple positions.
    # If this Aperture is turned into list, photutils (1.0) gives ValueError:
    #   ValueError: Input apertures must all have identical positions.
    # [single-position, multi-radius] case, the user will input MANY Aperture
    # objects in a list.
    #   In this case, the aperture must be flattened into a list.
    if not isinstance(aperture, Aperture):
        aperture = np.array(aperture).flatten()
        n_apertures = aperture.size
    else:
        try:
            n_apertures = len(aperture)
        except TypeError:
            # photutils 0.7+ has .isscalar to test this but I want to accept
            # older photutils too...
            n_apertures = 1

    flag_bad = True
    nbads = []
    bads = []
    if _mask is None:
        flag_bad = False
        bads = [0]*n_apertures
        nbads = [0]*n_apertures
        _mask = np.zeros_like(_arr).astype(bool)

    if mask is not None:
        _mask |= mask

    if error is not None:
        if verbose:
            print("Ignore any uncertainty extension in the original CCD, "
                  + "and use provided uncertainty map.")
        err = error.copy()
        if isinstance(err, CCDData):
            err = err.data
    else:
        try:
            err = _ccd.uncertainty.array
        except AttributeError:
            if verbose:
                warn("Uncertainty extension not found in ccd. Will not calculate errors.")
            err = np.zeros_like(_arr)

    if aparea_exact:
        # What this does is basically identical to area_overlap:
        # https://photutils.readthedocs.io/en/stable/api/photutils.aperture.PixelAperture.html#photutils.aperture.PixelAperture.area_overlap
        # I am just afraid of testing the code.
        _ones = np.ones_like(_arr)
        _area = aperture_photometry(_ones, aperture, mask=_mask, **kwargs)
        aparea = np.array([_area[c][0] for c in _area.colnames
                           if c.startswith("aperture_sum")])
    else:
        try:
            if n_apertures != 1:
                aparea = np.array([ap.area for ap in aperture])
            else:
                aparea = [aperture.area]
        except TypeError:  # prior to photutils 0.7
            if n_apertures != 1:
                aparea = np.array([ap.area() for ap in aperture])
            else:
                aparea = [aperture.area()]

    _phot = aperture_photometry(_arr, aperture, mask=_mask, error=err, **kwargs)
    # If we use ``_ccd``, photutils deal with the unit, and the lines below
    # will give a lot of headache for units. It's not easy since aperture can
    # be pixel units or angular units (Sky apertures).
    # ysBach 2018-07-26

    if flag_bad:
        try:
            for ap in aperture:
                apmask = ap.to_mask(method='exact')
                nbad = np.count_nonzero(apmask.multiply(_mask))
                bad = 1 if nbad > npix_mask_ap else 0
                nbads.append(nbad)
                bads.append(bad)
        except TypeError:  # scalar aperture
            apmask = aperture.to_mask(method='exact')
            nbad = np.count_nonzero(apmask.multiply(_mask))
            bad = 1 if nbad > npix_mask_ap else 0
            nbads.append(nbad)
            bads.append(bad)

    if annulus is not None:
        if sky_keys is None:
            skys = sky_fit(_arr, annulus, mask=_mask, method='mode',
                           mode_option='sex', sigma=3, maxiters=5, std_ddof=1)
        else:
            skys = sky_fit(_arr, annulus, mask=_mask, **sky_keys)
        for c in skys.colnames:
            _phot[c] = skys[c]
    else:
        _phot['msky'] = 0
        _phot['nsky'] = 1
        _phot['nrej'] = 0
        _phot['ssky'] = 0

    if isinstance(aperture, (list, tuple, np.ndarray)):
        # If multiple apertures at each position
        # Convert aperture_sum_xx columns into 1-column...
        n = len(aperture)
        apsums = []
        aperrs = []
        phot = QTable(meta=_phot.meta)

        for i, c in enumerate(_phot.colnames):
            if not c.startswith("aperture"):  # all other columns
                phot[c] = [_phot[c][0]]*n
            elif c.startswith("aperture_sum_err"):  # aperture_sum_err_xx
                aperrs.append(_phot[c][0])
            else:  # aperture_sum_xx
                apsums.append(_phot[c][0])

        phot["aperture_sum"] = apsums
        if aperrs:
            phot["aperture_sum_err"] = aperrs
        # I guess we should not have this..? :
        # else:
        #     phot = _phot

    else:
        phot = _phot

    if sky_min is not None:
        phot["msky"][phot["msky"] < sky_min] = sky_min

    phot['aparea'] = aparea
    phot["source_sum"] = phot["aperture_sum"] - aparea * phot["msky"]

    # see, e.g., http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?radprof.hlp
    # Poisson + RDnoise (Poisson includes signal + sky + dark) :
    var_errmap = phot["aperture_sum_err"]**2
    # Sum of aparea Gaussians (kind of random walk):
    var_skyrand = aparea * phot["ssky"]**2
    # The CLT error (although not correct, let me denote it as "systematic"
    # error for simplicity) of the mean estimation is ssky/sqrt(nsky), and that
    # is propagated for aparea pixels, so we have std = aparea*ssky/sqrt(nsky), so
    # variance is:
    # var_sky = (aparea * phot['ssky'])**2 / phot['nsky']
    # This error term is used in IRAF APPHOT, but this is wrong and thus
    # ignored here.

    phot["source_sum_err"] = np.sqrt(var_errmap + var_skyrand)
    snr = np.array(phot['source_sum'])/np.array(phot["source_sum_err"])
    phot["mag"] = -2.5*np.log10(phot['source_sum']/t_exposure)
    phot["merr"] = 2.5/np.log(10)*(1/snr)
    phot["snr"] = snr
    phot["bad"] = bads
    phot["nbadpix"] = nbads

    if pandas:
        return phot.to_pandas()
    else:
        return phot


# TODO: make this...
def apphot_ellip_sep(ccd, x, y, a, a_in, a_out, bpa=1, theta=0, t_exposure=None,
                     exposure_key="EXPTIME", error=None, mask=None, sky_keys={}, aparea_exact=False,
                     t_exposure_unit=u.s, verbose=False, pandas=False, **kwargs):
    ''' Similar to apphot_annulus but use sep to speedup.
    bpa : float
        b per a (ellipticity)
    '''
    try:
        import sep
    except ImportError:
        raise ImportError("sep is required for apphot_annulus_sep")

    _ccd = ccd.copy()

    if isinstance(_ccd, CCDData):
        _arr = _ccd.data
        _mask = _ccd.mask
        if t_exposure is None:
            try:
                t_exposure = _ccd.header[exposure_key]
            except (KeyError, IndexError):
                t_exposure = 1
                warn("The exposure time info not given and not found from the"
                     + f"header({exposure_key}). Setting it to 1 sec.")
    else:  # ndarray
        _arr = np.array(_ccd)
        _mask = None
        if t_exposure is None:
            t_exposure = 1
            warn("The exposure time info not given. Setting it to 1 sec.")

    if _mask is None:
        _mask = np.zeros_like(_arr).astype(bool)

    if mask is not None:
        _mask |= mask

    if error is not None:
        if verbose:
            print("Ignore any uncertainty extension in the original CCD and use provided error.")
        err = error.copy()
        if isinstance(err, CCDData):
            err = err.data
    else:
        try:
            err = _ccd.uncertainty.array
        except AttributeError:
            if verbose:
                warn("Couldn't find Uncertainty extension in ccd. "
                     + "Will not calculate errors.")
            err = np.zeros_like(_arr)

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    multipos = False

    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    elif x.size > 1:
        multipos = True

    a = np.atleast_1d(a)
    bpa = np.atleast_1d(bpa)
    theta = np.atleast_1d(theta)
    if (a.size > 1) + (bpa.size > 1) + (theta.size > 1) > 1:
        raise ValueError("Only one of a, bpa, theta can have size > 1.")

    num_apertures = max(a.size, bpa.size, theta.size)
    a = np.repeat(a, num_apertures)
    bpa = np.repeat(bpa, num_apertures)
    theta = np.repeat(theta, num_apertures)
    b = a*bpa

    a_in = np.atleast_1d(a_in)
    a_out = np.atleast_1d(a_out)
    if a_in.size > 1 or a_out.size > 1 or bpa.size > 1:
        raise ValueError("multiple annuli not allowed yet.")

    pass
