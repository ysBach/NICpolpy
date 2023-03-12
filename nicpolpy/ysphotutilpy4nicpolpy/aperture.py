import numpy as np
from astropy import units as u
from astropy.nddata import CCDData
from photutils.aperture import (CircularAnnulus, CircularAperture, EllipticalAnnulus,
                                EllipticalAperture)

__all__ = ["cutout_from_ap", "ap_to_cutout_position",
           "circ_ap_an", "ellip_ap_an",
           "radprof_pix", ]


def cutout_from_ap(ap, ccd):
    ''' Returns a Cutout2D object from bounding boxes of aperture/annulus.
    Parameters
    ----------
    ap : `photutils.Aperture`
        Aperture or annulus to cut the ccd.

    ccd : `astropy.nddata.CCDData` or ndarray
        The ccd to be cutout.
    '''
    from astropy.nddata import CCDData, Cutout2D
    if not isinstance(ccd, CCDData):
        ccd = CCDData(ccd, unit="adu")  # dummy unit

    positions = np.atleast_2d(ap.positions)
    try:
        bboxes = np.atleast_1d(ap.bbox)
    except AttributeError:
        bboxes = np.atleast_1d(ap.bounding_boxes)
    sizes = [bbox.shape for bbox in bboxes]
    cuts = []
    for pos, size in zip(positions, sizes):
        cuts.append(Cutout2D(ccd.data, position=pos, size=size))

    if len(cuts) == 1:
        return cuts[0]
    else:
        return cuts


def ap_to_cutout_position(ap, cutout2d):
    ''' Returns a new aperture/annulus only by updating ``positions``
    Parameters
    ----------
    ap : `photutils.Aperture`
        Aperture or annulus to update the ``.positions``.

    cutout2d : `astropy.nddata.Cutout2D`
        The cutout ccd to update ``ap.positions``.
    '''
    import copy
    newap = copy.deepcopy(ap)
    pos_old = np.atleast_2d(newap.positions)  # Nx2 positions
    newpos = []
    for pos in pos_old:
        newpos.append(cutout2d.to_cutout_position(pos))
    newap.positions = newpos
    return newap


"""
def cut_for_ap(to_move, based_on=None, ccd=None):
    ''' Cut ccd to ndarray from bounding box of ``based_on``.
    Useful for plotting aperture and annulus after cutting out centering
    on the object of interest.

    Parameters
    ----------
    to_move, based_on : `~photutils.Aperture`
        The aperture to be moved, and the reference.
    '''
    import copy

    moved = copy.deepcopy(to_move)

    if based_on is None:
        base = copy.deepcopy(to_move)
    else:
        base = copy.deepcopy(based_on)

    if np.atleast_2d(to_move.positions).shape[0] != 1:
        raise ValueError("multi-positions 'to_move' is not supported yet.")
    if np.atleast_2d(base.positions).shape[0] != 1:
        raise ValueError("multi-positions 'based_on' is not supported yet.")

    # for photutils before/after 0.7 compatibility...
    bbox = np.atleast_1d(base.bounding_boxes)[0]
    moved.positions = moved.positions - np.array([bbox.ixmin, bbox.iymin])

    if ccd is not None:
        from astropy.nddata import CCDData, Cutout2D
        if not isinstance(ccd, CCDData):
            ccd = CCDData(ccd, unit='adu')  # dummy unit
        # for photutils before/after 0.7 compatibility...
        pos = np.atleast_2d(moved.positions)[0]
        cut = Cutout2D(data=ccd.data, position=pos, size=bbox.shape)
        return moved, cut

    return moved

def cut_for_ap(to_move, based_on=None, ccd=None):
    ''' Cut ccd to ndarray from bounding box of ``based_on``.
    Useful for plotting aperture and annulus after cutting out centering
    on the object of interest.

    Parameters
    ----------
    to_move, based_on : `~photutils.Aperture`
        The aperture to be moved, and the reference.
    '''
    import copy

    moved = copy.deepcopy(to_move)

    if based_on is None:
        base = copy.deepcopy(to_move)
    else:
        base = copy.deepcopy(based_on)

    if ccd is not None:
        from astropy.nddata import CCDData
        if not isinstance(ccd, CCDData):
            ccd = CCDData(ccd, unit='adu')  # dummy unit

    pos_orig = np.atleast_2d(moved.positions)  # not yet moved
    pos_base = np.atleast_2d(base.positions)
    N_moved = pos_orig.shape[0]
    N_base = pos_base.shape[0]

    if N_base != 1 and N_moved != N_base:
        raise ValueError("based_on should have one 'positions' or "
                         + "have same number as 'move_to.positions'.")

    bboxes = np.atleast_1d(base.bounding_boxes)
    if base == 1:
        bboxes = np.repeat(bboxes, N_moved, 0)

    cuts = []
    for i, (position, bbox) in enumerate(zip(pos_orig, bboxes)):
        pos_cut = (position - np.array([bbox.ixmin, bbox.iymin]))

        moved.positions[i] = pos_cut
        if ccd is not None:
            from astropy.nddata import Cutout2D
            size = bbox.shape
            cut = Cutout2D(data=ccd.data, position=position, size=size)
            cuts.append(cut)

    if ccd is not None:
        if N_base == 1:
            return moved, cuts[0]
        else:
            return moved, cuts
    else:
        return moved
"""


def _sanitize_apsize(size=None, fwhm=None, factor=None, name='size', repeat=False):
    def __repeat(item, repeat=False, rep=2):
        if repeat and np.isscalar(item):
            return np.repeat(item, rep)
        else:
            return np.atleast_1d(item) if repeat else np.atleast_1d(item)[0]

    if size is None:
        try:
            fwhm = __repeat(fwhm, repeat=repeat, rep=2)
            factor = __repeat(factor, repeat=repeat, rep=2)
            return factor*fwhm
        except TypeError:
            raise ValueError(f"{name} is None; fwhm must be given.")
    else:
        size = __repeat(size, repeat=repeat, rep=2)
        return size


def circ_ap_an(
        positions,
        r_ap=None,
        r_in=None,
        r_out=None,
        fwhm=None,
        f_ap=1.5,
        f_in=4.,
        f_out=6.
):
    ''' A convenience function for pixel circular aperture/annulus
    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the
        following formats::

          * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
          * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs
          * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in pixel units

    r_ap, r_in, r_out : float, optional.
        The aperture, annular inner, and annular outer radii.

    fwhm : float, optional.
        The FWHM in pixel unit.

    f_ap, f_in, f_out: int or float, optional.
        The factors multiplied to ``fwhm`` to set the aperture radius, inner
        sky radius, and outer sky radius, respectively. Defaults are ``1.5``,
        ``4.0``, and ``6.0``, respectively, which are de facto standard values
        used by classical IRAF users.

    Returns
    -------
    ap, an : `~photutils.CircularAperture` and `~photutils.CircularAnnulus`
        The object aperture and sky annulus.
    '''
    r_ap = _sanitize_apsize(r_ap, fwhm=fwhm, factor=f_ap, name='r_ap')
    r_in = _sanitize_apsize(r_in, fwhm=fwhm, factor=f_in, name='r_in')
    r_out = _sanitize_apsize(r_out, fwhm=fwhm, factor=f_out, name='r_out')

    ap = CircularAperture(positions=positions, r=r_ap)
    an = CircularAnnulus(positions=positions, r_in=r_in, r_out=r_out)
    return ap, an


def ellip_ap_an(
        positions,
        r_ap=None,
        r_in=None,
        r_out=None,
        fwhm=None,
        theta=0.,
        f_ap=(1.5, 1.5),
        f_in=(4., 4.),
        f_out=(6., 6.)
):
    ''' A convenience function for pixel elliptical aperture/annulus
    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the following
        formats::

          * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
          * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs
          * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in pixel units

    r_ap, r_in, r_out: int or float, list or tuple of such, optional.
        The aperture, annular inner, and annular outer radii. If list-like, the
        0-th element is regarded as the "semi-major" axis, even though it is
        smaller than the 1-th element. Thus, ``a, b = r_xx[0], r_xx[1]``

    fwhm : float
        The FWHM in pixel unit.

    theta : float, optional
        The rotation angle in radians of the ellipse semimajor axis (0-th
        element of radii or f parameters, not necessarily the longer axis) from
        the positive ``x`` axis.  The rotation angle increases
        counterclockwise.
        Default: ``0``.

    f_ap, f_in, f_out: int or float, list or tuple of such, optional
        The factors multiplied to ``fwhm`` to set the aperture ``a`` and ``b``,
        inner sky ``a`` and ``b``, and outer sky ``a`` and ``b``, respectively.
        If scalar, it is assumed to be identical for both ``a`` and ``b``
        parameters. Defaults are ``(1.5, 1.5)``, ``(4.0, 4.0)``, and ``(6.0,
        6.0)``, respectively, which are de facto standard values used by
        classical IRAF users. If list-like, the 0-th element is regarded as the
        "semi-major" axis, even though it is smaller than the 1-th element.

    Returns
    -------
    ap, an : `~photutils.EllipticalAperture` and `~photutils.EllipticalAnnulus`
        The object aperture and sky annulus.
    '''
    a_ap, b_ap = _sanitize_apsize(r_ap, fwhm, factor=f_ap, name='r_ap', repeat=True)
    a_in, b_in = _sanitize_apsize(r_in, fwhm, factor=f_in, name='r_in', repeat=True)
    a_out, b_out = _sanitize_apsize(r_out, fwhm, factor=f_out, name='r_out', repeat=True)

    pt = dict(positions=positions, theta=theta)

    ap = EllipticalAperture(**pt, a=a_ap, b=b_ap)
    try:
        an = EllipticalAnnulus(**pt, a_in=a_in, a_out=a_out, b_in=b_in, b_out=b_out)
    except TypeError:  # Prior to photutils 1.0, b_in is not supported.
        an = EllipticalAnnulus(**pt, a_in=a_in, a_out=a_out, b_out=b_out)

    return ap, an


def radprof_pix(img, pos, mask=None, rmax=10, sort_dist=False):
    """Get radial profile (pixel values) of an object from n-D image.

    Parameters
    ----------
    img : CCDData or ndarray
        The image to be profiled.
    pos : array_like
        The xy coordinates of the center of the object (0-indexing).
    rmax : int, optional
        The maximum radius to be profiled.
    """
    if isinstance(img, CCDData):
        img = img.data
    elif not isinstance(img, np.ndarray):
        raise TypeError(f'img must be a CCDData or ndarray (now {type(img) = })')

    offset = np.array([int(max(0, _p - rmax)) for _p in pos])  # flooring by `int`
    slices = [slice(_o, min(_n, int(_p+rmax)+1))               # ceiling by `int` and +1
              for _o, _p, _n in zip(offset[::-1], pos[::-1], img.shape)]
    cut = img[tuple(slices)]
    pos_cut = np.array(pos) - offset
    grids = np.meshgrid(*[np.arange(_n) for _n in cut.shape])
    dists = np.sqrt(np.sum([(g - p)**2 for g, p in zip(grids, pos_cut[::-1])], axis=0)).T
    mask = (dists > rmax) if mask is None else ((dists > rmax) | mask)
    if sort_dist:
        sort_idx = np.argsort(dists[~mask])
        return dists[~mask][sort_idx], cut[~mask][sort_idx]
    return dists[~mask], cut[~mask]


def radprof_an(img, pos, rmax=10, dr=1, method="center"):
    """Get radial profile (annulus average) of an object from n-D image.
    """
    pass
