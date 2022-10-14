from warnings import warn
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from scipy.interpolate import UnivariateSpline

from .util import bezel_mask

__all__ = ["horizons_query",
           "HorizonsDiscreteEpochsQuery", "organize_ps1_and_isnear",
           "MASTQuery",
           "PanSTARRS1", "group_stars", "get_xy", "xyinFOV",
           "panstarrs_query"]


def horizons_query(
        id,
        epochs=None,
        sort_by='datetime_jd',
        start=None,
        stop=None,
        step='12h',
        location='500',
        id_type='smallbody',
        interpolate=None,
        interpolate_x='datetime_jd',
        k=1,
        s=0,
        output=None,
        format='csv',
        pandas=False,
        airmass_lessthan=99,
        solar_elongation=(0, 180),
        max_hour_angle=0,
        rate_cutoff=None,
        skip_daylight=False,
        refraction=False,
        refsystem='J2000',
        closest_apparition=False,
        no_fragments=False,
        quantities=','.join([str(i + 1) for i in range(43)]),
        cache=True,
        extra_precision=False
):
    """
    Parameters
    ----------
    id : str, required
        Name, number, or designation of the object to be queried.

    epochs : scalar, list-like, or dictionary, optional
        Either a list of epochs in JD or MJD format or a dictionary defining a
        range of times and dates; the range dictionary has to be of the form
        {``'start'``:'YYYY-MM-DD [HH:MM:SS]', ``'stop'``:'YYYY-MM-DD
        [HH:MM:SS]', ``'step'``:'n[y|d|m|s]'}. If no epochs are provided, the
        current time is used.

    sort_by : None, str or list of str, optional.
        The column keys to sort the table. It is recommended to sort by time,
        because it seems the default sorting by JPL HORIZONS is based on time,
        not the `epochs` (i.e., ``epochs = [1, 3, 2]`` would return an
        ephemerides table in the order of ``[1, 2, 3]``). Give `None` to skip
        sorting.

    start, stop, step: str, optional.
        If ``epochs=None``, it will be set as ``epochs = {'start':start,
        'stop':stop, 'step':step}``. If **eithter** `start` or `stop` is `None`
        or `epochs` is set to `None`, the current time is used for query.

    location : str or dict, optional
        Observer's location for ephemerides queries or center body name for
        orbital element or vector queries. Uses the same codes as JPL Horizons.
        If no location is provided, Earth's center is used for ephemerides
        queries and the Sun's center for elements and vectors queries.
        Arbitrary topocentic coordinates for ephemerides queries can be
        provided in the format of a dictionary. The dictionary has to be of the
        form {``'lon'``: longitude in deg (East positive, West negative),
        ``'lat'``: latitude in deg (North positive, South negative),
        ``'elevation'``: elevation in km above the reference ellipsoid,
        [``'body'``: Horizons body ID of the central body; optional; if this
        value is not provided it is assumed that this location is on Earth]}.

    id_type : str, optional
        Identifier type, options: ``'smallbody'``, ``'majorbody'`` (planets but
        also anything that is not a small body), ``'designation'``, ``'name'``,
        ``'asteroid_name'``, ``'comet_name'``, ``'id'`` (Horizons id number),
        or ``'smallbody'`` (find the closest match under any id_type).
        Default: ``'smallbody'``

    interpolate, interpolate_x : None, list of str, optinal.
        The column names to interpolate and the column for the ``x`` values to
        be used.
        default: ``interpolate=None, interpolate_x='datetime_jd'``.

    k : int, optional
        Degree of the smoothing spline.  Must be <= 5.
        Default is k=1, a linear spline.

    s : float or None, optional
        Positive smoothing factor used to choose the number of knots.
        Default is 0, i.e., the interpolation.

    output : None, path-like, optional.
        If you want to write the ephemerides, give this.

    format : str, optional.
        The output table format (see `~astropy.io.ascii.write`).

    pandas: bool, optional.
        If True, return a pandas DataFrame. Otherwise, return an astropy Table.
        Default: `False`.

    airmass_lessthan : float, optional
        Defines a maximum airmass for the query.
        Default: ``99``

    solar_elongation : tuple, optional
        Permissible solar elongation raqnge: (minimum, maximum).
        Default: ``(0, 180)``

    max_hour_angle : float, optional
        Defines a maximum hour angle for the query.
        Default: ``0``

    rate_cutoff : float, optional
        Angular range rate upper limit cutoff in arcsec/h.
        Default: `None` (disabled)

    skip_daylight : boolean, optional
        Crop daylight epochs in query.
        Default: `False`

    refraction : boolean
        If `True`, coordinates account for a standard atmosphere refraction
        model; if `False`, coordinates do not account for refraction (airless
        model);
        default: `False`

    refsystem : string
        Coordinate reference system: ``'J2000'`` or ``'B1950'``.
        default: ``'J2000'``

    closest_apparition : boolean, optional
        Only applies to comets. This option will choose the closest apparition
        available in time to the selected epoch; default: False. Do not use
        this option for non-cometary objects.

    no_fragments : boolean, optional
        Only applies to comets. Reject all comet fragments from selection;
        default: False. Do not use this option for non-cometary objects.

    quantities : integer or string, optional
        Single integer or comma-separated list in the form of a string
        corresponding to all the quantities to be queried from JPL Horizons
        using the coding according to the `JPL Horizons User Manual Definition
        of Observer Table Quantities
        <https://ssd.jpl.nasa.gov/?horizons_doc#table_quantities>`_
        Default: all quantities

    extra_precision : boolean, optional
        Enables extra precision in RA and DEC values.
        Default: `False`

    Returns
    -------
    obj : astroquery.jplhorizons.HorizonsClass
        The object used for query.

    eph : astropy Table
        The queried ephemerides from Horizons

    interpolated: dict
        The interpolation functions for the columns (specified by
        `interpolate`).


    Notes
    -----
    It uses ``scipy.interpolate.UnivariateSpline``.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
    """
    interpolated = {}

    if epochs is None:
        if start is None or stop is None:
            epochs = None
        else:
            epochs = dict(start=start, stop=stop, step=step)

    obj = Horizons(id=id, epochs=epochs, location=location, id_type=id_type)
    eph = obj.ephemerides(
        airmass_lessthan=airmass_lessthan,
        solar_elongation=solar_elongation,
        max_hour_angle=max_hour_angle,
        rate_cutoff=rate_cutoff,
        skip_daylight=skip_daylight,
        refraction=refraction,
        refsystem=refsystem,
        closest_apparition=closest_apparition,
        no_fragments=no_fragments,
        quantities=quantities,
        cache=cache,
        extra_precision=extra_precision,
        get_query_payload=False,
        get_raw_response=False
    )
    if interpolate is not None and interpolate_x is not None:
        if isinstance(interpolate, str):
            interpolate = [interpolate]
        x = eph[interpolate_x]
        for i in interpolate:
            interpolated[i] = UnivariateSpline(x, eph[i], k=k, s=s)

    if sort_by is not None:
        eph.sort(sort_by)

    if pandas:
        eph = eph.to_pandas()

    if output is not None:
        eph.write(Path(output), format=format)

    return obj, eph, interpolated


def mask_str(n_new, n_old, msg):
    dn = n_old - n_new
    print(f"{n_new:3d} objects remaining: {dn:3d} masked out of {n_old:3d} based on {msg:s}.")


class HorizonsDiscreteEpochsQuery:
    def __init__(self, targetname, location, epochs, id_type="smallbody"):
        '''
        Parameters
        ----------
        id : str
            Name, number, or designation of the object to be queried.

        location : str or dict
            Observer's location for ephemerides queries or center body name for
            orbital element or vector queries. Uses the same codes as JPL
            Horizons. If no location is provided, Earth's center is used for
            ephemerides queries and the Sun's center for elements and vectors
            queries. Arbitrary topocentic coordinates for ephemerides queries
            can be provided in the format of a dictionary. The dictionary has
            to be of the form {``'lon'``: longitude in deg (East positive, West
            negative), ``'lat'``: latitude in deg (North positive, South
            negative), ``'elevation'``: elevation in km above the reference
            ellipsoid, [``'body'``: Horizons body ID of the central body;
            optional; if this value is not provided it is assumed that this
            location is on Earth]}.

        epochs : scalar, list-like, or dictionary
            Either a list of epochs in JD or MJD format or a dictionary
            defining a range of times and dates; the range dictionary has to be
            of the form {``'start'``:'YYYY-MM-DD [HH:MM:SS]',
            ``'stop'``:'YYYY-MM-DD [HH:MM:SS]', ``'step'``:'n[y|d|m|s]'}. If no
            epochs are provided, the current time is used.

        id_type : str, optional
            Identifier type, options:
            ``'smallbody'``, ``'majorbody'`` (planets but also anything that is
            not a small body), ``'designation'``, ``'name'``,
            ``'asteroid_name'``, ``'comet_name'``, ``'id'`` (Horizons id
            number), or ``'smallbody'`` (find the closest match under any
            id_type).
            Default: ``'smallbody'``
        '''
        self.targetname = str(targetname)
        self.location = location
        self.epochs = np.asarray(epochs)
        self.id_type = id_type
        self.query_table = None
        self.uri = []

    def __str__(self):
        _str = "Query {:s} at location {} for given discrete epochs."
        return _str.format(self.targetname, self.location)

    def query(self, depoch=100, *args, **kwargs):
        '''
        Parameters
        ----------
        depoch : int, optional
            The number of discrete epochs to be chopped.

        args, kwargs : optional.
            Passed to ``.ephemerides()`` of ``Horizons``.
        '''
        Nepoch = np.shape(self.epochs)[0]
        Nquery = (Nepoch - 1) // depoch + 1
        tabs = []

        print(f'Query: {self.targetname} '
              + f'at {self.location} for {Nepoch} epochs''')

        if Nquery > 1:
            print(f"Query chopped into {Nquery} chunks: Doing ", end=' ')

        for i in range(Nquery):
            print(f"{i+1}...", end=' ')
            i_0 = i*depoch
            i_1 = (i + 1)*depoch
            epochs_i = self.epochs[i_0:i_1]

            obj = Horizons(
                id=self.targetname,
                location=self.location,
                epochs=epochs_i,
                id_type=self.id_type
            )
            eph = obj.ephemerides(*args, **kwargs)

            tabs.append(eph)
            self.uri.append(obj.uri)

        if len(tabs) == 1:
            self.query_table = tabs[0]

        elif len(tabs) > 1:
            self.query_table = vstack(tabs)

        print("Query done.")

        return self.query_table


PS1_DR1_DEL_FLAG = [
    0,   # FEW: Used within relphot; skip star.
    1,   # POOR: Used within relphot; skip star.
    2,   # ICRF_QSO: object IDed with known ICRF quasar
    3,   # HERN_QSO_P60: identified as likely QSO, P_QSO >= 0.60
    5,   # HERN_RRL_P60: identified as likely RR Lyra, P_RRLyra >= 0.60
    7,   # HERN_VARIABLE: identified as a variable based on ChiSq
    8,   # TRANSIENT: identified as a non-periodic (stationary) transient
    9,   # HAS_SOLSYS_DET: at least one detection identified with a known SSO
    10,  # MOST_SOLSYS_DET: most detections identified with a known SSO
    23,  # EXT: extended in our data (eg, PS)
    24   # EXT_ALT: extended in external data (eg, 2MASS)
]


def organize_ps1_and_isnear(
        ps1,
        header=None,
        bezel=0,
        nearby_obj_minsep=0*u.deg,
        group_crit_separation=0,
        select_filter_kw={},
        del_flags=PS1_DR1_DEL_FLAG,
        drop_by_Kron=True,
        calc_JC=True
):
    ''' Organizes the PanSTARRS1 object and check nearby objects.

    Parameters
    ----------
    ps1 : `~PanSTARRS1`
        The `~PanSTARRS1` object.

    header : `astropy.header.Header`, None, optional
        The header to extract WCS related information. If `None` (default), it
        will not drop any stars based on the field of view criterion.

    bezel : int, float, optional
        The bezel used to select stars inside the field of view.

    nearby_obj_minsep : float, `~astropy.Quantity`, optional.
        If there is any object closer than this value, a warning message will
        be printed.

    group_crit_separation : float, optional
        The critical separation parameter used in DAOGROUP algorithm
        (`~photutils.DAOGroup`) to select grouped stars.

    select_filter_kw : dict, optional
        The kwargs for `~PanSTARRS1.select_filter()` method.

    del_flags : list of int, optional
        The flags to be used for dropping objects based on ``"f_objID"`` of
        Pan-STARRS1 query.

    drop_by_Kron : bool, optional
        If `True` (default), drop the galaxies based on the Kron magnitude
        criterion suggested by PS1 (which works good only if i <~ 21):
        https://outerspace.stsci.edu/display/PANSTARRS/How+to+separate+stars+and+galaxies

    calc_JC : bool, optional
        Whether to calculate the Johnson-Cousins B V R_C filter magnitudes by
        the linear relationship given by Table 6 of Tonry J. et al. 2012, ApJ,
        750, 99., using g-r color. The following columns will be added to the
        table ``ps1.queried``:

          * ``"C_gr"``: The ``g-r`` color.
          * ``"dC_gr"``: The total error of ``g-r`` color. Not only the
            error-bar of the mean PSF magnitude (``"e_Xmag`` for filter
            ``X="g"`` and ``X="r"``), but also the intrinsic error-bar of each
            measurements (``"XmagStd" for filter ``X="g"`` and ``X="r"``) are
            considered, i.e., four error-bars are propagated by first order
            approximation (square sum and square rooted).
          * ``"Bmag"``, ``"Vmag"``, ``Rmag``:
            ``B = g + 0.213 + 0.587(g-r) (+- 0.034)``,
            ``V = r + 0.006 + 0.474(g-r) (+- 0.012)``,
            ``R = r - 0.138 -0.131(g-r) (+-0.015)``
          * ``"dBmag"``, ``"dVmag"``, ``"dRmag"``: The total error of above
            magnitudes. The scatter reported by Tonry et al. (e.g., 0.012 mag
            for V) is propagated with the first order error estimated from the
            magnitude calculation formula.

    Returns
    -------
    isnear : bool
        True if there is any nearby object from ``ps1.queried``.
    '''

    _ = ps1.query()

    # Select only those within FOV & bezel.
    # If you wanna keep those outside the edges, just set negative
    # ``bezel``.
    if header is not None:
        ps1.select_xyinFOV(header=header, bezel=bezel)

    # Check whether any object is near our target
    isnear = ps1.check_nearby(minsep=nearby_obj_minsep)
    if isnear:
        warn("There are objects near the target!")

    # Drop objects near to each other
    ps1.drop_star_groups(crit_separation=group_crit_separation)

    # Drop for preparing differential photometry
    ps1.drop_for_diff_phot(del_flags=del_flags, drop_by_Kron=drop_by_Kron)

    # remove columns that are of no interest
    ps1.select_filters(**select_filter_kw)

    ps1.queried["_r"] = ps1.queried["_r"].to(u.arcsec)
    ps1.queried["_r"].format = "%.3f"

    if calc_JC:
        ps1.queried["grcolor"] = ps1.queried["gmag"] - ps1.queried["rmag"]

        # Unfortunately the Std of color is unavailable from the columns
        # provided by PS1 DR1. But error of the mean can be estimated by
        # Gaussian error propagation
        var_g = ps1.queried["e_gmag"]**2
        var_r = ps1.queried["e_rmag"]**2
        dc_gr = np.sqrt(var_g + var_r)
        ps1.queried["e_grcolor"] = dc_gr

        pars = dict(Bmag=[0.213, 0.587, 0.034, "gmag"],
                    Vmag=[0.006, 0.474, 0.012, "rmag"],
                    Rmag=[-0.138, -0.131, 0.015, "rmag"])
        # filter_name = [B_0, B_1, B_sc of Tonry, mag used for conversion]
        for k, p in pars.items():
            ps1.queried[k] = (p[0] + p[1]*ps1.queried["grcolor"] + ps1.queried[p[3]])
            ps1.queried[f"e_{k}"] = np.sqrt((p[1]*ps1.queried["e_grcolor"])**2 + p[2]**2)

    return isnear


class MASTQuery():
    def __init__(self, catalog, table=None, data_release=None,
                 ra=None, dec=None, frame="icrs", skycoord=None, radius=None,
                 columns="*", sort_by=None, **filter_kw):
        """ MAST Catalog Query (astroquery.mast.Catalogs)

        Parameters
        ----------
        catalog : str
            The name of the catalog. Examples: ``"Galex"``, ``"HSC"``,
            ``"TIC"``, ``"Gaia"``, ``"Panstarrs"``.

        table, data_release: str
            The specification of table and data release. Example:
            ``catalog="Panstarrs", table="mean", data_release="dr2"`` By
            default, `data_release` is the most recent one, and `table` differs
            for each catalog.

        ra, dec, radius : float or `~astropy.Quantity`
            The central RA, DEC and the cone search radius. If not
            `~astropy.Quantity`, assuming it is in degrees unit.

        skycoord: `~astropy.coordinates.SkyCoord`
            The coordinate if the user wants higher freedom. `ra` and `dec`
            will be ignored.

        radius : float or `~astropy.Quantity`
            The cone search radius. If not `~astropy.Quantity`, assuming it is
            in degrees unit.

        columns : list of str, '*', optional
            The columns to be retrieved. ``"*"`` request all the columns.

        sort_by : str, list of tuples, optional.
            The column(s) to sort by. If a list of tuples, each tuple is
            direction and column name (e.g., ``[('desc', 'ra')]``, ``[('asc',
            'ra')]``).

        **filter_kw : dict, optional
            The column filters for astroquery.vizier.
            Examples: ``nStackDetections=[("gte", 2)]``

            ..note::
                To filter the query, criteria per column name are accepted. The
                AND operation is performed between all column name criteria,
                and the OR operation is performed within column name criteria.
                Per each column name parameter, criteria may consist of either
                a value or a list. The list may consist of a mix of values and
                tuples of criteria decorator (min, gte, gt, max, lte, lt, like,
                contains) and value.

            ..warning::
                The criteria doesn't work sometimes -- better to query all and
                filter afterwards..
                https://github.com/astropy/astroquery/issues/2355
        """
        if skycoord is not None:
            skycoord = skycoord.transform_to("icrs")
            ra = skycoord.ra
            dec = skycoord.dec

        _params = dict(
            ra=ra,
            dec=dec,
            radius=radius,
        )

        for k, v in _params.items():
            if v is None:
                continue
            if isinstance(v, u.Quantity):  # MAST uses only degrees unit for RA/DEC/radius
                _params[k] = v.to(u.deg).value

        self.coordinates = "{} {}".format(_params["ra"], _params["dec"])  # MAST str format...
        self.radius = _params["radius"]
        self.catalog = catalog
        self.table = table
        self.data_release = data_release
        self.sort_by = sort_by

        if isinstance(columns, str):
            if columns in ['*', '**']:
                self.columns = [columns]
            else:
                raise ValueError("If columns is str, it must be one of ['*', '**']")
        else:
            self.columns = columns

        self.filters = filter_kw

    def query(self):
        self.queried = Catalogs.query_criteria(
            coordinates=self.coordinates,
            radius=self.radius,
            catalog=self.catalog,
            table=self.table,
            data_release=self.data_release,
            columns=self.columns,
            sort_by=self.sort_by,
            **self.filters
        )


# TODO: Let it accept SkyCoord too
class PanSTARRS1:
    def __init__(self, ra, dec, frame="icrs", radius=None, inner_radius=None,
                 width=None, height=None, columns=["**", "+_r"],
                 column_filters={}):
        """ Query PanSTARRS @ VizieR using astroquery.vizier

        Parameters
        ----------
        ra, dec, radius : float or `~astropy.Quantity`
            The central RA, DEC and the cone search radius. If not
            `~astropy.Quantity`, assuming it is in degrees unit.

        frame : str
            The frame for the coordinates.
            Default: "icrs"

        inner_radius : cfloat or `~astropy.Quantity`
            When set in addition to `radius`, the queried region becomes
            annular, with outer radius `radius` and inner radius
            `inner_radius`. If not `~astropy.Quantity`, assuming it is in
            degrees unit.

        width : convertible to `~astropy.coordinates.Angle`
            The width of the square region to query. If not
            `~astropy.Quantity`, assuming it is in degrees unit.

        height : convertible to `~astropy.coordinates.Angle`
            When set in addition to ``width``, the queried region becomes
            rectangular, with the specified ``width`` and ``height``. If not
            `~astropy.Quantity`, assuming it is in degrees unit.

        columns : list of str, {'*', '**'}, optional
            The columns to be retrieved. The special column ``"*"`` requests
            just the default columns of a catalog; ``"**"`` (Default) would
            request all the columns. For sorting, use ``"+"`` in front of the
            column name. See the documentation:
            https://astroquery.readthedocs.io/en/latest/vizier/vizier.html#specifying-keywords-output-columns-and-constraints-on-columns

        column_filters : dict, optional
            The column filters for astroquery.vizier.
            Example can be ``{"gmag":"13.0..20.0", "e_gmag":"<0.10"}``.

        Return
        ------
        queried : astropy.table object
            The queried result.

        Notes
        -----
        All columns: http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=II/349
        """
        _params = dict(
            ra=ra,
            dec=dec,
            radius=radius,
            inner_radius=inner_radius,
            width=width,
            height=height,
        )

        for k, v in _params.items():
            if v is None:
                continue
            if not isinstance(v, u.Quantity):
                warn(f"{k} is not astropy Quantity: Assuming deg unit")
                _params[k] = v * u.deg

        self.ra = _params["ra"]
        self.dec = _params["dec"]
        self.radius = _params["radius"]
        self.inner_radius = _params["inner_radius"]
        self.width = _params["width"]
        self.height = _params["height"]
        self.frame = frame

        if isinstance(columns, str):
            if columns in ['*', '**']:
                self.columns = [columns]
            else:
                raise ValueError("If columns is str, it must be one of ['*', '**']")
        else:
            self.columns = columns

        self.column_filters = column_filters

    def query(self):
        vquery = Vizier(columns=self.columns, column_filters=self.column_filters, row_limit=-1)

        field = SkyCoord(ra=self.ra, dec=self.dec, frame=self.frame)

        self.queried = vquery.query_region(field,
                                           radius=self.radius,
                                           inner_radius=self.inner_radius,
                                           width=self.width,
                                           height=self.height,
                                           catalog="II/349/ps1")[0]

        return self.queried

    def select_xyinFOV(self, header=None, wcs=None, bezel=0, mode='all'):
        ''' Convert RA/DEC to xy (add columns) with rejection at bezels.
        Parameters
        ----------
        header : astropy.io.fits.Header, optional
            The header to extract WCS information. One and only one of `header`
            and `wcs` must be given.

        wcs : astropy.wcs.WCS, optional
            The WCS to convert the RA/DEC to XY. One and only one of `header`
            and `wcs` must be given.

        bezel: int or float, optional
            The bezel size to exclude stars at the image edges. If you want to
            keep some stars outside the edges, put negative values (e.g.,
            ``-5``).

        mode: 'all' or 'wcs', optional
            Whether to do the transformation including distortions (``'all'``)
            or only including only the core WCS transformation (``'wcs'``).
        '''
        self.queried = xyinFOV(table=self.queried, header=header, wcs=wcs,
                               ra_key="RAJ2000", dec_key="DEJ2000",
                               bezel=bezel, origin=0, mode=mode)

    def drop_for_diff_phot(self, del_flags=PS1_DR1_DEL_FLAG,
                           drop_by_Kron=True):
        ''' Drop objects which are not good for differential photometry.
        Parameters
        ----------
        del_flags : list of int, None, optional
            The flags to be used for dropping objects based on ``"f_objID"`` of
            Pan-STARRS1 query. These are the powers of 2 to identify the flag
            (e.g., 2 means ``2**2`` or flag ``4``). See Notes below for each
            flag. Set it to `None` to keep all the objects based on
            ``"f_objID"``.

        drop_by_Kron : bool, optional
            If `True` (default), drop the galaxies based on the Kron magnitude
            criterion suggested by PS1 (which works good only if i <~ 21):
            https://outerspace.stsci.edu/display/PANSTARRS/How+to+separate+stars+and+galaxies

        Notes
        -----
        H15 means Hernitschek+ 2015ApJ...801...45H.
          * FEW (1) : Used within relphot; skip star.
          * POOR (2) : Used within relphot; skip star.
          * ICRF_QSO (4) : object IDed with known ICRF quasar (may have ICRF
            position measurement)
          * HERN_QSO_P60 (8) : identified as likely QSO (H15), P_QSO >= 0.60
          * HERN_QSO_P05 (16) : identified as possible QSO (H15), P_QSO >= 0.05
          * HERN_RRL_P60 (32) : identified as likely RR Lyra (H15), P_RRLyra >=
            0.60
          * HERN_RRL_P05 (64) : identified as possible RR Lyra (H15), P_RRLyra
            >= 0.05
          * HERN_VARIABLE (128) : identified as a variable based on ChiSq (H15)
          * TRANSIENT (256) : identified as a non-periodic (stationary)
            transient
          * HAS_SOLSYS_DET (512) : at least one detection identified with a
            known solar-system object (asteroid or other).
          * MOST_SOLSYS_DET (1024 (2^10)) : most detections identified with a
            known solar-system object (asteroid or other).
          * LARGE_PM (2048) : star with large proper motion
          * RAW_AVE (4096) : simple weighted average position was used (no IRLS
            fitting)
          * FIT_AVE (8192) : average position was fitted
          * FIT_PM (16384) : proper motion model was fitted
          * FIT_PAR (32768) : parallax model was fitted
          * USE_AVE (65536) : average position used (not PM or PAR)
          * USE_PM (131072) : proper motion used (not AVE or PAR)
          * USE_PAR (262144) : parallax used (not AVE or PM)
          * NO_MEAN_ASTROM (524288) : mean astrometry could not be measured
          * STACK_FOR_MEAN (1048576 (2^20)) : stack position used for mean
            astrometry
          * MEAN_FOR_STACK (2097152) : mean astrometry used for stack position
          * BAD_PM (4194304) : failure to measure proper-motion model
          * EXT (8388608) : extended in our data (eg, PS)
          * EXT_ALT (16777216) : extended in external data (eg, 2MASS)
          * GOOD (33554432) : good-quality measurement in our data (eg,PS)
          * GOOD_ALT (67108864) : good-quality measurement in external data
            (eg, 2MASS)
          * GOOD_STACK (134217728) : good-quality object in the stack (>1 good
            stack measurement)
          * BEST_STACK (268435456) : the primary stack measurements are the
            best measurements
          * SUSPECT_STACK (536870912) : suspect object in the stack (no more
            than 1 good measurement, 2 or more suspect or good stack
            measurement)
          * BAD_STACK (1073741824 (2^30)) : poor-quality stack object (no more
            than 1 good or suspect measurement)

        Among the ``"f_objID"``, the following are better to be dropped because
        they are surely not usable for differential photometry:

            * 1, 2, 4, 8, 32, 128, 256, 512, 1024, 8388608, 16777216

        or in binary position (``del_flags``),

            * 0, 1, 2, 3, 5, 7, 8, 9, 10, 23, 24

        (plus maybe 2048(2^11) because centroiding may not work properly?)
        '''
        N_old = len(self.queried)
        if del_flags is not None:
            idx2remove = []
            for i, row in enumerate(self.queried):
                b_flag = list(f"{row['f_objID']:031b}")
                for bin_pos in del_flags:
                    if b_flag[-bin_pos] == '1':
                        idx2remove.append(i)
            self.queried.remove_rows(idx2remove)

            N_fobj = len(self.queried)
            mask_str(N_fobj, N_old, f"f_objID ({del_flags})")

            N_old = N_fobj

        if drop_by_Kron:
            dmag = (self.queried["imag"] - self.queried["iKmag"])
            mask = (dmag > 0.05)
            self.queried = self.queried[~mask]

            N_Kron = len(self.queried)
            mask_str(N_Kron, N_old, "the Kron magnitude criterion")

    def select_filters(
            self,
            filter_names=["g", "r", "i"],
            keep_columns=["_r", "objID", "f_objID", "RAJ2000", "DEJ2000", "x", "y"],
            n_mins=[0, 0, 0]
    ):
        ''' Abridges the columns depending on the specified filters.
        '''
        if not isinstance(filter_names, (list, tuple, np.ndarray)):
            filter_names = [filter_names]

        n_mins = np.atleast_1d(n_mins)
        if n_mins.shape[0] == 1:
            n_mins = np.repeat(n_mins, len(filter_names))
        elif n_mins.shape[0] != len(filter_names):
            raise ValueError("n_mins must be length 1 or same length as "
                             f"filter_names (now it's {len(filter_names)}).")

        selected_columns = keep_columns
        toremove_columns = []

        for filt in filter_names:
            selected_columns.append(f"N{filt}")
            selected_columns.append(f"{filt}mag")
            selected_columns.append(f"{filt}Kmag")
            selected_columns.append(f"{filt}Flags")
            selected_columns.append(f"{filt}PSFf")
            selected_columns.append(f"{filt}magStd")
            selected_columns.append(f"e_{filt}mag")
            selected_columns.append(f"e_{filt}Kmag")
            selected_columns.append(f"o_{filt}mag")
            selected_columns.append(f"b_{filt}mag")
            selected_columns.append(f"B_{filt}mag")

        for c in self.queried.columns:
            if c not in selected_columns:
                toremove_columns.append(c)

        self.queried.remove_columns(toremove_columns)

        N_old = len(self.queried)

        for i, filt in enumerate(filter_names):
            mask = np.array(self.queried[f"o_{filt}mag"]) < n_mins[i]
            self.queried = self.queried[~mask]

        N_new = len(self.queried)
        mask_str(N_new, N_old, f"o_{filter_names}mag >= {n_mins}")

    def check_nearby(self, minsep, maxmag=None, filter_names=["r"]):
        ''' Checkes whether there is any nearby object.

        Notes
        -----
        It checks the ``"_r"`` column of the ``PanSTARRS1`` queried result.
        Therefore, the query center should be the position where you want to
        check for any nearby object.

        Parameters
        ----------
        minsep : float or `~astropy.Quantity`
            The minimum separation to detect nearby object

        maxmag : int or float, optional
            The maximum magnitude value to mask objects. Objects fainter than
            this magnitude (Mean PSF magnitude) will be accepted even though it
            is nearby the search center.
        '''
        if isinstance(minsep, (float, int)):
            warn("minsep is not Quantity. Assuming degree unit.")
            minsep = minsep * u.deg
        elif not isinstance(minsep, u.Quantity):
            raise TypeError("minsep not understood.")

        if not isinstance(filter_names, (list, tuple, np.ndarray)):
            filter_names = [filter_names]

        chktab = self.queried.copy()

        if maxmag is not None:
            for filt in filter_names:
                chktab = chktab[chktab[filt] <= maxmag]
        minsep = minsep.to(chktab["_r"].unit).value
        isnear = (np.array(chktab["_r"]).min() <= minsep)
        return isnear

    def drop_star_groups(self, crit_separation):
        N_old = len(self.queried)
        grouped_rows = group_stars(table=self.queried, crit_separation=crit_separation,
                                   xcol="x", ycol="y", index_only=True)
        self.queried.remove_rows(grouped_rows)
        N_new = len(self.queried)
        mask_str(N_new, N_old,
                 (f"DAOGROUP with {crit_separation:.3f}-pixel critical separation."))


def group_stars(table, crit_separation, xcol="x", ycol="y", index_only=True):
    ''' Group stars using DAOGROUP algorithm and return row indices.

    Parameters
    ----------
    table : astropy.table.Table
        The queried result table.

    crit_separation : float or int
        Distance, in units of pixels, such that any two stars separated by less
        than this distance will be placed in the same group.

    xcol, ycol : str, optional
        The column names for x and y positions. This is necessary since
        `~photutils.DAOGroup accepts a table which has x y positions designated
        as ``"x_0"`` and ``"y_0"``.

    index : bool, optional
        Whether to return only the index of the grouped rows (group information
        is lost) or the full grouped table (after group_by).

    Notes
    -----
    Assuming the psf fwhm to be known, ``crit_separation`` may be set to
    ``k * fwhm``, for some positive real k.

    See Also
    --------
    photutils.DAOStarFinder

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at: http://adsabs.harvard.edu/abs/1987PASP...99..191S

    Return
    ------
    gtab: Table
        Returned when ``index_only=False``. The table underwent
        ``.group_by("group_id")``.

    grouped_rows: list
        Returned when ``index_only=True``.
        The indices of the rows which are "grouped" stars. You may remove such
        rows using ``table.remove_rows(grouped_rows)``.
    '''
    from photutils.psf.groupstars import DAOGroup
    tab = table.copy()

    tab[xcol].name = "x_0"
    tab[ycol].name = "y_0"
    gtab = DAOGroup(crit_separation=crit_separation)(tab).group_by("group_id")

    if not index_only:
        return gtab
    else:
        gid, gnum = np.unique(gtab["group_id"], return_counts=True)
        gmask = gid[gnum != 1]  # group id with > 1 stars
        grouped_rows = []
        for i, gid in enumerate(gtab["group_id"]):
            if gid in gmask:
                grouped_rows.append(i)
        return grouped_rows


def get_xy(header, ra, dec, unit=u.deg, origin=0, mode='all'):
    ''' Get image XY from the header WCS

    Parameters
    ----------
    header : astropy.io.fits.Header or pandas.DataFrame
        The header to extract WCS information.

    ra, dec: float or Quantity or array-like of such
        The coordinates to get XY position. If Quantity, ``unit`` will likely
        be ignored.

    unit: `~astropy.Quantity` or tuple of such
        Unit of the ``ra`` and ``dec`` given. It can be a tuple if they differ.

    origin: int, optional
       Whether to return 0 or 1-based pixel coordinates.

    mode: 'all' or 'wcs', optional
        Whether to do the transformation including distortions (``'all'``) or
        only including only the core WCS transformation (``'wcs'``).
    '''
    w = WCS(header)
    coo = SkyCoord(ra, dec, unit=unit)
    xy = SkyCoord.to_pixel(coo, wcs=w, origin=origin, mode=mode)
    return xy


# TODO: let it use something like `header_or_wcs`
def xyinFOV(
        table,
        header=None,
        wcs=None,
        shape=None,
        ra_key='ra',
        dec_key='dec',
        unit=None,
        bezel=0,
        bezel_x=None,
        bezel_y=None,
        origin=0,
        mode='all',
        return_mask=False,
        col_x="x",
        col_y="y",
        verbose=1
):
    ''' Convert RA/DEC to pixel with rejection at bezels

    Parameters
    ----------
    table : astropy.table.Table or pandas.DataFrame
        The queried result table.

    header : astropy.io.fits.Header, optional
        The header to extract WCS information. One and only one of `header` and
        `wcs` must be given.

    wcs : astropy.wcs.WCS, optional
        The WCS to convert the RA/DEC to XY. One and only one of `header` and
        `wcs` must be given.

    shape : 2-element tuple, optional
        The shape of the image (used to reject stars at bezels). If `header` or
        `wcs` is given, it is ignored and ``NAXISi`` keywords will be used.

    ra_key, dec_key : str, optional
        The column names containing RA/DEC.

    unit : `~astropy.Quantity`, optional
        The unit of the RA/DEC. If not given, it will be inferred from the
        table (only if astropy's QTable.)

    bezel : int, float, array-like, optional
        The bezel size. If array-like, it should be ``(lower, upper)``. If only
        this is given and `bezel_x` and/or `bezel_y` is/are `None`, it/both
        will be replaced by `bezel`. If you want to keep some stars outside the
        edges, put negative values (e.g., ``-5``).

    bezel_x, bezel_y : int, float, 2-array-like, optional
        The bezel (border width) for x and y axes. If array-like, it should be
        ``(lower, upper)``. Mathematically put, only objects with center
        ``(bezel_x[0] + 0.5 < center_x) & (center_x < nx - bezel_x[1] - 0.5)``
        (similar for y) will be selected. If you want to keep some stars
        outside the edges, put negative values (e.g., ``-5``).

    origin : int, optional
       Whether to return 0 or 1-based pixel coordinates.

    mode: 'all' or 'wcs', optional
        Whether to do the transformation including distortions (``'all'``) or
        only including only the core WCS transformation (``'wcs'``).

    col_x, col_y : str, optional
        The column names for x and y positions.
    '''
    bezel = np.atleast_1d(bezel)
    if bezel.shape == (1,):
        bezel = np.tile(bezel, 2)
    elif bezel.shape != (2,):
        raise ValueError(f"bezel must have shape (1,) or (2,). Now {bezel.shape}")

    _tab = table.copy()
    N_old = len(_tab)
    if isinstance(table, pd.DataFrame):
        _tab = Table.from_pandas(table)
        # Better to unify as astropy Table, because of the unit in RA/DEC may
        # be missing if converted to pandas..
    elif not isinstance(table, Table):
        raise TypeError("table must be either astropy Table or pandas DataFrame.")

    if wcs is None:
        wcs = WCS(header)

    if unit is None:
        coo = SkyCoord(_tab[ra_key], _tab[dec_key])
    else:
        coo = SkyCoord(_tab[ra_key], _tab[dec_key], unit=unit)

    x, y = coo.to_pixel(wcs=wcs, origin=origin, mode=mode)
    _tab[col_x] = x
    _tab[col_y] = y

    if shape is not None:
        nx, ny = shape[1], shape[0]
    else:
        if header is not None:
            nx, ny = header["NAXIS2"], header["NAXIS1"]
        elif wcs is not None:
            nx, ny = wcs._naxis
        else:  # If none of them is available, no way to reject stars at bezels.
            if verbose >= 1:
                print("No way to reject stars at bezels: provide `header` or `shape`.")
            if return_mask:
                return _tab, None
            return _tab

    mask = bezel_mask(x, y, nx, ny, bezel=bezel, bezel_x=bezel_x, bezel_y=bezel_y)
    _tab.remove_rows(mask)

    N_new = len(_tab)
    if verbose >= 1:
        mask_str(N_new, N_old, f"{bezel}-pixel bezel")

    if return_mask:
        return _tab, mask

    return _tab


def drop_for_diff_phot_ps(
        table,
        service="mast",
        del_flags=PS1_DR1_DEL_FLAG,
        drop_by_Kron=True
):
    ''' Drop objects which are not good for differential photometry.

    Parameters
    ----------
    table : pandas.DataFrame
        The queried table.

    service : str
        The service that provides the catalog. For PS1, MAST and Vizier have
        different column names.

    del_flags : list of int, None, optional
        The flags to be used for dropping objects based on ``"objInfoFlag"`` of
        Pan-STARRS1 query. These are the powers of 2 to identify the flag
        (e.g., 2 means ``2**2`` or flag ``4``). See Notes below for each
        flag. Set it to `None` to keep all the objects based on
        ``"objInfoFlag"``.

    drop_by_Kron : bool, optional
        If `True` (default), drop the galaxies based on the Kron magnitude
        criterion suggested by PS1 (which works good only if i <~ 21):
        https://outerspace.stsci.edu/display/PANSTARRS/How+to+separate+stars+and+galaxies

    Notes
    -----
    H15 means Hernitschek+ 2015ApJ...801...45H.
        * FEW (1) : Used within relphot; skip star.
        * POOR (2) : Used within relphot; skip star.
        * ICRF_QSO (4) : object IDed with known ICRF quasar (may have ICRF
        position measurement)
        * HERN_QSO_P60 (8) : identified as likely QSO (H15), P_QSO >= 0.60
        * HERN_QSO_P05 (16) : identified as possible QSO (H15), P_QSO >= 0.05
        * HERN_RRL_P60 (32) : identified as likely RR Lyra (H15), P_RRLyra >=
        0.60
        * HERN_RRL_P05 (64) : identified as possible RR Lyra (H15), P_RRLyra
        >= 0.05
        * HERN_VARIABLE (128) : identified as a variable based on ChiSq (H15)
        * TRANSIENT (256) : identified as a non-periodic (stationary)
        transient
        * HAS_SOLSYS_DET (512) : at least one detection identified with a
        known solar-system object (asteroid or other).
        * MOST_SOLSYS_DET (1024 (2^10)) : most detections identified with a
        known solar-system object (asteroid or other).
        * LARGE_PM (2048) : star with large proper motion
        * RAW_AVE (4096) : simple weighted average position was used (no IRLS
        fitting)
        * FIT_AVE (8192) : average position was fitted
        * FIT_PM (16384) : proper motion model was fitted
        * FIT_PAR (32768) : parallax model was fitted
        * USE_AVE (65536) : average position used (not PM or PAR)
        * USE_PM (131072) : proper motion used (not AVE or PAR)
        * USE_PAR (262144) : parallax used (not AVE or PM)
        * NO_MEAN_ASTROM (524288) : mean astrometry could not be measured
        * STACK_FOR_MEAN (1048576 (2^20)) : stack position used for mean
        astrometry
        * MEAN_FOR_STACK (2097152) : mean astrometry used for stack position
        * BAD_PM (4194304) : failure to measure proper-motion model
        * EXT (8388608) : extended in our data (eg, PS)
        * EXT_ALT (16777216) : extended in external data (eg, 2MASS)
        * GOOD (33554432) : good-quality measurement in our data (eg,PS)
        * GOOD_ALT (67108864) : good-quality measurement in external data
        (eg, 2MASS)
        * GOOD_STACK (134217728) : good-quality object in the stack (>1 good
        stack measurement)
        * BEST_STACK (268435456) : the primary stack measurements are the
        best measurements
        * SUSPECT_STACK (536870912) : suspect object in the stack (no more
        than 1 good measurement, 2 or more suspect or good stack
        measurement)
        * BAD_STACK (1073741824 (2^30)) : poor-quality stack object (no more
        than 1 good or suspect measurement)

    Among the ``"objInfoFlag"``, the following are better to be dropped because
    they are surely not usable for differential photometry:

        * 1, 2, 4, 8, 32, 128, 256, 512, 1024, 8388608, 16777216

    or in binary position (``del_flags``),

        * 0, 1, 2, 3, 5, 7, 8, 9, 10, 23, 24

    (plus maybe 2048(2^11) because centroiding may not work properly?)
    '''
    if service.lower() == "mast":
        col_flag = "objInfoFlag"
        # col_imag =
    elif service.lower() == "vizier":
        col_flag = "f_objID"
        col_imag = "imag"
        col_iKmag = "iKmag"

    N_old = len(table)
    if del_flags is not None:
        idx2remove = []
        for i, row in enumerate(table):
            b_flag = list(f"{row[col_flag]:031b}")
            for bin_pos in del_flags:
                if b_flag[-bin_pos] == '1':
                    idx2remove.append(i)
        table.drop(idx2remove, inplace=True)

        N_fobj = len(table)
        mask_str(N_fobj, N_old, f"{col_flag} ({del_flags})")

        N_old = N_fobj

    if drop_by_Kron:
        dmag = (table["imag"] - table["iKmag"])
        mask = (dmag > 0.05)
        table = table.loc[~mask]

        N_Kron = len(table)
        mask_str(N_Kron, N_old, "the Kron magnitude criterion")

    return table


"""
def sdss2BV(g, r, gerr=None, rerr=None):
    '''
    Pan-STARRS DR1 (PS1) uses AB mag.
    https://www.sdss.org/dr12/algorithms/fluxcal/#SDSStoAB
    Jester et al. (2005) and Lupton (2005):
    https://www.sdss.org/dr12/algorithms/sdssubvritransform/
    Here I used Lupton. Application to PS1, it seems like Jester - Lupton VS
    Lupton V mag is scattered around -0.013 +- 0.003 (minmax = -0.025, -0.005)
    --> Lupton conversion is fainter.
    V = g - 0.5784*(g - r) - 0.0038;  sigma = 0.0054
    '''
    if gerr is None:
        gerr = np.zeros_like(g)

    if rerr is None:
        rerr = np.zeros_like(r)

    V = g - 0.5784 * (g - r) - 0.0038
    dV = np.sqrt((1.5784 * gerr)**2 + (0.5784 * rerr)**2 + 0.0052**2)
    return V, dV
"""


def panstarrs_query(ra_deg, dec_deg, radius=None, inner_radius=None,
                    width=None, height=None, columns=None, column_filters={}):
    """
    DEPRECATED
    """
    print("panstarrs_query is deprecated. Use PanSTARRS1.")
    ps1 = PanSTARRS1(ra=ra_deg*u.deg, dec=dec_deg*u.deg,
                     radius=radius, inner_radius=inner_radius,
                     width=width, height=height,
                     columns=columns, column_filters=column_filters)

    return ps1.queried
