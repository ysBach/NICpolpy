"""
Currently only the half-wave plate angle (HWP angle) of 0, 22.5, 45,
67.5 combination is available.

Primitive naming:
<lin/circ/all>_<oe/sr>_<n>set
- lin/circ/all: linear, circular, or all Stoke's parameter will be
  determined.
- oe/sr: o- and e-ray (e.g., Wollaston or Savart prism is used) or
  single-ray version
- n: the number of HWP angles used. 3 or 4.
"""
import numpy as np
from .util import err_prop, convert_pct, convert_deg

__all__ = ["calc_stokes", "calc_pol_r",
           'calc_qu_4set', 'correct_eff', 'correct_off', 'correct_pa', 'calc_pol']


def calc_stokes(
        o_000, o_450, o_225, o_675, e_000, e_450, e_225, e_675,
        do_000=0, do_450=0, do_225=0, do_675=0,
        de_000=0, de_450=0, de_225=0, de_675=0,
        p_eff=1, dp_eff=0,
        rot_q=0, rot_u=0, q_off=0, u_off=0, dq_off=0, du_off=0,
        pa_off=0, dpa_off=0, pa_obs=0, pa_ccw=True,
        use_pct=False,
        use_deg=False,
        eminuso=True,
):
    """A one-line calculator of Stokes' parameters in 4-set linear polarimetry.

    Parameters
    ----------
    o_000, o_450, o_225, o_675, e_000, e_450, e_225, e_675 : float, ndarray
        The o-ray or e-ray intensities of 0, 22.5, 45, 67.5 degree HWP angles.
    do_000, do_450, do_225, do_675, de_000, de_450, de_225, de_675 : float, ndarray, optional.
        The o-ray or e-ray intensity errors of 0, 22.5, 45, 67.5 degree HWP angles.
    p_eff, dp_eff : float, optional.
        The polarimetric efficiency and its error.
    rot_q, rot_u : float, ndarray, optional.
        The instrumental rotation angle of the polarimetric efficiency.
    q_off, dq_off, u_off, du_off : float, optional.
        The offset of the q, u values (due to the polarization of the
        instrument) and corresponding erorrs.
     pa_off, dpa_off : float, optional.
        The offset of the optic's position angle (due to the instrument) and
        corresponding error.
    pa_obs : float, ndarray, optional.
        The position angle of the optics from the observation (usually in the
        FITS header information).
    pa_ccw : bool, optional.
        Whether the position angles (`pa_off` and `pa_obs`) are measured in
        counter-clockwise direction in the projected plane seen by the observer
        (i.e., the qraw-uraw (x: q_raw, y: u_raw) plane). If it is `True`,
        ``offset = pa_off - pa_obs`` will be used. Otherwise, ``offset =
        -(pa_off - pa_obs)`` will be used.
        Default: `True`.
    use_pct, use_deg : bool, optional.
        Whether to return percentage/degrees or natural unit. If `True`,
        `p_eff`, `dp_eff`, `q_off`, `u_off`, `dq_off`, `du_off` must be in
        percentage and `rot_q`, `rot_u`, `pa_off`, `dpa_off`, `pa_obs` must be
        in degrees.
    eminuso : bool, optional.
        Whether the q or u values are calculated in the way that "e-ray minus
        o-ray" convention. (See Notes)
        Default: `True`.

    Returns
    -------
    pol, thp, dpol, dthp : float, ndarray
        The polarization degree, polarization vector angle (angle from the
        North to East, CCW, of the strongest E-field vector) and error. The
        unit can be natural (if `use_pct` or `use_deg` is/are `False`) or
        percentage/degree (if `use_pct` or `use_deg` is/are `True`).

    Notes
    -----
    Why not o-ray minus e-ray, but e-ray minus o-ray is the default? For
    example, ``q = (I_e-I_o) / (I_e+I_o)`` if `eminuso` is `True`, where
    ``I_e`` and ``I_o`` are the e- and o-ray of 0 degree HWP observation,
    respectively. This is one of the widely used convention in observational
    polarimetry in solar system studies. In the real calculation, ``I_e =
    sqrt(e_000*o_045)`` and ``I_o = sqrt(o_000*e_045)`` are used to cancel out
    the effect of "systematic error by flat fielding". The opposite is the
    usual convention in physical sciences ("o-ray minus e-ray" convention).
    They differ by this way (opposite sign in the q and u values, i.e.,
    "polarization angle" convention is 90-degree offset) because the
    observers/experimenters are interested in the polarization angle with
    respect to the scattering plane normal vector, while the theoreticians are
    simply interested in the polarization angle with respect to the scattering
    plane.

    Examples
    --------
    >>> calc_stokes(1, 1, 1, 1, 2, 1, 1, 2,)

    """
    qu_dqu_raw = calc_qu_4set(o_000=o_000, o_450=o_450, o_225=o_225, o_675=o_675,
                              e_000=e_000, e_450=e_450, e_225=e_225, e_675=e_675,
                              do_000=do_000, do_450=do_450, do_225=do_225, do_675=do_675,
                              de_000=de_000, de_450=de_450, de_225=de_225, de_675=de_675,
                              out_pct=use_pct, eminuso=eminuso)
    qu_dqu_eff = correct_eff(*qu_dqu_raw, p_eff=p_eff, dp_eff=dp_eff,
                             in_pct=use_pct, out_pct=use_pct)
    qu_dqu_off = correct_off(*qu_dqu_eff, rot_q=rot_q, rot_u=rot_u,
                             q_off=q_off, u_off=u_off, dq_off=dq_off, du_off=du_off,
                             in_pct=use_pct, in_deg=use_deg, out_pct=use_pct)
    qu_dqu_pa = correct_pa(*qu_dqu_off, pa_off=pa_off, dpa_off=dpa_off, pa_obs=pa_obs,
                           pa_ccw=pa_ccw, in_pct=use_pct, in_deg=use_deg, out_pct=use_pct)
    pu_dpu = calc_pol(*qu_dqu_pa, in_pct=use_pct, out_pct=use_pct, out_deg=use_deg)

    return pu_dpu


def calc_qu_4set(
        o_000, o_450, o_225, o_675, e_000, e_450, e_225, e_675,
        do_000=0, do_450=0, do_225=0, do_675=0,
        de_000=0, de_450=0, de_225=0, de_675=0,
        out_pct=False,
        eminuso=True
):
    """ Calculate the q, u, dq, and du of the 4 sets (HWP angles) of O-E rays.

    Parameters
    ----------
    o_000, o_450, o_225, o_675, e_000, e_450, e_225, e_675 : float, ndarray
        The o-ray or e-ray intensities of 0, 22.5, 45, 67.5 degree HWP angles.
    do_000, do_450, do_225, do_675, de_000, de_450, de_225, de_675 : float, ndarray, optional.
        The o-ray or e-ray intensity errors of 0, 22.5, 45, 67.5 degree HWP angles.
    out_pct : bool, optional.
        If `True`, the output will be in percentage.
        Default: `False`.
    eminuso : bool, optional.
        Whether the q or u values are calculated in the way that "e-ray minus
        o-ray" convention. (See Notes)
        Default: `True`.

    Notes
    -----
    Why not o-ray minus e-ray, but e-ray minus o-ray is the default? For
    example, ``q = (I_e-I_o) / (I_e+I_o)`` if `eminuso` is `True`, where
    ``I_e`` and ``I_o`` are the e- and o-ray of 0 degree HWP observation,
    respectively. This is one of the widely used convention in observational
    polarimetry in solar system studies. In the real calculation, ``I_e =
    sqrt(e_000*o_045)`` and ``I_o = sqrt(o_000*e_045)`` are used to cancel out
    the effect of "systematic error by flat fielding". The opposite is the
    usual convention in physical sciences ("o-ray minus e-ray" convention).
    They differ by this way (opposite sign in the q and u values, i.e.,
    "polarization angle" convention is 90-degree offset) because the
    observers/experimenters are interested in the polarization angle with
    respect to the scattering plane normal vector, while the theoreticians are
    simply interested in the polarization angle with respect to the scattering
    plane.

    Returns
    -------
    q, u, dq, du : float, ndarray
        The q, u values and their errors. The unit can be natural (if `out_pct`
        is `False`) or percentage (if `out_pct` is `True`).
    """
    s_000 = err_prop(de_000/e_000, do_000/o_000)
    s_450 = err_prop(de_450/e_450, do_450/o_450)
    s_225 = err_prop(de_225/e_225, do_225/o_225)
    s_675 = err_prop(de_675/e_675, do_675/o_675)

    rq = np.sqrt((e_000/o_000)/(e_450/o_450))
    ru = np.sqrt((e_225/o_225)/(e_675/o_675))
    sign = 1 if eminuso else -1
    q = sign*(rq - 1)/(rq + 1)
    u = sign*(ru - 1)/(ru + 1)
    dq = (rq/(rq + 1)**2)*err_prop(s_000, s_450)
    du = (ru/(ru + 1)**2)*err_prop(s_225, s_675)
    q, u, dq, du = convert_pct(q, u, dq, du, already=False, convert2unit=out_pct)
    return q, u, dq, du


# TODO: make calc_qu_3set, which uses 0, 60, 120 degree data.
def correct_eff(
        q, u, dq=0, du=0,
        p_eff=1,
        dp_eff=0,
        in_pct=False,
        out_pct=False
):
    """ Correct the polarimetric efficiency.

    Parameters
    ----------
    q, u : float, ndarray
        The q, u values from the polarimetry.
    dq, du : float, ndarray, optional.
        The q, u errors from the polarimetry.
    p_eff, dp_eff : float, optional.
        The polarimetric efficiency and its error.
    in_pct : bool, optional.
        If True, the input will be in percentage.
        Default: `False`.
    out_pct : bool, optional.
        If True, the output will be in percentage.
        Default: `False`.

    Returns
    -------
    q_eff, u_eff, dq_eff, du_eff : float, ndarray
        The q, u values after the polarimetric efficiency correction and their
        errors. The unit can be natural (if `out_pct` is `False`) or percentage
        (if `out_pct` is `True`).
    """
    q, dq, u, du, p_eff, dp_eff = convert_pct(
        q, dq, u, du, p_eff, dp_eff, already=in_pct, convert2unit=False)

    q_eff = q/p_eff
    u_eff = u/p_eff

    dq_eff = np.abs(q_eff)*err_prop(dq/q, dp_eff/p_eff)
    du_eff = np.abs(u_eff)*err_prop(du/u, dp_eff/p_eff)

    q_eff, u_eff, dq_eff, du_eff = convert_pct(q_eff, u_eff, dq_eff, du_eff,
                                               already=False, convert2unit=out_pct)
    return q_eff, u_eff, dq_eff, du_eff


def correct_off(
        q, u, dq=0, du=0,
        rot_q=0, rot_u=0,
        q_off=0, u_off=0,
        dq_off=0, du_off=0,
        in_pct=False,
        in_deg=False,
        out_pct=False
):
    ''' Correct the instrument-induced polarization due to the offsets.

    Parameters
    ----------
    q, u : float, ndarray
        The q, u values from the polarimetry.
    dq, du : float, ndarray, optional.
        The q, u errors from the polarimetry.
    rot_q, rot_u : float, ndarray, optional.
        The instrumental rotation angle of the polarimetric efficiency.
    q_off, dq_off, u_off, du_off : float, optional.
        The offset of the q, u values (due to the polarization of the
        instrument) and corresponding erorrs.
    in_pct : bool, optional.
        If True, the input will be in percentage.
        Default: `False`.
    in_deg : bool, optional.
        If True, the input will be in degree.
        Default: `False`.
    out_pct : bool, optional.
        If True, the output will be in percentage.
        Default: `False`.

    Returns
    -------
    q_rot, u_rot, dq_rot, du_rot : float, ndarray
        The q, u values after the instrumental offset correction and their
        errors. The unit can be natural (if `out_pct` is `False`) or percentage
        (if `out_pct` is `True`).

    Notes
    -----
    Assumed: rotator angle (INSROT-like value) is assumed to have zero error.
    '''
    q, dq, u, du, q_off, dq_off, u_off, du_off = convert_pct(
        q, dq, u, du, q_off, dq_off, u_off, du_off, already=in_pct, convert2unit=False
    )
    rot_q, rot_u = convert_deg(rot_q, rot_u, already=in_deg, convert2unit=False)

    cos2q = np.cos(2*rot_q)
    sin2q = np.sin(2*rot_q)
    cos2u = np.cos(2*rot_u)
    sin2u = np.sin(2*rot_u)
    q_rot = q - (cos2q*q_off - sin2q*u_off)
    u_rot = u - (cos2u*q_off - sin2u*u_off)

    dq_rot = err_prop(dq, cos2q*dq_off, sin2q*du_off)
    du_rot = err_prop(du, cos2u*dq_off, sin2u*du_off)

    q_rot, u_rot, dq_rot, du_rot = convert_pct(q_rot, u_rot, dq_rot, du_rot,
                                               already=False, convert2unit=out_pct)
    return q_rot, u_rot, dq_rot, du_rot


def correct_pa(
        q, u, dq=0, du=0,
        pa_off=0,
        dpa_off=0,
        pa_obs=0,
        pa_ccw=True,
        in_pct=False,
        in_deg=False,
        out_pct=False
):
    '''Convert the q, u values from image coordinate to the celestial one.

    Notes
    -----
    Assumed: optic's position angle (PA or INST-PA-like value) is assumed to
    have zero error. The `pa_obs` value must be used with correct `pa_ccw`.
    For instance, Pirka MSI should use the header keyword ``"INST-PA"`` with
    `pa_ccw=True`, while NHAO NIC should use ``"PA"`` with `pa_ccw=False`.

    Parameters
    ----------
    q, u : float, ndarray
        The q, u values from the polarimetry.
    dq, du : float, ndarray, optional.
        The q, u errors from the polarimetry.
    pa_off, dpa_off : float, optional.
        The offset of the optic's position angle (due to the instrument) and
        corresponding error.
    pa_obs : float, ndarray, optional.
        The position angle of the optics from the observation (usually in the
        FITS header information).
    pa_ccw : bool, optional.
        Whether the position angles (`pa_off` and `pa_obs`) are measured in
        counter-clockwise direction in the projected plane seen by the observer
        (i.e., the qraw-uraw (x: q_raw, y: u_raw) plane). If it is `True`,
        ``offset = pa_off - pa_obs`` will be used. Otherwise, ``offset =
        -(pa_off - pa_obs)`` will be used.
        Default: `True`.
    in_pct : bool, optional.
        If True, the input will be in percentage.
        Default: `False`.
    in_deg : bool, optional.
        If True, the input will be in degree.
        Default: `False`.
    out_pct : bool, optional.
        If True, the output will be in percentage.
        Default: `False`.

    Returns
    -------
    q_inst, u_inst, dq_inst, du_inst : float, ndarray
        The q, u values after the position angle correction (i.e., in the
        celestial coordinate rather than the image coordinate) and their
        errors. The unit can be natural (if `out_pct` is `False`) or percentage
        (if `out_pct` is `True`).
    '''
    q, dq, u, du = convert_pct(q, dq, u, du, already=in_pct, convert2unit=False)
    pa_off, dpa_off, pa_obs = convert_deg(
        pa_off, dpa_off, pa_obs, already=in_deg, convert2unit=False
    )

    sign = 1 if pa_ccw else -1
    offset = sign*(pa_off - pa_obs)
    cos2o = np.cos(2*offset)
    sin2o = np.sin(2*offset)
    q_inst = cos2o*q + sin2o*u
    u_inst = -sin2o*q + cos2o*u

    dq_inst = err_prop(cos2o*dq, sin2o*du, 2*sin2o*q*dpa_off, 2*cos2o*u*dpa_off)
    du_inst = err_prop(sin2o*dq, cos2o*du, 2*cos2o*q*dpa_off, 2*sin2o*u*dpa_off)

    q_inst, u_inst, dq_inst, du_inst = convert_pct(q_inst, u_inst, dq_inst, du_inst,
                                                   already=False, convert2unit=out_pct)
    return q_inst, u_inst, dq_inst, du_inst


def calc_pol(
        q, u, dq=0, du=0,
        in_pct=False,
        out_pct=False,
        out_deg=False
):
    """ Calculate the polarization degree and error.
    Parameters
    ----------
    q, u : float, ndarray
        The q, u values from the polarimetry.
    dq, du : float, ndarray, optional.
        The q, u errors from the polarimetry.
    in_pct : bool, optional.
        If True, the input will be in percentage.
        Default: `False`.
    out_pct : bool, optional.
        If True, the output will be in percentage.
        Default: `False`.
    out_deg : bool, optional.
        If True, the output will be in degree.
        Default: `False`.

    Returns
    -------
    pol, thp, dpol, dthp : float, ndarray
        The polarization degree, polarization vector angle (angle from the
        North to East, CCW, of the strongest E-field vector) and error. The
        unit can be natural (if `out_pct` or `out_deg` is/are `False`) or
        percentage/degree (if `out_pct` or `out_deg` is/are `True`).
    """
    q, dq, u, du = convert_pct(q, dq, u, du, already=in_pct, convert2unit=False)
    pol = np.sqrt(q**2 + u**2)
    dpol = err_prop(q*dq, u*du)/pol
    thp = 0.5*np.arctan2(u, q)
    dthp = err_prop(q*du, u*dq)/(2*pol**2)

    pol, dpol = convert_pct(pol, dpol, already=False, convert2unit=out_pct)
    thp, dthp = convert_deg(thp, dthp, already=False, convert2unit=out_deg)
    return pol, thp, dpol, dthp


def calc_pol_r(
        pol, thp,
        dpol=0,
        dthp=0,
        suntargetpa=0,
        dsuntargetpa=0,
        in_pct=False,
        in_deg=False,
        out_pct=False,
        out_deg=False
):
    """ Calculate the "proper" polarization degree and error (following B.
    Lyot's definition).

    Parameters
    ----------
    pol, thp : float, ndarray
        The polarization degree and polarization vector angle (angle from the
        North to East, CCW, of the strongest E-field vector).
    dpol, dthp : float, ndarray, optional.
        The errors of the polarization degree and polarization vector angle.
    suntargetpa : float, optional.
        The position angle of the sun-target vector (projected to the sky
        plane) from the observer's position.
    dsuntargetpa : float, optional.
        The error of `suntargetpa` (see above). This is normally 0, unless the
        orbital element of the solar system body is very uncertain.
    in_pct, in_deg : bool, optional.
        If True, the input will be in percentage/degree.
        Default: `False`.
    out_pct, out_deg : bool, optional.
        If True, the output will be in percentage/degree.
        Default: `False`.

    Returns
    -------
    polr, thr, dpolr, dthr : float, ndarray
        The "proper" polarization degree, polarization vector angle (angle from
        the "scattering plane normal vector" to the strongest E-field vector,
        North to East direction), and their errors.
    """
    pol, dpol = convert_pct(pol, dpol, already=in_pct, convert2unit=False)
    thp, dthp, suntargetpa, dsuntargetpa = convert_deg(thp, dthp, suntargetpa, dsuntargetpa,
                                                       already=in_deg, convert2unit=False)
    thr = thp + suntargetpa
    # if np.array(thr).size == 1:
    #     thr += np.pi/2 if thr < 0 else 0
    #     thr -= np.pi/2 if thr > np.pi else 0
    # else:
    #     thr[thr < 0] += np.pi/2
    #     thr[thr > np.pi] -= np.pi/2
    cos2r = np.cos(2*thr)
    sin2r = np.sin(2*thr)
    polr = pol*cos2r
    dpolr = np.max([err_prop(dpol*cos2r, pol*(-2*sin2r)*dthp), dpol], axis=0)
    dthr = err_prop(dthp, dsuntargetpa)
    polr, dpolr = convert_pct(polr, dpolr, already=False, convert2unit=out_pct)
    thr, dthr = convert_deg(thr, dthr, already=False, convert2unit=out_deg)

    return polr, thr, dpolr, dthr


class PolObjMixin:
    def _set_qu(self):
        if self.mode == "lin_oe_4set":
            # The ratios, r for each HWP angle
            self.r000 = self.i000_e/self.i000_o
            self.r045 = self.i450_e/self.i450_o
            self.r225 = self.i225_e/self.i225_o
            self.r675 = self.i675_e/self.i675_o

            # noise-to-signal (dr/r) of I_e/I_o for each HWP angle
            self.ns000 = err_prop(self.di000_e/self.i000_e,
                                  self.di000_o/self.i000_o)
            self.ns450 = err_prop(self.di450_e/self.i450_e,
                                  self.di450_o/self.i450_o)
            self.ns225 = err_prop(self.di225_e/self.i225_e,
                                  self.di225_o/self.i225_o)
            self.ns675 = err_prop(self.di675_e/self.i675_e,
                                  self.di675_o/self.i675_o)

            # The q/u values
            self.r_q = np.sqrt(self.r000/self.r045)
            self.r_u = np.sqrt(self.r225/self.r675)
            self.q0 = (self.r_q - 1)/(self.r_q + 1)
            self.u0 = (self.r_u - 1)/(self.r_u + 1)

            # The errors
            s_q = err_prop(self.ns000, self.ns450)
            s_u = err_prop(self.ns225, self.ns675)
            self.dq0 = (self.r_q/(self.r_q + 1)**2 * s_q)
            self.du0 = (self.r_u/(self.r_u + 1)**2 * s_u)
        else:
            raise ValueError(f"{self.mode} not understood.")


'''
    @property
    def _set_qu(self):
        self.o_val = {}
        self.e_val = {}
        self.do_val = {}
        self.de_val = {}
        self.ratios = {}
        self.v_ratios = {}  # variances, not standard deviations
        self.q = {}
        self.u = {}
        self.dq = {}
        self.du = {}
        self.step = 0
        self.messages = {}

        for hwp in HWPS:
            idx = np.where(self.order == hwp)
            o_vals = self.orays[:, idx]
            e_vals = self.erays[:, idx]
            do_vals = self.dorays[:, idx]
            de_vals = self.derays[:, idx]
            self.o_val[hwp] = o_vals
            self.e_val[hwp] = e_vals
            self.do_val[hwp] = do_vals
            self.de_val[hwp] = de_vals
            self.ratios[hwp] = e_vals / o_vals
            self.v_ratios[hwp] = (do_vals/o_vals)**2 + (de_vals/e_vals)**2

        self.r_q = np.sqrt(self.ratios[HWPS[0]]/self.ratios[HWPS[2]])
        self.r_u = np.sqrt(self.ratios[HWPS[1]]/self.ratios[HWPS[3]])
        self.q[0] = (self.r_q - 1)/(self.r_q + 1)
        self.u[0] = (self.r_u - 1)/(self.r_u + 1)
        self.dq[0] = (self.r_q / (self.r_q + 1)**2
                      * np.sqrt(self.v_ratios[HWPS[0]]
                                + self.v_ratios[HWPS[2]])
                      )
        self.du[0] = (self.r_u / (self.r_u + 1)**2
                      * np.sqrt(self.v_ratios[HWPS[1]]
                                + self.v_ratios[HWPS[3]])
                      )
        self.messages[0] = (f"Initialize with input data (order: {self.order})"
                            + f" for {self.ndata} data-sets.")

    def set_check_2d_4(self, name, arr, ifnone=None):
        if (ifnone is not None) and (arr is None):
            a = ifnone
        else:
            a = np.atleast_2d(arr)
            if a.shape[1] != 4:
                raise ValueError(f"{name} must be a length 4 or (N, 4) array.")

        setattr(self, name, a)
'''


class LinPolOE4(PolObjMixin):
    def __init__(self, i000_o, i000_e, i450_o, i450_e,
                 i225_o, i225_e, i675_o, i675_e,
                 di000_o=None, di000_e=None, di450_o=None, di450_e=None,
                 di225_o=None, di225_e=None, di675_o=None, di675_e=None
                 ):
        """
        Parameters
        ----------
        ixxx_[oe] : array-like
            The intensity (in linear scale, e.g., sky-subtracted ADU) in
            the half-wave plate angle of ``xxx/10`` degree in the ``o``
            or ``e``-ray.
        dixxx_[oe] : array-like, optinal
            The 1-sigma error-bars of the corresponding ``ixxx_[oe]``.
            It must have the identical length as ``ixxx_[oe]`` if not
            None.
        """
        self.mode = "lin_oe_4set"
        self.i000_o = np.array(i000_o)
        self.i000_e = np.array(i000_e)
        self.i450_o = np.array(i450_o)
        self.i450_e = np.array(i450_e)
        self.i225_o = np.array(i225_o)
        self.i225_e = np.array(i225_e)
        self.i675_o = np.array(i675_o)
        self.i675_e = np.array(i675_e)

        if not (self.i000_o.shape == self.i000_e.shape
                == self.i450_o.shape == self.i450_e.shape
                == self.i225_o.shape == self.i225_e.shape
                == self.i675_o.shape == self.i675_e.shape):
            raise ValueError("all ixxx_<oe> must share the identical shape.")

        _dis = dict(di000_o=di000_o, di000_e=di000_e,
                    di450_o=di450_o, di450_e=di450_e,
                    di225_o=di225_o, di225_e=di225_e,
                    di675_o=di675_o, di675_e=di675_e)
        for k, v in _dis.items():
            if v is None:
                v = np.zeros_like(getattr(self, k[1:]))
            setattr(self, k, np.array(v))

        if not (self.di000_o.shape == self.di000_e.shape
                == self.di450_o.shape == self.di450_e.shape
                == self.di225_o.shape == self.di225_e.shape
                == self.di675_o.shape == self.di675_e.shape):
            raise ValueError("all dixxx_<oe> must share the identical shape.")

    # TODO: This should apply for any linear polarimetry using HWP..?
    # So maybe should move to Mixin class.
    def calc_pol(self, p_eff=1., dp_eff=0.,
                 q_inst=0., u_inst=0., dq_inst=0., du_inst=0.,
                 rot_instq=0., rot_instu=0.,
                 pa_inst=0., theta_inst=0., dtheta_inst=0.,
                 percent=True, degree=True):
        '''
        Parameters
        ----------
        p_eff, dp_eff : float, optional.
            The polarization efficiency and its error. Defaults to ``1``
            and ``0``.
        q_inst, u_inst, dq_inst, du_inst : float, optional
            The instrumental q (Stokes Q/I) and u (Stokes U/I) values
            and their errors. All defaults to ``0``.
        rot_instq, rot_instu: float, array-like, optional.
            The instrumental rotation. In Nayoro Pirka MSI manual, the
            average of ``INS-ROT`` for the HWP angle 0 and 45 at the
            start and end of exposure (total 4 values) are to be used
            for ``rot_instq``, etc. If array, it must have the same
            length as ``ixxx_[oe]``.
        pa_inst : float, array-like, optional.
            The position angle (North to East) of the instrument.
            If array-like, it must have the same length as
            ``ixxx_[oe]``.
        theta_inst, dtheta_inst : float, optinoal.
            The instrumental polarization rotation angle theta and its
            error.
        percent : bool, optional.
            Whether ``p_eff``, ``dp_eff``, ``q_inst``, ``dq_inst``,
            ``u_inst``, ``du_inst`` are in percent unit. Defaults to
            ``True``.
        degree : bool, optional.
            Whether ``rot_instq``, ``rot_instu``, ``theta_inst``,
            ``dtheta_inst`` are in degree unit. Otherwise it must be in
            radians. Defaults to ``True``.
        '''

        if percent:
            # polarization efficiency
            p_eff = p_eff/100
            dp_eff = dp_eff/100
            # instrumental polarization
            q_inst = q_inst/100
            u_inst = u_inst/100
            dq_inst = dq_inst/100
            du_inst = du_inst/100

        self.p_eff = p_eff
        self.dp_eff = dp_eff
        self.q_inst = q_inst
        self.u_inst = u_inst
        self.dq_inst = dq_inst
        self.du_inst = du_inst

        if degree:
            # instrument's rotation angle from FITS header
            rot_instq = np.deg2rad(rot_instq)
            rot_instu = np.deg2rad(rot_instu)
            # position angle and instrumental polarization angle
            pa_inst = np.deg2rad(pa_inst)
            theta_inst = np.deg2rad(theta_inst)
            dtheta_inst = np.deg2rad(dtheta_inst)

        self.rot_instq = rot_instq
        self.rot_instu = rot_instu
        self.pa_inst = pa_inst
        self.theta_inst = theta_inst
        self.dtheta_inst = dtheta_inst

        self._set_qu()

        self.q1 = self.q0/self.p_eff
        self.u1 = self.u0/self.p_eff
        self.dq1 = err_prop(self.dq0, np.abs(self.q1)*self.dp_eff)/self.p_eff
        self.du1 = err_prop(self.du0, np.abs(self.u1)*self.dp_eff)/self.p_eff

        # self.messages = ("Polarization efficiency corrected by "
        #                   + f"p_eff = {self.p_eff}, "
        #                   + f"dp_eff = {self.dp_eff}.")

        rotq = (np.cos(2*self.rot_instq), np.sin(2*self.rot_instq))
        rotu = (np.cos(2*self.rot_instu), np.sin(2*self.rot_instu))
        self.q2 = (self.q1 - (self.q_inst*rotq[0] - self.u_inst*rotq[1]))
        self.u2 = (self.u1 - (self.q_inst*rotu[1] + self.u_inst*rotu[0]))
        # dq_inst_rot = err_prop(self.dq_inst*rotq[0], self.du_inst*rotq[1])
        # du_inst_rot = err_prop(self.dq_inst*rotu[1], self.du_inst*rotu[0])
        self.dq2 = err_prop(self.dq1, self.dq_inst*rotq[0], self.du_inst*rotq[1])
        self.du2 = err_prop(self.du1, self.dq_inst*rotu[1], self.du_inst*rotu[0])

        theta = self.theta_inst - self.pa_inst
        rot = (np.cos(2*theta), np.sin(2*theta))
        self.q3 = +1*self.q2*rot[0] + self.u2*rot[1]
        self.u3 = -1*self.q2*rot[1] + self.u2*rot[0]
        self.dq3 = err_prop(rot[0]*self.dq2,
                            rot[1]*self.du2,
                            2*self.u3*dtheta_inst)
        self.du3 = err_prop(rot[1]*self.dq2,
                            rot[0]*self.du2,
                            2*self.q3*dtheta_inst)

        self.pol = np.sqrt(self.q3**2 + self.u3**2)
        self.dpol = err_prop(self.q3*self.dq3, self.u3*self.du3)/self.pol
        self.theta = 0.5*np.arctan2(self.u3, self.q3)
        self.dtheta = 0.5*self.dpol/self.pol

        if percent:
            self.pol *= 100
            self.dpol *= 100

        if degree:
            self.theta = np.rad2deg(self.theta)
            self.dtheta = np.rad2deg(self.dtheta)


def proper_pol(pol, theta, psang, degree=True):
    if not degree:
        theta = np.rad2deg(theta)
        psang = np.rad2deg(psang)

    dphi = psang + 90
    dphi = dphi - 180*(dphi//180)
    theta_r = theta - dphi
    pol_r = pol*np.cos(2*np.deg2rad(theta_r))
    if not degree:
        theta_r = np.deg2rad(theta_r)
    return pol_r, theta_r


"""
    @classmethod
    def from1d(self, orays, erays, dorays=None, derays=None,
               order=[0, 22.5, 45, 67.5]):
        '''
        Parameters
        ----------
        orays, erays: 1-d array-like or 2-d of shape ``(N, 4)``
            The oray and eray intensities in ``(N, 4)`` shape. The four
            elements are assumed to be in the order of HWP angle as in
            ``order``.
        order : array-like, optional.
            The HWP angle order in ``orays`` and ``erays``.
        '''
        # orays, erays, dorays, derays to 2-d array
        self.set_check_2d_4('orays', orays)
        self.set_check_2d_4('erays', erays)
        self.set_check_2d_4('dorays', dorays, ifnone=np.zeros_like(self.orays))
        self.set_check_2d_4('derays', derays, ifnone=np.zeros_like(self.erays))

        if not (self.orays.shape == self.erays.shape
                == self.dorays.shape == self.derays.shape):
            raise ValueError("orays, erays, dorays, derays must share "
                             + "the identical shape.")
        self.ndata = self.orays.shape[0]

        # check order
        self.order = np.atleast_1d(order)
        if self.order.ndim > 1:
            raise ValueError("order must be 1-d array-like.")
        elif not all(val in order for val in HWPS):
            raise ValueError(f"order must contain all the four of {HWPS}.")
        elif self.order.shape[0] != 4:
            raise ValueError(f"order must contain only the four of {HWPS}.")

    def correct_peff(self, p_eff, dp_eff=0, percent=True):
        if percent:
            p_eff = p_eff/100
            dp_eff = dp_eff/100
        self.p_eff = p_eff
        self.dp_eff = dp_eff

        self.step += 1
        idx = self.step

        self.q[idx] = self.q/self.p_eff
        self.u[idx] = self.u/self.p_eff
        sig_q = self.dq/self.p_eff
        sig_u = self.du/self.p_eff
        sig_p = self.dp_eff/self.p_eff
        self.dq[idx] = self.q[idx] * np.sqrt(sig_q**2 + sig_p**2)
        self.du[idx] = self.u[idx] * np.sqrt(sig_u**2 + sig_p**2)
        self.messages[idx] = ("Polarization efficiency corrected by "
                              + f"p_eff = {self.p_eff}, "
                              + f"dp_eff = {self.dp_eff}.")


        gain_corrected : bool, optional.
            Whether the values are in ADU (if ``False``) or electrons
            (if ``True``). Defaults to ``False``.
"""
