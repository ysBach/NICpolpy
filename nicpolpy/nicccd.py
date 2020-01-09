from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import sep
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian2D
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.time import Time

from ysfitsutilpy import (CCDData_astype, add_to_header, bdf_process, errormap,
                          load_ccd, propagate_ccdmask, set_ccd_gain_rdnoise,
                          trim_ccd)
from ysphotutilpy import (apphot_annulus, ellip_ap_an, fit_Gaussian2D,
                          sep_back, sep_extract)

from .preproc import (cr_reject_nic, find_fourier_peaks, fit_fourier,
                      vertical_correct)
from .util import (DARK_PATHS, FLAT_PATHS, FOURIERSECTS, GAIN, MASK_PATHS,
                   OBJSECTS, OBJSLICES, RDNOISE, USEFUL_KEYS, VERTICALSECTS,
                   infer_filter, split_oe)

# from photutils.centroids import GaussianConst2D


__all__ = ["NICPolImage"]


_PHOT_COLNAMES = ['id', 'xcenter', 'ycenter', 'aparea',
                  'aperture_sum', 'aperture_sum_err',
                  'msky', 'nrej', 'nsky', 'ssky',
                  'source_sum', 'source_sum_err',
                  'mag', 'merr', 'x_fwhm', 'y_fwhm', 'theta']

# TODO: Will it be better to put these ``_xxxx``` functions as @staticmethod?


def _nic_Gaussian2D_fit(ccd, obj, err, g_init=None, fbox=(40, 40), snr=3):
    '''
    obj: pd.DataFrame of sep output
        It must have been sorted such that the 0-th element is our target.
    '''
    # fwhm_rough = 10
    # pos = (obj['x'].values[0], obj['y'].values[0])
    # _, _, std = sigma_clipped_stats(ccd.data, sigma=3, maxiters=5, std_ddof=1)
    # cent = find_center_2dg(ccd=ccd, position_xy=pos, cbox_size=fwhm_rough,
    #                        csigma=3, maxiters=10, ssky=std,
    #                        max_shift=fwhm_rough, max_shift_step=fwhm_rough/2)

    # Set cutout region
    pos_init = (obj['x'].values[0], obj['y'].values[0])
    _, med, _ = sigma_clipped_stats(ccd.data, sigma=3, maxiters=5, std_ddof=1)
    cutkw = dict(position=pos_init, size=fbox)
    cut = Cutout2D(ccd.data, **cutkw)
    ecut = Cutout2D(err.data, **cutkw)
    mask = (cut.data < snr*ecut.data)
    d2fit = np.ma.array(data=cut.data - med, mask=mask)

    # Set initial Gaussian at the cutout coordinate
    pos_init_cut = cut.to_cutout_position(pos_init)
    if g_init is None:
        g_init = Gaussian2D(
            amplitude=obj['peak'].values[0],
            x_mean=pos_init_cut[0],
            y_mean=pos_init_cut[1],
            x_stddev=4,
            y_stddev=4,
            # fixed={'x_mean': True, 'y_mean': True},
            bounds={'x_stddev': (3, 6), 'y_stddev': (3, 6)}
        )

    # Fit to cutout fitting-box
    _res = fit_Gaussian2D(d2fit, g_init, sigma=ecut.data)
    gfit = _res[0]
    fitter = _res[1]
    # Restore the fitted position to original xy coordinate
    pos_cut = (gfit.x_mean.value, gfit.y_mean.value)
    gfit.x_mean.value, gfit.y_mean.value = cut.to_original_position(pos_cut)
    return gfit, fitter


def _sort_obj(ccd, obj, nobj):
    ''' Sort obj such that the 0-th is our target.
    '''
    s_nobj = ("{} objects found; Only the one closest to the FOV center "
              + "(segmentation map label = {}) will be used.")
    ny, nx = ccd.data.shape
    obj['_r'] = np.sqrt((obj['x'] - nx/2)**2 + (obj['y'] - ny/2)**2)
    obj.sort_values('_r', inplace=True)
    s = s_nobj.format(nobj, obj['segm_label'].values[0])
    add_to_header(ccd.header, 'h', s)


def _eap_phot_fit(ccd, err, gfit, f_ap, f_in, f_out):
    fitted = dict(
        positions=(gfit.x_mean.value, gfit.y_mean.value),
        fwhm=(gfit.x_fwhm, gfit.y_fwhm),
        theta=gfit.theta.value
    )
    ap, an = ellip_ap_an(**fitted, f_ap=f_ap, f_in=f_in, f_out=f_out)
    phot = apphot_annulus(ccd, ap, an, error=err, pandas=True)
    phot['x_fwhm'] = fitted['fwhm'][0]
    phot['y_fwhm'] = fitted['fwhm'][1]
    phot['theta'] = fitted['theta']
    return ap, an, phot


def _eap_phot_sep(ccd, obj, fwhm, err, f_ap, f_in, f_out):
    theta = obj['theta'].values[0]
    ap, an = ellip_ap_an(positions=(obj['x'].values[0], obj['y'].values[0]),
                         fwhm=fwhm, theta=theta,
                         f_ap=f_ap, f_in=f_in, f_out=f_out)
    phot = apphot_annulus(ccd, ap, an, error=err, pandas=True)
    phot['x_fwhm'] = fwhm[0]
    phot['y_fwhm'] = fwhm[1]
    phot['theta'] = theta
    return ap, an, phot


def _sep_fwhm(obj):
    _a = obj['a'].values[0]
    _b = obj['b'].values[0]
    scale_fwhm = 2*np.sqrt(2*np.log(2))
    fwhm = np.array([_a, _b])*scale_fwhm
    return fwhm


def _append_to_phot(phot, header, fpath, nobj=1, keys=USEFUL_KEYS):
    if keys is None or len(keys) == 0:
        pass

    colnames = phot.columns
    for k in keys:
        if k in colnames:
            continue
        v = header[k]
        phot[k] = v
    phot['nobj'] = nobj
    phot['file'] = fpath


class NICPolImage:
    def __init__(self, fpath, filt=None, verbose=True):
        self.file = Path(fpath)
        # for future calculation, set float32 dtype:
        self.ccd_raw = CCDData_astype(load_ccd(fpath), 'float32')
        self.filt = infer_filter(self.ccd_raw, filt=filt, verbose=verbose)
        set_ccd_gain_rdnoise(self.ccd_raw,
                             gain=GAIN[self.filt],
                             rdnoise=RDNOISE[self.filt])
        self.gain = self.ccd_raw.gain
        self.rdnoise = self.ccd_raw.rdnoise

    def preproc(self, mdarkpath='default',
                mflatpath_o='default', mflatpath_e='default',
                mask='default', do_verti=True, do_fouri=False,
                verti_fitting_sections=None,
                verti_method='median',
                verti_sigclip_kw=dict(sigma=2, maxiters=5),
                fouri_peak_infer_section="[300:500, :]",
                fouri_peak_sigclip_kw={
                    'sigma_lower': np.inf, 'sigma_upper': 3},
                fouri_min_freq=1/100,
                fouri_max_peaks=3,
                fouri_npool=8,
                fouri_fitting_y_sections=None,
                fouri_subtract_x_sections=None,
                do_crrej=True,
                verbose=False,
                verbose_crrej=False,
                crrej_kw=None
                ):
        """ Do full preprocess to NHAO NIC.
        Parameters
        ----------
        mdarkpath, mflatpath : path-like, None, 'default', optional.
            The paths to master dark and flat. If ``'default'``
            (default), NICpolpy's default dark and flats (in the
            ``dark/`` and ``flat/`` directories in the package) will be
            used. To turn off dark subtraction or flat correction, set
            ``None``.

        mask : ndarray, None, 'default', optional.
            The mask to be used. If ``'default'`` (default), NICpolpy's
            default mask (``mask/`` directory in the package) will be
            used. To turn off any masking, set ``None``.



        Returns
        -------

        Example
        -------
        >>>

        Note
        ----

        """
        if mdarkpath == 'default':
            if verbose:
                print("mdarkpath is 'default': using NICpolpy default dark.")
            self.mdarkpath = DARK_PATHS[self.filt]

        if mflatpath_o == 'default':
            if verbose:
                print("mflatpath_o is 'default': using NICpolpy default flat.")
            self.mflatpath_o = FLAT_PATHS[self.filt]['o']

        if mflatpath_e == 'default':
            if verbose:
                print("mflatpath_e is 'default': using NICpolpy default flat.")
            self.mflatpath_e = FLAT_PATHS[self.filt]['e']

        if mask == 'default':
            if verbose:
                print("mask is 'default': using NICpolpy default mask.")
            # Set mask for high-dark / low-flat pixels
            pixmask = fits.open(MASK_PATHS[self.filt])[0].data.astype(bool)
            self.mask = propagate_ccdmask(self.ccd_raw, pixmask)
        else:
            if mask is None and verbose:
                print("mask is None: no mask will be used (not recommended).")
            self.mask = mask

        self.ccd_proc = self.ccd_raw.copy()

        # ===============================================================
        # * 0. Subtract Dark (regard bias is also subtracted)
        # ===============================================================
        self.ccd_dark = load_ccd(self.mdarkpath)
        self.ccd_bdxx = bdf_process(
            self.ccd_proc,
            mdarkpath=self.mdarkpath,
            dark_scale=True,
            verbose_bdf=verbose,
            verbose_crrej=verbose_crrej
        )
        self.ccd_proc = self.ccd_bdxx.copy()

        # ===============================================================
        # * 1. Subtract vertical patterns.
        # ===============================================================
        if do_verti:
            if verti_fitting_sections is None:
                verti_fitting_sections = VERTICALSECTS

            self.verti_corr_kw = dict(
                fitting_sections=verti_fitting_sections,
                method=verti_method,
                sigclip_kw=verti_sigclip_kw,
                dtype='float32',       # hard-coded
                return_pattern=False,  # hard-coded
                update_header=True,    # hard-coded
                verbose=verbose
            )
            self.ccd_vs = vertical_correct(self.ccd_proc, **self.verti_corr_kw)
            self.ccd_proc = self.ccd_vs.copy()

        # ==============================================================
        # * 2. Subtract Fourier pattern
        # ==============================================================
        if do_fouri:
            self.ccd_fc = self.ccd_proc.copy()

            if fouri_fitting_y_sections is None:
                fouri_fitting_y_sections = FOURIERSECTS[self.filt]

            if fouri_subtract_x_sections is None:
                fouri_subtract_x_sections = "[513:900]"

            self.fouri_trim_kw = dict(
                fits_section=fouri_peak_infer_section
            )

            self.fouri_freq_kw = dict(
                sigclip_kw=fouri_peak_sigclip_kw,
                min_freq=fouri_min_freq,
                max_peaks=fouri_max_peaks
            )
            s_fouri_peak = (
                f"Fourier frequencies obtained from {fouri_peak_infer_section}"
                + f" by sigma-clip with {fouri_peak_sigclip_kw} "
                + f"minimum frequency = {fouri_min_freq} and "
                + f"only upto maximum {fouri_max_peaks} peaks. "
                + "See FIT-AXIS, FIT-NF, FIT-Fiii."
            )

            self.fouri_corr_kw = dict(
                npool=fouri_npool,
                fitting_y_sections=fouri_fitting_y_sections,
                subtract_x_sections=fouri_subtract_x_sections,
                apply_crrej_mask=False,  # hard-coded
                apply_sigclip_mask=True  # hard-coded
            )
            s_fouri_corr = (
                "Fourier series fitted for x positions (in FITS format) "
                + f"{fouri_subtract_x_sections}, using y sections "
                + f"(in FITS format) {fouri_fitting_y_sections}."
            )

            s_fouri_sub = "The obtained Fourier pattern subtracted"
            self.ccd_fc.header["FIT-AXIS"] = (
                "COL",
                "The direction to which Fourier series is fitted."
            )

            # * 2-1. Find Fourier peaks
            # ----------------------------------------------------------
            _t = Time.now()
            f_reg = trim_ccd(self.ccd_fc, **self.fouri_trim_kw)
            self.freqs = find_fourier_peaks(f_reg.data, axis=0,
                                            **self.fouri_freq_kw)

            self.ccd_fc.header["FIT-NF"] = (
                len(self.freqs), "Number of frequencies used for fit.")
            for k, f in enumerate(self.freqs):
                self.ccd_fc.header[f"FIT-F{k+1:03d}"] = (
                    f, f"[1/pix] The {k+1:03d}-th frequency")
            add_to_header(self.ccd_fc.header, 'h', s_fouri_peak,
                          verbose=verbose, t_ref=_t)

            # * 2-2. Fit Fourier series
            # ----------------------------------------------------------
            _t = Time.now()
            res = fit_fourier(
                self.ccd_fc.data,
                self.freqs,
                mask=self.mask,
                filt=self.filt,
                **self.fouri_corr_kw
            )
            self.pattern = res[0]
            self.popts = res[1]
            add_to_header(self.ccd_fc.header, 'h', s_fouri_corr,
                          verbose=verbose, t_ref=_t)

            # * 2-3. Subtract Fourier pattern
            # ----------------------------------------------------------
            _t = Time.now()
            self.ccd_fc.data -= self.pattern
            add_to_header(self.ccd_fc.header, 'h', s_fouri_sub,
                          verbose=verbose, t_ref=_t)

            self.ccd_proc = self.ccd_fc.copy()

        # ==============================================================
        # * 3. Split o-/e-ray
        # ==============================================================
        self.ccd_o_bdxx, self.ccd_e_bdxx = split_oe(
            self.ccd_proc,
            filt=self.filt,
            verbose=verbose
        )
        self.ccd_o_proc = self.ccd_o_bdxx.copy()
        self.ccd_e_proc = self.ccd_e_bdxx.copy()

        self.ccd_dark_o = trim_ccd(
            self.ccd_dark, OBJSECTS[self.filt][0], verbose=verbose)
        self.ccd_dark_e = trim_ccd(
            self.ccd_dark, OBJSECTS[self.filt][1], verbose=verbose)
        self.mask_o = self.mask[OBJSLICES[self.filt][0]]
        self.mask_e = self.mask[OBJSLICES[self.filt][1]]
        self.mask_o_proc = self.mask_o.copy()
        self.mask_e_proc = self.mask_e.copy()

        # ==============================================================
        # * 4. Flat correct
        # ==============================================================
        self.ccd_o_bdfx = bdf_process(
            self.ccd_o_bdxx,
            mflatpath=self.mflatpath_o,
            verbose_bdf=verbose,
            verbose_crrej=verbose_crrej
        )
        self.ccd_e_bdfx = bdf_process(
            self.ccd_e_bdxx,
            mflatpath=self.mflatpath_e,
            verbose_bdf=verbose,
            verbose_crrej=verbose_crrej
        )
        self.ccd_o_proc = self.ccd_o_bdfx.copy()
        self.ccd_e_proc = self.ccd_e_bdfx.copy()

        # ==============================================================
        # * 5. Error calculation
        # ==============================================================
        s = "Pixel error calculated for both o/e-rays"
        _t = Time.now()
        self.flat_o = load_ccd(self.mflatpath_o)
        self.flat_e = load_ccd(self.mflatpath_e)
        # rough estimations for the dark standard deviations
        self.dark_std_o = (self.ccd_dark_o.uncertainty.array
                           * np.sqrt(self.ccd_dark_o.header['NCOMBINE']))
        self.dark_std_e = (self.ccd_dark_e.uncertainty.array
                           * np.sqrt(self.ccd_dark_e.header['NCOMBINE']))
        self.err_o = errormap(
            self.ccd_o_bdxx,
            gain_epadu=self.gain,
            rdnoise_electron=self.rdnoise,
            subtracted_dark=self.ccd_dark_o.data,
            dark_std=self.dark_std_o,
            flat=self.flat_o.data,
            flat_err=self.flat_o.uncertainty.array
        )
        self.err_e = errormap(
            self.ccd_e_bdxx,
            gain_epadu=self.gain,
            rdnoise_electron=self.rdnoise,
            subtracted_dark=self.ccd_dark_e.data,
            dark_std=self.dark_std_e,
            flat=self.flat_e.data,
            flat_err=self.flat_e.uncertainty.array
        )

        add_to_header(self.ccd_o_bdfx.header, 'h', s, t_ref=_t)

        # ==============================================================
        # * 6. CR-rejection
        # ==============================================================
        crkw = dict(crrej_kw=crrej_kw, verbose=verbose_crrej,
                    full=True, update_header=True)
        self.ccd_o_bdfc, self.mask_o_cr, self.crrej_kw = cr_reject_nic(
            self.ccd_o_bdfx,
            mask=self.mask_o,
            **crkw
        )
        self.ccd_e_bdfc, self.mask_e_cr, _ = cr_reject_nic(
            self.ccd_e_bdfx,
            mask=self.mask_e,
            **crkw
        )
        self.ccd_o_proc = self.ccd_o_bdfc.copy()
        self.ccd_e_proc = self.ccd_e_bdfc.copy()
        self.mask_o_proc = self.mask_o_proc | self.mask_o_cr
        self.mask_e_proc = self.mask_e_proc | self.mask_e_cr

    def find_obj(self, thresh=3, bezel_x=(40, 40), bezel_y=(150, 100),
                 box_size=(64, 64), filter_size=(12, 12),
                 minarea=100, verbose=True,
                 **extract_kw):
        """
        Note
        ----
        This includes ``sep``'s `extract`` and ``background``.
        Equivalent processes in photutils may include ``detect_sources``
        and ``source_properties``, and ``Background2D``, respectively.

        Parameters
        ----------
        thresh : float, optional.
            The SNR threshold. It is not an absolute pixel value because
            internally the ``self.err_o`` and ``self.err_e`` will be
            used.

        bezel_x, bezel_y : int, float, list of such, optional.
            The x and y bezels, in ``[lower, upper]`` convention.

        box_size : int or array-like (int) optional.
            The background smooting box size. Default is ``(64, 64)``
            for NIC. **Note**: If array-like, order must be ``[height,
            width]``, i.e., y and x size.

        filter_size : int or array-like (int) optional.
            The 2D median filter size. Default is ``(12, 12)`` for NIC.
            **Note**: If array-like, order must be ``[height, width]``,
            i.e., y and x size.

        minarea : int, optional
            Minimum number of pixels required for an object. Default is
            100 for NIC.

        # gauss_fbox : int, float, array-like of such, optional.
        #     The fitting box size to fit a Gaussian2D function to the
        #     objects found by ``sep``. This is done to automatically set
        #     aperture sizes of the object.

        Returns
        -------

        Example
        -------
        >>>

        Note
        ----

        """
        bkg_kw = dict(maskthresh=0.0, filter_threshold=0.0,
                      box_size=box_size, filter_size=filter_size)
        ext_kw = dict(thresh=thresh, minarea=minarea,
                      bezel_x=bezel_x, bezel_y=bezel_y, **extract_kw)
        sepv = sep.__version__
        s_bkg = f"Background estimated from sep (v {sepv}) with {bkg_kw}."
        s_obj = "Objects found from sep (v {}) with {}."

        _t = Time.now()
        self.bkg_o = sep_back(self.ccd_o_proc.data, mask=self.mask_o, **bkg_kw)
        add_to_header(self.ccd_o_proc.header, 'h', s_bkg,
                      verbose=verbose, t_ref=_t)

        _t = Time.now()
        self.obj_o, self.seg_o = sep_extract(
            self.ccd_o_proc.data,
            bkg=self.bkg_o,
            err=self.err_o,
            mask=self.mask_o,
            **ext_kw)
        _s = s_obj.format(sepv, ext_kw)
        add_to_header(self.ccd_o_proc.header, 'h', _s,
                      verbose=verbose, t_ref=_t)

        _t = Time.now()
        self.bkg_e = sep_back(self.ccd_e_proc.data, mask=self.mask_e, **bkg_kw)
        add_to_header(self.ccd_e_proc.header, 'h', s_bkg,
                      verbose=verbose, t_ref=_t)

        _t = Time.now()
        self.obj_e, self.seg_e = sep_extract(
            self.ccd_e_proc.data,
            bkg=self.bkg_e,
            err=self.err_e,
            mask=self.mask_e,
            **ext_kw)
        _s = s_obj.format(sepv, ext_kw)
        add_to_header(self.ccd_e_proc.header, 'h', _s,
                      verbose=verbose, t_ref=_t)

        self.nobj_o = len(self.obj_o)
        self.nobj_e = len(self.obj_e)

        for ccd, obj, n, oe in zip([self.ccd_o_proc, self.ccd_e_proc],
                                   [self.obj_o, self.obj_e],
                                   [self.nobj_o, self.nobj_e],
                                   ['o', 'e']
                                   ):
            if n < 1 and verbose:
                warn(f"No object found for {oe}-ray of {self.file}!", Warning)
            elif n > 1:
                _sort_obj(ccd, obj, nobj=n)

# def ellipphot_sep(self, f_ap=(2., 2.), f_in=(4., 4.), f_out=(6., 6.),
#                   g_init_o=None, g_init_e=None, keys=USEFUL_KEYS,
#                   verbose=True):
#     '''
#     Parameters
#     ----------
#     f_ap, f_in, f_out: int or float, array-like of such, optional.
#         The factors multiplied to ``fwhm`` to set the aperture ``a``
#         and ``b``, inner sky ``a`` and ``b``, and outer sky ``a``
#         and ``b``, respectively. If scalar, it is assumed to be
#         identical for both ``a`` and ``b`` parameters. Defaults are
#         ``(1.5, 1.5)``, ``(4.0, 4.0)``, and ``(6.0, 6.0)``,
#         respectively, which are de facto standard values used by
#         classical IRAF users.
#     g_init_o, g_init_e : astropy FunctionalModel2D, None, optional.
#         The Gaussian initial guess of the PSF of objects in o- and
#         e-ray, respectively.
#     keys : list of str, None, optional.
#         The list of header keywords to be appended to the
#         ``self.phot_o`` and ``self.phot_e``.
#     Note
#     ----
#     The sep A/B paramters are for the ellipse to describe the
#     "boundary". We need flux-describing Gaussian, so I need to do 2D
#     gaussian fitting.
#     '''
#     s_phot = ('Photometry done for elliptical aperture/annulus with '
#               + f"f_ap = {f_ap}, f_in = {f_in}, f_out = {f_out}"
#               + "for FWHM = ({:.3f}, {:.3f})")
#     fs = dict(f_ap=f_ap, f_in=f_in, f_out=f_out)

#     if self.nobj_o < 1:
#         self.fwhm_o = (np.nan, np.nan)
#         self.ap_o = None
#         self.an_o = None
#         self.phot_o = pd.DataFrame([[np.nan]*len(_PHOT_COLNAMES)],
#                                    columns=_PHOT_COLNAMES)
#         _append_to_phot(self.phot_o, self.ccd_o_proc.header,
#                         fpath=self.file.name, nboj=self.nobj_o, keys=keys)

#     else:
#         _t = Time.now()
#         self.ap_o, self.an_o, self.phot_o = _eap_phot(
#             self.ccd_o_proc,
#             self.err_o,
#             self.gfit_o,
#             **fs
#         )
#         s = s_phot.format(*self.fwhm_o)
#         _append_to_phot(self.phot_o, self.ccd_o_proc.header,
#                         fpath=self.file.name, nobj=self.nobj_o, keys=keys)
#         add_to_header(self.ccd_o_proc.header, 'h', s,
#                       verbose=verbose, t_ref=_t)

#     if self.nobj_e < 1:
#         self.gfit_e = None
#         self.fitter_e = None
#         self.fwhm_e = (np.nan, np.nan)
#         self.ap_e = None
#         self.an_e = None
#         self.phot_e = pd.DataFrame([[np.nan]*len(_PHOT_COLNAMES)],
#                                    columns=_PHOT_COLNAMES)
#         _append_to_phot(self.phot_o, self.ccd_o_proc.header,
#                         fpath=self.file.name, nboj=self.nobj_e, keys=keys)

#     else:
#         _t = Time.now()
#         self.gfit_e, self.fitter_e = _nic_Gaussian2D_fit(
#             self.ccd_e_proc,
#             self.obj_e,
#             err=self.err_e,
#             g_init=g_init_e
#         )
#         # self.gfit_e.x_fwhm = self.gfit_e.x_stddev*gaussian_sigma_to_fwhm
#         # self.gfit_e.y_fwhm = self.gfit_e.y_stddev*gaussian_sigma_to_fwhm
#         self.fwhm_e = (self.gfit_e.x_fwhm, self.gfit_e.y_fwhm)
#         add_to_header(self.ccd_e_proc.header, 'h', s_fit,
#                       verbose=verbose, t_ref=_t)

#         _t = Time.now()
#         self.ap_e, self.an_e, self.phot_e = _eap_phot(
#             self.ccd_e_proc,
#             self.err_e,
#             self.gfit_e,
#             **fs
#         )
#         s = s_phot.format(*self.fwhm_e)
#         _append_to_phot(self.phot_e, self.ccd_e_proc.header,
#                         fpath=self.file.name, nobj=self.nobj_e, keys=keys)
#         add_to_header(self.ccd_e_proc.header, 'h', s,
#                       verbose=verbose, t_ref=_t)

    def ellipphot_fit(self, f_ap=(2., 2.), f_in=(4., 4.), f_out=(6., 6.),
                      g_init_o=None, g_init_e=None, keys=USEFUL_KEYS,
                      verbose=True):
        '''
        Parameters
        ----------
        f_ap, f_in, f_out: int or float, array-like of such, optional.
            The factors multiplied to ``fwhm`` to set the aperture ``a``
            and ``b``, inner sky ``a`` and ``b``, and outer sky ``a``
            and ``b``, respectively. If scalar, it is assumed to be
            identical for both ``a`` and ``b`` parameters. Defaults are
            ``(1.5, 1.5)``, ``(4.0, 4.0)``, and ``(6.0, 6.0)``,
            respectively, which are de facto standard values used by
            classical IRAF users.
        g_init_o, g_init_e : astropy FunctionalModel2D, None, optional.
            The Gaussian initial guess of the PSF of objects in o- and
            e-ray, respectively.
        keys : list of str, None, optional.
            The list of header keywords to be appended to the
            ``self.phot_o`` and ``self.phot_e``.
        Note
        ----
        The sep A/B paramters are for the ellipse to describe the
        "boundary". We need flux-describing Gaussian, so I need to do 2D
        gaussian fitting.
        '''
        from astropy.table import QTable
        s_fit = 'Gaussian2D function fitted.'
        s_phot = ('Photometry done for elliptical aperture/annulus with '
                  + f"f_ap = {f_ap}, f_in = {f_in}, f_out = {f_out}"
                  + "for FWHM = ({:.3f}, {:.3f})")
        fs = dict(f_ap=f_ap, f_in=f_in, f_out=f_out)

        if self.nobj_o < 1:
            self.gfit_o = None
            self.fitter_o = None
            self.fwhm_o = (np.nan, np.nan)
            self.ap_o = None
            self.an_o = None
            self.phot_o = pd.DataFrame([[np.nan]*len(_PHOT_COLNAMES)],
                                       columns=_PHOT_COLNAMES)
            _append_to_phot(self.phot_o, self.ccd_o_proc.header,
                            fpath=self.file.name, nboj=self.nobj_o, keys=keys)

        else:
            _t = Time.now()
            self.gfit_o, self.fitter_o = _nic_Gaussian2D_fit(
                self.ccd_o_proc,
                self.obj_o,
                err=self.err_o,
                g_init=g_init_o
            )
            # self.gfit_o.x_fwhm = self.gfit_o.x_stddev*gaussian_sigma_to_fwhm
            # self.gfit_o.y_fwhm = self.gfit_o.y_stddev*gaussian_sigma_to_fwhm
            self.fwhm_o = (self.gfit_o.x_fwhm, self.gfit_o.y_fwhm)
            add_to_header(self.ccd_o_proc.header, 'h', s_fit,
                          verbose=verbose, t_ref=_t)

            _t = Time.now()
            self.ap_o, self.an_o, self.phot_o = _eap_phot_fit(
                self.ccd_o_proc,
                self.err_o,
                self.gfit_o,
                **fs
            )
            s = s_phot.format(*self.fwhm_o)
            _append_to_phot(self.phot_o, self.ccd_o_proc.header,
                            fpath=self.file.name, nobj=self.nobj_o, keys=keys)
            add_to_header(self.ccd_o_proc.header, 'h', s,
                          verbose=verbose, t_ref=_t)

        if self.nobj_e < 1:
            self.gfit_e = None
            self.fitter_e = None
            self.fwhm_e = (np.nan, np.nan)
            self.ap_e = None
            self.an_e = None
            self.phot_e = pd.DataFrame([[np.nan]*len(_PHOT_COLNAMES)],
                                       columns=_PHOT_COLNAMES)
            _append_to_phot(self.phot_e, self.ccd_e_proc.header,
                            fpath=self.file.name, nboj=self.nobj_e, keys=keys)

        else:
            _t = Time.now()
            self.gfit_e, self.fitter_e = _nic_Gaussian2D_fit(
                self.ccd_e_proc,
                self.obj_e,
                err=self.err_e,
                g_init=g_init_e
            )
            # self.gfit_e.x_fwhm = self.gfit_e.x_stddev*gaussian_sigma_to_fwhm
            # self.gfit_e.y_fwhm = self.gfit_e.y_stddev*gaussian_sigma_to_fwhm
            self.fwhm_e = (self.gfit_e.x_fwhm, self.gfit_e.y_fwhm)
            add_to_header(self.ccd_e_proc.header, 'h', s_fit,
                          verbose=verbose, t_ref=_t)

            _t = Time.now()
            self.ap_e, self.an_e, self.phot_e = _eap_phot_fit(
                self.ccd_e_proc,
                self.err_e,
                self.gfit_e,
                **fs
            )
            s = s_phot.format(*self.fwhm_e)
            _append_to_phot(self.phot_e, self.ccd_e_proc.header,
                            fpath=self.file.name, nobj=self.nobj_e, keys=keys)
            add_to_header(self.ccd_e_proc.header, 'h', s,
                          verbose=verbose, t_ref=_t)

    def ellipphot_sep(self, fwhm=(11., 11.), fix_fwhm=False,
                      f_ap=(2., 2.), f_in=(4., 4.), f_out=(6., 6.),
                      keys=USEFUL_KEYS, verbose=True):
        '''
        Parameters
        ----------
        fwhm : tuple of float, optional.
            The fixed FWHM in pixels in x and y directions.
        f_ap, f_in, f_out: int or float, array-like of such, optional.
            The factors multiplied to ``fwhm`` to set the aperture ``a``
            and ``b``, inner sky ``a`` and ``b``, and outer sky ``a``
            and ``b``, respectively. If scalar, it is assumed to be
            identical for both ``a`` and ``b`` parameters. Defaults are
            ``(1.5, 1.5)``, ``(4.0, 4.0)``, and ``(6.0, 6.0)``,
            respectively, which are de facto standard values used by
            classical IRAF users.
        g_init_o, g_init_e : astropy FunctionalModel2D, None, optional.
            The Gaussian initial guess of the PSF of objects in o- and
            e-ray, respectively.
        keys : list of str, None, optional.
            The list of header keywords to be appended to the
            ``self.phot_o`` and ``self.phot_e``.
        Note
        ----
        The sep A/B paramters are for the ellipse to describe the
        "boundary". We need flux-describing Gaussian, so I need to do 2D
        gaussian fitting.
        '''
        s_phot = ('Photometry done for elliptical aperture/annulus with '
                  + f"f_ap = {f_ap}, f_in = {f_in}, f_out = {f_out} "
                  + "for FWHM = ({:.3f}, {:.3f})")
        fs = dict(f_ap=f_ap, f_in=f_in, f_out=f_out)

        if self.nobj_o < 1:
            self.fwhm_o = (np.nan, np.nan)
            self.ap_o = None
            self.an_o = None
            self.phot_o = pd.DataFrame([[np.nan]*len(_PHOT_COLNAMES)],
                                       columns=_PHOT_COLNAMES)
            _append_to_phot(self.phot_o, self.ccd_o_proc.header,
                            fpath=self.file.name, nobj=self.nobj_o, keys=keys)

        else:
            _t = Time.now()
            if fix_fwhm:
                self.fwhm_o = fwhm
            else:
                self.fwhm_o = _sep_fwhm(self.obj_o)
            self.ap_o, self.an_o, self.phot_o = _eap_phot_sep(
                ccd=self.ccd_o_proc,
                fwhm=self.fwhm_o,
                obj=self.obj_o,
                err=self.err_o,
                **fs
            )
            _append_to_phot(self.phot_o, self.ccd_o_proc.header,
                            fpath=self.file.name, nobj=self.nobj_o, keys=keys)
            s = s_phot.format(*self.fwhm_o)
            add_to_header(self.ccd_o_proc.header, 'h', s,
                          verbose=verbose, t_ref=_t)

        if self.nobj_e < 1:
            self.fwhm_e = (np.nan, np.nan)
            self.ap_e = None
            self.an_e = None
            self.phot_e = pd.DataFrame([[np.nan]*len(_PHOT_COLNAMES)],
                                       columns=_PHOT_COLNAMES)
            _append_to_phot(self.phot_e, self.ccd_e_proc.header,
                            fpath=self.file.name, nobj=self.nobj_e, keys=keys)

        else:
            _t = Time.now()
            if fix_fwhm:
                self.fwhm_e = fwhm
            else:
                self.fwhm_e = _sep_fwhm(self.obj_e)
            self.ap_e, self.an_e, self.phot_e = _eap_phot_sep(
                ccd=self.ccd_e_proc,
                fwhm=self.fwhm_e,
                obj=self.obj_e,
                err=self.err_e,
                **fs
            )
            _append_to_phot(self.phot_e, self.ccd_e_proc.header,
                            fpath=self.file.name, nobj=self.nobj_e, keys=keys)
            s = s_phot.format(*self.fwhm_e)
            add_to_header(self.ccd_e_proc.header, 'h', s,
                          verbose=verbose, t_ref=_t)

    def close(self):
        allk = list(vars(self).keys())
        for k in allk:
            delattr(self, k)


'''
    def pattern_correct(self, mask=None, do_verti=True, do_fouri=True,
                        verti_fitting_sections=None,
                        verti_method='median',
                        verti_sigclip_kw=dict(sigma=3, maxiters=5),
                        fouri_peak_infer_section="[300:500, :]",
                        fouri_peak_sigclip_kw={
                            'sigma_lower': np.inf, 'sigma_upper': 3},
                        fouri_min_freq=1/100,
                        fouri_max_peaks=3,
                        fouri_npool=5,
                        fouri_fitting_y_sections=None,
                        fouri_subtract_x_sections=None,
                        ):
        self.additional_mask = mask

        # 1. Subtract vertical patterns.
        if do_verti:
            if verti_fitting_sections is None:
                verti_fitting_sections = VERTICALSECTS

            self.verti_fitting_sections = verti_fitting_sections
            self.verti_method = verti_method
            self.verti_sigclip_kw = verti_sigclip_kw
            vskw = dict(dtype='float32',
                        return_pattern=False,
                        update_header=True)
            self.ccd_vs = vertical_correct(
                self.ccd_raw,
                fitting_sections=self.verti_fitting_sections,
                method=self.verti_method,
                sigclip_kw=self.verti_sigclip_kw,
                **vskw  # hard-coded
            )
            self.ccd_proc = self.ccd_vs.copy()

        if do_fouri:
            # 2. Find Fourier peaks
            if fouri_fitting_y_sections is None:
                fouri_fitting_y_sections = FOURIERSECTS[self.filt]

            if fouri_subtract_x_sections is None:
                fouri_subtract_x_sections = "[513:900]"

            self.fouri_peak_infer_section = fouri_peak_infer_section
            self.fouri_peak_sigclip_kw = fouri_peak_sigclip_kw
            self.fouri_min_freq = fouri_min_freq
            self.fouri_max_peaks = fouri_max_peaks
            self.fouri_fitting_y_sections = fouri_fitting_y_sections
            self.fouri_subtract_x_sections = fouri_subtract_x_sections

            fourier_region = trim_ccd(
                self.ccd_vs,
                fits_section=self.fouri_peak_infer_section
            )
            self.freqs = find_fourier_peaks(
                fourier_region,
                axis=0,
                min_freq=self.fouri_min_freq,
                max_peaks=self.fouri_max_peaks,
                sigclip_kw=self.fouri_peak_sigclip_kw
            )

            # 3. Fit Fourier series
            fckw = dict(apply_crrej_mask=False, apply_sigclip_mask=True)
            res = fit_fourier(
                self.ccd_vs.data,
                npool=fouri_npool,
                mask=self.additional_mask,
                filt=self.filt,
                fitting_y_sections=self.fouri_fitting_y_sections,
                subtract_x_sections=self.fouri_subtract_x_sections,
                **fckw  # hard-coded
            )
            self.pattern = res[0]
            self.popts = res[1]
            self.ccd_fc = self.ccd_vs.copy()
            self.ccd_fc.data -= self.pattern
            self.ccd_proc = self.ccd_fc.copy()

    def correct_fourier(self, mask=None, crrej_kw=None,
                        crrej_verbose=True, output=None, overwrite=False,
                        peak_infer_section="[900:, :]", max_peaks=3,
                        subtract_section_upper="[513:, 513:]",
                        subtract_section_lower="[513:, :512]",
                        fitting_section_lower="[:, :250]",
                        fitting_section_upper="[:, 800:]",
                        sigclip_kw={'sigma_lower': np.inf, 'sigma_upper': 3},
                        flat_err=0.0, calc_var=True,
                        **kwargs):
        # This crrej is merely to make mask to pixels for Fourier fitting.
        # self.ccd_raw_cr = cr_reject_nic(self.ccd_raw, filt=self.filt,
        #                                 update_header=True,
        #                                 crrej_kw=crrej_kw,
        #                                 verbose=crrej_verbose)
        # try:
        #     self.ccd_raw_cr.mask = self.ccd_raw.mask | self.ccd_raw_cr.mask
        # except TypeError:
        #     pass

        self.additional_mask = mask
        self.peak_infer_section = peak_infer_section
        self.fitting_section_lower = fitting_section_lower
        self.fitting_section_upper = fitting_section_upper
        self.fourier_peak_sigclip_kw = sigclip_kw
        self.fourier_extrapolation_kw = kwargs
        res = fouriersub(ccd=self.ccd_raw, mask=self.additional_mask,
                         peak_infer_section=self.peak_infer_section,
                         max_peaks=max_peaks,
                         subtract_section_upper=subtract_section_upper,
                         subtract_section_lower=subtract_section_lower,
                         fitting_section_lower=self.fitting_section_lower,
                         fitting_section_upper=self.fitting_section_upper,
                         sigclip_kw=self.fourier_peak_sigclip_kw,
                         full=True, **self.fourier_extrapolation_kw)
        self.ccd_fc = res[0]
        self.pattern = res[1]
        self.popt = res[2]
        self.freq = res[3]

        # If ``ccd=self.ccd_raw_cr`` in fouriersub, the resulting CCD
        # will contain the pattern-subtracted value AFTER CR-REJ, which
        # is not desirable sometimes.
        self.ccd_fc.data = self.ccd_raw.data - self.pattern
        ##########################
        # Remove the above line if you used ``ccd=self.ccd_raw``, etc.

        if output is not None:
            self.ccd_fc.write(Path(output), overwrite=overwrite)

        if calc_var:
            self.var_fc = (self.ccd_fc.data*(flat_err + 1/self.gain)
                           + self.rdnoise**2)
'''
