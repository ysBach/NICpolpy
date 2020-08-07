from itertools import product
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import sep
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian2D
from astropy.nddata import CCDData, Cutout2D, VarianceUncertainty
from astropy.stats import sigma_clipped_stats
from astropy.time import Time

import imcombinepy as imc
from ysfitsutilpy import (LACOSMIC_KEYS, CCDData_astype, add_to_header, bdf_process, bezel_ccd, errormap,
                          fitsxy2py, imcopy, load_ccd, propagate_ccdmask, set_ccd_gain_rdnoise, stack_FITS,
                          trim_ccd, medfilt_bpm)
from ysphotutilpy import (apphot_annulus, ellip_ap_an, fit_Gaussian2D, sep_back, sep_extract)

from .preproc import (cr_reject_nic, find_fourier_peaks, fit_fourier, vertical_correct)
from .util import (DARK_PATHS, FLAT_PATHS, FOURIERSECTS, GAIN, MASK_PATHS, OBJSECTS, OBJSLICES, RDNOISE,
                   USEFUL_KEYS, VERTICALSECTS, infer_filter, split_oe, summary_nic)

__all__ = [
    "NICPolDir",
    "NICPolImage"]


_PHOT_COLNAMES = ['id', 'xcenter', 'ycenter', 'aparea',
                  'aperture_sum', 'aperture_sum_err',
                  'msky', 'nrej', 'nsky', 'ssky',
                  'source_sum', 'source_sum_err',
                  'mag', 'merr', 'x_fwhm', 'y_fwhm', 'theta']


# class NICCCDBase:
#     def __init__(self, fpath, filt=None, verbose=True):
#         self.file = Path(fpath)
#         # for future calculation, set float32 dtype:
#         self.ccd_raw = CCDData_astype(load_ccd(fpath), 'float32')
#         self.filt = infer_filter(self.ccd_raw, filt=filt, verbose=verbose)
#         set_ccd_gain_rdnoise(
#             self.ccd_raw,
#             gain=GAIN[self.filt],
#             rdnoise=RDNOISE[self.filt]
#         )
#         self.gain = self.ccd_raw.gain
#         self.rdnoise = self.ccd_raw.rdnoise
#         self.ccdlog = dict(raw=self.ccd_raw)
#         self.proc_dark = False  # dark subtraction
#         self.proc_dark_by = None  # 'dark': dark frame, 'cr': crrej, 'sky': sky - sky_cr \approx dark
#         self.proc_flat = False  # flat division
#         self.proc_sky = False  # sky subtraction \approx fringe subtraction.


# class NICPolScience(NICCCDBase):
#     def __init__(self, fpath, filt=None, verbose=True):
#         super().__init__(fpath=fpath, filt=filt, verbose=verbose)

#         # for frame in ['dark', 'flat', 'sky']:
#         #     framepath = getattr(self, frame + 'path')
#         #     if framepath is not None:
#         #         frameobj = NICCCDBase(fpath=framepath, verbose=verbose)
#         #         setattr(self, frame, frameobj)
#         #     else:
#         #         setattr(self, frame, None)

#     def preproc_by_sky(self, mflatpath=None, skypath=None, mflat=None, sky=None, crrej_kw=None,
#                        verbose_crrej=False, sky_scale_section="[20:130, 40:140]", scale_fun=np.median,
#                        verbose=False):
#         self.mflatpath = mflatpath
#         self.skypath = skypath
#         self.sky_scale_section = sky_scale_section

#         if mflat is not None:
#             self.mflat = mflat
#         else:
#             self.mflat = load_ccd(self.mflatpath)

#         if sky is not None:
#             self.sky = sky
#         else:
#             self.sky = load_ccd(self.skypath)

#         # Better to use default LACOSMIC because we want to strongly remove
#         # not only CR but also hot pixels.
#         if crrej_kw is None:
#             crrej_kw = LACOSMIC_KEYS.copy()
#             crrej_kw["sepmed"] = True

#         self.ccd = self.ccd_raw.copy()
#         _t = Time.now()
#         self.sky_cr = cr_reject_nic(self.sky, crrej_kw=crrej_kw,
#                                     verbose=verbose_crrej)
#         self.mdark = CCDData(data=self.ccd_raw.data - self.sky_cr.data, header=self.sky.header.copy())
#         add_to_header(self.ccd.header, 'h', "Dark estimated by sky - sky_crrej", verbose=verbose, t_ref=_t)

#         scale_obj = scale_fun(imcopy(self.ccd, self.sky_scale_section).data)
#         scale_sky = scale_fun(imcopy(self.sky_cr, self.sky_scale_section).data)
#         scale = scale_obj/scale_sky
#         self.ccd.data -= scale*self.sky_cr.data
#         add_to_header(self.ccd.header, 'h', f"CR-rejected sky frame is subtracted with scale {scale:.4f}",
#                       verbose=verbose, t_ref=_t)

#         self.ccd = bdf_process(self.ccd, mdark=self.mdark, mflat=self.mflat)
#         var = (self.ccd_raw.data*self.gain + self.rdnoise**2)/self.gain**2
#         self.ccd.uncertainty = VarianceUncertainty(var, unit='adu^2')


# class NICPolSky(NICCCDBase):
#     def __init__(self, fpath, filt=None, verbose=True):
#         super().__init__(fpath=fpath, filt=filt, verbose=verbose)

class NICPolDirMixin:
    @staticmethod
    def _save(ccd, original_path, object_name, savedir, combined=False, verbose=False):
        def _set_fname(original_path, object_name, combined=False):
            ''' If combined, COUNTER, POL-AGL1, INSROT are meaningless, so remove these.
            '''
            splitted = Path(original_path).name.split('_')
            del splitted[3:-4]  # remove both "Earthshine" and "Earthshine_sky"
            if combined:
                splitted.pop(2)  # remove counter
                del splitted[-3:-1]  # remove POL-AGL1, INSROT
                splitted.insert(2, object_name)
            else:
                splitted.insert(3, object_name)
            return "_".join(splitted)

        ccd.header["OBJECT"] = object_name
        newpath = savedir/_set_fname(original_path, object_name, combined=combined)
        ccd.write(newpath, overwrite=True)
        if verbose:
            print(f"Writing FITS to {newpath}")
        return ccd, newpath

    @staticmethod
    def _set_mflat(summary_flat, filt, oe, flatdir, flat_min_value=0.):
        ''' Note that it returns ndarray, not CCDData.
        '''
        if summary_flat is None:
            return 1, None

        if flatdir is not None:
            mflatpath = stack_FITS(summary_table=summary_flat, loadccd=False, verbose=False,
                                   type_key=["FILTER", "OERAY"],
                                   type_val=[filt.upper(), oe])
            if len(mflatpath) > 1:
                raise ValueError(f"More than 1 flat for (FILTER, OERAY) = ({filt}, {oe}) found.")
            elif len(mflatpath) == 0:
                raise ValueError(f"No FITS file for (FILTER, OERAY) = ({filt}, {oe}) found.")
            mflatpath = mflatpath[0]
            mflat = load_ccd(mflatpath).data
            mflat[mflat < flat_min_value] = 1.
        else:
            mflatpath = None
            mflat = 1
        return mflat, mflatpath


class NICPolDir(NICPolDirMixin):
    def __init__(self, location, rawdir="raw", caldir="calib", tmpcal="tmp_calib", tmpred="tmp_reduc",
                 flatdir=None, verbose=False):
        self.location = Path(location)
        if rawdir is None:
            self.rawdir = self.location
        else:
            self.rawdir = self.location/rawdir

        if not self.rawdir.exists():
            raise FileNotFoundError("Raw data directory not found.")

        self.caldir = self.location/caldir
        self.tmpcal = self.location/tmpcal
        self.tmpred = self.location/tmpred
        self.caldir.mkdir(parents=True, exist_ok=True)
        self.tmpcal.mkdir(parents=True, exist_ok=True)
        self.tmpred.mkdir(parents=True, exist_ok=True)

        self.flatdir = Path(flatdir) if flatdir is not None else None
        if not self.flatdir.exists():
            raise FileNotFoundError("Flat directory not found.")

        self.keys = USEFUL_KEYS + ["OERAY"]

        if self.flatdir is not None:
            self.summary_flat = summary_nic(self.flatdir/"*.fits", keywords=self.keys, verbose=verbose)
        else:
            self.summary_flat = None

        if verbose:
            print("Extracting header info...")

        self.summary_raw = summary_nic(self.rawdir/"*.fits", keywords=self.keys, verbose=verbose)

        self.skydata = {}
        self.sky_dark_exists = False
        self.pure_dark_exists = False
        self.sciframes_exptimes = {}
        for objname in np.unique(self.summary_raw["OBJECT"]):
            if objname.endswith("_sky"):
                self.skydata[objname] = {}
                self.sky_dark_exists = True
            elif not self.pure_dark_exists and objname.lower() == "dark":
                self.pure_dark_exists = True
            elif objname.lower() not in ['test', 'flat', 'bias']:  # just in case...
                mask_sci = self.summary_raw["OBJECT"] == objname
                self.sciframes_exptimes[objname] = np.unique(self.summary_raw[mask_sci]["EXPTIME"])

        for skyname in self.skydata.keys():
            skymask = self.summary_raw["OBJECT"] == skyname
            self.skydata[skyname]["summary"] = self.summary_raw[skymask]
            self.skydata[skyname]["exptimes"] = np.unique(self.summary_raw[skymask]["EXPTIME"])

        if self.pure_dark_exists:
            self.summary_dark = self.summary_raw[self.summary_raw["OBJECT"] == "DARK"]
        else:
            self.summary_dark = None

    def prepare_calib(self, dark_min=2., sky_scale_section="[20:130, 40:140]", fringe_min_value=0.0,
                      flat_min_value=0.0,
                      verbose=True, verbose_combine=False, verbose_bdf=False, verbose_crrej=False):
        self.dark_min = dark_min
        self.sky_scale_section = sky_scale_section
        self.sky_scale_slice = fitsxy2py(self.sky_scale_section)
        self.paths_puredark = {}
        self.paths_skydark = {}
        self.paths_skyfringe = {}
        self.paths_flat = {}
        self.fringe_min_value = fringe_min_value
        self.flat_min_value = flat_min_value

        crrej_kw = None
        print(f"Output and intermediate files are saved to {self.caldir} and {self.tmpcal}.")
        if self.flatdir is None:
            print("Better to specify flat files by flatdir.")

        if verbose:
            print("Estimating DARK from sky frames if exists... ")

        for filt, oe in product('JHK', 'oe'):
            mflat, mflatpath = self._set_mflat(self.summary_flat, filt, oe, self.flatdir, self.flat_min_value)
            self.paths_flat[(filt, oe)] = mflatpath
            for skyname, skydict in self.skydata.items():
                for exptime in skydict['exptimes']:
                    sky_fpaths = stack_FITS(summary_table=skydict['summary'], loadccd=False, verbose=verbose,
                                            type_key=["FILTER", "OERAY", "EXPTIME"],
                                            type_val=[filt.upper(), oe, exptime])

                    # == Estimate DARK from sky frames ===================================================== #
                    skycomb_paths = []
                    sky_dark_paths = []
                    for fpath in sky_fpaths:
                        _t = Time.now()
                        sky = load_ccd(fpath)
                        # Sky / Flat
                        # flat division to prevent artificial CR rejection:
                        sky_f = sky.copy()
                        sky_f.data = sky.data/mflat
                        sky_f, _ = self._save(sky_f, fpath, f"{skyname}_FLAT", self.tmpcal)

                        # (Sky/Flat)_cr
                        sky_f_cr = cr_reject_nic(sky_f, crrej_kw=crrej_kw, verbose=verbose_crrej)
                        sky_f_cr, _ = self._save(sky_f_cr, fpath, f"{skyname}_FLAT_CRREJ", self.tmpcal)

                        # (Sky/Flat)_cr * Flat
                        sky_cr = sky_f_cr.copy()
                        sky_cr.data *= mflat
                        sky_cr, _ = self._save(sky_cr, fpath, f"{skyname}_FLAT_CRREJ_DEFLATTED", self.tmpcal)

                        # Dark ~ Sky - (Sky/Flat)_cr * Flat
                        sky_dark = sky_f_cr.copy()  # retain CRREJ header info
                        sky_dark.data = sky.data - sky_f_cr.data*mflat
                        sky_dark.data[sky_dark.data < dark_min] = 0
                        add_to_header(sky_dark.header, 'h',
                                      ("Dark estimated by combine(sky - (sky/flat)_cr*flat) "
                                       + f"and dark_min of {dark_min} applied."),
                                      t_ref=_t, verbose=verbose_combine)
                        _, sky_dark_path = self._save(sky_dark, fpath, f"{skyname}_SKYDARK", self.tmpcal)
                        sky_dark_paths.append(sky_dark_path)

                    # == Combine estimated DARK ============================================================ #
                    if verbose:
                        print("    Combine the estimated DARK from _sky frames", end='... ')

                    _t = Time.now()
                    comb_sky_dark = imc.fitscombine(sky_dark_paths,
                                                    ombine='med',
                                                    reject='sc',
                                                    sigma=3,
                                                    verbose=verbose_combine)
                    add_to_header(comb_sky_dark.header, 'h', "Combined FITS files",
                                  t_ref=_t, verbose=verbose_combine)
                    comb_sky_dark, comb_sky_dark_path = self._save(comb_sky_dark, sky_fpaths[0],
                                                                   f"{skyname}_SKYDARK", self.caldir,
                                                                   combined=True)
                    self.paths_skydark[(skyname, filt, oe, exptime)] = comb_sky_dark_path

                    if verbose:
                        print("Done.")

                    # == Make fringe frames for each sky =================================================== #
                    if verbose:
                        print("    Make and combine the SKYFRINGES (flat corrected) from _sky frames",
                              end='...')

                    for fpath in sky_fpaths:
                        sky = load_ccd(fpath)
                        # give mdark/mflat so that the code does not read the FITS files repeatedly:
                        sky = bdf_process(sky,
                                          mdark=comb_sky_dark,
                                          mflat=CCDData(mflat, unit='adu'),
                                          mdarkpath=comb_sky_dark_path,
                                          mflatpath=mflatpath,
                                          verbose_bdf=verbose_bdf,
                                          verbose_crrej=verbose_crrej)
                        sky, sky_tocomb_path = self._save(sky, fpath, f"{skyname}_FRINGE", self.tmpcal)
                        skycomb_paths.append(sky_tocomb_path)

                    # == Combine sky fringes =============================================================== #
                    # combine dark-subtracted and flat-corrected sky frames to get the fringe pattern by sky
                    # emission lines:
                    _t = Time.now()
                    logpath = self.caldir/f"{filt.lower()}_{skyname}_{exptime:.1f}_{oe}_combinelog.csv"
                    comb_sky_fringe = imc.fitscombine(skycomb_paths,
                                                      combine='med',
                                                      reject='sc',
                                                      sigma=3,
                                                      scale='avg',
                                                      scale_to_0th=False,
                                                      scale_section=sky_scale_section,
                                                      verbose=verbose_combine,
                                                      logfile=logpath)
                    add_to_header(comb_sky_fringe.header, 'h', "Combined sky fringe FITS files",
                                  t_ref=_t, verbose=verbose_combine)

                    # Normalize using the section
                    _t = Time.now()
                    norm_value = np.mean(comb_sky_fringe.data[self.sky_scale_slice])
                    comb_sky_fringe.data /= norm_value
                    comb_sky_fringe.data[comb_sky_fringe.data < self.fringe_min_value] = 0
                    add_to_header(comb_sky_fringe.header, 'h',
                                  "Normalized by mean of NORMSECT (NORMVALU), replaced value < FRINMINV to 0",
                                  t_ref=_t, verbose=verbose_combine)
                    comb_sky_fringe.header["NORMSECT"] = sky_scale_section
                    comb_sky_fringe.header["NORMVALU"] = norm_value
                    comb_sky_fringe.header["FRINMINV"] = self.fringe_min_value

                    _, comb_sky_fringe_path = self._save(comb_sky_fringe, skycomb_paths[0],
                                                         f"{skyname}_SKYFRINGE", self.caldir, combined=True)
                    self.paths_skyfringe[(skyname, filt, oe, exptime)] = comb_sky_fringe_path

                    if verbose:
                        print("Done.")

        # == Combine dark if DARK exists =================================================================== #
        if verbose:
            print("Combining DARK frames if exists", end='... ')

        if self.pure_dark_exists:
            exptimes = np.unique(self.summary_dark["EXPTIME"])
            for filt, oe, exptime in product('JHK', 'oe', exptimes):
                _t = Time.now()
                dark_fpaths = stack_FITS(summary_table=self.summary_dark, loadccd=False, verbose=False,
                                         type_key=["FILTER", "OERAY", "EXPTIME"],
                                         type_val=[filt.upper(), oe, exptime])
                comb = imc.fitscombine(dark_fpaths,
                                       combine='med',
                                       reject="sc",
                                       sigma=(3, 3),
                                       dtype='float32',
                                       verbose=verbose_combine)
                comb.data[comb.data < dark_min] = 0
                add_to_header(comb.header, 'h', f"Images combined and dark_min {dark_min} applied.",
                              t_ref=_t, verbose=verbose)
                _, comb_dark_path = self._save(comb, dark_fpaths[0], "DARK", self.caldir, combined=True)
                self.paths_puredark[(filt, oe, exptime)] = comb_dark_path
        elif verbose:
            print("No DARK found.", end=' ')

        if verbose:
            print("Done.")

    def preproc(self, reddir="reduced", prefer_skydark=False, pixel_mask_method="medfilt_bpm",
                med_ratio_clip=[0.5, 2], std_ratio_clip=[-3, 5],
                medfilt_kw=dict(cadd=1.e-10, size=5, mode='reflect', cval=0.0,
                                origin=0, sigma=3., maxiters=5, std_ddof=1, dtype='float32'),
                do_crrej_pos=True, do_crrej_neg=True,
                verbose=True, verbose_bdf=False, verbose_bpm=False, verbose_crrej=False):
        '''
        '''
        self.reddir = self.location/reddir
        self.reddir.mkdir(parents=True, exist_ok=True)
        self.summary_cal = summary_nic(self.caldir/"*.fits", keywords=self.keys, verbose=verbose)
        self.darks = {}
        self.fringes = {}

        for filt, oe in product('JHK', 'oe'):
            mflatpath = self.paths_flat[(filt, oe)]
            if mflatpath is None:
                mflat = None
            else:
                mflat = load_ccd(mflatpath)
                mflat.data[mflat.data < self.flat_min_value] = 1.
            for objname, exptimes in self.sciframes_exptimes.items():
                for exptime in exptimes:
                    if objname.lower().endswith("_sky") or objname.upper() in ["DARK", "TEST"]:
                        continue  # if sky/dark frames

                    # == Setup dark and fringe ============================================================= #
                    if prefer_skydark:
                        try:
                            mdarkpath = self.paths_skydark[(f"{objname}_sky", filt, oe, exptime)]
                            mdark = load_ccd(mdarkpath)
                        except (KeyError, IndexError, FileNotFoundError):
                            if verbose:
                                print(f"prefer_skydark but skydark for ({objname}_sky, {filt}, {oe}, "
                                      + f"{exptime}) not found. Trying to use pure dark.")
                            try:
                                mdarkpath = self.paths_puredark[(filt, oe, exptime)]
                                mdark = load_ccd(mdarkpath)
                            except (KeyError, IndexError, FileNotFoundError):
                                mdarkpath = None
                                mdark = None
                                if verbose:
                                    print("No dark file found. Turning off dark subtraction.")

                    else:
                        try:
                            mdarkpath = self.paths_puredark[(filt, oe, exptime)]
                            mdark = load_ccd(mdarkpath)
                        except (KeyError, IndexError, FileNotFoundError):
                            if verbose:
                                print(f"Pure dark for ({filt}, {oe}, {exptime}) not found. "
                                      + f"Trying to use SKYDARK of ({objname}_sky, {filt}, {oe}, {exptime})",
                                      end='... ')
                            try:
                                mdarkpath = self.paths_skydark[(f"{objname}_sky", filt, oe, exptime)]
                                mdark = load_ccd(mdarkpath)
                                if verbose:
                                    print("Loaded successfully.")
                            except (KeyError, IndexError, FileNotFoundError):
                                mdarkpath = None
                                mdark = None
                                if verbose:
                                    print("No dark file found. Turning off dark subtraction.")

                    try:
                        mfringepath = self.paths_skyfringe[(f"{objname}_sky", filt, oe, exptime)]
                        mfringe = load_ccd(mfringepath)
                    except (KeyError, IndexError, FileNotFoundError):
                        mfringepath = None
                        mfringe = None
                        if verbose:
                            print("No finge file found. Turning off fringe subtraction.")

                    # == Reduce data ======================================================================= #
                    raw_fpaths = stack_FITS(summary_table=self.summary_raw, loadccd=False, verbose=False,
                                            type_key=["FILTER", "OBJECT", "OERAY", "EXPTIME"],
                                            type_val=[filt.upper(), objname, oe, exptime])

                    for fpath in raw_fpaths:
                        if verbose:
                            print(fpath)
                        rawccd = load_ccd(fpath)
                        # 1. Do Dark and Flat.
                        # 2. Subtract Fringe
                        if mdark is not None or mflat is not None or mfringe is not None:
                            redccd = bdf_process(rawccd,
                                                 mdarkpath=mdarkpath,
                                                 mflatpath=mflatpath,
                                                 mdark=mdark,
                                                 mflat=mflat,
                                                 mfringe=mfringe,
                                                 mfringepath=mfringepath,
                                                 fringe_scale_fun=np.mean,
                                                 fringe_scale_section=self.sky_scale_section,
                                                 verbose_bdf=verbose_bdf)
                            objname_orig = redccd.header['OBJECT']
                            objname = f"{redccd.header['OBJECT']}"
                            if mdark is not None:
                                objname += "D"
                            if mflat is not None:
                                objname += "F"
                            if mfringe is not None:
                                objname += "Fr"
                            redccd, _ = self._save(redccd, fpath, objname, self.tmpred, verbose=verbose)

                        else:
                            redccd = rawccd.copy()

                        # -- if crrej case:
                        #   3. Do CRrej for HIGHLY POSITIVE values (remaining dark & CR).
                        #   4. Do CRrej for HIGHLY NEGATIVE values (oversubtracted dark?)
                        #   Both CR rejections use sigfrac=0.5 (LACosmic's default) and objlim determined
                        #   empirically.
                        #   For positive rejection, it is better to be done before the fringe subtraction in
                        #   principle. However, when I tested, CRrej before fringe subtraction gave almost no
                        #   rejection of hot pixels, so I just did Fringe subtraction before positive CRrej.
                        #       ysBach 2020-08-06 16:17:03 (KST: GMT+09:00)
                        if pixel_mask_method == "crrej":
                            if verbose:
                                print(f"{pixel_mask_method} will be used.")
                            # positive CR rejection
                            if do_crrej_pos:
                                redccd = cr_reject_nic(redccd,
                                                       crrej_kw=dict(objlim=1, sigfrac=0.5),
                                                       verbose=verbose_crrej)
                                objname += "C"
                                redccd, _ = self._save(redccd, fpath, objname, self.tmpred, verbose=verbose)

                            # # FRINGE subtraction
                            # if mfringe is not None:
                            #     redccd = bdf_process(redccd,
                            #                         mfringe=mfringe,
                            #                         mfringepath=mfringepath,
                            #                         fringe_scale_fun=np.mean,
                            #                         fringe_scale_section=self.sky_scale_section,
                            #                         verbose_bdf=verbose_bdf)
                            #     objname += "Fr"
                            #     redccd, _ = self._save(redccd, fpath, objname, self.tmpred, verbose=verbose

                            # negative CR rejection
                            if do_crrej_neg:
                                add_to_header(redccd.header, 'h', verbose=verbose_bdf,
                                              s="Negated the pixel values to remove 'negative' CR.")
                                redccd.data *= -1
                                redccd = cr_reject_nic(redccd,
                                                       crrej_kw=dict(objlim=5, sigfrac=0.5),
                                                       verbose=verbose_crrej)
                                redccd.data *= -1
                                add_to_header(redccd.header, 'h', verbose=verbose_bdf,
                                              s="Negated the pixel values to restore the original sign.")

                                objname += "Cneg"
                                redccd, _ = self._save(redccd, fpath, objname, self.tmpred, verbose=verbose)

                        # -- if medfilt case:
                        #   3. Calculate median-filtered frame.
                        #   4. Calculate
                        #       RATIO = original/medfilt,
                        #       SIGRATIO = (original - medfilt) / mean(original)
                        #   5. Replace pixels where (RATIO > medfilt_ratio) & (SIGRATIO > medfilt_sig_ratio)
                        elif pixel_mask_method == "medfilt_bpm":
                            if verbose:
                                print(f"{pixel_mask_method} will be used.")

                            res = medfilt_bpm(redccd,
                                              std_section=self.sky_scale_section,
                                              med_ratio_clip=med_ratio_clip,
                                              std_ratio_clip=std_ratio_clip,
                                              **medfilt_kw,
                                              verbose=verbose_bpm,
                                              full=True)
                            redccd, posmask, negmask, med_filt, med_ratio, std_ratio = res

                            tmpccd = redccd.copy()
                            tmpccd.data = med_filt
                            medfilt_objname = objname + "_MedFilt"
                            _ = self._save(tmpccd, fpath, medfilt_objname, self.tmpred, verbose=verbose_bpm)

                            tmpccd.data = med_ratio
                            medratio_objname = objname + "_MedRatio"
                            _ = self._save(tmpccd, fpath, medratio_objname, self.tmpred, verbose=verbose_bpm)

                            tmpccd.data = std_ratio
                            stdratio_objname = objname + "_StdRatio"
                            _ = self._save(tmpccd, fpath, stdratio_objname, self.tmpred, verbose=verbose_bpm)

                            tmpccd.data = (1*posmask + 2*negmask).astype(np.uint8)
                            fullmask_objname = objname + "_MASK_pos1neg2"
                            _ = self._save(tmpccd, fpath, fullmask_objname, self.tmpred, verbose=verbose_bpm)

                        elif verbose:
                            print(f"{pixel_mask_method} NOT understood !!")

                        # -- Save final result
                        # Duplicated save as "Cneg" file above. The user may just DELETE tmpred folder in the
                        # future if it is unnecessary.
                        _ = self._save(redccd, fpath, objname_orig, self.reddir, verbose=verbose)

                        if verbose:
                            print()

                    if verbose:
                        print()


# # -------------------------------------------------------------------------------------------------------- #

#         for objname, filt, oe, exptime in product(self.objnames, 'JHK', 'oe', exptimes):
#             if objname.lower().endswith("_sky") or objname.upper() in ["DARK", "TEST"]:  # if sky/dark frames
#                 continue

#             if prefer_skydark:
#                 try:
#                     sky_dark_path = self.paths_skydark[(f"{objname}_sky", filt, oe, exptime)]
#                     self.darks[(objname, filt, oe, exptime)] = dict(mdark=load_ccd(sky_dark_path),
#                                                                     mdarkpath=sky_dark_path)
#                 except (KeyError, IndexError, FileNotFoundError):
#                     print(f"prefer_skydark but skydark for ({objname}_sky, {filt}, {oe}, {exptime}) "
#                           + "not found. Trying to use pure dark.")
#                     try:
#                         pure_dark_path = self.paths_puredark[(filt, oe, exptime)]
#                         self.darks[(objname, filt, oe, exptime)] = dict(mdark=load_ccd(pure_dark_path),
#                                                                         mdarkpath=pure_dark_path)
#                     except (KeyError, IndexError, FileNotFoundError):
#                         self.darks[(objname, filt, oe, exptime)] = dict(mdark=None, mdarkpath=None)
#                         print(f"No dark file found. Turning off dark subtraction.")

#             else:
#                 try:
#                     pure_dark_path = self.paths_puredark[(filt, oe, exptime)]
#                     self.darks[(objname, filt, oe, exptime)] = dict(mdark=load_ccd(pure_dark_path),
#                                                                     mdarkpath=pure_dark_path)
#                 except (KeyError, IndexError, FileNotFoundError):
#                     print(f"Pure dark for ({filt}, {oe}, {exptime}) not found."
#                           + f"Trying to use SKYDARK of ({objname}_sky, {filt}, {oe}, {exptime}).")
#                     try:
#                         sky_dark_path = self.paths_skydark[(f"{objname}_sky", filt, oe, exptime)]
#                         self.darks[(objname, filt, oe, exptime)] = dict(mdark=load_ccd(sky_dark_path),
#                                                                         mdarkpath=sky_dark_path)
#                     except (KeyError, IndexError, FileNotFoundError):
#                         self.darks[(objname, filt, oe, exptime)] = dict(mdark=None, mdarkpath=None)
#                         print(f"No dark file found. Turning off dark subtraction.")

#             try:
#                 sky_fringe_path = self.paths_skyfringe[(f"{objname}_sky", filt, oe, exptime)]
#                 self.fringes[(objname, filt, oe, exptime)] = dict(mfringe=load_ccd(sky_fringe_path),
#                                                                   mfringepath=sky_fringe_path)
#             except (KeyError, IndexError, FileNotFoundError):
#                 self.fringes[(objname, filt, oe, exptime)] = dict(mfringe=None, mfringepath=None)
#                 print("No finge file found. Turning off fringe subtraction.")

#         useful_obj = []  # e.g., Earthshine, Vesta, NGC1234, ...
#         for objname in self.objnames:
#             if not objname.lower().endswith("_sky") and not objname.upper() in ["DARK", "TEST"]:
#                 useful_obj.append(objname)


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


def _sort_obj(ccd, obj, nobj, verbose):
    ''' Sort obj such that the 0-th is our target.
    '''
    s_nobj = ("{} objects found; Only the one closest to the FOV center "
              + "(segmentation map label = {}) will be used.")
    ny, nx = ccd.data.shape
    obj['_r'] = np.sqrt((obj['x'] - nx/2)**2 + (obj['y'] - ny/2)**2)
    obj.sort_values('_r', inplace=True)
    s = s_nobj.format(nobj, obj['segm_label'].values[0])
    add_to_header(ccd.header, 'h', s, verbose=verbose)


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


# TODO: let user to skip dark correction.
class NICPolImage:
    def __init__(self, fpath, ccd=None, filt=None, verbose=True):
        self.file = Path(fpath)
        # for future calculation, set float32 dtype:
        self.ccd_raw = CCDData_astype(load_ccd(fpath), 'float32')
        self.right_half = self.ccd_raw.data.shape[1] == 512
        self.filt = infer_filter(self.ccd_raw, filt=filt, verbose=verbose)
        set_ccd_gain_rdnoise(
            self.ccd_raw,
            gain=GAIN[self.filt],
            rdnoise=RDNOISE[self.filt]
        )
        self.gain = self.ccd_raw.gain
        self.rdnoise = self.ccd_raw.rdnoise
        self.sect_o = OBJSECTS(self.right_half)[self.filt][0]
        self.sect_e = OBJSECTS(self.right_half)[self.filt][1]
        self.sl_o = OBJSLICES(self.right_half)[self.filt][0]
        self.sl_e = OBJSLICES(self.right_half)[self.filt][1]

    def preproc(self, mdarkpath='default',
                mflatpath_o='default', mflatpath_e='default',
                mask='default', do_verti=True, do_fouri=False,
                verti_fitting_sections=None,
                verti_method='median',
                verti_sigclip_kw=dict(sigma=2, maxiters=5),
                fouri_peak_infer_section="[300:500, :]",
                fouri_peak_sigclip_kw={
                    'sigma_lower': 3, 'sigma_upper': 3},
                fouri_min_freq=1/100,
                fouri_max_peaks=3,
                fouri_npool=8,
                fouri_fitting_y_sections=None,
                fouri_subtract_x_sections=None,
                do_crrej=True,
                verbose=False,
                verbose_crrej=False,
                crrej_kw=None,
                bezel_x=(0, 0),
                bezel_y=(0, 0),
                replace=np.nan
                ):
        """ Do full preprocess to NHAO NIC.
        Note
        ----
        A CCDData or array in the shape of original FITS file is denoted
        with underbar as self._xxx.

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


        bezel_x, bezel_y : array-like of int
            The x and y bezels, in ``[lower, upper]`` convention. If
            ``0``, no replacement happens.
        replace : int, float, nan, optional.
            The value to replace the pixel value where it should be
            masked. If ``None``, the ccds will be trimmed.
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
        else:
            self.mdarkpath = mdarkpath

        if mflatpath_o == 'default':
            if verbose:
                print("mflatpath_o is 'default': using NICpolpy default flat.")
            self.mflatpath_o = FLAT_PATHS[self.filt]['o']
        else:
            self.mflatpath_o = mflatpath_o

        if mflatpath_e == 'default':
            if verbose:
                print("mflatpath_e is 'default': using NICpolpy default flat.")
            self.mflatpath_e = FLAT_PATHS[self.filt]['e']
        else:
            self.mflatpath_e = mflatpath_e

        if mask == 'default':
            if verbose:
                print("mask is 'default': using NICpolpy default mask.")
            # Set mask for high-dark / low-flat pixels
            pixmask = fits.open(MASK_PATHS[self.filt])[0].data.astype(bool)
            self._mask = propagate_ccdmask(self.ccd_raw, pixmask)
        else:
            if mask is None and verbose:
                print("mask is None: no mask will be used (not recommended).")
            self._mask = mask

        self._ccd_proc = self.ccd_raw.copy()
        # Do not split oe of original CCD because of vertical pattern
        # subtraction and Fourier pattern subtraction must be done based
        # on the full CCD frame!
        self.mask_o_proc = self._mask[self.sl_o]
        self.mask_e_proc = self._mask[self.sl_e]
        vb = dict(verbose=verbose)

        # ================================================================================================== #
        # * 0. Subtract Dark (regard bias is also subtracted)
        # ================================================================================================== #
        if self.mdarkpath is not None:
            self._ccd_dark = load_ccd(self.mdarkpath)
            self.ccd_dark_o = trim_ccd(self._ccd_dark, self.sect_o, **vb)
            self.ccd_dark_e = trim_ccd(self._ccd_dark, self.sect_e, **vb)
        else:  # e.g., path is given as None
            self._ccd_dark = None
            self.ccd_dark_o = None
            self.ccd_dark_e = None

        self.ccd_bdxx = bdf_process(
            self._ccd_proc,
            mdarkpath=self.mdarkpath,
            dark_scale=True,
            verbose_bdf=verbose,
            verbose_crrej=verbose_crrej
        )
        self._ccd_proc = self.ccd_bdxx.copy()

        # ================================================================================================== #
        # * 1. Subtract vertical patterns.
        # ================================================================================================== #
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
                **vb
            )
            self._ccd_vs = vertical_correct(
                self._ccd_proc, **self.verti_corr_kw)
            self._ccd_proc = self._ccd_vs.copy()

        # ================================================================================================== #
        # * 2. Subtract Fourier pattern
        # ================================================================================================== #
        if do_fouri:
            self._ccd_fc = self._ccd_proc.copy()

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
            self._ccd_fc.header["FIT-AXIS"] = (
                "COL",
                "The direction to which Fourier series is fitted."
            )

            # * 2-1. Find Fourier peaks
            # -----------------------------------------------------------------
            _t = Time.now()
            f_reg = trim_ccd(self._ccd_fc, **self.fouri_trim_kw)
            self.freqs = find_fourier_peaks(f_reg.data, axis=0,
                                            **self.fouri_freq_kw)

            self._ccd_fc.header["FIT-NF"] = (
                len(self.freqs), "Number of frequencies used for fit.")
            for k, f in enumerate(self.freqs):
                self._ccd_fc.header[f"FIT-F{k+1:03d}"] = (
                    f, f"[1/pix] The {k+1:03d}-th frequency")
            add_to_header(self._ccd_fc.header, 'h', s_fouri_peak,
                          **vb, t_ref=_t)

            # * 2-2. Fit Fourier series
            # ------------------------------------------------------------------------------------------------
            _t = Time.now()
            res = fit_fourier(
                self._ccd_fc.data,
                self.freqs,
                mask=self._mask,
                filt=self.filt,
                **self.fouri_corr_kw
            )
            self._pattern_fc = res[0]
            self.popts = res[1]
            add_to_header(self._ccd_fc.header, 'h', s_fouri_corr,
                          **vb, t_ref=_t)

            # * 2-3. Subtract Fourier pattern
            # ------------------------------------------------------------------------------------------------
            _t = Time.now()
            self._ccd_fc.data -= self._pattern_fc
            add_to_header(self._ccd_fc.header, 'h', s_fouri_sub,
                          **vb, t_ref=_t)

            self._ccd_proc = self._ccd_fc.copy()

        # ================================================================================================== #
        # * 3. Split o-/e-ray
        # ================================================================================================== #
        self.ccd_o_bdxx, self.ccd_e_bdxx = split_oe(
            self._ccd_proc,
            filt=self.filt,
            right_half=self.right_half,
            **vb
        )
        self.ccd_o_proc = self.ccd_o_bdxx.copy()
        self.ccd_e_proc = self.ccd_e_bdxx.copy()

        # ================================================================================================== #
        # * 4. Flat correct
        # ================================================================================================== #
        if self.mflatpath_o is not None:
            self.ccd_o_bdfx = bdf_process(
                self.ccd_o_bdxx,
                mflatpath=self.mflatpath_o,
                verbose_bdf=verbose,
                verbose_crrej=verbose_crrej
            )
            self.ccd_o_proc = self.ccd_o_bdfx.copy()
        else:
            self.ccd_o_bdfx = None
            # Do not update ccd_o_proc

        if self.mflatpath_e is not None:
            self.ccd_e_bdfx = bdf_process(
                self.ccd_e_bdxx,
                mflatpath=self.mflatpath_e,
                verbose_bdf=verbose,
                verbose_crrej=verbose_crrej
            )
            self.ccd_e_proc = self.ccd_e_bdfx.copy()
        else:
            self.ccd_e_bdfx = None
            # Do not update ccd_e_proc

        # ================================================================================================== #
        # * 5. Error calculation
        # ================================================================================================== #
        s = "Pixel error calculated for both o/e-rays"
        _t = Time.now()
        try:
            self.ccd_flat_o = load_ccd(self.mflatpath_o)
            flat_o_data = self.ccd_flat_o.data
            flat_o_err = self.ccd_flat_o.uncertainty.array
        except (FileNotFoundError, ValueError):
            self.ccd_flat_o = None
            flat_o_data = None
            flat_o_err = None

        try:
            self.ccd_flat_e = load_ccd(self.mflatpath_e)
            flat_e_data = self.ccd_flat_e.data
            flat_e_err = self.ccd_flat_e.uncertainty.array
        except (FileNotFoundError, ValueError):
            self.ccd_flat_e = None
            flat_e_data = None
            flat_e_err = None

        try:
            # if self._ccd_dark is not None:
            # rough estimations for the dark standard deviations
            self.dark_std_o = (self.ccd_dark_o.uncertainty.array
                               * np.sqrt(self.ccd_dark_o.header['NCOMBINE']))
            self.dark_std_e = (self.ccd_dark_e.uncertainty.array
                               * np.sqrt(self.ccd_dark_e.header['NCOMBINE']))
            subtracted_dark_o = self.ccd_dark_o.data
            subtracted_dark_e = self.ccd_dark_e.data
        except (TypeError, AttributeError):
            # else:
            self.dark_std_o = None
            self.dark_std_e = None
            subtracted_dark_o = None
            subtracted_dark_e = None

        self.err_o = errormap(
            self.ccd_o_bdxx,
            gain_epadu=self.gain,
            rdnoise_electron=self.rdnoise,
            subtracted_dark=subtracted_dark_o,
            dark_std=self.dark_std_o,
            flat=flat_o_data,
            flat_err=flat_o_err
        )
        self.err_e = errormap(
            self.ccd_e_bdxx,
            gain_epadu=self.gain,
            rdnoise_electron=self.rdnoise,
            subtracted_dark=subtracted_dark_e,
            dark_std=self.dark_std_e,
            flat=flat_e_data,
            flat_err=flat_e_err
        )

        add_to_header(self.ccd_o_proc.header, 'h', s, t_ref=_t)
        add_to_header(self.ccd_e_proc.header, 'h', s, t_ref=_t)

        # ================================================================================================== #
        # * 6. CR-rejection
        # ================================================================================================== #
        if do_crrej:
            crkw = dict(
                crrej_kw=crrej_kw,
                filt=self.filt,
                verbose=verbose_crrej,
                full=True,
                update_header=True
            )
            self.ccd_o_bdfc, self.mask_o_cr, self.crrej_kw = cr_reject_nic(
                self.ccd_o_proc,
                mask=self.mask_o_proc,
                **crkw
            )
            self.ccd_e_bdfc, self.mask_e_cr, _ = cr_reject_nic(
                self.ccd_e_proc,
                mask=self.mask_e_proc,
                **crkw
            )
            self.ccd_o_proc = self.ccd_o_bdfc.copy()
            self.ccd_e_proc = self.ccd_e_bdfc.copy()
            self.mask_o_proc = self.mask_o_proc | self.mask_o_cr
            self.mask_e_proc = self.mask_e_proc | self.mask_e_cr

        # ================================================================================================== #
        # * 8. Trim by bezel widths
        # ================================================================================================== #
        bzkw = dict(bezel_x=bezel_x, bezel_y=bezel_y, replace=replace)
        self.ccd_o_proc = bezel_ccd(self.ccd_o_proc, **bzkw, verbose=verbose)
        self.ccd_e_proc = bezel_ccd(self.ccd_e_proc, **bzkw, verbose=verbose)
        self.mask_o_proc = bezel_ccd(self.mask_o_proc, **bzkw, verbose=verbose)
        self.mask_e_proc = bezel_ccd(self.mask_e_proc, **bzkw, verbose=verbose)

        # # ==============================================================
        # # * 8. Add "list" version of CCDs
        # # ==============================================================
        # # By not copying, they're linked together and updated simult.ly.
        # self.ccd_bdfx  = [self.ccd_o_bdfx  , self.ccd_e_bdfx  ]
        # self.ccd_bdfc  = [self.ccd_o_bdfc  , self.ccd_e_bdfc  ]
        # self.ccd_proc  = [self.ccd_o_proc  , self.ccd_e_proc]
        # self.mask_proc = [self.mask_o_proc , self.mask_e_proc ]
        # self.err       = [self.err_o       , self.err_e       ]
        # self.flat      = [self.ccd_flat_o  , self.ccd_flat_e  ]
        # self.dark      = [self.ccd_dark_o  , self.ccd_dark_e  ]

    def edgereplace(self, bezel_x=(5, 5), bezel_y=(5, 5), replace=np.nan):
        ''' Replace edge values to null for better zscale display.
        Parameters
        ----------
        bezel_x, bezel_y : array-like of int
            The x and y bezels, in ``[lower, upper]`` convention. If
            ``0``, no replacement happens.
        replace : int, float, nan, optional.
            The value to replace the pixel value where it should be
            masked. If ``None``, the ccds will be trimmed.
        '''
        pass

    # def edgemask(self, bezel_x=(10, 10), bezel_y=(10, 10),
    #              sigma_lower=1, sigma_upper=1,
    #              maxiters=10, edge_ksigma=3, replace=np.nan):
    #     ''' Replace edge values to null for better zscale display.
    #     Parameters
    #     ----------
    #     bezel_x, bezel_y : int, float, list of such, optional.
    #         The x and y bezels, in ``[lower, upper]`` convention.
    #     replace : int, float, nan, None, optional.
    #         The value to replace the pixel value where it should be
    #         masked. If ``None``, nothing is replaced, but only the
    #         ``self.mask_proc`` will have been updated.
    #     '''
    #     def _idxmask(maskarr):
    #         try:
    #             idx_mask = np.max(np.where(maskarr))
    #         except ValueError:  # sometimes no edge is detected
    #             idx_mask = 0
    #         return idx_mask

    #     sc_kw = dict(sigma_lower=sigma_lower,
    #                  sigma_upper=sigma_upper,
    #                  maxiters=maxiters)
    #     self._edge_sigclip_mask = [None, None]

    #     # Iterate thru o-/e-ray
    #     for i, (ccd, mask) in enumerate(zip(self.ccd_proc, self.mask_proc)):
    #         ny, nx = ccd.data.shape
    #         _, med, std = sigma_clipped_stats(ccd.data, mask, **sc_kw)
    #         scmask = (ccd.data < (med - edge_ksigma*std))
    #         self._edge_sigclip_mask[i] = scmask
    #         # Mask for more than half of the total N is masked
    #         xmask = np.sum(scmask, axis=0) > ny/2
    #         ymask = np.sum(scmask, axis=1) > nx/2

    #         # Sometimes "low level row/col" may occur other than edge.
    #         # Find whether the edge is at left/right of x and
    #         # upper/lower of y.
    #         isleft = (np.sum(xmask[:nx//2]) > np.sum(xmask[nx//2:]))
    #         islowr = (np.sum(ymask[:ny//2]) > np.sum(ymask[ny//2:]))

    #         # * Set mask for x-axis edge
    #         if isleft:
    #             ix = np.min([_idxmask(xmask[:nx//2]), bezel_x[0]])
    #             sx = (slice(None, None, None), slice(None, ix, None))
    #         else:
    #             ix = np.min([_idxmask(xmask[nx//2:]), nx - bezel_x[0]])
    #             sx = (slice(None, None, None), slice(ix, None, None))

    #         # * Set mask for y-axis edge
    #         if islowr:
    #             iy = np.min([_idxmask(ymask[:ny//2]), bezel_y[0]])
    #             sy = (slice(None, iy, None), slice(None, None, None))
    #         else:
    #             iy = np.max([_idxmask(ymask[ny//2:]), ny - bezel_y[1]])
    #             sy = (slice(iy, None, None), slice(None, None, None))

    #         mask[sx] = True
    #         mask[sy] = True
    #         if replace is not None:
    #             ccd.data[sx] = replace
    #             ccd.data[sy] = replace

    def find_obj(self, thresh=3, bezel_x=(40, 40), bezel_y=(200, 120),
                 box_size=(64, 64), filter_size=(12, 12), deblend_cont=1,
                 minarea=100, verbose=True,
                 **extract_kw):
        """
        Note
        ----
        This includes ``sep``'s ``extract`` and ``background``.
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

        deblend_cont : float, optional
            Minimum contrast ratio used for object deblending. To
            entirely disable deblending, set to 1.0.

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
                      deblend_cont=deblend_cont, bezel_x=bezel_x,
                      bezel_y=bezel_y, **extract_kw)
        sepv = sep.__version__
        s_bkg = f"Background estimated from sep (v {sepv}) with {bkg_kw}."
        s_obj = "Objects found from sep (v {}) with {}."

        _t = Time.now()
        self.bkg_o = sep_back(
            self.ccd_o_proc.data,
            mask=self.mask_o_proc,
            **bkg_kw
        )
        add_to_header(self.ccd_o_proc.header, 'h', s_bkg,
                      verbose=verbose, t_ref=_t)

        _t = Time.now()
        self.obj_o, self.seg_o = sep_extract(
            self.ccd_o_proc.data,
            bkg=self.bkg_o,
            err=self.err_o,
            mask=self.mask_o_proc,
            **ext_kw)
        _s = s_obj.format(sepv, ext_kw)
        add_to_header(self.ccd_o_proc.header, 'h', _s,
                      verbose=verbose, t_ref=_t)

        _t = Time.now()
        self.bkg_e = sep_back(
            self.ccd_e_proc.data,
            mask=self.mask_e_proc,
            **bkg_kw
        )
        add_to_header(self.ccd_e_proc.header, 'h', s_bkg,
                      verbose=verbose, t_ref=_t)

        _t = Time.now()
        self.obj_e, self.seg_e = sep_extract(
            self.ccd_e_proc.data,
            bkg=self.bkg_e,
            err=self.err_e,
            mask=self.mask_e_proc,
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
            ccd.header["NOBJ-SEP"] = (n, "Number of objects found from SEP.")
            if n < 1:
                warn(f"No object found for {oe}-ray of {self.file}!", Warning)
            elif n > 1:
                _sort_obj(ccd, obj, nobj=n, verbose=verbose)
                warn(f"No object found for {oe}-ray of {self.file}!", Warning)

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
            self._ccd_vs = vertical_correct(
                self.ccd_raw,
                fitting_sections=self.verti_fitting_sections,
                method=self.verti_method,
                sigclip_kw=self.verti_sigclip_kw,
                **vskw  # hard-coded
            )
            self.ccd_proc = self._ccd_vs.copy()

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
                self._ccd_vs,
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
                self._ccd_vs.data,
                npool=fouri_npool,
                mask=self.additional_mask,
                filt=self.filt,
                fitting_y_sections=self.fouri_fitting_y_sections,
                subtract_x_sections=self.fouri_subtract_x_sections,
                **fckw  # hard-coded
            )
            self._pattern_fc = res[0]
            self.popts = res[1]
            self._ccd_fc = self._ccd_vs.copy()
            self._ccd_fc.data -= self._pattern_fc
            self.ccd_proc = self._ccd_fc.copy()

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
        self._ccd_fc = res[0]
        self._pattern_fc = res[1]
        self.popt = res[2]
        self.freq = res[3]

        # If ``ccd=self.ccd_raw_cr`` in fouriersub, the resulting CCD
        # will contain the pattern-subtracted value AFTER CR-REJ, which
        # is not desirable sometimes.
        self._ccd_fc.data = self.ccd_raw.data - self._pattern_fc
        ##########################
        # Remove the above line if you used ``ccd=self.ccd_raw``, etc.

        if output is not None:
            self._ccd_fc.write(Path(output), overwrite=overwrite)

        if calc_var:
            self.var_fc = (self._ccd_fc.data*(flat_err + 1/self.gain)
                           + self.rdnoise**2)
'''
