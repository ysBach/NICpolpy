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
from astropy.visualization import ImageNormalize, SqrtStretch, ZScaleInterval
from photutils.aperture import CircularAnnulus, CircularAperture
from ysfitsutilpy import (CCDData_astype, add_to_header, bdf_process,
                          bezel_ccd, errormap, fitsxy2py, imcombine, load_ccd,
                          medfilt_bpm, propagate_ccdmask, select_fits,
                          set_ccd_gain_rdnoise, trim_ccd)
from ysphotutilpy import (LinPolOE4, apphot_annulus, ellip_ap_an,
                          fit_Gaussian2D, sep_back, sep_extract, sky_fit)

from .preproc import (cr_reject_nic, find_fourier_peaks, fit_fourier,
                      vertical_correct)
from .util import (DARK_PATHS, FLAT_PATHS, FOURIERSECTS, GAIN, MASK_PATHS,
                   OBJSECTS, OBJSLICES, RDNOISE, USEFUL_KEYS, VERTICALSECTS,
                   infer_filter, parse_fpath, split_oe, summary_nic)

try:
    import fitsio
    HAS_FITSIO = True
except ImportError:
    warn("python version of fitsio is strongly recommended (https://github.com/esheldon/fitsio/tree/master/)")
    HAS_FITSIO = False

__all__ = [
    "NICPolDir", "NICPolPhot", "read_pols",
    "NICPolImage"]


_PHOT_COLNAMES = ['id', 'xcenter', 'ycenter', 'aparea',
                  'aperture_sum', 'aperture_sum_err',
                  'msky', 'nrej', 'nsky', 'ssky',
                  'source_sum', 'source_sum_err',
                  'mag', 'merr', 'x_fwhm', 'y_fwhm', 'theta']

MEDCOMB_SC3_F4 = dict(combine='med', reject='sc', sigma=3, maxiters=50, use_cfitsio=True, dtype='float32')


def DONE2HDR(header, verbose):
    add_to_header(header, 'h', verbose=verbose, fmt=None, s="{:-^72s}".format(' DONE'))


def FINDFITS(tab, filt, oe, exptime=None, objname=None, loadccd=False, verbose=False):
    type_key = ["FILTER", "OERAY"]
    type_val = [filt.upper(), oe.lower()]
    for k, v in zip(["EXPTIME", "OBJECT"], [exptime, objname]):
        if v is not None:
            type_key.append(k)
            type_val.append(v)
    return select_fits(summary_table=tab, type_key=type_key, type_val=type_val,
                       loadccd=loadccd, verbose=verbose)


def SAVENIC(ccd, original_path, object_name, savedir, combined=False, verbose=False):
    def _set_fname(original_path, object_name, combined=False):
        ''' If combined, COUNTER, POL-AGL1, INSROT are meaningless, so remove these.
        '''
        es = parse_fpath(original_path)
        es['OBJECT'] = object_name
        if combined:
            fstem = '_'.join([es['filt'], es['yyyymmdd'], es['OBJECT'], es['EXPTIME'], es['oe']])
        else:
            fstem = '_'.join(es.values())
        return fstem + '.fits'

    ccd.header["OBJECT"] = object_name
    newpath = savedir/_set_fname(original_path, object_name, combined=combined)
    ccd.write(newpath, overwrite=True)
    if verbose:
        print(f"Writing FITS to {newpath}")
    return ccd, newpath


class NICPolDirMixin:
    @staticmethod
    def mkpuredark(tab, caldir, filt, oe, exptime, dark_min=10., verbose_combine=False):
        _t = Time.now()
        dark_fpaths = FINDFITS(tab, filt, oe, exptime=exptime, verbose=False)
        comb = imcombine(dark_fpaths, **MEDCOMB_SC3_F4, verbose=verbose_combine)
        comb.data[comb.data < dark_min] = 0
        add_to_header(comb.header,
                      'h',
                      f"Images combined and dark_min {dark_min} applied.",
                      t_ref=_t,
                      verbose=verbose_combine)
        _, comb_dark_path = SAVENIC(comb, dark_fpaths[0], "DARK", caldir, combined=True)
        return comb_dark_path

    @staticmethod
    def mkskydark_single(fpath, tmpcal, skyname, mflat, dark_min=10., skydark_medfilt_bpm_kw={},
                         verbose_bdf=False, verbose_combine=False):
        _t = Time.now()
        sky = load_ccd(fpath)
        add_to_header(sky.header, 'h', verbose=verbose_bdf, fmt=None,
                      s="{:=^72s}".format(' Estimating DARK from this sky frame '))
        # Sky / Flat
        # flat division to prevent artificial CR rejection:
        sky_f = sky.copy()
        sky_f.data = sky.data/mflat
        sky_f, _ = SAVENIC(sky_f, fpath, f"{skyname}_FLAT", tmpcal)

        # (Sky/Flat)_cr
        # sky_f_cr = cr_reject_nic(sky_f, crrej_kw=crrej_kw, verbose=verbose_crrej)
        sky_f_cr = medfilt_bpm(sky_f, **skydark_medfilt_bpm_kw)
        sky_f_cr, _ = SAVENIC(sky_f_cr, fpath, f"{skyname}_FLAT_CRREJ", tmpcal)

        # (Sky/Flat)_cr * Flat
        sky_cr = sky_f_cr.copy()
        sky_cr.data *= mflat
        sky_cr, _ = SAVENIC(sky_cr, fpath, f"{skyname}_FLAT_CRREJ_DEFLATTED", tmpcal)

        # Dark ~ Sky - (Sky/Flat)_cr * Flat
        sky_dark = sky_f_cr.copy()  # retain CRREJ header info
        sky_dark.data = sky.data - sky_f_cr.data*mflat
        sky_dark.data[sky_dark.data < dark_min] = 0
        add_to_header(
            sky_dark.header, 'h', t_ref=_t, verbose=verbose_combine,
            s=("Dark from this frame estimated by sky - (sky/flat)_cr*flat "
                + f"and replaced pixel value < {dark_min} = 0.")
        )

        add_to_header(
            sky_dark.header, 'h', verbose=verbose_bdf, fmt=None,
            s="{:=^72s}".format(' Similar SKYDARK frames will be combined ')
        )

        _, sky_dark_path = SAVENIC(sky_dark, fpath, f"{skyname}_SKYDARK", self.tmpcal)

        return sky_dark_path

    @staticmethod
    def mkskydark_comb(fpaths, caldir, skyname, verbose_combine=False, verbose_bdf=False):
        comb_sky_dark = imcombine(fpaths, **MEDCOMB_SC3_F4, verbose=verbose_combine)
        DONE2HDR(comb_sky_dark.header, verbose_bdf)
        _, comb_sky_dark_path = SAVENIC(comb_sky_dark, fpaths[0],
                                        f"{skyname}_SKYDARK", caldir, combined=True)
        return comb_sky_dark_path

    @staticmethod
    def mkfringe_single(fpath, tmpcal, skyname, mdark, mflat, mdarkpath, mflatpath,
                        verbose_bdf=False, verbose_crrej=False):
        sky = load_ccd(fpath)
        # give mdark/mflat so that the code does not read the FITS files repeatedly:
        add_to_header(sky.header, 'h', verbose=verbose_bdf, fmt=None,
                      s="{:=^72s}".format(' Estimating FRINGE from this sky frame '))
        sky_fringe = bdf_process(sky,
                                 mdark=mdark,
                                 mflat=CCDData(mflat, unit='adu'),
                                 mdarkpath=mdarkpath,
                                 mflatpath=mflatpath,
                                 verbose_bdf=verbose_bdf,
                                 verbose_crrej=verbose_crrej)
        add_to_header(sky_fringe.header, 'h', verbose=verbose_bdf, fmt=None,
                      s="{:=^72s}".format(' Similar SKYFRINGE frames will be combined '))

        _, sky_tocomb_path = SAVENIC(sky_fringe, fpath, f"{skyname}_FRINGE", tmpcal)
        return sky_tocomb_path

    @staticmethod
    def mkfringe_comb(fpaths, logpath, skyname, caldir, scale_section, scale='avg',
                      scale_to_0th=False, fringe_min_value=0.0, verbose_combine=False, verbose_bdf=False):
        # FRINGE must not be smoothed as remaining DARK signal may reside here.
        comb_sky_fringe = imcombine(fpaths,
                                    **MEDCOMB_SC3_F4,
                                    scale=scale,
                                    scale_to_0th=scale_to_0th,
                                    scale_section=scale_section,
                                    verbose=verbose_combine,
                                    logfile=logpath)

        # Normalize using the section
        _t = Time.now()
        norm_value = np.mean(comb_sky_fringe.data[fitsxy2py(scale_section)])
        comb_sky_fringe.data /= norm_value
        comb_sky_fringe.data[comb_sky_fringe.data < fringe_min_value] = 0
        add_to_header(comb_sky_fringe.header, 'h', t_ref=_t, verbose=verbose_combine,
                      s="Normalized by mean of NORMSECT (NORMVALU), replaced value < FRINMINV to 0")
        comb_sky_fringe.header["NORMSECT"] = scale_section
        comb_sky_fringe.header["NORMVALU"] = norm_value
        comb_sky_fringe.header["FRINMINV"] = fringe_min_value
        DONE2HDR(comb_sky_fringe.header, verbose_bdf)

        _, comb_sky_fringe_path = SAVENIC(comb_sky_fringe, fpaths[0],
                                          f"{skyname}_SKYFRINGE", caldir, combined=True)
        return comb_sky_fringe_path

    @staticmethod
    def _set_mflat(summary_flat, filt, oe, flatdir, flat_min_value=0.):
        ''' Note that it returns ndarray, not CCDData.
        '''
        if summary_flat is None:
            return 1, None

        if flatdir is not None:
            mflatpath = FINDFITS(summary_flat, filt, oe, verbose=False)
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

    @staticmethod
    def _set_dark(prefer_skydark, paths_skydark, paths_puredark, objname, filt, oe, exptime, verbose):
        if prefer_skydark:
            try:
                mdarkpath = paths_skydark[(f"{objname}_sky", filt, oe, exptime)]
                mdark = load_ccd(mdarkpath)
            except (KeyError, IndexError, FileNotFoundError):
                if verbose:
                    print(f"prefer_skydark but skydark for ({objname}_sky, {filt}, {oe}, "
                          + f"{exptime}) not found. Trying to use pure dark.")
                try:
                    mdarkpath = paths_puredark[(filt, oe, exptime)]
                    mdark = load_ccd(mdarkpath)
                except (KeyError, IndexError, FileNotFoundError):
                    mdarkpath = None
                    mdark = None
                    if verbose:
                        print("\nNo dark file found. Turning off dark subtraction.")

        else:
            try:
                mdarkpath = paths_puredark[(filt, oe, exptime)]
                mdark = load_ccd(mdarkpath)
            except (KeyError, IndexError, FileNotFoundError):
                if verbose:
                    print(f"Pure dark for ({filt}, {oe}, {exptime}) not found. "
                          + f"Trying to use SKYDARK of ({objname}_sky, {filt}, {oe}, {exptime})",
                          end='... ')
                try:
                    mdarkpath = paths_skydark[(f"{objname}_sky", filt, oe, exptime)]
                    mdark = load_ccd(mdarkpath)
                    if verbose:
                        print("Loaded successfully.")
                except (KeyError, IndexError, FileNotFoundError):
                    mdarkpath = None
                    mdark = None
                    if verbose:
                        print("No dark file found. Turning off dark subtraction.")
        return mdark, mdarkpath

    @staticmethod
    def _set_fringe(paths_skyfringe, objname, filt, oe, exptime, verbose):
        try:
            mfringepath = paths_skyfringe[(f"{objname}_sky", filt, oe, exptime)]
            mfringe = load_ccd(mfringepath)
        except (KeyError, IndexError, FileNotFoundError):
            mfringepath = None
            mfringe = None
            if verbose:
                print("No finge file found. Turning off fringe subtraction.")
        return mfringe, mfringepath

    @staticmethod
    def _find_obj(arr, var,
                  thresh_tests=[30, 20, 10, 6, 5, 4, 3], bezel_x=(30, 30), bezel_y=(180, 120),
                  box_size=(64, 64), filter_size=(12, 12), deblend_cont=1,
                  minarea=314,
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
        bkg, obj, segm

        """
        bkg_kw = dict(maskthresh=0.0, filter_threshold=0.0, box_size=box_size, filter_size=filter_size)
        bkg = sep_back(arr, **bkg_kw)
        sepv = sep.__version__
        s_bkg = f"Background estimated from sep (v {sepv}) with {bkg_kw}."

        thresh_tests = np.sort(np.atleast_1d(thresh_tests))[::-1]
        for thresh in thresh_tests:
            ext_kw = dict(thresh=thresh, minarea=minarea, deblend_cont=deblend_cont,
                          bezel_x=bezel_x, bezel_y=bezel_y, **extract_kw)

            obj, seg = sep_extract(arr, bkg=bkg, var=var, **ext_kw)
            nobj = len(obj)
            if nobj < 1:
                continue
            else:
                s_obj = f"Objects found from sep (v {sepv}) with {ext_kw}."
                break

        found = nobj >= 1
        if not found:
            s_obj = f"NO object found from sep (v {sepv}) with {ext_kw}."

        return bkg, obj, seg, s_bkg, s_obj, found


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

        # ==  make directories if not exist ================================================================ #
        self.caldir.mkdir(parents=True, exist_ok=True)
        self.tmpcal.mkdir(parents=True, exist_ok=True)
        self.tmpred.mkdir(parents=True, exist_ok=True)

        # Check flatdir
        self.flatdir = Path(flatdir) if flatdir is not None else None
        if (self.flatdir is not None) and (not self.flatdir.exists()):
            raise FileNotFoundError("Flat directory not found.")

        self.keys = USEFUL_KEYS + ["OERAY"]

        # == collect header info of FLATs ================================================================== #
        if self.flatdir is not None:
            self.summary_flat = summary_nic(self.flatdir/"*.fits", keywords=self.keys, verbose=verbose)
        else:
            self.summary_flat = None

        if verbose:
            print("Extracting header info...")

        # == collect header info of RAWs =================================================================== #
        self.summary_raw = summary_nic(self.rawdir/"*.fits", keywords=self.keys, verbose=verbose)

        # == Find object frames, object_sky frames, and find the unique EXPTIMEs. ========================== #
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

        # == Check if DARK exists ========================================================================== #
        if self.pure_dark_exists:
            self.summary_dark = self.summary_raw[self.summary_raw["OBJECT"] == "DARK"]
        else:
            self.summary_dark = None

    def prepare_calib(self, dark_min=10., sky_scale_section="[20:130, 40:140]", fringe_min_value=0.0,
                      flat_min_value=0.0,
                      skydark_medfilt_bpm_kw=dict(size=5, std_section="[40:100, 40:140]", std_model='std',
                                                  med_sub_clip=[None, 10], med_rat_clip=None,
                                                  std_rat_clip=[None, 3]),
                      verbose=True, verbose_combine=False, verbose_bdf=False, verbose_crrej=False):
        '''
        Parameters
        ----------
        skydark_medfilt_bpm_kw : dict, optional.
            The median filtered bad pixel masking algorithm (MBPM)
            parameters if SKYDARK must be extracted. The lower clips are
            all `None` so that the SKYDARK frame contains **no** pixel
            smaller than local median minus ``med_sub_clip[1]``.

        fringe_min_value : float, optional.
            All pixels smaller than this value in the fringe map
            (super-bad pixels) are replaced with this. Setting
            ``fringe_min_value=0`` removes all negative pixels from
            fringe, so the fringe-subtracted frame will contain negative
            pixels (super-bad pixels) unchanged, so that easily replaced
            in preprocessing.
        '''
        self.dark_min = dark_min
        self.sky_scale_section = sky_scale_section
        self.sky_scale_slice = fitsxy2py(self.sky_scale_section)
        self.paths_puredark = {}
        self.paths_skydark = {}
        self.paths_skyfringe = {}
        self.paths_flat = {}
        self.fringe_min_value = fringe_min_value
        self.flat_min_value = flat_min_value

        print(f"Output and intermediate files are saved to {self.caldir} and {self.tmpcal}.")
        if self.flatdir is None:
            print("Better to specify flat files by flatdir.")

        if verbose:
            print("Estimating DARK from sky frames if exists... ")

        # == Combine dark if DARK exists =================================================================== #
        if verbose:
            print("Combining DARK frames if exists", end='... ')

        if self.pure_dark_exists:
            exptimes = np.unique(self.summary_dark["EXPTIME"])
            for filt, oe, exptime in product('JHK', 'oe', exptimes):
                comb_dark_path = self.mkpuredark(
                    tab=self.summary_dark,
                    caldir=self.caldir,
                    filt=filt,
                    oe=oe,
                    exptime=exptime,
                    dark_min=self.dark_min,
                    verbose_combine=verbose_combine
                )
                self.paths_puredark[(filt, oe, exptime)] = comb_dark_path
            if verbose:
                print("Done.")
        elif verbose:
            print("No pure DARK found. Trying SKYDARK", end='... ')

        # == Reduce SKY frames if exists =================================================================== #
        for filt, oe in product('JHK', 'oe'):
            mflat, mflatpath = self._set_mflat(self.summary_flat, filt, oe, self.flatdir, self.flat_min_value)
            self.paths_flat[(filt, oe)] = mflatpath
            for skyname, skydict in self.skydata.items():
                for exptime in skydict['exptimes']:
                    sky_fpaths = FINDFITS(skydict['summary'], filt, oe, exptime, verbose=verbose)
                    skycomb_paths = []
                    try:
                        # == Check to see if there is PUREDARK ============================================= #
                        mdarkpath = self.paths_puredark[(filt, oe, exptime)]
                        print(self.paths_puredark[(filt, oe, exptime)])
                    except (KeyError, IndexError):
                        # == Estimate DARK from sky (SKYDARK) if no PUREDARK found ========================= #
                        sky_dark_paths = []
                        for fpath in sky_fpaths:
                            # -- estimate SKYDARK for ALL _sky frames
                            sky_dark_path = self.mkskydark_single(
                                fpath=fpath,
                                tmpcal=self.tmpcal,
                                skyname=skyname,
                                mflat=mflat,
                                dark_min=self.dark_min,
                                skydark_medfilt_bpm_kw=skydark_medfilt_bpm_kw,
                                verbose_bdf=verbose_bdf,
                                verbose_combine=verbose_combine
                            )
                            sky_dark_paths.append(sky_dark_path)

                        # -- Combine estimated SKYDARK (_sky) frames
                        if verbose:
                            print("    Combine the estimated DARK from _sky frames", end='... ')

                        comb_sky_dark_path = self.mkskydark_comb(
                            fpaths=sky_dark_paths,
                            caldir=self.caldir,
                            skyname=skyname,
                            verbose_combine=verbose_combine,
                            verbose_bdf=verbose_bdf
                        )
                        self.paths_skydark[(skyname, filt, oe, exptime)] = comb_sky_dark_path
                        mdarkpath = comb_sky_dark_path

                        if verbose:
                            print("Done.")

                    # == Make fringe frames for each sky =================================================== #
                    if verbose:
                        print("    Make and combine the SKYFRINGES (flat corrected) from _sky frames",
                              end='...')

                    mdark = load_ccd(mdarkpath)
                    for fpath in sky_fpaths:
                        sky_tocomb_path = self.mkfringe_single(
                            fpath=fpath,
                            tmpcal=self.tmpcal,
                            skyname=skyname,
                            mdark=mdark,
                            mflat=mflat,
                            mdarkpath=mdarkpath,
                            mflatpath=mflatpath,
                            verbose_bdf=verbose_bdf,
                            verbose_crrej=verbose_crrej
                        )
                        skycomb_paths.append(sky_tocomb_path)

                    # == Combine sky fringes =============================================================== #
                    # combine dark-subtracted and flat-corrected sky frames to get the fringe pattern by sky
                    # emission lines:
                    logpath = self.caldir/f"{filt.lower()}_{skyname}_{exptime:.1f}_{oe}_combinelog.csv"
                    comb_sky_fringe_path = self.mkfringe_comb(
                        fpaths=skycomb_paths,
                        logpath=logpath,
                        skyname=skyname,
                        caldir=self.caldir,
                        scale_section=self.sky_scale_section,
                        scale='avg',
                        scale_to_0th=False,
                        fringe_min_value=self.fringe_min_value,
                        verbose_combine=verbose_combine,
                        verbose_bdf=verbose_bdf
                    )
                    self.paths_skyfringe[(skyname, filt, oe, exptime)] = comb_sky_fringe_path

                    if verbose:
                        print("Done.")

    def preproc(self, reddir="reduced", prefer_skydark=False,
                med_rat_clip=[0.5, 2], std_rat_clip=[-3, 3], bezel_x=[20, 20], bezel_y=[20, 20],
                medfilt_kw=dict(med_sub_clip=None, size=5),
                do_crrej_pos=True, do_crrej_neg=True,
                verbose=True, verbose_bdf=False, verbose_bpm=False, verbose_crrej=False, verbose_phot=False
                ):
        '''
        '''
        self.reddir = self.location/reddir
        self.reddir.mkdir(parents=True, exist_ok=True)
        self.summary_cal = summary_nic(self.caldir/"*.fits", keywords=self.keys, verbose=verbose)
        self.darks = {}
        self.fringes = {}

        for filt, oe in product('JHK', 'oe'):
            mflatpath = self.paths_flat[(filt, oe)]
            gain = GAIN[filt]
            rdnoise = RDNOISE[filt]
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
                    mdark, mdarkpath = self._set_dark(
                        prefer_skydark=prefer_skydark,
                        paths_skydark=self.paths_skydark,
                        paths_puredark=self.paths_puredark,
                        objname=objname,
                        filt=filt,
                        oe=oe,
                        exptime=exptime,
                        verbose=verbose
                    )
                    mfringe, mfringepath = self._set_fringe(
                        paths_skyfringe=self.paths_skyfringe,
                        objname=objname,
                        filt=filt,
                        oe=oe,
                        exptime=exptime,
                        verbose=verbose
                    )

                    # == Reduce data and do photometry ===================================================== #
                    raw_fpaths = FINDFITS(self.summary_raw, filt, oe, exptime, objname)

                    for fpath in raw_fpaths:
                        if verbose:
                            print('\n{}'.format(fpath))
                        rawccd = load_ccd(fpath)
                        set_ccd_gain_rdnoise(rawccd, gain=gain, rdnoise=rdnoise)
                        add_to_header(rawccd.header, 'h', verbose=verbose_bdf,
                                      s="{:=^72s}".format(' Preprocessing start '), fmt=None)
                        # set_ccd_gain_rdnoise(rawccd)
                        # 1. Do Dark and Flat.
                        # 2. Subtract Fringe
                        if (mdark is not None) or (mflat is not None) or (mfringe is not None):
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
                            objname = redccd.header['OBJECT']
                            proc = ''
                            if mdark is not None:
                                proc += "D"
                            if mflat is not None:
                                proc += "F"
                            if mfringe is not None:
                                proc += "Fr"

                            if proc != '':
                                objname = '_'.join([objname, proc])

                            redccd, _ = SAVENIC(redccd, fpath, objname, self.tmpred, verbose=verbose)
                            objname_proc = objname
                        else:
                            redccd = rawccd.copy()

                        #   3. Calculate median-filtered frame.
                        #   4. Calculate
                        #       RATIO = original/medfilt,
                        #       SIGRATIO = (original - medfilt) / mean(original)
                        #   5. Replace pixels by Median BPM (MBPM) algorithm
                        if verbose:
                            print("Median filter bad pixel masking (MBPM) will be used.")

                        redccd, res = medfilt_bpm(redccd,
                                                  std_section=self.sky_scale_section,
                                                  med_rat_clip=med_rat_clip,
                                                  std_rat_clip=std_rat_clip,
                                                  **medfilt_kw,
                                                  verbose=verbose_bpm,
                                                  full=True)

                        tmpccd = redccd.copy()
                        tmpccd.data = res['med_filt']
                        medfilt_objname = objname + "_MedFilt"
                        _ = SAVENIC(tmpccd, fpath, medfilt_objname, self.tmpred, verbose=verbose_bpm)

                        tmpccd.data = res['med_sub']
                        med_sub_objname = objname + "_MedSub"
                        _ = SAVENIC(tmpccd, fpath, med_sub_objname, self.tmpred, verbose=verbose_bpm)

                        tmpccd.data = res['med_rat']
                        med_rat_objname = objname + "_MedRatio"
                        _ = SAVENIC(tmpccd, fpath, med_rat_objname, self.tmpred, verbose=verbose_bpm)

                        tmpccd.data = res['std_rat']
                        std_rat_objname = objname + "_StdRatio"
                        _ = SAVENIC(tmpccd, fpath, std_rat_objname, self.tmpred, verbose=verbose_bpm)

                        tmpccd.data = (1*res['posmask'] + 2*res['negmask']).astype(np.uint8)
                        fullmask_objname = objname + "_MASK_pos1neg2"
                        _ = SAVENIC(tmpccd, fpath, fullmask_objname, self.tmpred, verbose=verbose_bpm)

                        DONE2HDR(redccd.header, verbose_bdf)

                        # -- Uncertainty calculation
                        # Use the raw one, i.e., BEFORE dark, flat, sky, crrej, etc.
                        # var = error^2
                        _t = Time.now()
                        var = (rawccd.data/gain     # Photon noise = signal + dark + sky (fringe) BEFORE flat
                               + (rdnoise/gain)**2  # readout noise
                               + 1/12               # digitization (see Eq 12 and below of Merline+Howell 95)
                               ).astype('float32')
                        # sometimes negative pixel exists and gives NaN is sqrt is taken...
                        # redccd.uncertainty = StdDevUncertainty(np.sqrt(var))
                        redccd.uncertainty = VarianceUncertainty(var)

                        add_to_header(redccd.header, 'h', verbose=verbose_bdf, t_ref=_t,
                                      s=("1-sigma VARIANCE calculated "
                                         + f"by GAIN ({gain}) and RDNOISE ({rdnoise});"
                                         + " see ext=1 (EXTNAME = 'UNCERT')"))

                        # -- Save final result
                        redccd = bezel_ccd(redccd, bezel_x=bezel_x, bezel_y=bezel_y, replace=None,
                                           verbose=verbose)
                        _ = SAVENIC(redccd, fpath, objname_proc, self.reddir, verbose=verbose)

                    if verbose:
                        print()


class NICPolPhot(NICPolDirMixin):
    def __init__(self, location, objnames=None, reddir="reduced",
                 p_eff=dict(J=98., H=95., K=92.), dp_eff=dict(J=6., H=7., K=12.),
                 theta_inst=dict(J=0.5, H=1.3, K=-0.7), dtheta_inst=dict(J=1.3, H=3.1, K=6.3),
                 q_inst=dict(J=0.0, H=0.03, K=-0.02), u_inst=dict(J=-0.01, H=-0.03, K=-0.07),
                 dq_inst=dict(J=0.29, H=0.52, K=0.30), du_inst=dict(J=0.29, H=0.55, K=0.31),
                 correct_dqdu_stddev_to_stderr=True
                 ):
        self.location = Path(location)
        if reddir is None:
            self.reddir = self.location
        else:
            self.reddir = self.location/reddir

        if not self.reddir.exists():
            raise FileNotFoundError("Reduced data directory not found.")

        self.p_eff = p_eff
        self.dp_eff = dp_eff
        self.theta_inst = theta_inst
        self.dtheta_inst = dtheta_inst
        # TODO: Currently instrumental polarizaiton correction is turned off.
        # self.q_inst = q_inst
        # self.u_inst = u_inst
        # if correct_dqdu_stddev_to_stderr:
        #     for filt in "JHK":
        #         # Convert stddev to the standard error of the mean estimator (see TakahashiJ+2018)
        #         dq_inst[filt] = dq_inst[filt]/np.sqrt(150 + 15 + 15)
        #         du_inst[filt] = du_inst[filt]/np.sqrt(150 + 15 + 15)
        # self.dq_inst = dq_inst
        # self.du_inst = du_inst

        # We need not make a summary, because filenames contain all such information.
        self.redfpaths = list(self.reddir.glob("*.fits"))
        self.redfpaths.sort()
        self.objnames = objnames
        self.parsed = pd.DataFrame.from_dict([parse_fpath(fpath) for fpath in self.redfpaths])
        #                                     ^^ 0.7-0.8 us per iteration, and few ms for DataFrame'ing

        if self.objnames is not None:
            objmask = self.parsed['OBJECT'].str.split('_', expand=True)[0].isin(np.atleast_1d(self.objnames))
            self.parsed = self.parsed[objmask]
            self.objfits = np.array(self.redfpaths)[objmask]
        else:
            self.objfits = self.redfpaths

        self.parsed.insert(loc=0, column='file', value=self.objfits)
        self.parsed['counter'] = self.parsed['counter'].astype(int)
        self.parsed['set'] = np.tile(1 + np.arange(len(self.parsed)/3)//8, 3).astype(int)
        #                                               (nimg/filter) // files/set=8
        self.parsed['PA'] = self.parsed['PA'].astype(float)
        self.parsed['INSROT'] = self.parsed['INSROT'].astype(float)
        self.parsed['IMGROT'] = self.parsed['IMGROT'].astype(float)
        self.grouped = self.parsed.groupby(['filt', 'set'])

    def photpol(self, radii, figdir=None, thresh_tests=[30, 20, 10, 6, 5, 4, 3, 2, 1, 0],
                sky_r_in=60, sky_r_out=90, skysub=True,
                satlevel=dict(J=7000, H=7000, K=7000), sat_npix=dict(J=5, H=5, K=5),
                verbose_bkg=False, verbose_obj=False, verbose=True,
                obj_find_kw=dict(bezel_x=(30, 30), bezel_y=(180, 120), box_size=(64, 64),
                                 filter_size=(12, 12), deblend_cont=1, minarea=100),
                output_pol=None):
        self.phots = {}
        self.positions = {}
        self.pols = {}

        if figdir is not None:
            self.figdir = Path(figdir)
            self.figdir.mkdir(parents=True, exist_ok=True)
            savefig = True
        else:
            self.figdir = None
            savefig = False

        self.radii = np.atleast_1d(radii)

        self.skyan_kw = dict(r_in=sky_r_in, r_out=sky_r_out)
        self.Pol = {}

        for (filt, set_id), df in self.grouped:
            if len(df) < 8:  # if not a full 4 images/set is obtained
                continue
            for i, (_, row) in enumerate(df.iterrows()):
                fpath = row['file']
                if HAS_FITSIO:
                    # Using FITSIO reduces time 0.3 s/set --> 0.1 s/set (1 set = 8 FITS files).
                    arr = fitsio.FITS(fpath)[0].read()
                    var = fitsio.FITS(fpath)[1].read()  # variance only from photon noise.
                else:
                    _ccd = load_ccd(fpath)
                    arr = _ccd.data
                    var = _ccd.uncertainty

                find_res = self._find_obj(arr, var=var, thresh_tests=thresh_tests, **obj_find_kw)
                bkg, obj, seg, s_bkg, s_obj, found = find_res
                if found:
                    pos_x, pos_y = [obj['x'][0]], [obj['y'][0]]
                    saturated = False

                    cut = Cutout2D(data=arr, position=(pos_x[0], pos_y[0]), size=51)
                    n_saturated = np.count_nonzero(cut.data > satlevel[filt.upper()])
                    saturated = n_saturated > sat_npix[filt.upper()]
                    if saturated:
                        if verbose:
                            print(f"{n_saturated} pixels above satlevel {satlevel[filt.upper()]} (at {filt})"
                                  + "; Do not do any photometry. ")
                        objsum = np.nan*np.ones_like(radii)
                        varsum = np.nan*np.ones_like(radii)

                    else:
                        if verbose_bkg:
                            print(s_bkg)
                        if verbose_obj:
                            print(s_obj)
                        if verbose:
                            dx = pos_x[0] - arr.shape[1]/2
                            dy = pos_y[0] - arr.shape[0]/2
                            print(f"{fpath.name}: "  # 0-th object at
                                  + f"(x, y) = ({pos_x[0]:6.3f}, {pos_y[0]:7.3f}), "  # [0-indexing]
                                  + f"(dx, dy) = ({dx:+6.3f}, {dy:+6.3f})"  # from image center
                                  )

                        objsum, varsum, _ = sep.sum_circle(arr, var=var, x=pos_x, y=pos_y, r=self.radii)
                        if skysub:
                            ones = np.ones_like(arr)
                            aparea, _, _ = sep.sum_circle(ones, x=pos_x, y=pos_y, r=self.radii)
                            sky_an = CircularAnnulus((pos_x, pos_y), **self.skyan_kw)
                            sky = sky_fit(arr, annulus=sky_an)
                            objsum -= sky['msky']*aparea
                            varsum += (sky['ssky']*aparea)**2/sky['nsky'] + aparea*sky['ssky']**2

                else:
                    objsum = np.nan*np.ones_like(radii)
                    varsum = np.nan*np.ones_like(radii)
                    # pos_x, pos_y = [arr.shape[1]/2], [arr.shape[0]/2]
                    # s_obj += "\n Using IMAGE CENTER as a fixed object center"
                    # if verbose:
                    #     print("Object not found. Using IMAGE CENTER as a fixed object center.")

                self.phots[filt, set_id, row['POL-AGL1'], row['oe']] = dict(objsum=objsum, varsum=varsum)
                self.positions[filt, set_id, row['POL-AGL1'], row['oe']] = (pos_x[0], pos_y[0])

            # self.ratio_valu = {}
            # self.ratio_vari = {}
            # for ang in ['00.0', '45.0', '22.5', '67.5']:
            #     phot_o = self.phots[filt, set_id, ang, 'o']
            #     phot_e = self.phots[filt, set_id, ang, 'e']
            #     self.ratio_valu[ang] = phot_e['objsum']/phot_o['objsum']
            #     # sum of (err/apsum)^2 = variance/apsum^2
            #     self.ratio_vari[ang] = (phot_e['varsum']/phot_e['objsum']**2
            #                             + phot_o['varsum']/phot_o['objsum']**2)

            # self.rq_valu = np.sqrt(self.ratio_valu['00.0']/self.ratio_valu['45.0'])
            # self.ru_valu = np.sqrt(self.ratio_valu['22.5']/self.ratio_valu['67.5'])
            # self.q_valu = (self.rq_valu - 1)/(self.rq_valu + 1)
            # self.u_valu = (self.ru_valu - 1)/(self.ru_valu + 1)
            # self.q_vari = (self.rq_valu/(self.rq_valu + 1)**2)**2*(self.ratio_vari['00.0']
            #                                                        + self.ratio_vari['45.0'])
            # self.u_vari = (self.ru_valu/(self.ru_valu + 1)**2)**2*(self.ratio_vari['22.5']
            #                                                        + self.ratio_vari['67.5'])

            # pol_valu = np.sqrt(self.q_valu**2 + self.u_valu**2)
            # pol_err = np.sqrt(self.q_valu**2*self.q_vari + self.u_valu**2*self.u_vari)/pol_valu
            # th_valu = 0.5*np.rad2deg(np.arctan2(self.u_valu, self.q_valu))
            # th_err = 0.5*np.rad2deg(pol_err/pol_valu)
            self.Pol[filt, set_id] = LinPolOE4(
                i000_o=self.phots[filt, set_id, '00.0', 'o']['objsum'],
                i000_e=self.phots[filt, set_id, '00.0', 'e']['objsum'],
                i450_o=self.phots[filt, set_id, '45.0', 'o']['objsum'],
                i450_e=self.phots[filt, set_id, '45.0', 'e']['objsum'],
                i225_o=self.phots[filt, set_id, '22.5', 'o']['objsum'],
                i225_e=self.phots[filt, set_id, '22.5', 'e']['objsum'],
                i675_o=self.phots[filt, set_id, '67.5', 'o']['objsum'],
                i675_e=self.phots[filt, set_id, '67.5', 'e']['objsum'],
                di000_o=np.sqrt(self.phots[filt, set_id, '00.0', 'o']['varsum']),
                di000_e=np.sqrt(self.phots[filt, set_id, '00.0', 'e']['varsum']),
                di450_o=np.sqrt(self.phots[filt, set_id, '45.0', 'o']['varsum']),
                di450_e=np.sqrt(self.phots[filt, set_id, '45.0', 'e']['varsum']),
                di225_o=np.sqrt(self.phots[filt, set_id, '22.5', 'o']['varsum']),
                di225_e=np.sqrt(self.phots[filt, set_id, '22.5', 'e']['varsum']),
                di675_o=np.sqrt(self.phots[filt, set_id, '67.5', 'o']['varsum']),
                di675_e=np.sqrt(self.phots[filt, set_id, '67.5', 'e']['varsum'])
            )

            self.Pol[filt, set_id].calc_pol(
                p_eff=self.p_eff[filt.upper()], dp_eff=self.dp_eff[filt.upper()],
                theta_inst=self.theta_inst[filt.upper()], dtheta_inst=self.dtheta_inst[filt.upper()],
                pa_inst=np.mean(df['PA']),
                rot_instq=np.mean(df['INSROT'][:4]), rot_instu=np.mean(df['INSROT'][-4:]),
                q_inst=0, u_inst=0, dq_inst=0, du_inst=0,
                degree=True, percent=True
            )

            self.pols[filt, set_id] = dict(pol=self.Pol[filt, set_id].pol,
                                           dpol=self.Pol[filt, set_id].dpol,
                                           theta=self.Pol[filt, set_id].theta,
                                           dtheta=self.Pol[filt, set_id].dtheta)

        self.phots = (pd.DataFrame.from_dict(self.phots)).T  # ~ few ms for tens of sets
        self.pols = (pd.DataFrame.from_dict(self.pols)).T    # ~ few ms for tens of sets

        if savefig:
            from matplotlib import pyplot as plt
            from matplotlib import rcParams
            from mpl_toolkits.axes_grid1 import ImageGrid

            # We need to do it in a separate cell. See:
            # https://github.com/jupyter/notebook/issues/3385
            plt.style.use('default')
            rcParams.update({'font.size': 12})

            zs_v = {}
            mm_v = {}
            for filt in 'jhk':
                avgarr = 0
                fpaths = self.parsed[self.parsed['filt'] == filt]['file']
                for fpath in fpaths:
                    avgarr += fitsio.FITS(fpath)[0].read()
                avgarr /= len(fpaths)
                zs = ImageNormalize(avgarr, interval=ZScaleInterval())
                mm_v[filt] = dict(vmin=avgarr.min(), vmax=avgarr.max())
                zs_v[filt] = dict(vmin=zs.vmin, vmax=zs.vmax)

            imgrid_kw = dict(
                nrows_ncols=(1, 8),
                axes_pad=(0.15, 0.05),
                label_mode="1",  # "L"
                share_all=True,
                cbar_location="bottom",
                cbar_mode="single",  # "each",
                cbar_size="0.5%",
                cbar_pad="1.2%"
            )

            fig = plt.figure(figsize=(9, 7))
            grid_z, grid_s = None, None
            for (filt, set_id), df in self.grouped:
                if len(df) < 8:  # if not a full 4 images/set is obtained
                    continue
                del grid_z, grid_s
                # sub-gridding one of the Axes (see nrows_ncols)
                grid_z = ImageGrid(fig, '211', **imgrid_kw)
                grid_s = ImageGrid(fig, '212', **imgrid_kw)

                for i, (_, row) in enumerate(df.iterrows()):
                    fpath = row['file']
                    # Using FITSIO reduces time 0.3 s/set --> 0.1 s/set (1set = 8 FITS files).
                    arr = fitsio.FITS(fpath)[0].read()
                    pos = self.positions[filt, set_id, row['POL-AGL1'], row['oe']]

                    ap1 = CircularAperture(pos, r=self.radii.min())
                    ap_mid = CircularAperture(pos, r=(self.radii.min() + self.radii.max())/2)
                    ap2 = CircularAperture(pos, r=self.radii.max())
                    an = CircularAnnulus(pos, **self.skyan_kw)

                    ax_z = grid_z[i]
                    ax_s = grid_s[i]
                    ax_z.set_title("{}˚, {}".format(row['POL-AGL1'], row['oe']),
                                   fontsize='small')

                    im_z = ax_z.imshow(arr, origin='lower', **zs_v[filt])
                    im_s = ax_s.imshow(arr, origin='lower', **mm_v[filt],
                                       norm=ImageNormalize(stretch=SqrtStretch()))
                    for ax in [ax_z, ax_s]:
                        ap1.plot(ax, color='r', ls='--')
                        ap2.plot(ax, color='r', ls='--')
                        ap_mid.plot(ax, color='r')
                        an.plot(ax, color='w')

                grid_z.cbar_axes[0].colorbar(im_z)
                grid_s.cbar_axes[0].colorbar(im_s)
                plt.suptitle(f"{filt.upper()}, {row['OBJECT']}, Set #{set_id:03d}", fontfamily='monospace')
                # plt.tight_layout()
                plt.savefig(self.figdir/f"{filt}_{row['yyyymmdd']}_{row['OBJECT']}_{set_id:03d}.pdf",
                            bbox_inches='tight', dpi=300)
                fig.clf()

        if output_pol is not None:
            pols2save = self.pols.copy()
            for c in pols2save.columns:
                for idx in pols2save.index:
                    pols2save[c][idx] = pols2save[c][idx].tolist()

            pols2save.to_csv(output_pol, index=True)

        return self.pols


def read_pols(polpath):
    from numpy import nan
    # ^If not imported like this, ``eval('[1.1, nan, 2.3]')`` will raise an error.

    pols = pd.read_csv(polpath, header=0, index_col=[0, 1])
    for col in pols.columns:
        for idx in pols.index:
            pols[col][idx] = np.array(eval(pols[col][idx]))
    return pols


# == Below are included only for archieval purpose (for codes prior to about 2020 August.) ================= #
# ---------------------------------------------------------------------------------------------------------- #
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
            `None`.

        mask : ndarray, None, 'default', optional.
            The mask to be used. If ``'default'`` (default), NICpolpy's
            default mask (``mask/`` directory in the package) will be
            used. To turn off any masking, set `None`.


        bezel_x, bezel_y : array-like of int
            The x and y bezels, in ``[lower, upper]`` convention. If
            ``0``, no replacement happens.
        replace : int, float, nan, optional.
            The value to replace the pixel value where it should be
            masked. If `None`, the ccds will be trimmed.
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
            masked. If `None`, the ccds will be trimmed.
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
    #         masked. If `None`, nothing is replaced, but only the
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
