from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.visualization import (ImageNormalize, LinearStretch,
                                   ZScaleInterval, simple_norm)
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import t as tdist

from .ysfitsutilpy4nicpolpy import (CCDData_astype, cmt2hdr, find_extpix,
                                    find_satpix, give_stats, imslice,
                                    is_list_like, load_ccd, make_summary,
                                    slicefy)

__all__ = ["iterator",
           "HDR_KEYS", "PLANCOL_INIT",
           "MATCHER", "GROUPER",
           "OBJSECTS", "NICSECTS", "OBJSLICES",
           "NICSLICES", "NHAO_LOCATION", "NIC_CRREJ_KEYS",
           "GAIN", "RDNOISE", "FLATERR",
           "QINSTOFF", "DQINSTOFF", "UINSTOFF", "DUINSTOFF",
           "PEFF", "DPEFF", "PAOFF", "DPAOFF", "SATLEVEL", "_natural_unit_inst_params",
           "BPM_KW", "infer_filter", "split_oe", "split_quad",
           "add_maxsat",
           "_sanitize_hdr",
           "_set_fstem", "_set_fstem_proc",
           #    "parse_fpath",
           "_save", "_summary_path_parse",
           "_set_dir_iol", "_sanitize_objects", "_sanitize_fits",
           "_load_as_dict", "_find_calframe",
           "_save_or_load_summary", "_select_summary_rows",
           "summary_nic", "zscale_lims", "norm_imshow", "vrange_sigc", "colorbaring",
           "thumb_with_stat", "thumb_with_satpix",
           "plot_nightly_check_fig",
           "outliers_gesd"
           ]


def iterator(it, show_progress=True):
    if show_progress:
        try:
            from tqdm import tqdm
            toiter = tqdm(it)
        except ImportError:
            toiter = it
    else:
        toiter = it
    return toiter


# **************************************************************************************** #
#                 BASIC PARAMTERS RELATED TO NHAO NIC (INTERNAL API USE ONLY)              #
# **************************************************************************************** #
HDR_KEYS = {
    "all": [
        "FRAMEID", "COUNTER", "DATE-OBS", "EXPTIME", "UT-END", "DATA-TYP", "OBJECT",
        "FILTER", "POL-AGL1", "PA", "INSROT", "IMGROT", "WAVEPLAT",
        "SHUTTER", "AIRMASS", "ZD", "ALTITUDE", "AZIMUTH", "RA2000", "DEC2000",
        # "DITH_NUM", "DITH_NTH", "DITH_RAD",  # Dithering-related: only for imaging mode.
        # "NAXIS1", "NAXIS2",
        # "BIN-FCT1", "BIN-FCT2",  # I haven't seen any case when BINNING was done...
        "DOM-HUM", "DOM-TMP", "OUT-HUM", "OUT-TMP", "OUT-WND", "WEATHER",
        "NICTMP1", "NICTMP2", "NICTMP3", "NICTMP4", "NICTMP5", "NICHEAT", "DET-ID"
    ],
    "simple": [
        "COUNTER", "DATE-OBS", "OBJECT", "EXPTIME",
        "FILTER", "POL-AGL1", "PA", "INSROT", "IMGROT",
        "AIRMASS", "ALTITUDE", "AZIMUTH", "SETID"
    ],
    "minimal": [
        "COUNTER", "DATE-OBS", "OBJECT", "EXPTIME",
        "FILTER", "POL-AGL1", "SETID"
    ],
    "dark": [
        "FRAMEID", "DATE-OBS", "EXPTIME", "DATA-TYP", "OBJECT", "FILTER",
        "NICTMP1", "NICTMP2", "NICTMP3", "NICTMP4", "NICTMP5", "NICHEAT"
    ],
    "flat": [  # domeflat only
        "FRAMEID", "DATE-OBS", "EXPTIME", "DATA-TYP", "OBJECT",
        "FILTER", "POL-AGL1", "PA", "INSROT", "IMGROT"
    ],
    "frin": [
        "FRAMEID", "DATE-OBS", "EXPTIME", "DATA-TYP", "OBJECT",
        "FILTER", "POL-AGL1", "PA", "INSROT", "IMGROT",
        "AIRMASS", "ALTITUDE", "AZIMUTH"
    ],
    "mask": ["FILTER"],
    "mdark": ["DATE-OBS", "OBJECT", "FILTER", "EXPTIME"],
    "mflat": ["DATE-OBS", "OBJECT", "FILTER"],
    "mfrin": ["DATE-OBS", "OBJECT", "FILTER", "EXPTIME", "POL-AGL1", "OERAY"],
    "ifrin": ["DATE-OBS", "OBJECT", "FILTER", "EXPTIME", "POL-AGL1", "OERAY", "FRINCID"],
    "mmask": ["DATE-OBS", "FILTER"],
    "imask": ["DATE-OBS", "FILTER"],
    "dmask": ["DATE-OBS", "FILTER", "EXPTIME"],
}

MATCHER = dict(
    dark=dict(fullmatch={"OBJECT": "DARK"}, flags=0, querystr=None),
    flat=dict(fullmatch={"OBJECT": "FLAT"}, flags=0, querystr=None),
    frin=dict(fullmatch={"OBJECT": ".*\_[sS][kK][yY]"}, flags=0, querystr=None),
    mask=dict(fullmatch={"OBJECT": "MASK"}, flags=0, querystr=None),
)

GROUPER = dict(
    dark=["FILTER", "EXPTIME"],
    flat=["FILTER"],
    frin=["FILTER", "OBJECT", "OERAY"],
    mask=["FILTER"],
)

PLANCOL_INIT = dict(
    dark=dict(columns=["REMOVEIT"], values=[0], dtypes=[int]),
    mask=dict(columns=["REMOVEIT"], values=[0], dtypes=[int]),
    frin=dict(columns=["REMOVEIT"], values=[0], dtypes=[int]),
)
SORT_BY = ["FILTER", "COUNTER", "DATE-OBS", "POL-AGL1" , "OERAY", "OBJECT"]
SORT_MAP = {"J": 0, "H": 1, "K": 2, "o": 0, "e": 1}


def _fits2sl(fits_sect):
    pyth_slice = {}
    for k, sects in fits_sect.items():
        pyth_slice[k] = []
        for sect in sects:
            pyth_slice[k].append(slicefy(sect))
    return pyth_slice


# **************************************************************************************** #
#                             NIC SPECIFIC PARAMETERS/FUNCTIONS                            #
# **************************************************************************************** #
def OBJSECTS(right_half=False):
    if right_half:
        #                 150       430          150      430
        return dict(J=["[22:171, 306:735]", "[207:356, 306:735]"],
                    H=["[52:201, 331:760]", "[229:378, 331:760]"],
                    K=["[48:197, 341:770]", "[217:366, 341:770]"])

    else:
        #                 150       430          150      430
        return dict(J=["[534:683, 306:735]", "[719:868, 306:735]"],
                    H=["[564:713, 331:760]", "[741:890, 331:760]"],
                    K=["[560:709, 341:770]", "[729:878, 341:770]"])


def OBJSLICES(right_half=False):
    # OBJSLICES()[0/1] = o-/e-ray,
    # OBJSLICES()[0][0/1] = o-ray, axis 0/1 (y-axis/x-axis)
    return _fits2sl(OBJSECTS(right_half))


NICSECTS = dict(lower="[:, :512]", upper="[:, 513:]", left="[:512, :]", right="[513:, :]")

VERTICALSECTS = ["[:, 100:250]", "[:, 850:974]"]

# TODO: Let these be functions with default arg as "recent" so the most recent
# measurement of these are
# returned.
GAIN = dict(J=9.9, H=9.8, K=9.5)
RDNOISE = dict(J=37, H=36, K=35)
FLATERR = dict(J=0.02, H=0.02, K=0.02)

SATLEVEL = dict(J=8000, H=8000, K=8000)

# http://www.nhao.jp/~nic/nic_wiki/index.php?%E5%81%8F%E5%85%89%E8%A6%B3%E6%B8%AC%E6%80%A7%E8%83%BD
QINSTOFF = dict(J=0.00, H=0.03, K=-0.02)  # instrumental Q offset
DQINSTOFF = dict(J=0.29, H=0.52, K=0.30)
UINSTOFF = dict(J=-0.01, H=-0.03, K=-0.07)  # instrumental U offset
DUINSTOFF = dict(J=0.29, H=0.55, K=-0.31)
PEFF = dict(J=98, H=95, K=92)  # %, polarization efficiency, (P_meas)/(literature P)
DPEFF = dict(J=6, H=7, K=12)
PAOFF = dict(J=0.5, H=1.3, K=-0.7)  # position angle, Deg (PA_meas) - (literature PA)
DPAOFF = dict(J=1.3, H=3.1, K=6.3)  # Degrees

# == Nominal =========================================================== #
# QINSTOFF = dict(J=0., H=0., K=0.)  # instrumental Q offset
# DQINSTOFF = dict(J=0., H=0., K=0.)
# UINSTOFF = dict(J=0., H=0., K=0.)  # instrumental U offset
# DUINSTOFF = dict(J=0., H=0., K=0.)
# PEFF = dict(J=100, H=100, K=100)  # %, polarization efficiency, (P_meas)/(literature P)
# DPEFF = dict(J=0, H=0, K=0)
# PAOFF = dict(J=0., H=0., K=0.)  # position angle, Deg (PA_meas) - (literature PA)
# DPAOFF = dict(J=0., H=0., K=0.)  # Degrees


def _natural_unit_inst_params(filt, nocalib=False):
    if nocalib:
        p_eff = 1
        dp_eff = 0
        q_off = 0
        u_off = 0
        dq_off = 0
        du_off = 0
        pa_off = 0
        dpa_off = 0
    else:
        p_eff = PEFF[filt]/100
        dp_eff = DPEFF[filt]/100
        q_off = QINSTOFF[filt]/100
        u_off = UINSTOFF[filt]/100
        dq_off = DQINSTOFF[filt]/100
        du_off = DUINSTOFF[filt]/100
        pa_off = PAOFF[filt]*np.pi/180
        dpa_off = DPAOFF[filt]*np.pi/180
    return p_eff, dp_eff, q_off, u_off, dq_off, du_off, pa_off, dpa_off


NIC_CRREJ_KEYS = dict(
    sepmed=False,
    sigclip=4.5,
    sigfrac=5.,
    objlim=1,
    satlevel=30000,  # arbitrary large number (>> SATLEVEL)
    niter=4,
    cleantype='medmask',
    fs="median",
    # psffwhm=15,
    # psfsize=31,
    # psfbeta=4.765,
)

NICSLICES = {}
VERTICALSLICES = []

for k, sect in NICSECTS.items():
    NICSLICES[k] = slicefy(sect)

for sect in VERTICALSECTS:
    VERTICALSLICES.append(slicefy(sect))

# The default yfu.medfilt_bpm parameters, to detect bad pixels and cosmic-rays
# (e.g., in vpfv process)
# It uses standard deviation (not gain/rdnoise/snoise) by infinite 3-sig clipping.
BPM_KW = dict(
    size=5,
    med_sub_clip=[-5, 5],
    med_rat_clip=[0.5, 2],
    std_rat_clip=[-5, 5],
    logical="and",
    std_model="std",
    std_section="[50:450, 100:900]",
    sigclip_kw=dict(sigma=3., maxiters=50, std_ddof=1)
)

NHAO_LOCATION = dict(lon=134.3356, lat=35.0253, elevation=0.449)


# **************************************************************************************** #
#                             FUNCTIONS FOR NHAO NIC FITS FILES                            #
# **************************************************************************************** #
def infer_filter(ccd, filt=None, verbose=1):
    if filt is None:
        try:
            filt = ccd.header["FILTER"]
            if verbose >= 1:
                print(f"Assuming filter is '{filt}' from header.")
        except (KeyError, AttributeError):
            raise TypeError("Filter cannot be inferred from the given ccd.")
    return filt


def split_oe(
        ccd,
        filt=None,
        right_half=False,
        verbose=1,
        return_dict=False,
        update_header=True
):
    filt = infer_filter(ccd, filt=filt, verbose=verbose >= 1)
    try:
        ccd_o = imslice(ccd, trimsec=OBJSECTS(right_half)[filt][0])
        ccd_e = imslice(ccd, trimsec=OBJSECTS(right_half)[filt][1])
    except AttributeError:  # input is ndarray, not ccd
        return split_oe(CCDData(data=ccd, unit=u.adu), filt=filt, right_half=right_half,
                        verbose=verbose, return_dict=return_dict, update_header=False)
    if update_header:
        ccd_o.header["OERAY"] = ("o", "O-ray or E-ray. Either 'o' or 'e'.")
        ccd_e.header["OERAY"] = ("e", "O-ray or E-ray. Either 'o' or 'e'.")
        if right_half:
            for _c in [ccd_o, ccd_e]:
                _c.header["LTV1"] += 512

    if return_dict:
        return dict(o=ccd_o, e=ccd_e)
    else:
        return (ccd_o, ccd_e)


def split_quad(ccd):
    ''' Split the 2-D CCD to four pieces (quadrants).

    Returns [upper left, upper right, lower left, lower right]
    '''
    nx = ccd.data.shape[1]
    i_half = nx//2

    quads = [ccd.data[-i_half:, :i_half],
             ccd.data[-i_half:, -i_half:],
             ccd.data[:i_half, :i_half],
             ccd.data[:i_half, -i_half:]]
    return quads


def add_maxsat(ccd, mmaskpath, npixs=(5, 5), bezels=((20, 20), (20, 20)), verbose=1):
    """
    """
    _t = Time.now()
    filt = infer_filter(ccd, verbose=verbose >= 1)
    ccds = split_oe(ccd, filt=filt, verbose=verbose >= 1)
    if mmaskpath is not None:
        mask = load_ccd(mmaskpath, ccddata=False)
    else:
        mask = np.zeros(ccd.data.shape, dtype=np.bool)

    satlevel = SATLEVEL[filt.upper()]
    kw = dict(bezels=bezels, update_header=False, verbose=verbose)
    for _ccd, sl, oe in zip(ccds, OBJSLICES()[filt.upper()], "oe"):
        exts = find_extpix(_ccd, mask=mask[sl], npixs=npixs, **kw)
        nsat = np.count_nonzero(
            find_satpix(_ccd, mask=mask[sl], satlevel=satlevel, **kw)
        )
        for ext, mm in zip(exts, ["min", "max"]):
            if ext is not None:
                for i, extval in enumerate(ext):
                    ccd.header.set(f"{mm.upper()}V{i+1:03d}{oe}",
                                   extval,
                                   f"{mm} pixel value in {oe}-ray region")
        ccd.header[f"NSATPIX{oe}"] = (nsat, f"No. of saturated pixels in {oe}-ray region")
        ccd.header["SATLEVEL"] = (satlevel, "Saturation: pixels >= this value")

    bezstr = "" if bezels is None else f" and bezel: {bezels} in xyz order"
    cmt2hdr(ccd.header, 'h', verbose=verbose >= 1, time_fmt=None,
            s=(f"Extrema pixel values found N(smallest, largest) = {npixs} excluding "
                + f"mask ({mmaskpath}){bezstr}. See MINViii[OE] and MAXViii[OE]."))
    cmt2hdr(ccd.header, 'h', verbose=verbose >= 1, t_ref=_t,
            s=(f"Saturated pixels found based on satlevel = {satlevel}, excluding "
               + f"mask ({mmaskpath}){bezstr}. See NSATPIX and SATLEVEL."))

    return ccd


# **************************************************************************************** #
#                               HEADER, FILENAME, SANITIZERS                               #
# **************************************************************************************** #
def _sanitize_hdr(ccd):
    # NOTE: It overwrites the existing comments of each header keyword.
    hdr = ccd.header

    # == Set counter ===================================================================== #
    try:
        counter = hdr["COUNTER"]
        yyyymmdd = hdr["YYYYMMDD"]
    except (ValueError, KeyError, TypeError):
        frameid = hdr['FRAMEID']
        try:
            filtyymmdd, counter = frameid.split('_')
            # This yyyymmdd is neither JST (Japan), UT, nor TELINFO.
            yyyymmdd = '20' + filtyymmdd[1:]
            # yyyymmdd = hdr["DATE_LT"].replace("-", "")
            # try:
            #     # Start of exposure, if exists
            #     yyyymmdd = Time(hdr['DATE-OBS'], format='isot').strftime('%Y%m%d')
            # except KeyError:
            #     # if only the FITS creation date (after the exposure) is present
            #     # Example: 2018 Flat data
            #     yyyymmdd = Time(hdr['TELINFO'], format='iso').strftime('%Y%m%d')
        except ValueError:
            # test images has, e.g., frameid = 'h'
            #    -> ValueError not enough values to unpack)
            yyyymmdd = ''
            counter = 9999
        hdr["YYYYMMDD"] = (yyyymmdd, "Date from FRAMEID")
        hdr['COUNTER'] = (counter, "Image counter of the day, 1-indexing; 9999=TEST")

    # if "COUNTER" not in hdr:
    #     try:
    #         counter = fpath.stem.split("_")[1]
    #         counter = counter.split(".")[0]
    #         # e.g., hYYMMDD_dddd.object.pcr.fits
    #     except IndexError:  # e.g., test images (``h.fits```)
    #         counter = 9999
    #     hdr["COUNTER"] = (counter, "Image counter of the day, 1-indexing; 9999=TEST")

    # == Update DATE-OBS ================================================================= #
    if "DATE-OBS" in hdr:
        # "YYYY-MM-DDThh:mm:ss" has length 19
        if len(hdr["DATE-OBS"]) < 19:
            # If I add `or "T" not in hdr["DATE-OBS"]` as an additional
            # condition, ~ 1s of overhead is introduced per each FITS...!?
            # 2021-12-06 17:53:16 (KST: GMT+09:00) ysBach
            hdr["DATE-OBS"] = (hdr["DATE-OBS"] + "T" + hdr["UT-STR"],
                               "Start of exposure (UTC)")
    else:
        # Some frames, e.g., 2018-03-14, doesn't have DATE-OBS. Maybe due to a bug?
        # Although it is not a perfect remedy, let's do this:
        yyyymmdd = hdr["YYYYMMDD"]
        time_0 = Time.strptime(yyyymmdd + hdr["TEXINFO"][11:19], "%Y%m%d%H:%M:%S")
        time_2 = time_0 + 2*u.s  # execution of the readout_x_xM.sh
        time_4 = time_0 + 4*u.s  # expsoure starts (UT-STR)
        time_e = time_4 + hdr["EXPTIME"]*u.s + 2.5*u.s  # final readout (UT-END)
        cmt = "Estimated by NICpolpy"
        hdr["DATE-OBS"] = (time_2.strftime("%Y-%m-%dT%H:%M:%S"),
                           cmt + "(Start of Exposure)")
        hdr["DATE_UTC"] = (time_2.strftime("%Y-%m-%d"), cmt)
        hdr["TIME_UTC"] = (time_2.strftime("%H:%M:%S"), cmt)
        # hdr["DATE-OBS"] = (time_4.strftime("%Y-%m-%d"), cmt)
        hdr["UT-STR"] = (time_4.strftime("%H:%M:%S"), cmt)
        hdr["UT-END"] = (time_e.strftime("%H:%M:%S"), cmt)

    # == Update warning-invoking parts =================================================== #
    try:
        hdr["MJD-STR"] = float(hdr["MJD-STR"])
        hdr["MJD-END"] = float(hdr["MJD-END"])
    except KeyError:
        pass

    try:
        del hdr["MJD-OBS"]
    except KeyError:
        pass

    # # == Update DATE-OBS to the true start of exposure ================================= #
    # t_str = Time(hdr["DATE-OBS"] + "T" + hdr["UT-STR"], format="isot")
    # t_end = Time(hdr["DATE-OBS"] + "T" + hdr["UT-END"], format="isot")
    # t_mid = Time((t_str + t_end)/2, format="jd")
    # hdr["DATE-OBS"] = (t_str.isot, "ISO-8601 time, start of exposure")
    # hdr.set("DATE-MID", t_mid.isot,
    #                "ISO-8601 time, middle of exposure", after="DATE-OBS")
    # hdr.set("DATE-END", t_end.isot, "ISO-8601 time, end of exposure", after="DATE-MID")


def _set_fstem(hdr, setid=None, proc2add=None):
    '''
    Original files have
        ``<FILTER (j, h, k)><System YYMMDD>_<COUNTER:04d>.fits``
    The output fstem will be
        ``<FILTER (j, h, k)>_<System YYYYMMDD>_<COUNTER:04d>
          _<OBJECT>_<EXPTIME:.1f>_<POL-AGL1:04.1f>_<INSROT:+04.0f>-PROC-xxxx``
    '''
    # because of POL-AGL1, I cannot use yfu's renaming scheme...
    # ysBach 2020-05-15 16:22:37 (KST: GMT+09:00)
    try:
        polmode = hdr['SHUTTER'] in ['pol', 'close']
        # SHUTTER can be open|close|pol
        # we need to crop flat/pol/sky (SHUTTER=pol) and dark (SHUTTER=close)
    except (ValueError, KeyError, TypeError):  # Just in case SHUTTER has problem..??
        polmode = False

    try:
        polagl1 = f"{hdr['POL-AGL1']:04.1f}"
    except (ValueError, KeyError, TypeError):  # non-pol has no POL-AGL1 or = 'x'
        polagl1 = "xxxx"

    proc = "-PROC-" + "".join(hdr["PROCESS"].split("-")) if "PROCESS" in hdr else ""
    proc += "" if proc2add is None else f"-PROC-{''.join(proc2add.split('-'))}"

    if setid is None:
        setid = hdr["SETID"] if "SETID" in hdr else ""

    outstem = (
        f"{hdr['FILTER'].lower()}"  # h, j, k
        + f"_{hdr['YYYYMMDD']}"      # YYYY-MM-DD
        + f"_{int(hdr['COUNTER']):04d}"
        + f"_{hdr['EXPTIME']:.1f}"
        + f"_{polagl1}"
        + f"_{hdr['OBJECT']}"
        + proc
        + f"_{setid}"
    )
    # try:
    #     insrot = f"{hdr['INSROT']:+04.0f}"
    # except (ValueError, KeyError, TypeError):  # Just in case there is no INSROT
    #     insrot = "xxxx"

    # try:
    #     imgrot = f"{hdr['IMGROT']:+04.0f}"
    # except (ValueError, KeyError, TypeError):  # Just in case there is no IMGROT
    #     imgrot = "xxxx"

    # try:
    #     pa = f"{hdr['PA']:+06.1f}"
    # except (ValueError, KeyError, TypeError):  # Just in case there is no PA
    #     pa = "xxxxxx"

    # outstem = f"{outstem}_{polagl1}_{insrot}_{imgrot}_{pa}"

    return outstem, polmode


def _set_fstem_proc(fpath, ccd):
    fstem = fpath.stem.split("-PROC-")[0]
    return (
        fstem
        + "-PROC-"
        + ccd.header["PROCESS"].replace("-", "")  # "".join(ccd.header["PROCESS"].split("-"))
        + f"_{ccd.header['SETID']}_{ccd.header['OERAY']}"
    )


def _save(ccd, savedir, fstem, return_path=False):
    if savedir is None:
        return None

    outpath = Path(savedir)/f"{fstem}.fits"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    ccd.write(outpath, overwrite=True, output_verify="fix")
    if return_path:
        return outpath


# def parse_fpath(fpath, return_dict=True):
#     elements = Path(fpath).stem.split("-PROC-")[0].split("_")
#     oe = Path(fpath).stem.split("_")[-1]
#     if oe not in ['o', 'e']:
#         oe = 'oe'

#     try:
#         if elements[-2].startswith("set"):
#             setid = elements[-2][-3:]
#         else:
#             setid = None
#     except KeyError:
#         setid = None

#     obj = '_'.join(elements[3:-5])

#     if return_dict:
#         return {
#             'filt': elements[0],
#             'yyyymmdd': elements[1],
#             'counter': elements[2],
#             'OBJECT': obj,
#             'EXPTIME': elements[-5],
#             'POL-AGL1': elements[-4],
#             'INSROT': elements[-3],
#             'IMGROT': elements[-2],
#             'PA': elements[-1],
#             'setid': setid,
#             'oe': oe
#         }
#     else:
#         return elements[:3] + [obj] + elements[-6:]


def _sanitize_fits(
        fpath,
        dir_out,
        proc2add=None,
        dtype=None,
        skip_if_exists=True,
        verbose=0,
        process_title=" Basic preprocessing start ",
        assert_almost_equal=False,
        setid=None
):
    skipit = False
    fpath = Path(fpath)
    ccd_orig = load_ccd(fpath)
    _t = Time.now()

    # == Set output stem ================================================================= #
    _sanitize_hdr(ccd_orig)
    outstem, _ = _set_fstem(ccd_orig.header, setid=setid, proc2add=proc2add)

    # == Skip if conditions meet ========================================================= #
    if skip_if_exists and (dir_out/f"{outstem}.fits").exists():
        skipit = True
        return load_ccd(fpath), outstem, skipit

    if setid is not None:
        ccd_orig.header["SETID"] = (setid, "Pol mode set number of OBJECT on the night")

    # == Start preprocessing ============================================================= #
    cmt2hdr(ccd_orig.header, "h", verbose=verbose >= 1, time_fmt=None,
            s="{:=^72s}".format(process_title))

    # -- First, change the bit
    # Copy the CCDData in either case
    if dtype is not None:
        nccd = CCDData_astype(ccd_orig, dtype=dtype, copy=True)

        if ccd_orig.dtype != nccd.dtype:
            cmt2hdr(nccd.header, "h", t_ref=_t, verbose=verbose >= 1,
                    s=f"Changed dtype (BITPIX): {ccd_orig.dtype} --> {nccd.dtype}")
    else:
        nccd = ccd_orig

    # -- Then check if identical
    if assert_almost_equal:
        # It takes < ~20 ms for 1000x1000 32-bit VS 16-bit int images on MBP
        # 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB
        # (2400MHz DDR4), Radeon Pro 560X (4GB)]
        #   ysBach 2020-05-15 16:06:08 (KST: GMT+09:00)
        np.testing.assert_almost_equal(
            ccd_orig.data - nccd.data,
            np.zeros(nccd.data.shape)
        )
    return nccd, outstem, skipit


def _set_dir_iol(dir_in, dir_out, dir_log=None):
    """ Set directory name for input, output, and logs.
    """
    _dir_in = Path(dir_in)
    _dir_out = Path(dir_out)
    if _dir_in == _dir_out:
        raise ValueError(
            "input and output directories (dir_in, dir_out) must be different."
        )
    if dir_log is None:
        _dir_log = _dir_out.parent
    else:
        _dir_log = Path(dir_log)
    return _dir_in, _dir_out, _dir_log


def _sanitize_objects(
        objects,
        objects_exclude=False,
        default_objects_exclude=["DARK", "FLAT", "TEST"]
):
    """ Return object names and whether it's for exclusive or inclusive.
    """
    if objects is None:
        _objects = default_objects_exclude
        objects_exclude = True
    elif is_list_like(objects):
        _objects = [str(obj) for obj in objects]
    else:
        try:
            _objects = [str(objects)]
        except TypeError as e:
            raise e("objects must be None, str, or list of str.")
    return _objects, objects_exclude


def _load_as_dict(path, keys, verbose):
    """ Load all "path/*.fits" as a dict with keys.
    """
    try:
        path = Path(path)
        if not path.exists():
            path = None
            if verbose >= 1:
                print(f"File not found at given path ({path})")
    except TypeError:
        path = None

    if path is None:
        ccds = None
        paths = None
        if verbose >= 2:
            print("SKIPPED")
    else:
        path = Path(path)
        _summary = summary_nic(f"{path}/*.fits")
        if _summary is None:
            return path, None, None
        ccds = {}
        paths = {}
        for _, row in _summary.iterrows():
            key = tuple([row[k] for k in keys])
            if len(key) == 1:
                key = key[0]

            if ccds.get(key) is None:
                ccds[key] = load_ccd(row['file'])
                paths[key] = row['file']
            else:  # If more than 2, make as list
                if isinstance(ccds[key], list):  # If already list: append
                    ccds[key].append(load_ccd(row["file"]))
                    paths[key].append(row['file'])
                else:  # If not list yet: make as list
                    ccds[key] = [ccds[key], load_ccd(row["file"])]
                    paths[key] = [paths[key], row['file']]

        if verbose >= 2:
            print(f"loaded for combinations {keys} = {list(ccds.keys())}")

    return path, ccds, paths


def _find_calframe(calframes, calpaths, values, calname="Calibration frame", verbose=0):
    """ Find the calibration frame and path from given dicts for dict values.
    values:
      * dark: ["FILTER", "EXPTIME"]
      * flat: ["FILTER"]
      * mask: ["FILTER"]
      * fringe: ["FILTER", "OBJECT", "EXPTIME"]

    """
    if calframes is None:
        calpath = None
        calframe = None
    else:
        try:
            calframe = calframes[values]
            calpath = calpaths[values]
        except (KeyError, IndexError):
            calpath = None
            calframe = None
            if verbose >= 1:
                print(
                    f"    {calname} not found for the combination [{values}]: Skip process."
                )
    return calframe, calpath


# **************************************************************************************** #
#                                   SUMMARY FILE RELATED                                   #
# **************************************************************************************** #
def _summary_path_parse(parent, given_name, default_name):
    return Path(parent)/default_name if given_name is None else Path(given_name)


def _summary_find_setid(summary):
    """Find the SETID from the rawest summary table.
    """
    if "SETID" in summary.columns:
        return summary
    df = summary.copy()
    df.sort_values(by="FRAMEID", inplace=True)
    df["SETID"] = "X000"  # e.g., DARK frames are set to have setid="X000"
    for filt in "JHK":
        df_f = df.loc[df["FILTER"] == filt]
        for objname in df_f["OBJECT"].unique():
            if objname.upper() in ["DARK", "TEST"]:
                continue
            df_fo = df_f.loc[df_f["OBJECT"] == objname].copy()
            for agl, qu in zip([0.0, 45.0, 22.5, 67.5], ["q", "q", "u", "u"]):
                df_foa = df_fo.loc[df_fo["POL-AGL1"].isin([int(agl), agl])].copy()
                df_foa["SETID"] = 1 + np.arange(len(df_foa) - 0.1, dtype=int)
                df_foa["SETID"] = df_foa["SETID"].map(f"{qu}{{:03d}}".format)
                df.update(df_foa)
    return df


def _save_or_load_summary(
        fits_dir,
        summary_path,
        keywords=HDR_KEYS["all"],
        skip_if_exists=True,
        rm_nonpol=True,
        rm_test=True,
        add_setid=True,
        verbose=0
):
    """ Load the summary_path if exists ; make a summary at the path o.w.

    Parameters
    ----------
    fits_dir : path-like
        The directory of the FITS files

    summary_path : path-like
        The path to read the summary table if exists; otherwise, make a summary
        file.

    skip_if_exists : bool, optional.
        If `True` (default), the existing summary file is read. Otherwise, the
        summary file will be overridden.

    rm_nonpol : bool, optional.
        If `True` (default), non-polarimetric mode frames will be removed from
        the summary. Non-pol frames are determined by ``"WAVEPLAT" == "out" &&
        "DATA-TYP" != "DARK``.

    rm_test : bool, optional.
        If `True` (default), FITS files with ``OBJECT`` of ``"TEST"`` will be
        ignored.

    add_setid: bool, optional.
        Whether to add the `"SETID"` column to the summary table.
    """
    def __rm_nonpol(summary):
        try:
            nonpol_mask = ((summary["WAVEPLAT"].str.lower() == "out")
                           & (summary["DATA-TYP"] != "DARK"))
            df = summary[~nonpol_mask]
        except KeyError:
            df = summary
        return df

    def __rm_test(summary):
        try:
            test_mask = ((summary["OBJECT"].str.lower() == "test"))
            df = summary[~test_mask]
        except KeyError:
            df = summary
        return df

    def __setup_summary(summary, rm_nonpol, rm_test, add_setid):
        if rm_nonpol:
            summary = __rm_nonpol(summary)
        if rm_test:
            summary = __rm_test(summary)
        if add_setid:
            summary = _summary_find_setid(summary)
        return summary

    nsumpath = Path(summary_path)
    if skip_if_exists and nsumpath.exists():
        summary = pd.read_csv(nsumpath)
        summary = __setup_summary(summary, rm_nonpol, rm_test, add_setid)
        if verbose >= 0:
            print(f"Loading the existing summary CSV file from {nsumpath}")
    else:
        nsumpath.parent.mkdir(parents=True, exist_ok=True)
        if verbose >= 0:
            print(f"Making summary CSV file to {nsumpath}")
        summary = summary_nic(fits_dir/"*.fits", keywords=keywords, verbose=verbose >= 2)
        summary = __setup_summary(summary, rm_nonpol, rm_test, add_setid)
        summary.to_csv(nsumpath, index=False)
    return summary


def _select_summary_rows(
        summary,
        include_dark=True,
        include_flat=True,
        objects=None,
        objects_exclude=False
):
    ''' load the summary file and select rows.

    Parameters
    ----------
    objects : regex str
        Select rows where OBJECT *contains* the given regex.
    '''
    objs, _ = _sanitize_objects(objects, objects_exclude, default_objects_exclude=[])

    _mask = np.ones(len(summary), dtype=bool)
    if objs is not None:  # only corresponding ojbects
        _mask |= summary["OBJECT"].str.contains("|".join(objs), case=False)
    if include_dark:  # ALL of dark must be used
        _mask |= summary["OBJECT"].str.fullmatch("dark", case=False)
    if include_flat:  # ALL of flat must be used
        _mask |= summary["OBJECT"].str.fullmatch("flat", case=False)
    # If objs is None, summary[_mask_objt] will be nothing but the full summary.

    return summary[_mask]


def summary_nic(
        inputs,
        output=None,
        keywords=HDR_KEYS["all"],
        skip_if_exists=False,
        rm_nonpol=True,
        rm_test=True,
        add_setid=True,
        verbose=0,
        **kwargs
):
    '''Simple wrapper for ysfitsutilpy.make_summary.
    Note
    ----
    Identical to ysfitsutilpy.make_summary but with NIC-related keywords.

    Parameters
    ----------
    inputs : glob pattern, list-like of path-like, list-like of CCDData
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or list of
        files (each element must be path-like or CCDData). Although it is not a
        good idea, a mixed list of CCDData and paths to the files is also
        acceptable.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple of
        str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    verify_fix : bool, optional.
        Whether to do ``.verify('fix')`` to all FITS files to avoid
        VerifyError. It may take some time if turned on. Default is `False`.

    fname_option : str {'absolute', 'relative', 'name'}, optional
        Whether to save full absolute/relative path or only the filename.

    output : str or path-like, optional
        The directory and file name of the output summary file.

    format : str, optional
        The astropy.table.Table output format. Only works if ``pandas`` is
        `False`.

    keywords : list or str(``"*"``), optional
        The list of the keywords to extract (keywords should be in str).

    example_header : None or path-like, optional
        The path including the filename of the output summary text file. If
        specified, the header of the 0-th element of ``inputs`` will be
        extracted (if glob-pattern is given, the 0-th element is random, so be
        careful) and saved to `example_header`. Use `None` (default) to skip
        this.

    sort_by : str, optional
        The column name to sort the results. It can be any element of
        `keywords` or `'file'`, which sorts the table by the file name.
    '''
    def __rm_nonpol(summary):
        try:
            nonpol_mask = ((summary["WAVEPLAT"].str.lower() == "out")
                           & (summary["DATA-TYP"] != "DARK"))
            df = summary[~nonpol_mask]
        except KeyError:
            df = summary
        return df

    def __rm_test(summary):
        try:
            test_mask = ((summary["OBJECT"].str.lower() == "test"))
            df = summary[~test_mask]
        except KeyError:
            df = summary
        return df

    def __setup_summary(summary, rm_nonpol, rm_test, add_setid):
        if rm_nonpol:
            summary = __rm_nonpol(summary)
        if rm_test:
            summary = __rm_test(summary)
        if add_setid and "SETID" not in summary.columns:
            summary = _summary_find_setid(summary)
        by = [c for c in SORT_BY if c in summary]
        summary.sort_values(by + ["file"], key=lambda x: x.map(SORT_MAP),
                            inplace=True, ignore_index=True)

        return summary

    if output is not None:
        if skip_if_exists:
            try:
                summary = pd.read_csv(output)
                summary = __setup_summary(summary, rm_nonpol, rm_test, add_setid)
                if verbose >= 1:
                    print(f"Loading the existing summary CSV file from {output}")
                return summary
            except FileNotFoundError:
                pass

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if verbose >= 1:
            print(f"Making summary CSV file to {output}")

    summary = make_summary(
        inputs=inputs,
        keywords=keywords,
        verbose=verbose >= 1,
        **kwargs
    )
    if summary is not None:
        summary = __setup_summary(summary, rm_nonpol, rm_test, add_setid)
        summary.drop("filesize", inplace=True, axis=1)
        summary.to_csv(output, index=False)
    return summary


# **************************************************************************************** #
# **                                      PLOITTING                                     ** #
# **************************************************************************************** #
# Fuctions below are adopted from ysvisutilpy
#   https://github.com/ysBach/ysvisutilpy
def zscale_lims(data, zscale_bezels=((20, 20), (20, 20))):
    # NOTE: simple usage for NIC image - no support for stretch.
    if zscale_bezels is not None:
        zs = ImageNormalize(data[zscale_bezels[1][0]:-zscale_bezels[1][1],
                                 zscale_bezels[0][0]:-zscale_bezels[0][1]],
                            interval=ZScaleInterval())
    else:
        zs = ImageNormalize(data, interval=ZScaleInterval())
    return zs.vmin, zs.vmax


def colorbaring(fig, ax, im, fmt="%.0f", orientation='horizontal',
                formatter=FormatStrFormatter, **kwargs):
    cb = fig.colorbar(im, ax=ax, orientation=orientation,
                      format=formatter(fmt), **kwargs)
    return cb


def vrange_sigc(data, factors=3, mask=None, mask_value=None, sigma=3.0, sigma_lower=None,
                sigma_upper=None, maxiters=5, cenfunc='median', stdfunc='std', std_ddof=0,
                axis=None, grow=False, as_dict=True):
    _, med, std = sigma_clipped_stats(data, mask=mask, mask_value=mask_value, sigma=sigma,
                                      sigma_lower=sigma_lower, sigma_upper=sigma_upper,
                                      maxiters=maxiters, cenfunc=cenfunc, stdfunc=stdfunc,
                                      std_ddof=std_ddof, axis=axis, grow=grow)
    factors = np.atleast_1d(factors)
    if factors.size == 1:
        factors = np.repeat(factors, 2)
    if as_dict:
        return dict(vmin=med - factors[0]*std, vmax=med + factors[1]*std)
    return (med - factors[0]*std, med + factors[1]*std)


def znorm(image, stretch=LinearStretch(), **kwargs):
    return ImageNormalize(image, interval=ZScaleInterval(**kwargs), stretch=stretch)


def norm_imshow(
    ax,
    data,
    origin="lower",
    stretch="linear",
    power=1.0,
    asinh_a=0.1,
    min_cut=None,
    max_cut=None,
    min_percent=None,
    max_percent=None,
    percent=None,
    clip=True,
    log_a=1000,
    invalid=-1.0,
    zscale=False,
    vmin=None,
    vmax=None,
    **kwargs
):
    """Do normalization and do imshow"""
    if vmin is not None and min_cut is not None:
        warn("vmin will override min_cut.")

    if vmax is not None and max_cut is not None:
        warn("vmax will override max_cut.")

    if zscale:
        zs = ImageNormalize(data, interval=ZScaleInterval())
        min_cut = vmin = zs.vmin
        max_cut = vmax = zs.vmax

    if vmin is not None or vmax is not None:
        im = ax.imshow(data, origin=origin, vmin=vmin, vmax=vmax, **kwargs)
    else:
        im = ax.imshow(
            data,
            origin=origin,
            norm=simple_norm(
                data=data,
                stretch=stretch,
                power=power,
                asinh_a=asinh_a,
                min_cut=min_cut,
                max_cut=max_cut,
                min_percent=min_percent,
                max_percent=max_percent,
                percent=percent,
                clip=clip,
                log_a=log_a,
                invalid=invalid
            ),
            **kwargs)
    return im


def thumb_with_stat(
        fpath,
        outdir,
        dpi=72,
        percentiles=[0.1, 99.9],
        N_extrema=3,
        zscale_bezels=((20, 20), (20, 20)),
        figsize=(7, 3),
        ext="pdf",
        origin='lower',
        stretch='linear',
        power=1.0,
        asinh_a=0.1,
        min_cut=None,
        max_cut=None,
        min_percent=None,
        max_percent=None,
        percent=None,
        clip=True,
        log_a=1000,
        zscale=False,
        vmin=None,
        vmax=None,
        skip_if_exists=True,
        **kwargs
):
    fpath = Path(fpath)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir/f"{fpath.stem}.{ext}"
    if skip_if_exists and outpath.exists():
        return

    data = load_ccd(fpath, ccddata=False)  # quick load using fitsio.read
    if zscale_bezels is not None:
        _data = data[zscale_bezels[1][0]:-zscale_bezels[1][1],
                     zscale_bezels[0][0]:-zscale_bezels[0][1]]
    else:
        _data = data
    _st = give_stats(_data, percentiles=percentiles, N_extrema=N_extrema)
    del _st["slices"]

    infostr = []
    for k, v in _st.items():
        if isinstance(v, float):
            infostr.append(f"{k:<12s}: {v:.4f}")
        elif is_list_like(v):  # percentiles
            vstrs = str(v)[1:-1].replace(",", "").split()  # e.g., ["0.1", "1", "10", "100"]
            vstr = [f"{float(v):2.4f}" for v in vstrs]
            # [1:-1] to convert '0.1' to 0.1
            infostr.append(f"{k:<12s}: {vstr}")
        else:
            infostr.append(f"{k:<12s}: {v}")
    infostr = '\n'.join(infostr)

    fig, axs = plt.subplots(1, 2, figsize=figsize)
    norm_imshow(axs[0], data,
                origin=origin, stretch=stretch, power=power,
                asinh_a=asinh_a, min_cut=min_cut, max_cut=max_cut,
                min_percent=min_percent, max_percent=max_percent,
                percent=percent, clip=clip, log_a=log_a, zscale=zscale,
                vmin=vmin, vmax=vmax, **kwargs)
    axs[1].axis("off")
    axs[1].text(0, 0.1, infostr, fontsize=8, fontfamily="monospace")
    plt.suptitle(fpath.name, fontsize=8, fontfamily="monospace")

    plt.tight_layout()

    plt.savefig(outpath, dpi=dpi)
    plt.cla()
    plt.clf()
    plt.close("all")


def thumb_with_satpix(
        df,
        masks,
        outdir,
        basename,
        skip_if_exists=True,
        figsize=(5.5, 3),
        gap_value=-100,
        bezels=((20, 20), (20, 20)),
        ext="pdf",
        dpi=72,
        show_progress=True
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _df = df.copy()
    for col in ["NSATPIXO", "NSATPIXE"]:
        if col not in df:
            _df[col] = 0
    ny = OBJSLICES()["J"][0][0].stop - OBJSLICES()["J"][0][0].start
    nx = OBJSLICES()["J"][0][1].stop - OBJSLICES()["J"][0][1].start
    # ^ assuming nx, ny are the same for J, H, K

    for counter in iterator(_df["COUNTER"].unique(), show_progress=show_progress):
        df_jhk = (_df.loc[_df["COUNTER"] == counter])  # 3 rows (JHK)
        df_jhk.reset_index(drop=False, inplace=True)
        objname = df_jhk.loc[0, "OBJECT"]
        setid = df_jhk.loc[0, "SETID"]
        polagl = df_jhk.loc[0, "POL-AGL1"]
        # imgrot = df_jhk.loc[0, "IMGROT"]
        # insrot = df_jhk.loc[0, "INSROT"]
        # pa = df_jhk.loc[0, "PA"]
        outpath = outdir/(f"{basename}_{counter:04d}_{objname:s}_{setid}.{ext}")
        do_not_draw = (skip_if_exists and Path(outpath).exists())
        data = gap_value * np.ones((ny, 6*(nx + 1)))
        satu = np.zeros((ny, 6*(nx + 1)))
        nsats = []
        title = []
        for i, filt in enumerate("JHK"):
            row = df_jhk.loc[df_jhk["FILTER"] == filt].squeeze()
            filt = row["FILTER"]
            if len(row) == 0:
                continue
            img = load_ccd(row["file"], ccddata=False)
            for j, (sl, oe) in enumerate(zip(OBJSLICES()[filt], "oe")):
                img_oe = img[sl]
                satpix_oe = find_satpix(
                    img_oe,
                    mask=None if masks is None else masks.get(filt)[sl],
                    satlevel=SATLEVEL[filt],
                    bezels=bezels
                )
                nsatpix_oe = np .sum(satpix_oe)
                zmin, zmax = zscale_lims(img_oe, zscale_bezels=bezels)
                i_beg = (2*i + j)*(nx + 1)
                i_end = (2*i + j + 1)*(nx + 1) - 1
                data[:, i_beg:i_end] = (img_oe - zmin)/(zmax - zmin)
                satu[:, i_beg:i_end] = satpix_oe
                nsats.append(nsatpix_oe)
                title.append(f"{row['FILTER']}{oe}")
                _df.loc[row["index"], f"NSATPIX{oe.upper()}"] = nsatpix_oe
            # ^ This is thumbnail, not science image, so normalize for better visibility

        if do_not_draw:
            continue
        satupos = np.where(satu)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(data, origin="lower", vmin=0, vmax=1)
        ax.plot(satupos[1], satupos[0], "r+")
        ax.set_title(
            f"{int(counter):04d} ({objname:s} {setid:s} {polagl:.1f}˚)\n"
            + "Each img ZSCALE'd. Order: {}\n".format(", ".join(title))
            + f"# of saturated pix excluding MMASK: {nsats}",
            fontsize=8,
            fontfamily="monospace"
        )
        plt.savefig(outpath, dpi=dpi)
        plt.cla()
        plt.clf()
        plt.close("all")
    return _df


def plot_nightly_check_fig(summary, figtitle, output):
    _summary = summary.copy()
    try:
        _summary = summary[summary["OERAY"] == "o"]
        # o/e share identical values which are used below.
    except KeyError:
        pass

    try:
        _summary = summary[summary["FILTER"] == "J"]
    except KeyError:
        pass

    t_str = Time((_summary["DATE-OBS"]).tolist(), format='isot')
    fig, axs = plt.subplots(5, 1, figsize=(12, 9), sharex=True, gridspec_kw=None)
    plotkw = dict(ms=4, ls='')
    markers = ['|', 'x', '1', '2', '3']
    tdate = t_str.plot_date

    for m, key in zip(markers, ["DOM-TMP", "OUT-TMP"]):
        axs[0].plot_date(tdate, _summary[key], marker=m, label=key, **plotkw)
    axs[0].set(ylabel="Temperatures [˚C]")
    axs[0].legend(framealpha=0, fontsize=10)

    for m, key in zip(markers, ["DOM-HUM", "OUT-HUM"]):
        axs[1].plot_date(tdate, _summary[key], marker=m, label=key, **plotkw)
    axs[1].set(ylabel="Humidities [%]", ylim=(0, 100))
    axs[1].legend(framealpha=0, fontsize=10)

    for m, key in zip(markers, ["OUT-WND"]):
        axs[2].plot_date(tdate, _summary[key], marker=m, label=key, **plotkw)
    axs[2].set(ylabel="Wind [m/s]", ylim=(0, axs[2].get_ylim()[1]))
    axs[2].legend(framealpha=0, fontsize=10, ncol=5)

    for num in range(5):
        axs[3].plot_date(tdate, _summary[f"NICTMP{num + 1}"],
                         marker=markers[num], label=f"NICTMP{num + 1}", **plotkw)
    axs[3].set(ylabel="Temperatures [K]")
    axs[3].legend(framealpha=0, fontsize=10, ncol=5)

    for m, key in zip(markers, ["WEATHER", "NICHEAT"]):
        axs[4].plot_date(tdate, _summary[key], marker=m, label=key, **plotkw)
    axs[4].set(
        ylabel="",
        ylim=(max(-0.1, axs[4].get_ylim()[0]),
              min(1.1, axs[4].get_ylim()[1]))
    )
    axs[4].legend(framealpha=0, fontsize=10, loc=2)

    ax_id = axs[4].twinx()
    ax_id.plot_date(tdate, _summary["FRAMEID"].str.split("_").str[1].astype(int),
                    marker='+', color='r', label="FRAMEID", **plotkw)
    ax_id.legend(framealpha=0, fontsize=10, loc=4)
    ax_id.grid(axis='y')
    for ax, lab, color in zip([axs[4], ax_id], ['WEATHER/NICHEAT', 'FRAMEID'], ['k', 'r']):
        ax.set_ylabel(lab, color=color)
        ax.tick_params(axis='y', color=color, labelcolor=color)

    axs[-1].set(xlabel="Time of start of exposure (header UT-STR)")
    plt.suptitle(figtitle)

    plt.tight_layout()
    fig.align_ylabels(axs)
    fig.align_xlabels(axs)
    plt.savefig(output)
    return fig, axs


def outliers_gesd(
        x: list | np.ndarray,
        outliers: int = 5,
        hypo: bool = False,
        report: bool = False,
        alpha: float = 0.05) -> np.ndarray:
    """ Directly from https://github.com/maximtrp/scikit-posthocs/pull/56
    The generalized (Extreme Studentized Deviate) ESD test is used
    to detect one or more outliers in a univariate data set that follows
    an approximately normal distribution [1]_.

    Parameters
    ----------
    x : Union[List, np.ndarray]
        An array, any object exposing the array interface, containing
        data to test for outliers.
    outliers : int = 5
        Number of potential outliers to test for. Test is two-tailed, i.e.
        maximum and minimum values are checked for potential outliers.
    hypo : bool = False
        Specifies whether to return a bool value of a hypothesis test result.
        Returns True when we can reject the null hypothesis. Otherwise, False.
        Available options are:
        1) True - return a hypothesis test result.
        2) False - return a filtered array without an outlier (default).
    report : bool = False
        Specifies whether to print a summary table of the test.
    alpha : float = 0.05
        Significance level for a hypothesis test.

    Returns
    -------
    np.ndarray
        Returns the filtered array if alternative hypo is True, otherwise an
        unfiltered (input) array.

    Notes
    -----
    .. [1] Rosner, Bernard (May 1983), Percentage Points for a Generalized
        ESD Many-Outlier Procedure,Technometrics, 25(2), pp. 165-172.

    Examples
    --------
    >>> data = np.array([-0.25, 0.68, 0.94, 1.15, 1.2, 1.26, 1.26, 1.34,
        1.38, 1.43, 1.49, 1.49, 1.55, 1.56, 1.58, 1.65, 1.69, 1.7, 1.76,
        1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.1, 2.14, 2.15,
        2.23, 2.24, 2.26, 2.35, 2.37, 2.4, 2.47, 2.54, 2.62, 2.64, 2.9,
        2.92, 2.92, 2.93, 3.21, 3.26, 3.3, 3.59, 3.68, 4.3, 4.64, 5.34,
        5.42, 6.01])
    >>> outliers_gesd(data, 5)
    array([-0.25,  0.68,  0.94,  1.15,  1.2 ,  1.26,  1.26,  1.34,  1.38,
            1.43,  1.49,  1.49,  1.55,  1.56,  1.58,  1.65,  1.69,  1.7 ,
            1.76,  1.77,  1.81,  1.91,  1.94,  1.96,  1.99,  2.06,  2.09,
            2.1 ,  2.14,  2.15,  2.23,  2.24,  2.26,  2.35,  2.37,  2.4 ,
            2.47,  2.54,  2.62,  2.64,  2.9 ,  2.92,  2.92,  2.93,  3.21,
            3.26,  3.3 ,  3.59,  3.68,  4.3 ,  4.64])
    >>> outliers_gesd(data, outliers = 5, report = True)
    H0: no outliers in the data
    Ha: up to 5 outliers in the data
    Significance level:  α = 0.05
    Reject H0 if Ri > Critical Value (λi)
    Summary Table for Two-Tailed Test
    ---------------------------------------
          Exact           Test     Critical
      Number of      Statistic    Value, λi
    Outliers, i      Value, Ri          5 %
    ---------------------------------------
              1          3.119        3.159
              2          2.943        3.151
              3          3.179        3.144 *
              4           2.81        3.136
              5          2.816        3.128
    """
    rs, ls = np.zeros(outliers, dtype=float), np.zeros(outliers, dtype=float)
    ms = []

    data_proc = np.copy(x)
    argsort_index = np.argsort(data_proc)
    data = data_proc[argsort_index]
    n = data_proc.size

    # Lambda values (critical values): do not depend on the outliers.
    nol = np.arange(outliers)  # the number of outliers
    df = n - nol - 2  # degrees of freedom
    t_ppr = tdist.ppf(1 - alpha / (2 * (n - nol)), df)
    ls = ((n - nol - 1) * t_ppr) / np.sqrt((df + t_ppr**2) * (n - nol))

    for i in np.arange(outliers):

        abs_d = np.abs(data_proc - np.mean(data_proc))

        # R-value calculation
        R = np.max(abs_d) / np.std(data_proc, ddof=1)
        rs[i] = R

        # Masked values
        lms = ms[-1] if len(ms) > 0 else []
        ms.append(
            lms + np.where(data == data_proc[np.argmax(abs_d)])[0].tolist())

        # Remove the observation that maximizes |xi - xmean|
        data_proc = np.delete(data_proc, np.argmax(abs_d))

    if report:

        report = ["H0: no outliers in the data",
                  "Ha: up to " + str(outliers) + " outliers in the data",
                  "Significance level:  α = " + str(alpha),
                  "Reject H0 if Ri > Critical Value (λi)", "",
                  "Summary Table for Two-Tailed Test",
                  "---------------------------------------",
                  "      Exact           Test     Critical",
                  "  Number of      Statistic    Value, λi",
                  "Outliers, i      Value, Ri      {:5.3g} %".format(100*alpha),
                  "---------------------------------------"]

        for i, (r, l) in enumerate(zip(rs, ls)):
            report.append('{: >11s}'.format(str(i+1)) +
                          '{: >15s}'.format(str(np.round(r, 3))) +
                          '{: >13s}'.format(str(np.round(l, 3))) +
                          (" *" if r > l else ""))

        print("\n".join(report))

    # Remove masked values
    # for which the test statistic is greater
    # than the critical value and return the result

    if any(rs > ls):
        if hypo:
            data[:] = False
            data[ms[np.max(np.where(rs > ls))]] = True
            # rearrange data so mask is in same order as incoming data
            data = np.vstack((data, np.arange(0, data.shape[0])[argsort_index]))
            data = data[0, data.argsort()[1, ]]
            data = data.astype('bool')
        else:
            data = np.delete(data, ms[np.max(np.where(rs > ls))])

    return data