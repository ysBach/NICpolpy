from pathlib import Path

import numpy as np
import pandas as pd
from ysfitsutilpy import (LACOSMIC_KEYS, fitsxy2py, is_list_like, load_ccd,
                          make_summary, trim_ccd)

__all__ = ["iterator",
           "USEFUL_KEYS", "OBJSECTS", "NICSECTS", "OBJSLICES", "NICSLICES",
           "NHAO_LOCATION", "NIC_CRREJ_KEYS", "GAIN", "RDNOISE",
           "infer_filter", "split_oe", "split_quad",
           "_set_fstem", "parse_fpath",
           "_set_dir_iol", "_sanitize_objects", "_save_or_load_summary",
           "_load_as_dict", "_find_calframe"
           ]


USEFUL_KEYS = [
    "FRAMEID", "DATE-OBS", "UT-STR", "EXPTIME", "UT-END", "DATA-TYP", "OBJECT",
    "FILTER", "POL-AGL1", "PA", "INSROT", "IMGROT", "WAVEPLAT",
    "SHUTTER", "AIRMASS", "ZD", "ALTITUDE", "AZIMUTH",
    "DITH_NUM", "DITH_NTH", "DITH_RAD",
    "NAXIS1", "NAXIS2", "BIN-FCT1", "BIN-FCT2", "RA2000", "DEC2000",
    "DOM-HUM", "DOM-TMP", "OUT-HUM", "OUT-TMP", "OUT-WND", "WEATHER",
    "NICTMP1", "NICTMP2", "NICTMP3", "NICTMP4", "NICTMP5", "NICHEAT", "DET-ID"
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


def OBJSECTS(right_half=False):
    if right_half:
        #                 150       440          150      440
        return dict(J=["[28:178, 300:740]", "[213:363, 300:740]"],
                    H=["[53:203, 325:765]", "[233:383, 325:765]"],
                    K=["[48:198, 335:775]", "[218:368, 335:775]"])

    else:
        #                 150       420          150      440
        return dict(J=["[540:690, 300:740]", "[725:875, 300:740]"],
                    H=["[565:715, 325:765]", "[745:895, 325:765]"],
                    K=["[560:710, 335:775]", "[730:880, 335:775]"])


NICSECTS = dict(lower="[:, :512]", upper="[:, 513:]", left="[:512, :]", right="[513:, :]")

VERTICALSECTS = ["[:, 100:250]", "[:, 850:974]"]

GAIN = dict(J=9.2, H=9.8, K=9.4)
RDNOISE = dict(J=50, H=75, K=83)

NIC_CRREJ_KEYS = LACOSMIC_KEYS.copy()
NIC_CRREJ_KEYS["sepmed"] = True
NIC_CRREJ_KEYS['satlevel'] = np.inf
NIC_CRREJ_KEYS['objlim'] = 5
NIC_CRREJ_KEYS['sigfrac'] = 5
NIC_CRREJ_KEYS['cleantype'] = 'median'


def _fits2sl(fits_sect):
    pyth_slice = {}
    for k, sects in fits_sect.items():
        pyth_slice[k] = []
        for sect in sects:
            pyth_slice[k].append(fitsxy2py(sect))
    return pyth_slice


def OBJSLICES(right_half=False):
    return _fits2sl(OBJSECTS(right_half))


NICSLICES = {}
VERTICALSLICES = []

for k, sect in NICSECTS.items():
    NICSLICES[k] = fitsxy2py(sect)

for sect in VERTICALSECTS:
    VERTICALSLICES.append(fitsxy2py(sect))


NHAO_LOCATION = dict(lon=134.3356, lat=35.0253, elevation=0.449)


def infer_filter(ccd, filt=None, verbose=True):
    if filt is None:
        try:
            filt = ccd.header["FILTER"]
            if verbose:
                print(f"Assuming filter is '{filt}' from header.")
        except (KeyError, AttributeError):
            raise TypeError("Filter cannot be inferred from the given ccd.")
    return filt


def split_oe(ccd, filt=None, right_half=False, verbose=True):
    filt = infer_filter(ccd, filt=filt, verbose=verbose)
    ccd_o = trim_ccd(ccd, fits_section=OBJSECTS(right_half)[filt][0])
    ccd_o.header["OERAY"] = ("o", "O-ray or E-ray. Either 'o' or 'e'.")
    ccd_e = trim_ccd(ccd, fits_section=OBJSECTS(right_half)[filt][1])
    ccd_e.header["OERAY"] = ("e", "O-ray or E-ray. Either 'o' or 'e'.")
    if right_half:
        for _c in [ccd_o, ccd_e]:
            _c.header["LTV1"] += 512
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


def _set_fstem(hdr):
    '''
    Original files have
        ``<FILTER (j, h, k)><System YYMMDD>_<COUNTER:04d>.fits``
    The output fstem will be
        ``<FILTER (j, h, k)>_<System YYYYMMDD>_<COUNTER:04d>
          _<OBJECT>_<EXPTIME:.1f>_<POL-AGL1:04.1f>_<INSROT:+04.0f>``
    '''
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
    except ValueError:  # test images has, e.g., frameid = 'h' --> ValueError not enough values to unpack)
        yyyymmdd = ''
        counter = 9999

    hdr['COUNTER'] = (counter, "Image counter of the day, 1-indexing; 9999=TEST")

    outstem = (
        f"{hdr['FILTER'].lower()}"  # h, j, k
        + f"_{yyyymmdd}"      # YYYY-MM-DD
        + f"_{int(counter):04d}"
        + f"_{hdr['OBJECT']}"
        + f"_{hdr['EXPTIME']:.1f}"
    )

    # because of POL-AGL1, I cannot use yfu's renaming scheme...
    # ysBach 2020-05-15 16:22:37 (KST: GMT+09:00)
    try:
        polmode = hdr['SHUTTER'] in ['pol', 'close']
        # SHUTTER can be open|close|pol
        # we need to crop flat/pol/sky (SHUTTER=pol) and dark (SHUTTER=close)
    except (ValueError, KeyError, TypeError):  # Just in case SHUTTER has problem..??
        polmode = False

    try:
        outstem += f"_{hdr['POL-AGL1']:04.1f}"
    except (ValueError, KeyError, TypeError):  # non-pol has no POL-AGL1 or = 'x'
        outstem += "_xxxx"

    try:
        outstem += f"_{hdr['INSROT']:+04.0f}"
    except (ValueError, KeyError, TypeError):  # Just in case there is no INSROT
        outstem += "_xxxx"

    try:
        outstem += f"_{hdr['IMGROT']:+04.0f}"
    except (ValueError, KeyError, TypeError):  # Just in case there is no IMGROT
        outstem += "_xxxx"

    try:
        outstem += f"_{hdr['PA']:+06.1f}"
    except (ValueError, KeyError, TypeError):  # Just in case there is no PA
        outstem += "_xxxxxx"

    return outstem, polmode


def parse_fpath(fpath, return_dict=True):
    elements = Path(fpath).stem.split("-PROC-")[0].split("_")
    if elements[-1] not in ['o', 'e']:
        elements.append('oe')

    obj = '_'.join(elements[3:-6])

    if return_dict:
        return {
            'filt': elements[0],
            'yyyymmdd': elements[1],
            'counter': elements[2],
            'OBJECT': obj,
            'EXPTIME': elements[-6],
            'POL-AGL1': elements[-5],
            'INSROT': elements[-4],
            'IMGROT': elements[-3],
            'PA': elements[-2],
            'oe': elements[-1],
        }
    else:
        return elements[:3] + [obj] + elements[-6:]


def _set_dir_iol(dir_in, dir_out, dir_log=None):
    """ Set directory name for input, output, and logs.
    """
    dir_in = Path(dir_in)
    dir_out = Path(dir_out)
    if dir_log is None:
        dir_log = dir_out.parent
    else:
        dir_log = Path(dir_log)
    return dir_in, dir_out, dir_log


def _sanitize_objects(objects, objects_exclude=False):
    """ Return object names and whether it's for exclusive or inclusive.
    """
    if objects is None:
        objects = ["DARK", "FLAT", "TEST"]
        objects_exclude = True
    elif is_list_like(objects):
        objects = [str(obj) for obj in objects]
    else:
        try:
            objects = [str(objects)]
        except TypeError:
            raise TypeError("objects must be None, str, or list of str.")
    return objects, objects_exclude


def _save_or_load_summary(fits_dir, summary_path, keywords=USEFUL_KEYS,
                          skip_if_exists=True, ignore_nonpol=True, verbose=0):
    """ Load the summary_path if exists ; make a summary at the path o.w.
    """
    def __rm_nonpol(summary):
        nonpol_mask = ((summary["WAVEPLAT"] == "out") & (summary["DATA-TYP"] != "DARK"))
        summary = summary[~nonpol_mask]
        return summary

    summary_path = Path(summary_path)
    if skip_if_exists and summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = __rm_nonpol(summary) if ignore_nonpol else summary
        if verbose >= 0:
            print(f"Loading the existing summary CSV file from {summary_path}")
    else:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose >= 0:
            print(f"Making summary CSV file to {summary_path}")
        summary = summary_nic(fits_dir/"*.fits", keywords=keywords, verbose=verbose >= 2)
        summary = __rm_nonpol(summary) if ignore_nonpol else summary
        summary.to_csv(summary_path, index=False)
    return summary


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
            ccds[key] = load_ccd(row['file'])
            paths[key] = row['file']
        if verbose >= 2:
            print(f"loaded for combinations {keys} = {list(ccds.keys())}")
    return path, ccds, paths


def _find_calframe(calframes, calpaths, values, calname, verbose=False):
    """ Find the calibration frame and path from given dicts for dict values.
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
            if verbose:
                print(f"{calname} not found for the combination [{values}]: Skip process.")
    return calframe, calpath


def summary_nic(inputs, output=None, keywords=USEFUL_KEYS, pandas=True, verbose=False, **kwargs):
    '''Simple wrapper for ysfitsutilpy.make_summary.
    Note
    ----
    Identical to ysfitsutilpy.make_summary but with (1) NIC-related keywords, (2) default pandas=True,
    and (3) default verbose=False.

    Parameters
    ----------
    inputs : glob pattern, list-like of path-like, list-like of CCDData
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or list of files (each element must
        be path-like or CCDData). Although it is not a good idea, a mixed list of CCDData and paths to the
        files is also acceptable.

    extension: int, str, (str, int)
        The extension of FITS to be used. It can be given as integer (0-indexing) of the extension,
        ``EXTNAME`` (single str), or a tuple of str and int: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used.

    verify_fix : bool, optional.
        Whether to do ``.verify('fix')`` to all FITS files to avoid VerifyError. It may take some time
        if turned on. Default is `False`.

    fname_option : str {'absolute', 'relative', 'name'}, optional
        Whether to save full absolute/relative path or only the filename.

    output : str or path-like, optional
        The directory and file name of the output summary file.

    format : str, optional
        The astropy.table.Table output format. Only works if ``pandas`` is `False`.

    keywords : list or str(``"*"``), optional
        The list of the keywords to extract (keywords should be in str).

    example_header : None or path-like, optional
        The path including the filename of the output summary text file. If specified, the header of
        the 0-th element of ``inputs`` will be extracted (if glob-pattern is given, the 0-th element is
        random, so be careful) and saved to ``example_header``. Use `None` (default) to skip this.

    pandas : bool, optional
        Whether to return pandas. If `False`, astropy table object is returned. It will save csv
        format regardless of ``format``.

    sort_by : str, optional
        The column name to sort the results. It can be any element of ``keywords`` or ``'file'``, which
        sorts the table by the file name.
    '''
    return make_summary(inputs=inputs, output=output, keywords=keywords,
                        pandas=pandas, verbose=verbose, **kwargs)
