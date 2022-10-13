import shutil
from dataclasses import dataclass, field
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats

from .ysfitsutilpy4nicpolpy import (LACOSMIC_CRREJ, ccdred, crrej, df_selector,
                                    fixpix, inputs2list, listify, load_ccd,
                                    make_reduc_planner)

from .prepare import _do_16bit_vertical, _do_vfv, make_darkmask
from .preproc import fringe_scale_mask, make_cal
from .util import (BPM_KW, GAIN, GROUPER, HDR_KEYS, MATCHER, NIC_CRREJ_KEYS,
                   PLANCOL_INIT, RDNOISE, SORT_MAP, _load_as_dict, _save,
                   _set_fstem_proc, iterator, split_oe, summary_nic,
                   thumb_with_satpix, thumb_with_stat)

__all__ = ["NICPolReduc"]


def exists(path):
    if path is None:
        return False
    return Path(path).exists()


def do_crrej(ccd, row, default=LACOSMIC_CRREJ, verbose=False):
    return crrej(
        ccd,  # NOTE: Already FIXPIX'ed, so no need to use mask again here.
        gain=row.get("gain", default.get("gain")),
        rdnoise=row.get("rdnoise", default.get("rdnoise")),
        sigclip=row.get("sigclip", default.get("sigclip")),
        sigfrac=row.get("sigfrac", default.get("sigfrac")),
        objlim=row.get("objlim", default.get("objlim")),
        satlevel=row.get("satlevel", default.get("satlevel")),
        niter=row.get("niter", default.get("niter")),
        sepmed=row.get("sepmed", default.get("sepmed")),
        cleantype=row.get("cleantype", default.get("cleantype")),
        fs=row.get("fs", default.get("fs")),
        psffwhm=row.get("psffwhm", default.get("psffwhm")),
        psfsize=row.get("psfsize", default.get("psfsize")),
        psfbeta=row.get("psfbeta", default.get("psfbeta")),
        verbose=verbose,
    )


def sigma_clipped_med(data, **kwargs):
    return sigma_clipped_stats(data, **kwargs)[1]


def _save_thumb_with_stat(df, outdir, verbose=0, show_progress=True, **kwargs):
    if verbose >= 1:
        print(f"Saving thumbnails to {outdir}")
    for _, row in iterator(df.iterrows(), show_progress=show_progress):
        thumb_with_stat(
            Path(row["file"]), outdir=outdir, skip_if_exists=False, **kwargs
        )


def _save_thumb_with_satpix(
    df, masks, outdir, basename, verbose=0, show_progress=True, **kwargs
):
    if verbose >= 1:
        print(f"Saving thumbnails to {outdir}")
    return thumb_with_satpix(
        df,
        masks=masks,
        outdir=outdir,
        basename=basename,
        show_progress=show_progress,
        skip_if_exists=False,
        **kwargs,
    )


def dict_get(ccddict, pathdict, row, keys):
    keys = listify(keys)
    keys = keys[0] if len(keys) == 1 else tuple(keys)
    ccd = None if ccddict is None else ccddict.get(row[keys])
    path = None if pathdict is None else pathdict.get(row[keys])
    return ccd, path


def _get_plan_frm(df, col, split_oe_ccd=False, pix_min_value=None, pix_replace_value=1):
    if df is None or col not in df:
        return {}

    ccds = {}
    for fpath in df[col].unique():
        if isinstance(fpath, str):
            ccd = load_ccd(fpath)
            if pix_min_value is not None:
                ccd.data[ccd.data < pix_min_value] = pix_replace_value
            if split_oe_ccd:
                ccds[fpath] = split_oe(ccd, return_dict=True, verbose=False)
            else:
                ccds[fpath] = ccd
        else:
            ccds[fpath] = None

    return ccds


class NICPolReducMixin:
    def _read_plan(self, name, removeit=True):
        fpath = self.planpaths.get(name)
        if fpath is None or not exists(fpath):
            if self.verbose >= 1:
                print("SKIP: Planner does not exist for {}".format(name))
            return None
        else:
            plan = pd.read_csv(fpath)
            if removeit and "REMOVEIT" in plan:
                plan = plan.loc[plan["REMOVEIT"] == 0].copy()
            return plan

    def _read_summ(self, name):
        fpath = self.summpaths.get(name)
        if fpath is None:
            return None
        else:
            return pd.read_csv(fpath) if fpath.exists() else None

    def _check_skip(self, planred, name):
        """
        skipex[red/plan]
          skipex = "skip if exists"
          red    = reduced data summary (`_summ_*.csv` files)
          plan   = reduction plan files (`_plan_*.csv` files)
        """
        # Update the attributes
        self._skipexred = {k: exists(v) for k, v in self.summpaths.items()}
        self._skipexplan = {k: exists(v) for k, v in self.planpaths.items()}
        _name = name if isinstance(name, str) else f"lv{name}"

        # Check if the step is skipped
        if planred.lower().startswith("r"):
            if self._skipexred[name]:
                if self.verbose >= 1:
                    print(f"SKIP: Summary exists for {_name} at {self.summpaths[name]}")
                return self._read_summ(name)

        elif planred.lower().startswith("p"):
            if self._skipexplan[name]:
                if self.verbose >= 1:
                    print(f"SKIP: Planer exists for {_name} at {self.planpaths[name]}")
                return self._read_plan(name)

    # def _make_cal_from_plan(self, name, fmts, combine_kw, **kwargs):
    #     plan = self._read_plan(name)
    #     _ = make_cal(
    #         framename=name,
    #         plan_in=plan,
    #         dir_out=self.dirs[name],
    #         group_key=GROUPER[name[-4:]],
    #         fmts=fmts,
    #         combine_kw=combine_kw,
    #         verbose=self.verbose,
    #         **kwargs
    #     )
    #     return plan

    def _make_reduc_plan(
        self, plan, add_crrej=True, add=None,
    ):
        names = dict(
            mdark="DARKFRM", mflat="FLATFRM", mfrin="FRINFRM", mmask="MASKFILE"
        )
        cal_summary = []
        newcolname = []
        match_by = []
        if add is not None:
            for name in listify(add):
                cal_summ = self._read_summ(name)
                if cal_summ is None:
                    continue
                if name == "mfrin":
                    cal_summ["OBJECT"] = cal_summ["OBJECT"].str.split(
                        "_[sS][kK][yY]", expand=True
                    )[0]
                cal_summary.append(cal_summ)
                newcolname.append(names[name])
                match_by.append(GROUPER[name[1:5]])

        plan = make_reduc_planner(
            plan,
            cal_summary=cal_summary,
            newcolname=newcolname,
            match_by=match_by,
            ifmany="time",
            timecol="DATE-OBS",
            timefmt="isot",
        )
        if add_crrej:
            plan["do_crrej"] = True
            plan["gain"] = plan["FILTER"].map(GAIN)
            plan["rdnoise"] = plan["FILTER"].map(RDNOISE)
            for k, v in NIC_CRREJ_KEYS.items():
                plan[k] = v

        return plan

    def _save_plan(self, plan, name, iscal=True, do_not_modify=False, thumb_kw=None):
        # name includes "m"
        if len(plan) == 0:
            if self.verbose >= 1:
                print(f"No data found for {name}.")
            self.summpaths[name] = None
            self.planpaths[name] = None
            plan = None
        else:
            if iscal:
                coldic = PLANCOL_INIT[name[1:5]]
                for col, val in zip(coldic["columns"], coldic["values"]):
                    plan[col] = val
                for col, dtype in zip(coldic["columns"], coldic["dtypes"]):
                    plan[col] = plan[col].astype(dtype)
                plan.sort_values(
                    GROUPER[name[1:5]], inplace=True, key=lambda x: x.map(SORT_MAP)
                )
                plan.reset_index(drop=True, inplace=True)

            plan.to_csv(self.planpaths[name], index=False)
            if self.verbose >= 1:
                print(f"Saved to {self.planpaths[name]}")
            if thumb_kw is not None:
                _save_thumb_with_stat(
                    plan, self.plan_thumb_dirs[name], verbose=self.verbose, **thumb_kw
                )
            if self.verbose and not do_not_modify:
                if do_not_modify:
                    print("GOOD2GO! Proceed to the next step.")
                else:
                    print(f"WAIT! Modify\n\t{self.planpaths[name]}")
        return plan

    def _save_summ(self, name, thumb_kw=None, thumb_dir=None, add_keys=None):
        hdrkeys = HDR_KEYS.get(name)
        if hdrkeys is None:  # 2, 3, 4
            namestr = "lv" + str(name)
            hdrkeys = HDR_KEYS["simple"].copy()
        else:
            namestr = name
        if add_keys is not None:
            hdrkeys += add_keys

        _summ = summary_nic(
            self.dirs[name] / "*.fits",
            keywords=hdrkeys,
            output=self.summpaths[name],
            verbose=self.verbose - 1,
            add_setid=False,
        )
        if len(_summ) == 0:
            if self.verbose >= 1:
                print(f"No frame found for {namestr}.")
            self.summpaths[name] = None
            _summ = None
        elif thumb_kw is not None:
            thumb_dir = self.thumb_dirs[name] if thumb_dir is None else Path(thumb_dir)
            _save_thumb_with_stat(_summ, thumb_dir, verbose=self.verbose, **thumb_kw)
        return _summ


# FIXME: _raw_v is not allowed to be saved.
@dataclass
class NICPolReduc(NICPolReducMixin):
    """ Reduce NIC polaeimteric data (not calibration frames, e.g., dome
    flats), mostly hard-coded.

    For planners (using `make_*_plan`), "REMOVEIT" column can be any value for
    your own logging purposes. But only the rows with "REMOVEIT" = 0 will be
    used for reduction. You may further add other columns to the planner if you
    wish, but they'll be ignored.
    """

    name: str
    inputs: str
    mdarks: str = field(default=None)
    mfrins: str = field(default=None)
    mflats: str = field(default="cal-flat/*.fits")
    imasks: str = field(default="masks/*.fits")
    top_log: str = field(default="__logs")
    top_lv1: str = field(default="_lv1")
    top_lv2: str = field(default="_lv2")
    top_lv3: str = field(default="_lv3")
    top_lv4: str = field(default="_lv4")
    # Making skip_if_exists_* = False is used only for DEBUGGING purposes.
    # skip_if_exists_lv1: bool = field(default=True)
    # skip_if_exists_lv2plan: bool = field(default=True)
    # skip_if_exists_lv2: bool = field(default=True)
    # skip_if_exists_lv3plan: bool = field(default=True)
    # skip_if_exists_lv3: bool = field(default=True)
    # skip_if_exists_lv4plan: bool = field(default=True)
    # skip_if_exists_lv4: bool = field(default=True)
    # skip_if_exists_mdarkplan: bool = field(default=True)
    # skip_if_exists_mdark: bool = field(default=True)
    # skip_if_exists_mfrinplan: bool = field(default=True)
    # skip_if_exists_mfrin: bool = field(default=True)
    # skip_if_exists_mmaskplan: bool = field(default=True)
    # skip_if_exists_mmask: bool = field(default=True)
    # skip_step can be turned on by the user.
    # FIXME: At this moment, any skip_*=True will raise ERRORs. Tune where __skip() is used.
    skip_step_lv2: bool = field(default=False)
    skip_step_lv3: bool = field(default=False)
    skip_step_lv4: bool = field(default=False)
    skip_step_mdark: bool = field(default=False)
    skip_step_dmask: bool = field(default=False)
    skip_step_mfrin: bool = field(default=False)
    skip_step_mmask: bool = field(default=False)
    verbose: int = field(default=1)
    # TODO: save thuimbnails for reduced frames

    def __post_init__(self):
        self.skip = {
            2: self.skip_step_lv2,
            3: self.skip_step_lv3,
            4: self.skip_step_lv4,
            "mdark": self.skip_step_mdark,
            "dmask": self.skip_step_dmask,
            "mfrin": self.skip_step_mfrin,
            "mmask": self.skip_step_mmask,
        }

        def __skip(key, value, skipdict=self.skip):
            if skipdict.get(key):  # True
                return None
            else:  # Either False or None
                return value

        # If self.skip[key] is True, then self.dirs[key] will be None.
        # Then that directory will not be mase based on the `d.mkdir` line below.

        self.dirs = {
            1: __skip(1, Path(self.top_lv1) / self.name),
            2: __skip(2, Path(self.top_lv2) / self.name),
            3: __skip(3, Path(self.top_lv3) / self.name),
            4: __skip(4, Path(self.top_lv4) / self.name),
            "log": Path(self.top_log) / self.name,
            "mflat": __skip("mflat", Path(self.top_log) / self.name / "cal-mflat"),
            "dark1": __skip("mdark", Path(self.top_lv1) / self.name / "dark"),
            "mdark": __skip("mdark", Path(self.top_log) / self.name / "cal-mdark"),
            "dmask": __skip("dmask", Path(self.top_log) / self.name / "cal-dmask"),
            "mfrin": __skip("mfrin", Path(self.top_log) / self.name / "cal-mfrin"),
            "ifrin": __skip("mfrin", Path(self.top_lv4) / self.name / "frin"),
            "imask": __skip("imask", Path(self.top_log) / self.name / "cal-imask"),
            "mmask": __skip("mmask", Path(self.top_log) / self.name / "cal-mmask"),
        }

        self.plan_thumb_dirs = {
            "mdark": __skip("mdark", self.dirs["log"] / "thumbs_mdark_plan"),
            "mflat": __skip("mflat", self.dirs["log"] / "thumbs_mflat_plan"),
            "mfrin": __skip("mfrin", self.dirs["log"] / "thumbs_mfrin_plan"),
            "dmask": __skip("dmask", self.dirs["log"] / "thumbs_dmask_plan"),
            "mmask": __skip("mmask", self.dirs["log"] / "thumbs_mmask_plan"),
        }

        self.thumb_dirs = {
            1: __skip(1, self.dirs[1] / "thumbs"),
            2: __skip(2, self.dirs[2] / "thumbs"),
            3: __skip(3, self.dirs[3] / "thumbs"),
            4: __skip(4, self.dirs[4] / "thumbs"),
            "mdark": __skip("mdark", self.dirs["log"] / "thumbs_mdark"),
            "mfrin": __skip("mfrin", self.dirs["log"] / "thumbs_mfrin"),
            "dmask": __skip("dmask", self.dirs["log"] / "thumbs_dmask"),
            "mmask": __skip("mmask", self.dirs["log"] / "thumbs_mmask")
            # "mflat": __skip("mflat", self.dirs["mflat"]/"planthumbs"),
        }

        del self.top_lv1, self.top_lv2, self.top_lv3, self.top_lv4, self.top_log

        # for dd in [self.dirs, self.thumb_dirs, self.plan_thumb_dirs]:
        #     [d.mkdir(parents=True, exist_ok=True) for d in dd.values() if d is not None]

        self.summpaths = {  # summary files
            0: self.dirs["log"] / f"{self.name}_summ_lv0.csv",
            1: __skip(1, self.dirs["log"] / f"{self.name}_summ_lv1.csv"),
            2: __skip(2, self.dirs["log"] / f"{self.name}_summ_lv2.csv"),
            3: __skip(3, self.dirs["log"] / f"{self.name}_summ_lv3.csv"),
            4: __skip(4, self.dirs["log"] / f"{self.name}_summ_lv4.csv"),
            "mdark": __skip("mdark", self.dirs["log"] / f"{self.name}_summ_MDARK.csv"),
            "mflat": __skip("mflat", self.dirs["log"] / f"{self.name}_summ_MFLAT.csv"),
            "mfrin": __skip("mfrin", self.dirs["log"] / f"{self.name}_summ_MFRIN.csv"),
            "ifrin": __skip("ifrin", self.dirs["log"] / f"{self.name}_summ_IFRIN.csv"),
            "imask": __skip("imask", self.dirs["log"] / f"{self.name}_summ_IMASK.csv"),
            "dmask": __skip("dmask", self.dirs["log"] / f"{self.name}_summ_DMASK.csv"),
            "mmask": __skip("mmask", self.dirs["log"] / f"{self.name}_summ_MMASK.csv"),
        }
        self.planpaths = {  # Reduction plan files
            2: __skip(2, self.dirs["log"] / f"{self.name}_plan-lv2.csv"),
            3: __skip(3, self.dirs["log"] / f"{self.name}_plan-lv3.csv"),
            4: __skip(4, self.dirs["log"] / f"{self.name}_plan-lv4.csv"),
            "mdark": __skip("mdark", self.dirs["log"] / f"{self.name}_plan-MDARK.csv"),
            # "mflat": self.dirs["log"]/f"{self.name}_plan-MFLAT.csv",
            # ^ making flat not supported under NICPolReduc yet
            "ifrin": __skip("ifrin", self.dirs["log"] / f"{self.name}_plan-IFRIN.csv"),
            "mfrin": __skip("mfrin", self.dirs["log"] / f"{self.name}_plan-MFRIN.csv"),
            "dmask": __skip("dmask", self.dirs["log"] / f"{self.name}_plan-DMASK.csv"),
            "mmask": __skip("mmask", self.dirs["log"] / f"{self.name}_plan-MMASK.csv"),
        }

        _ = summary_nic(
            inputs=self.inputs,
            output=self.summpaths[0],
            keywords=HDR_KEYS["all"],  # hard-coded
            rm_test=True,  # hard-coded
            rm_nonpol=True,  # hard-coded
            add_setid=True,  # hard-coded
            verbose=self.verbose - 1,
        )

        for name in ["mdark", "mfrin", "mflat", "imask"]:
            _list = inputs2list(getattr(self, f"{name}s"))
            if not exists(self.summpaths[name]) and _list:  # Not empty
                newpaths = []
                self.dirs[name].mkdir(parents=True, exist_ok=True)
                # -- Copy to log directory
                for fpath in _list:
                    newpath = self.dirs[name] / Path(fpath).name
                    shutil.copy(fpath, newpath)
                    newpaths.append(newpath)
                # -- Add summary if master frames are given
                _ = summary_nic(
                    newpaths,
                    keywords=HDR_KEYS[name],
                    output=self.summpaths[name],
                    add_setid=False,  # master calibration frames need no SETID.
                )
                # ^ TODO: there will be FEW master frames, so just ignore skip....

    def plan_mdark(
        self,
        method="median",
        sigclip_kw=dict(sigma=2, maxiters=5),
        fitting_sections=None,
        show_progress=True,
        thumb_kw=dict(ext="pdf", dpi=72, zscale=True, percentiles=[0.01, 99.9, 99.99]),
    ):
        """ Processes lv1 for DARKs, make plan for MDARK
        method, sigclip_kw, fitting_sections for vertical correction.
        Since we are taking median along ~50+ pixels, updated mask using
        nightly dark is expected to have minimal improvement (if any). Thus,
        for lv1 DARK frames, we do not consider the DMASK to update MMASK.

        ~ 30s/(20*6 FITS) on MBP 16" [2021, macOS 12.0.1, M1Pro, 8P+2E core,
        GPU 16-core, RAM 16GB]
        """
        if (loaded := self._check_skip("plan", "mdark")) is not None:
            return loaded

        _dark_lv0 = df_selector(
            self._read_summ(0),
            **MATCHER["dark"],
            columns=["file", "DATE-OBS"] + GROUPER["dark"],
        )
        if len(_dark_lv0) == 0:
            if self.verbose:
                print("No dark frame found.")
            for attr in [
                "dirs",
                "plan_thumb_dirs",
                "thumb_dirs",
                "summpaths",
                "planpaths",
            ]:
                for key in ["mdark", "dmask"]:
                    getattr(self, attr)[key] = None
                try:
                    getattr(self, attr)["mmask"] = getattr(self, attr)["imask"]
                except KeyError:
                    getattr(self, attr)["mmask"] = None
            return None

        _, masks, maskpaths = _load_as_dict(
            self.dirs["imask"], GROUPER["mask"], verbose=self.verbose - 1
        )

        for _, row in iterator(_dark_lv0.iterrows(), show_progress=show_progress):
            mask, maskpath = dict_get(masks, maskpaths, row, GROUPER["mask"])
            _do_16bit_vertical(
                fpath=row["file"],
                dir_out=self.dirs["dark1"],
                mask=mask,
                maskpath=maskpath,
                npixs=(10, 10),  # hard-coded for DARK frames
                bezels=(20, 20),  # hard-coded for DARK frames
                process_title=" Level 1 (vertical pattern removal) ",
                method=method,
                sigclip_kw=sigclip_kw,
                fitting_sections=fitting_sections,
                dtype="float32",  # hard-coded
                multi2for_int16=False,  # hard-coded
                skip_if_exists=self._skipexred[1],
                verbose=self.verbose - 2,
            )
        _plan = summary_nic(self.dirs["dark1"] / "*.fits", **MATCHER["dark"],)[
            ["file", "DATE-OBS"] + GROUPER["dark"]
        ]

        return self._save_plan(_plan, "mdark", iscal=True, thumb_kw=thumb_kw)

    def comb_mdark_dmask(
        self,
        combine_kw=dict(combine="median", reject="sc", sigma=3),
        dark_thresh=(-10, 50),
        dark_percentile=(0.01, 99.99),
        thumb_kw=dict(
            ext="pdf", dpi=72, vmin=0, vmax=50, percentiles=[0.01, 99.9, 99.99]
        ),
        thumb_kw_dmask=dict(ext="pdf", dpi=72, vmin=0, vmax=1),
    ):
        """ Make MDARK and DMASK frames
        """
        if (loaded := self._check_skip("red", "mdark")) is not None:
            return loaded

        if not exists(self.planpaths["mdark"]):
            if self.verbose:
                print("No DARK frames found. Skipping...")
            return None

        _ = make_cal(
            framename="mdark",
            plan_in=self._read_plan("mdark"),
            dir_out=self.dirs["mdark"],
            group_key=GROUPER["dark"],
            fmts="{:s}_{:.1f}s",
            combine_kw=combine_kw,
            verbose=self.verbose,
        )
        _summ = self._save_summ("mdark", thumb_kw)

        # Make darkmask files
        if _summ is not None:
            for _, row in _summ.iterrows():
                make_darkmask(
                    row["file"],
                    output=self.dirs["dmask"] / f"darkmask_{Path(row['file']).name}",
                    thresh=dark_thresh,
                    percentile=dark_percentile,
                )
            _ = self._save_summ("dmask", thumb_kw=thumb_kw_dmask)

        return _summ

    def plan_mmask(
        self,
        thumb_kw=dict(ext="pdf", percentiles=[99.9, 99.99], dpi=72, vmin=0, vmax=1),
    ):
        """
        """
        if (loaded := self._check_skip("plan", "mmask")) is not None:
            return loaded

        if not exists(self.summpaths["mdark"]):  # if no DARK frames found
            if self.verbose:
                print("No dark found. Skipping DMASK (using IMASK as MMASK)")
            self.dirs["mmask"] = self.dirs["imask"]
            self.summpaths["mmask"] = self.summpaths["imask"]
            return None

        _df = pd.concat(
            [
                summary_nic(
                    self.dirs["imask"] / "*.fits", add_setid=False, verbose=False
                ),
                summary_nic(
                    self.dirs["dmask"] / "*.fits", add_setid=False, verbose=False
                ),
            ]
        )
        # use summary_nic rather than _read_summ, because columns do not match.
        _plan = _df[["file", "DATE-OBS"] + GROUPER["mask"]].copy()
        return self._save_plan(_plan, "mmask", thumb_kw=thumb_kw)

    def comb_mmask(
        self,
        combine_kw=dict(combine="sum", reject=None),
        thumb_kw=dict(ext="pdf", dpi=72, vmin=0, vmax=1),
    ):
        if (loaded := self._check_skip("red", "mmask")) is not None:
            return loaded

        if not exists(self.planpaths["mmask"]):
            if self.verbose:
                print("Using IMASK as MMASK. Skipping...")
            return None

        if combine_kw.get("dtype") is not None:
            warn("dtype in `combine_kw` will be ignored.", UserWarning)

        combine_kw["dtype"] = "uint8"
        _ = make_cal(
            framename="mmask",
            plan_in=self._read_plan("mmask"),
            dir_out=self.dirs["mmask"],
            group_key=GROUPER["mask"],
            fmts="{:s}",
            combine_kw=combine_kw,
            hedit_keys=["OBJECT"],
            hedit_values=["MASK"],
            verbose=self.verbose,
        )
        return self._save_summ("mmask", thumb_kw=thumb_kw)

    # TODO: plot nightly log plots
    def proc_lv1(
        self,
        fullmatch=None,
        flags=0,
        querystr=None,
        npixs=(5, 5),
        bezels=((20, 20), (20, 20)),
        method="median",
        sigclip_kw=dict(sigma=2, maxiters=5),
        fitting_sections=None,
        rm_nonpol=True,
        rm_test=True,
        show_progress=True,
        thumb_kw=dict(
            ext="pdf",
            dpi=72,
            gap_value=-100,
            bezels=((20, 20), (20, 20)),
            figsize=(5.5, 3),
        ),
    ):
        """Prepare level1 data (vertical pattern removal) except DARK.

        ~ 30s/450FITS on MBP 16" [2021, macOS 12.0.1, M1Pro, 8P+2E core, GPU
        16-core, RAM 16GB]
        ~ 25s/360 on MBP 14" [2021, macOS 12.2.1, M1Pro 6P+2E, 32G, GPU 14c]
        """
        if (loaded := self._check_skip("red", 1)) is not None:
            return loaded

        _df_lv0 = summary_nic(
            inputs=self.inputs,
            keywords=HDR_KEYS["all"],  # hard-coded
            rm_test=rm_test,
            rm_nonpol=rm_nonpol,
            add_setid=True,  # hard-coded
            verbose=self.verbose - 1,
        )
        # remove DARK frames
        _df_lv0 = df_selector(_df_lv0, **MATCHER["dark"], negate_fullmatch=True)
        # Select based on user input
        _df_lv0 = df_selector(
            _df_lv0, fullmatch=fullmatch, flags=flags, querystr=querystr
        )
        _, masks, maskpaths = _load_as_dict(
            self.dirs["mmask"], GROUPER["mask"], verbose=self.verbose - 1
        )

        for _, row in iterator(_df_lv0.iterrows(), show_progress=show_progress):
            mask, maskpath = dict_get(masks, maskpaths, row, GROUPER["mask"])
            _do_16bit_vertical(
                fpath=row["file"],
                dir_out=self.dirs[1],
                mask=mask,
                maskpath=maskpath,
                npixs=npixs,
                bezels=bezels,
                process_title=" Level 1 (vertical pattern removal) ",
                method=method,
                sigclip_kw=sigclip_kw,
                fitting_sections=fitting_sections,
                dtype="float32",  # hard-coded
                multi2for_int16=False,  # hard-coded
                setid=row["SETID"],
                skip_if_exists=self._skipexred[1],
                verbose=self.verbose - 2,
            )

        return self._save_summ(1, add_keys=["NSATPIXO", "NSATPIXE"])

    def plan_lv2(
        self,
        maxnsat_oe=5,
        maxnsat_qu=10,
        thumb_kw=dict(
            ext="pdf",
            dpi=72,
            gap_value=-100,
            bezels=((20, 20), (20, 20)),
            figsize=(5.5, 3),
        ),
    ):
        """ lv2 takes a bit of time due to fixpix, mbpm & FFT, so filter frames here.
        The user may have to manually "un-REMOVEIT" some _sky frames (fringe
        frames), if they have REMOVEIT > 0 due to the CR pixels.

        Parameters
        ----------
        maxnsat_oe : int, optional
            The maximum number of saturated pixels on each ray (o-/e-ray)
            region of a single frame (exclusive). Code: 1 (2**0)
        maxnsat_qu : int, optional
            The maximum number of saturated pixels within the o-/e-rays of
            either q (POL-AGL1 0. and 45.) or u (POL-AGL1 22.5 and 67.5) frames
            of the same set (exclusive) . Code: 2 (2**1)
        """
        if (loaded := self._check_skip("plan", 2)) is not None:
            return loaded

        _df = self._read_summ(1)
        _plan = _df[["file"] + HDR_KEYS["simple"]].copy()
        # if rm_dark:
        #     _plan = df_selector(_plan, **MATCHER["dark"], negate_fullmatch=True)
        _, masks, _ = _load_as_dict(
            self.dirs["mmask"], GROUPER["mask"], verbose=self.verbose - 1
        )
        _plan = _save_thumb_with_satpix(
            _plan,
            masks=masks,
            outdir=self.thumb_dirs[1],
            basename=self.name,
            show_progress=True,
            **thumb_kw,
        )

        _plan["REMOVEIT"] = 0
        if maxnsat_oe is not None:
            _plan["REMOVEIT"] = 1 * (
                (_plan["NSATPIXO"] > maxnsat_oe) | (_plan["NSATPIXE"] > maxnsat_oe)
            )

        ofs = [
            (o, f, s, a)
            for o in _plan["OBJECT"].unique()
            for f in "JHK"
            for s in _plan["SETID"].unique()
            for a in [[0.0, 45.0, 0, 45], [22.5, 67.5]]
        ]
        if maxnsat_qu is not None:
            for objname, filt, setid, agls in ofs:
                df_i = _plan.loc[
                    (_plan["OBJECT"] == objname)
                    & (_plan["FILTER"] == filt)
                    & (_plan["SETID"] == setid)
                    & (_plan["POL-AGL1"].isin(agls))
                ]
                removeit = 2 * ((df_i["NSATPIXO"] + df_i["NSATPIXE"]) > maxnsat_qu)
                _plan.loc[df_i.index, "REMOVEIT"] += removeit.max()
        _plan["REMOVEIT"] = _plan["REMOVEIT"].astype(int)
        return self._save_plan(_plan, 2, iscal=False)

    def proc_lv2(
        self,
        do_fourier=True,
        do_fixpix_before_fourier=True,
        do_mbpm_before_fourier=True,
        cut_wavelength=100,
        vertical_again=True,
        bpm_kw=BPM_KW,
        show_progress=True,
    ):
        """ Fourier pattern removal
        ~ 13 min/350FITS on MBP 16" [2021, macOS 12.0.1, M1Pro, 8P+2E core, GPU
        16-core, RAM 16GB
        """
        # if not do_fourier:  # If False, do nothing but use lv1 as if it is lv2
        #     self._check_skip()
        #     if self.verbose >= 1:
        #         print("Skipping lv2...")
        #     self.summpaths[2] = self.summpaths[1]
        #     self.dirs[2] = self.dirs[1]
        if (loaded := self._check_skip("red", 2)) is not None:
            return loaded

        _df1 = self._read_plan(2)
        _, masks, maskpaths = _load_as_dict(
            self.dirs["mmask"], GROUPER["mask"], verbose=self.verbose - 1
        )

        for _, row in iterator(_df1.iterrows(), show_progress=show_progress):
            mask, maskpath = dict_get(masks, maskpaths, row, GROUPER["mask"])
            _do_vfv(
                row["file"],
                dir_out=self.dirs[2],
                mask=mask,
                maskpath=maskpath,
                process_title=" Level 2 (Fourier pattern removal) ",
                vertical_already=True,
                do_fourier=do_fourier,
                do_fixpix_before_fourier=do_fixpix_before_fourier,
                do_mbpm_before_fourier=do_mbpm_before_fourier,
                cut_wavelength=cut_wavelength,
                bpm_kw=bpm_kw,
                vertical_again=vertical_again,
                skip_if_exists=self._skipexred[2],
                setid=row["SETID"],
                verbose=self.verbose - 2,
            )

        return self._save_summ(2, add_keys=["NSATPIXO", "NSATPIXE"])

    def plan_lv3(
        self,
        use_lv1=False,
        thumb_kw=dict(
            ext="pdf",
            dpi=72,
            gap_value=-100,
            bezels=((20, 20), (20, 20)),
            figsize=(5.5, 3),
        ),
    ):
        if (loaded := self._check_skip("plan", 3)) is not None:
            return loaded

        _df2 = self._read_summ(1) if use_lv1 else self._read_summ(2)
        _, masks, _ = _load_as_dict(
            self.dirs["mmask"], GROUPER["mask"], verbose=self.verbose - 1
        )
        _ = _save_thumb_with_satpix(
            _df2, masks=masks, outdir=self.thumb_dirs[2], basename=self.name, **thumb_kw
        )
        # ^ strictly speaking, this does not represent SATURATION as it has
        #   already processed through the Fourier pattern removal. However, the
        #   total number of SATPIX will not differ a lot, so I used it to
        #   simplify the code.

        _df2 = _df2[["file"] + HDR_KEYS["minimal"] + ["NSATPIXO", "NSATPIXE"]].copy()
        _plan = self._make_reduc_plan(
            _df2, add_crrej=False, add=["mdark", "mflat", "mmask"]
        )

        return self._save_plan(_plan, 3, iscal=False)

    def proc_lv3(
        self,
        flat_mask=0.3,
        flat_fill=0.3,
        thumb_kw=dict(ext="pdf", dpi=72, zscale=True, percentiles=[0.1, 99.9, 99.99]),
    ):
        """
        ~ 3 min/350FITS on MBP 16" [2021, macOS 12.0.1, M1Pro, 8P+2E core, GPU
        16-core, RAM 16GB]
        flat_mask and flat_fill are used only because they make the "bezel"
        regions to behave abnormally. The inner part of the fov has already
        been masked in imask and will be fixed afterwards by interpolation.
        """
        if (loaded := self._check_skip("red", 3)) is not None:
            return loaded

        plan = self._read_plan(3)
        plan = plan.loc[plan["REMOVEIT"] == 0].copy()
        mdarks = _get_plan_frm(plan, "DARKFRM")
        mflats = _get_plan_frm(plan, "FLATFRM")
        mmasks = _get_plan_frm(plan, "MASKFILE", split_oe_ccd=True)

        for _, row in iterator(plan.iterrows(), show_progress=True):
            fpath = Path(row["file"])
            ccd = load_ccd(fpath)
            mdarkpath = row.get("DARKFRM")
            mflatpath = row.get("FLATFRM")
            mmaskpath = row.get("MASKFILE")
            ccd = ccdred(
                ccd,
                mdark=mdarks.get(mdarkpath),
                mdarkpath=mdarkpath if isinstance(mdarkpath, str) else None,
                mflat=mflats.get(mflatpath),
                mflatpath=mflatpath if isinstance(mflatpath, str) else None,
                flat_mask=flat_mask,
                flat_fill=flat_fill,
                dark_scale=False,
                verbose_bdf=self.verbose >= 2,
            )
            ccd.header["LV2FRM"] = str(fpath)
            ccd.header["LVNOW"] = (3, "The current level; see LViFRM for history.")

            ccds = split_oe(ccd, return_dict=True, verbose=self.verbose - 2)
            for oe, ccd_oe in ccds.items():
                try:
                    mask = mmasks.get(mmaskpath)[oe]
                except TypeError:  # mmask is None
                    mask = None
                ccd_oe = fixpix(
                    ccd_oe,
                    mask=mask,
                    maskpath=mmaskpath,
                    priority=(1, 0),
                    verbose=self.verbose >= 2,
                )
                ccd_oe.header.set(
                    "NSATPIX",
                    ccd_oe.header.get(f"NSATPIX{oe.upper()}"),
                    "No. of saturated pix in this region",
                    after="NSATPIXE",
                )
                _save(ccd_oe, self.dirs[3], _set_fstem_proc(fpath, ccd_oe))

        return self._save_summ(3, add_keys=["OERAY"], thumb_kw=thumb_kw)

    def plan_ifrin(
        self, add_crrej=True,
    ):
        """
        CR rejection is not negligible in FRINGE frames.
        Thus, a separate plan will be made with name "ifrin" (initial fringe).
        Corresponding Thumbnails availabe from self.thumb_dirs[3]
        """
        if (loaded := self._check_skip("plan", "ifrin")) is not None:
            return loaded

        plan = df_selector(
            self._read_summ(3),
            **MATCHER["frin"],
            columns=["file", "DATE-OBS"] + GROUPER["frin"] + ["SETID"],
        )
        plan = self._make_reduc_plan(plan, add_crrej=add_crrej)
        # ^ add=None: dark, flat, mask(FIXPIX) already processed in lv3
        plan["FRINCID"] = plan["FILTER"].str.cat(
            [
                plan["OBJECT"],
                plan["SETID"].map(lambda x: x[1:]),
                #  plan["POL-AGL1"].map("{:04.1f}".format),
                plan["OERAY"],
            ],
            sep="_",
        )
        plan.drop("SETID", axis=1, inplace=True)

        return self._save_plan(plan, "ifrin", iscal=True)

    def plan_mfrin(
        self, thumb_kw=dict(ext="pdf", dpi=72, zscale=True, percentiles=[1, 95, 99])
    ):
        """ DFC process on fringe frames
        """
        if (loaded := self._check_skip("red", "mfrin")) is not None:
            return loaded

        if (plan := self._read_plan("ifrin")) is None:
            return None

        for combid in iterator(plan["FRINCID"].unique(), show_progress=True):
            plan_i = plan[plan["FRINCID"] == combid]
            for _, row in plan_i.iterrows():
                fpath = Path(row["file"])
                ccd = load_ccd(fpath)
                ccd.header["FRINCID"] = (combid, "Master fringe Combine-ID")
                if row["do_crrej"]:
                    # ccd is Already FIXPIX'ed, so no need to use mask again here.
                    ccd, _ = do_crrej(ccd, row, verbose=self.verbose >= 2)
                _save(ccd, self.dirs["ifrin"], _set_fstem_proc(fpath, ccd))

        plan = summary_nic(
            self.dirs["ifrin"] / "*.fits",
            **MATCHER["frin"],
            keywords=HDR_KEYS["ifrin"],
            add_setid=False,
        )[["file", "DATE-OBS"] + GROUPER["frin"] + ["FRINCID"]]
        return self._save_plan(plan, "mfrin", thumb_kw=thumb_kw)

    def comb_mfrin(
        self,
        combine_kw=dict(
            combine="median",
            reject="mm",
            n_minmax=(0, 1),
            scale="avg_sc",
            scale_section="[20:100, 20:400]",
            scale_to_0th=False,
        ),
        thumb_kw=dict(ext="pdf", dpi=72, zscale=True, percentiles=[1, 95, 99]),
    ):
        """
        Notes
        -----
        Use minmax with 1 maximum removal as defualt --- This is because of the
        "presistence" (remaining electrons) after bright exosures. The pixels
        where the object was located will be slightly brighter than the blank
        sky for several minutes (decaying over time, not over exposures). The
        effect of it is expected to be minimal, but just in case...
        """
        if (loaded := self._check_skip("red", "mfrin")) is not None:
            return loaded

        if (plan_in := self._read_plan("mfrin")) is None:
            return None

        _ = make_cal(
            framename="mfrin",
            plan_in=plan_in,
            dir_out=self.dirs["mfrin"],
            group_key="FRINCID",
            fmts="{:s}",
            combine_kw=combine_kw,
            verbose=self.verbose,
        )

        return self._save_summ("mfrin", thumb_kw=thumb_kw)

    def plan_lv4(
        self,
        add_crrej=True,
        add_mfrin=True,
        fullmatch=None,
        flags=0,
        querystr=None,
        negate_fullmatch=False,
        thumb_kw=dict(
            ext="pdf", dpi=72, gap_value=-100, bezels=(20, 20), figsize=(5.5, 3)
        ),
    ):
        if (loaded := self._check_skip("red", 4)) is not None:
            return loaded

        _df3 = self._read_summ(3)
        plan = df_selector(
            df_selector(
                _df3, **MATCHER["frin"], negate_fullmatch=True
            ),  # remove fringe
            fullmatch=fullmatch,
            flags=flags,
            negate_fullmatch=negate_fullmatch,
            querystr=querystr,
            columns=["file"] + HDR_KEYS["minimal"] + ["OERAY"],
        )
        plan = self._make_reduc_plan(
            plan, add_crrej=add_crrej, add="mfrin" if add_mfrin else None
        )

        return self._save_plan(plan, 4, iscal=False)

    def proc_lv4(
        self,
        frin_bezels=20,
        frin_sep_kw=dict(minarea=np.pi * 5 ** 2),
        frin_scale_kw=dict(sigma=2.5),
        thumb_kw=dict(ext="pdf", dpi=72, zscale=True, percentiles=[0.1, 99.9, 99.99]),
    ):
        if (loaded := self._check_skip("red", 4)) is not None:
            return loaded

        if (plan := self._read_plan(4)) is None:
            return None

        plan = self._read_plan(4)
        frinpaths = plan["FRINFRM"].unique() if "FRINFRM" in plan.columns else []
        frins = {}
        for fpath in frinpaths:
            if isinstance(fpath, str):
                frins[fpath] = load_ccd(fpath)  # fpath is not NaN
                # N.B. np.isnan(fpaths) will raise error because of np.isnan(str)

        for _, row in iterator(plan.iterrows(), show_progress=True):
            fpath = Path(row["file"])
            ccd = load_ccd(fpath)
            mfrinpath = row.get("FRINFRM")
            if not isinstance(mfrinpath, str):
                mfrinpath = None  # in case it is NaN
            mfrin = frins.get(mfrinpath)
            mfrin_scale_reg = fringe_scale_mask(
                mfrin, bezels=frin_bezels, **frin_sep_kw
            )

            ccd = ccdred(
                ccd,
                mfrin=mfrin,
                mfrinpath=mfrinpath,
                fringe_scale=sigma_clipped_med,
                fringe_scale_region=mfrin_scale_reg,
                fringe_scale_kw=frin_scale_kw,
                verbose_bdf=self.verbose >= 2,
            )
            ccd.header["LV3FRM"] = str(fpath)
            ccd.header["LVNOW"] = (4, "The current level; see LViFRM for history.")

            if row["do_crrej"]:
                # ccd is Already FIXPIX'ed, so no need to use mask again here.
                ccd, _ = do_crrej(ccd, row, verbose=self.verbose >= 2)

            _save(ccd, self.dirs[4], _set_fstem_proc(fpath, ccd))

        return self._save_summ(
            4, add_keys=["OERAY", "NSATPIX", "CRNPIX"], thumb_kw=thumb_kw
        )
