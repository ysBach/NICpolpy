from pathlib import Path

import nicpolpy as nic
import pandas as pd
from matplotlib import pyplot as plt

import ysfitsutilpy as yfu

try:
    NO_TQDM = False
    from tqdm import tqdm
except ImportError:
    NO_TQDM = True

# %%
top = Path("your/folder/path/from_current_pwd_or_absolute_path_to_it")
cal = Path("calibrated/data/will_be_saved_in_this_folder")
Path.mkdir(cal, exist_ok=True, parents=True)
allfits = list(top.glob("*.fits"))
allfits.sort()
summary = yfu.make_summary(allfits,
                           keywords=nic.USEFUL_KEYS,
                           pandas=True,
                           output=f"{top.name}.csv")

object_name = 'Vesta'
object_mask = summary['OBJECT'].str.lower() == object_name.lower()
obj_fpaths = summary[object_mask]['file'].values
# %%
# single image example:
nicimg = nic.NICPolImage(obj_fpaths[0], verbose=True)
nicimg.preproc(do_verti=True,
               verti_fitting_sections=None,
               verti_method='median',
               verti_sigclip_kw=dict(sigma=2, maxiters=5),
               do_fouri=False,
               do_crrej=True,
               verbose=True,
               verbose_crrej=True)
nicimg.find_obj(thresh=3, verbose=True)
nicimg.ellipphot_sep(f_ap=(2, 2), verbose=True)

# Also try:
for k in list(vars(nicimg).keys()):
    print(k)

# and try such as this:
nicimg.ccd_o_bdfx.write('test_bdfx.fits', overwrite=True)

# %%
# multiple image example:
phots_o = []
phots_e = []

if NO_TQDM:
    for fpath in obj_fpaths:
        fpath = Path(fpath)
        out = [cal/f"{fpath.stem}_o.fits", cal/f"{fpath.stem}_e.fits"]
        #
        nicimg = nic.NICPolImage(fpath, verbose=False)
        nicimg.preproc(do_fouri=False, verbose_crrej=False, verbose=False)
        nicimg.find_obj(thresh=1, verbose=False)
        nicimg.ellipphot_sep(f_ap=(2, 2), fwhm=(11., 11.),
                             fix_fwhm=False, verbose=False)

        for outpath, ccd in zip(out, [nicimg.ccd_o_proc, nicimg.ccd_e_proc]):
            ccd.write(outpath, overwrite=True)

        phots_o.append(nicimg.phot_o)
        phots_e.append(nicimg.phot_e)
        # If you want to plot, uncomment these:
        # fig, axs = plt.subplots(1, 2, figsize=(5, 5),
        #                         sharex=False, sharey=False, gridspec_kw=None)

        # _stats_o = yfu.give_stats(nicimg.ccd_o_proc.data - nicimg.bkg_o)
        # _stats_e = yfu.give_stats(nicimg.ccd_e_proc.data - nicimg.bkg_e)
        # min_o, max_o = _stats_o['zmin'], _stats_o['zmax']
        # min_e, max_e = _stats_e['zmin'], _stats_e['zmax']
        # vv = dict(vmin=min(min_o, min_e),
        #           vmax=max(max_o, max_e),
        #           origin='lower')
        # axs[0].imshow(nicimg.ccd_o_proc.data - nicimg.bkg_o, **vv)
        # axs[1].imshow(nicimg.ccd_e_proc.data - nicimg.bkg_e, **vv)
        # try:
        #     nicimg.ap_o.plot(axs[0], color='r')
        #     nicimg.an_o.plot(axs[0], color='w')
        # except (TypeError, AttributeError):
        #     pass
        # try:
        #     nicimg.ap_e.plot(axs[1], color='r')
        #     nicimg.an_e.plot(axs[1], color='w')
        # except (TypeError, AttributeError):
        #     pass
        #     plt.tight_layout()
        # fig.align_ylabels(axs)
        # fig.align_xlabels(axs)
        # plt.savefig(cal/f"{fpath.stem}.pdf")
        # plt.close('all')

        nicimg.close()

else:
    for fpath in tqdm(obj_fpaths):
        fpath = Path(fpath)
        out = [cal/f"{fpath.stem}_o.fits", cal/f"{fpath.stem}_e.fits"]
        #
        nicimg = nic.NICPolImage(fpath, verbose=False)
        nicimg.preproc(do_fouri=False, verbose_crrej=False, verbose=False)
        nicimg.find_obj(thresh=1, verbose=False)
        nicimg.ellipphot_sep(f_ap=(2, 2), fwhm=(11., 11.),
                             fix_fwhm=False, verbose=False)
        #
        for outpath, ccd in zip(out, [nicimg.ccd_o_proc, nicimg.ccd_e_proc]):
            ccd.write(outpath, overwrite=True)

        phots_o.append(nicimg.phot_o)
        phots_e.append(nicimg.phot_e)

        # If you want to plot, uncomment these:
        # fig, axs = plt.subplots(1, 2, figsize=(5, 5),
        #                         sharex=False, sharey=False, gridspec_kw=None)

        # _stats_o = yfu.give_stats(nicimg.ccd_o_proc.data - nicimg.bkg_o)
        # _stats_e = yfu.give_stats(nicimg.ccd_e_proc.data - nicimg.bkg_e)
        # min_o, max_o = _stats_o['zmin'], _stats_o['zmax']
        # min_e, max_e = _stats_e['zmin'], _stats_e['zmax']
        # vv = dict(vmin=min(min_o, min_e),
        #           vmax=max(max_o, max_e),
        #           origin='lower')
        # axs[0].imshow(nicimg.ccd_o_proc.data - nicimg.bkg_o, **vv)
        # axs[1].imshow(nicimg.ccd_e_proc.data - nicimg.bkg_e, **vv)
        # try:
        #     nicimg.ap_o.plot(axs[0], color='r')
        #     nicimg.an_o.plot(axs[0], color='w')
        # except (TypeError, AttributeError):
        #     pass
        # try:
        #     nicimg.ap_e.plot(axs[1], color='r')
        #     nicimg.an_e.plot(axs[1], color='w')
        # except (TypeError, AttributeError):
        #     pass
        #     plt.tight_layout()
        # fig.align_ylabels(axs)
        # fig.align_xlabels(axs)
        # plt.savefig(cal/f"{fpath.stem}.pdf")
        # plt.close('all')

        nicimg.close()

phots_o_df = pd.concat(phots_o, sort=True)
phots_e_df = pd.concat(phots_e, sort=True)
phots_o_df.to_csv(f"{top.name}_o.csv")
phots_e_df.to_csv(f"{top.name}_e.csv")
