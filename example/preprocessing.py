from pathlib import Path

import nicpolpy as nic

import ysfitsutilpy as yfu

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
for fpath in obj_fpaths:
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
