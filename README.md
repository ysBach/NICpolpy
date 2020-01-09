# NICpolpy
Nishi Harima Astronomical Observatory (NHAO)'s Near-Infrared Camera (NIC) Polarimetry photometry pipeline.

Under-development by Yoonsoo P. Bach at Seoul National University, South Korea, since late 2019.



## 1. Installation
You will need
1. Python **3.6+** (recommended: [Anaconda 3](https://www.anaconda.com/distribution/#download-section))
2. Dependencies (I will assume you used Anaconda 3; otherwise, use pip3):

### 1-1. First time
<details><summary>For the <b>first</b> installation (click):</summary>
<pre>
# On terminal
conda install -c astropy astroquery photutils ccdproc astroscrappy
conda install -c openastronomy sep
cd ~            # whatever directory you want
mkdir github    # whatever name you want
git clone https://github.com/ysBach/ysfitsutilpy.git
cd ysfitsutilpy && python setup.py install && cd ..
git clone https://github.com/ysBach/ysphotutilpy.git
cd ysphotutilpy && python setup.py install && cd ..
git clone https://github.com/ysBach/NICpolpy.git
cd NICpolpy && python setup.py install && cd ..
</pre>
</details>

### 1-2. After the first
If you need to update any of these, just do
```
cd ~/github/ysfitsutilpy && git pull && python setup.py install
cd ~/github/ysphotutilpy && git pull && python setup.py install
cd ~/github/NICpolpy && git pull && python setup.py install
```



## 2. Basic Usage
The following will require **< 100 MB memory** for processing.

The speed is roughly **``100 ± 10 frames/second``** on MBP 2018 15" (2.6 GHz i7), and CPU load was ~ **20-30 %**.

<details><summary><b>code</b> (click):</summary>
<p>
<pre>
from pathlib import Path
import nicpolpy as nic
import ysfitsutilpy as yfu
#%%
top = Path("your/folder/path/from_current_pwd_of_python_or_absolute_path_to_it")
cal = Path("calibrated/data/will_be_saved_in_this_folder")
Path.mkdir(cal, exist_ok=True, parents=True)
allfits = list(top.glob("*.fits"))
allfits.sort()
summary = yfu.make_summary(allfits,
                        keywords=nic.USEFUL_KEYS,
                        pandas=True,
                        output=f"{top.name}.csv")
#%%
object_name = 'Vesta'
object_mask = summary['OBJECT'].str.lower() == object_name.lower()
obj_fpaths = summary[object_mask]['file'].values
#%%
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
#%%
# multiple image example:
for fpath in obj_fpaths:
    fpath = Path(fpath)
    out = [cal/f"{fpath.stem}_o.fits", cal/f"{fpath.stem}_e.fits"]
    #
    nicimg = nic.NICPolImage(fpath, verbose=False)
    nicimg.preproc(do_fouri=False, verbose_crrej=False, verbose=False)
    nicimg.find_obj(thresh=1, verbose=False)
    nicimg.ellipphot_sep(f_ap=(2, 2), fwhm=(11., 11.), fix_fwhm=False, verbose=False)
    #
    for outpath, ccd in zip(out, [nicimg.ccd_o_proc, nicimg.ccd_e_proc]):
        ccd.write(outpath, overwrite=True)
</pre>
</p>
</detail>

I tried my best to put the most detailed log to the FITS header, so *please refer to the verbose output as well as FITS header*.

<details><summary> <b>Also try</b> (click):</summary>
<p>
<pre>
for k in list(vars(nicimg).keys()):
    print(k)
# Try such as
nicimg.ccd_o_bdfx.write('test_bdfx.fits', overwrite=True)
</pre>
</p>
</detail>

All the preprocessing intermediate results are stored, with appropriate header information.



## Note
Some data from NHAO NIC is in 32-bit format, using twice the storage than required. You may use the following snippet to **convert those into 16-bit** without losing any dynamic range.

<details><summary> <b>code</b> (click):</summary>
<p>
<pre>
from pathlib import Path
import numpy as np
from astropy.io import fits
import ysfitsutilpy as yfu
import nicpolpy as nic
#%%
top = Path("folder/where/your_data_is_stored")
out = Path("output/path")
allfits = list(top.glob("**/*.fits"))
#allfits = (top/"4_Vesta_20191218_NHAO_NIC/").glob('*.fits')
#%%
for fpath in allfits:
    # select only raw data
    if ".pcr." in str(fpath):
        continue
    ccd_32bit = yfu.load_ccd(fpath)
    ccd_16bit = ccd_32bit.copy()
    ccd_16bit = yfu.CCDData_astype(ccd_16bit, dtype='int16')
    outpath = out/f"{fpath.name}"
    # Or you can tune like this:
    # outname = (f"{fpath.stem}"
    #            + f"_{ccd_16bit.header['OBJECT']}"
    #            + f"_{ccd_16bit.header['EXPTIME']:.1f}.fits")
    # try:
    #     counter = fpath.stem.split('_')[1]
    # except IndexError:
    #     counter = None
    # ccd_16bit.header['COUNTER'] = (counter, "Image counter of the day, 1-indexing")
    # outpath = Path(*fpath.parts[6:-1])/outname
    # try:
    #     ccd_16bit.header["MJD-STR"] = float(ccd_16bit.header["MJD-STR"])
    #     ccd_16bit.header["MJD-END"] = float(ccd_16bit.header["MJD-END"])
    # except KeyError:
    #     pass
    try:
        ccd_16bit.write(outpath, overwrite=True)
    except FileNotFoundError:
        outpath.parent.mkdir(parents=True)
        ccd_16bit.write(outpath, overwrite=True)
</pre>
</p>
</detail>

