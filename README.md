# NICpolpy
 (Currently it is under a very freqeunt and abrupt updates. Please contact me directly if you want to use it: ysbach93@gmail.com . We are planning to finalize it within 2022.)

Nishi Harima Astronomical Observatory (NHAO)'s Near-Infrared Camera (NIC) Polarimetry photometry pipeline. Under-development by Yoonsoo P. Bach at Seoul National University, South Korea, since late 2019.

Few things special for NHAO NIC polarimetric mode are that
1. **MASK**s are assumed to be present prior to any data reduction.
2. **FLAT**s are assumed to be present prior to any data reduction. FLATs are not taken every night. It is taken only rarely, so the majority of this package is assuming you already have the master flats for each FILTER.
3. **DARK**s are not taken every night. Sometimes it's missing, but sometimes you have DARK frames. Thus, the code has a flexibility for the user to combine nightly DARK.
4. Unfortunately, dark current on hot pixels often do not follow linear law (pixel value is not proportional to EXPTIME of DARK frames). Therefore, it is best to simply mask such pixels and interpolate the pixel value based on nearby ones, rather than relying on DARK frames.
5. As NIC has NIR sensor, temperature is critical to hot pixels. Thus a difficulty is that the locations/severity of such "bad" pixels may vary not only over time, but also on the efficiency of the cooling system. Although it's rare, the system does suffer from cooling problem, and therefore, the MASK frames must be differ on such nights (this can even be seen from visual inspection).

The data nomenclature (**May change later!!!**):
1. LV0: The very raw data (32-bit int, not 16-bit; so wasting double the storage, unfortunately..).
2. LV1: After vertical pattern subtraction (32-bit int)
3. LV2: Fourier pattern removal. (32-bit float; <- the **"raw" data** if it were not for artificial patterns...)
4. LV3: After DARK/FLAT correction and FIXPIX using MASK frames. The nominal "preprocessed" image (32-bit float).
5. LV4: After CR rejection, additional vertical pattern correction, FRINGE subtraction, (32-bit float).
    - Rarely CR rejection distorts the image severely by detecting too many cosmic rays (see CRNPIX in the header).
    - The sky in IR (JHK bands) can change rather quickly, so that the fringe subtraction may only increase the artifact.
    - Thus, I splitted LV4 from LV3.

(
below is just an idea, not actually implemented:
- N.B. In the vertical pattern subtraction by median value along the column, the output may contain integer + 0.5 pixel value. Meanwhile, NIC has saturation at well below 10k ADU, and therefore, the range of ``-32,768`` to ``32,767`` is more than enough to store all meaningful data. Combining these two information, `NICpolpy` **multiplies 2** to the vertical-pattern-subtracted images, and store it as `int16` to save storage by half for this intermediate data. Just in case, by default, any pixel larger than 15000 (`maxval`) or smaller than -15000 (`minval`) will be replaced by -32768 (`blankval` or ``"BLANK"`` in FITS header).
)

After reduction, you may freely remove LV1 and 2 data to save your storage. They are intermediate data produced just in case. A size of single FITS frame is
- LV0: 4.2 MB
- LV1: 4.2 MB
- LV2: 4.2 MB
- LV3: 280 kB * 2 = 560 kB (o-/e-ray splitted)
- LV4: 280 kB * 2 = 560 kB (o-/e-ray splitted)
- log: 12.8 MB (MFLAT) + 3.3 MB (IMASK) + [~ 15 MB/DARK_EXPTIME] + [~3.3 MB/DARKMASK]
In total, the log directory will be likely ~ 50 MB. For 10-set observation at NIC, i.e., 40 frames per filter = 120 FITS frames, will have LV0 ~ LV1 ~ LV2 ~ 0.5 GB, LV3 ~ LV4 ~ 0.1 GB thus in total, <~ 2 GB.

Names:
* mdark, mflat, mfrin : master dark, flat, fringe
* ifrin : the initial fringes (LV3, i.e., after dark/flat corrections). The master fringe is the combination of the ifrin frames.
* imask, dmask, mmask : initial input mask (given a priori), dark mask (based on nightly dark frames), master mask (made by combining imask and nightly dark, if exists)
* skip : Skip certain process if the output file (csv, FITS, etc) already exists. Otherwise, proceed making them.

## 1. Installation
You will need Python **3.8+** (recommended: [Anaconda 3](https://www.anaconda.com/distribution/#download-section)).

<details><summary>For the <b>first</b> installation (click):</summary>
<p>
<pre>
# On terminal
conda install -c astropy astroquery photutils ccdproc astroscrappy
conda install -c openastronomy sep
conda install -c conda-forge fitsio  # Windows may fail - please just ignore.
cd ~            # whatever directory you want
mkdir github    # whatever name you want
git clone https://github.com/ysBach/ysfitsutilpy.git
cd ysfitsutilpy && python setup.py install && cd ..
git clone https://github.com/ysBach/ysphotutilpy.git
cd ysphotutilpy && python setup.py install && cd ..
git clone https://github.com/ysBach/NICpolpy.git
cd NICpolpy && python setup.py install && cd ..
</p>
</pre>
</details>

### *After* the first installation
If you need to update any of these, just do
```
cd ~/github/ysfitsutilpy && git pull && python setup.py install
cd ~/github/ysphotutilpy && git pull && python setup.py install
cd ~/github/NICpolpy && git pull && python setup.py install
```



## 2. Basic Usage
TBD