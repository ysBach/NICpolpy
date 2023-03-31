# NICpolpy

[![DOI](https://zenodo.org/badge/219398338.svg)](https://zenodo.org/badge/latestdoi/219398338)

 (ysbach93@gmail.com)

ℹ️ For the most recent documentation, please always refer to [GitHub](https://github.com/ysBach/NICpolpy).

## What is this?
NHAO (Nishi-Harima Astronomical Observatory) has NIC (Near-Infrared Camera). On top of imaging mode, NIC has a three-filter (JHKs) simultaneous dual-beam polarimetric mode. This package is for the polarimetric mode data reduction pipeline (image preprocessing, excluding photometry at the moment). Under-development by @ysBach since late 2019...

## TL;DR
1. On terminal: ``$ pip install NICpolpy``
1. Download flat/mask from the [SM repo](https://github.com/ysBach/nicpolpy_sag22sm).

Then refer to:
* Practical usage example: [SM repo/example](https://github.com/ysBach/nicpolpy_sag22sm/tree/main/example).
* Theoretical/implementation details: [Bach Y. P. et al. (2022) SAG](http://www.nhao.jp/research/starsandgalaxies/05.html#2022J-4).

## Citation ✅
Please consider one or both of the following citation(s) (BibTeX):

1. ``NICpolpy`` Zenodo (when you just want to mention which package was used).
```
@software{nicpolpy_v013,
  author       = {Yoonsoo P. Bach},
  title        = {ysBach/NICpolpy: NICpolpy v0.1.3},
  month        = dec,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {publish},
  doi          = {10.5281/zenodo.7391454},
  url          = {https://doi.org/10.5281/zenodo.7391454}
}
```
2. The implementation details document ([SAG official website](http://www.nhao.jp/research/starsandgalaxies/05.html#2022J-4), peer-reviewed, non-SCI)
```
@ARTICLE{2022_SAG_NICpolpy,
       author = {{Bach}, Yoonsoo P. and {Ishiguro}, Masateru and {Takahashi}, Jun and {Geem}, Jooyeon},
        title = "{Data Reduction Process and Pipeline for the NIC Polarimetry Mode in Python, NICpolpy}",
      journal = {Stars and Galaxies (arXiv:2212.14167)},
     keywords = {methods: data analysis, methods: observational, techniques: image processing, techniques: polarimetric},
         year = 2022,
        month = dec,
       volume = {5},
          eid = {4},
        pages = {4},
archivePrefix = {arXiv},
       eprint = {2212.14167},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv221214167B},
}
```

-----
-----
**DETAILS**:

-----
-----

## 1. Installation
To use this package, you need to have the pre-made master flat and initial mask frames. They are downloadable at the [SM repo](https://github.com/ysBach/nicpolpy_sag22sm). There, you can also find the example usage of ``NICpolpy``.

Requirements:
* Python **3.7+** (recommended: **3.10**)
  * You may use [Anaconda 3](https://www.anaconda.com/distribution/#download-section).
* numpy
* scipy
* bottleneck
* astroscrappy
* sep
* astroquery
* tqdm
* pandas > 1.0
* astropy >= 5.0
* photutils >= 0.7
* ccdproc >= 1.3

Simply do

    $ pip install nicpolpy

or if you prefer conda install:
```
# On terminal
conda install -c astropy astropy astroquery photutils ccdproc astroscrappy
conda install -c openastronomy sep
conda install -c conda-forge fitsio  # Windows may fail - please just ignore.
pip install nicpolpy
```

## 2. Descriptions
For detailed descriptions about image reduction steps, please refer to [Bach Y. P. et al. (2022) SAG](http://www.nhao.jp/research/starsandgalaxies/05.html#2022J-4). Below are simple summary of that publication.

### 1-1. A Short Note
Few things special for NHAO NIC polarimetric mode:
1. **MASK** means the default bad-pixel map.
  * Assumed to be **present prior to any data reduction**.
2. **FLAT** means the master flat field image (normalized to 1)
  * Assumed to be **present prior to any data reduction**.
  * FLATs are not taken every night. It is taken only rarely, so the majority of this package is assuming you already have the master flats for each FILTER.
3. **DARK**s means the *nightly* dark frames.
  * Dark frames are not taken every night. It's often missing. Thus, the code has a flexibility for the user to combine nightly dark frames, use dark of different nights (by providing relative paths), or completely ignore dark subtraction process.
4. Unfortunately, dark current often do not follow linear law (pixel value is not proportional to EXPTIME). Therefore, it is best to simply mask hot dark pixels and interpolate the pixel value based on nearby ones at the last stage, rather than relying on DARK frames.
5. As dark current changes abruptly over the temperature, a difficulty is that the locations/severity of such "bad" pixels may vary not only over time, but also on the efficiency of the cooling system. Although it's rare, the system *can* suffer from cooling problem, and therefore, the MASK frames must be differ on such nights (this can even be seen from visual inspection). Sometimes the pixels should be masked are permanently changed.

* ``mdark``, ``mflat``, ``mfrin`` : master dark, flat, fringe
* ``ifrin`` : the initial fringes (LV3, i.e., after dark/flat corrections). The master fringe is the combination of the ifrin frames.
* ``imask``, ``dmask``, ``mmask`` : initial input mask (given a priori), dark mask (based on nightly dark frames), master mask (made by combining imask and nightly dark, if exists)

### 1-2. A Short Summary of Data Reduction Steps
(``lv`` means "level")
1. ``lv0``: The original, very raw data (32-bit int, not 16-bit; so wasting double the storage, unfortunately..).
2. ``lv1``: After vertical pattern subtraction (32-bit int)
3. ``lv2``: Fourier pattern removal. (32-bit float)
   - ``lv2`` is *the* **"raw" data**, if it were not for those artificial patterns.
   - Thus, now the remaining reduction processes are similar to usual observations.
4. ``lv3``: DARK/FLAT correction and FIXPIX using MASK frames. The nominal "preprocessed" image (32-bit float).
   - FIXPIX means the interpolation of pixels indicated by MASK. The name originates from IRAF.PROTO.FIXPIX task.
5. ``lv4``: After CR rejection and FRINGE subtraction, (32-bit float).
    - Rarely, CR rejection corrupts the image severely by detecting too many cosmic rays (see CRNPIX in the header). If such thing happens, you may want to either turn off CR rejection, or manually find the best parameters for the CR rejection.
    - The sky in IR (JHK bands) can change rather quickly, so that the fringe subtraction may only increase the artifact. Also, fringe subtraction has only marginal effect in the final Stokes' parameter (BachYP+2022, in prep). Thus, we recommend skip the fringe subtraction.

<details><summary><u>An idea (click)</u></summary>
<p>

Below is just an idea, not actually implemented:
- In the vertical pattern subtraction by median value along the column, the output may contain integer + 0.5 pixel value. Meanwhile, NIC has saturation at well below 10k ADU, and therefore, the range of ``-32,768`` to ``32,767`` is more than enough to store all meaningful data. Combining these two information, `NICpolpy` **multiplies 2** to the vertical-pattern-subtracted images, and store it as `int16` to save storage by half for this intermediate data. Just in case, by default, any pixel larger than 15000 (`maxval`) or smaller than -15000 (`minval`) will be replaced by -32768 (`blankval` or ``"BLANK"`` in FITS header).


</p>
</details>

### 1-3. A Short Summary of the Output Files/Directories
After reduction, you may freely remove LV1 and 2 data to save your storage. They are intermediate data produced just in case. A size of single FITS frame is
- ``lv0``: 4.2 MB
- ``lv1``: 4.2 MB
- ``lv2``: 4.2 MB
- ``lv3``: 280 kB * 2 = 560 kB (o-/e-ray splitted)
- ``lv4``: 280 kB * 2 = 560 kB (o-/e-ray splitted)
- ``logs/``: 12.8 MB (MFLAT) + 3.3 MB (IMASK) + [~ 15 MB/DARK_EXPTIME] + [~3.3 MB/DARKMASK] + something more...
In total, the log directory (by default ``__logs/``) will be likely \~ 50 MB. For 10-set observation at NIC, i.e., 40 frames per filter = 120 FITS frames, will have LV0 \~ LV1 \~ LV2 \~ 0.5 GB, LV3 \~ LV4 \~ 0.1 GB thus in total, <\~ 2 GB.


# TODO
* Refactor ``ysfitsutilpy`` and ``ysphotutilpy``
  * Currently, ``NICpolpy`` contains a snapshot of ``ysfitsutilpy`` and ``ysphotutilpy``, which is an undesirable way (especially because of the other package dependency). If there is a critical need for significantly updating NICpolpy in the future, I may re-implement these.