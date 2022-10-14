# NICpolpy
 (ysbach93@gmail.com)

Nishi-Harima Astronomical Observatory (NHAO)'s Near-Infrared Camera (NIC) Polarimetric mode data reduction pipeline (image preprocessing, excluding photometry at the moment). Under-development by Yoonsoo P. Bach at Seoul National University, South Korea, since late 2019.


## 0. Installation
To use this package, you need to have the pre-made master flat and initial mask frames. They are downloadable at [this repository](https://github.com/ysBach/nicpolpy_sag22sm). There, you can also find the example usage of ``NICpolpy``.

You will need Python **3.7+** (recommended: **3.8+**, [Anaconda 3](https://www.anaconda.com/distribution/#download-section)). You also need the following packages:
* numpy
* scipy
* bottleneck
* astroscrappy
* sep
* astroquery
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

## 1. Descriptions


For detailed descriptions about image reduction steps, please refer to BachYP et al. (2022) SAG (Stars And Galaxies), in prep (you may freely contact via email above). Below are simple summary of that publication.

### 1-1. A Short Summary of Data Reduction Steps
Few things special for NHAO NIC polarimetric mode:

1. **MASK**s are assumed to be present prior to any data reduction.
2. **FLAT**s are assumed to be present prior to any data reduction. FLATs are not taken every night. It is taken only rarely, so the majority of this package is assuming you already have the master flats for each FILTER.
3. **DARK**s are not taken every night. Sometimes it's missing, but sometimes you have DARK frames. Thus, the code has a flexibility for the user to combine nightly DARK.
4. Unfortunately, dark current on hot pixels often do not follow linear law (pixel value is not proportional to EXPTIME). Therefore, it is best to simply mask such pixels and interpolate the pixel value based on nearby ones, rather than relying on DARK frames.
5. As NIC has NIR sensor, temperature is critical to hot pixels. Thus a difficulty is that the locations/severity of such "bad" pixels may vary not only over time, but also on the efficiency of the cooling system. Although it's rare, the system *can* suffer from cooling problem, and therefore, the MASK frames must be differ on such nights (this can even be seen from visual inspection).

The data nomenclature (``lv`` means "level"):
1. ``lv0``: The very raw data (32-bit int, not 16-bit; so wasting double the storage, unfortunately..).
2. ``lv1``: After vertical pattern subtraction (32-bit int)
3. ``lv2``: Fourier pattern removal. (32-bit float)
   - ``lv2`` is *the* **"raw" data**, if it were not for those artificial patterns.
   - Thus, now the remaining reduction processes are similar to usual observations.
4. ``lv3``: DARK/FLAT correction and FIXPIX using MASK frames. The nominal "preprocessed" image (32-bit float).
5. ``lv4``: After CR rejection and FRINGE subtraction, (32-bit float).
    - Rarely, CR rejection corrupts the image severely by detecting too many cosmic rays (see CRNPIX in the header).
    - The sky in IR (JHK bands) can change rather quickly, so that the fringe subtraction may only increase the artifact. Also, fringe subtraction has only marginal effect in the final Stokes' parameter (BachYP+2022, in prep). Thus, we recommend skip the fringe subtraction.

<details><summary><u>A note (click)<\u></summary>
<p>

Below is just an idea, not actually implemented:
- In the vertical pattern subtraction by median value along the column, the output may contain integer + 0.5 pixel value. Meanwhile, NIC has saturation at well below 10k ADU, and therefore, the range of ``-32,768`` to ``32,767`` is more than enough to store all meaningful data. Combining these two information, `NICpolpy` **multiplies 2** to the vertical-pattern-subtracted images, and store it as `int16` to save storage by half for this intermediate data. Just in case, by default, any pixel larger than 15000 (`maxval`) or smaller than -15000 (`minval`) will be replaced by -32768 (`blankval` or ``"BLANK"`` in FITS header).


</p>
</details>

### 1-2. A Short Summary of the Output Files/Directories
After reduction, you may freely remove LV1 and 2 data to save your storage. They are intermediate data produced just in case. A size of single FITS frame is
- ``lv0``: 4.2 MB
- ``lv1``: 4.2 MB
- ``lv2``: 4.2 MB
- ``lv3``: 280 kB * 2 = 560 kB (o-/e-ray splitted)
- ``lv4``: 280 kB * 2 = 560 kB (o-/e-ray splitted)
- ``logs/``: 12.8 MB (MFLAT) + 3.3 MB (IMASK) + [~ 15 MB/DARK_EXPTIME] + [~3.3 MB/DARKMASK] + something more...
In total, the log directory (by default ``__logs/``) will be likely \~ 50 MB. For 10-set observation at NIC, i.e., 40 frames per filter = 120 FITS frames, will have LV0 \~ LV1 \~ LV2 \~ 0.5 GB, LV3 \~ LV4 \~ 0.1 GB thus in total, <\~ 2 GB.

Names:
* mdark, mflat, mfrin : master dark, flat, fringe
* ifrin : the initial fringes (LV3, i.e., after dark/flat corrections). The master fringe is the combination of the ifrin frames.
* imask, dmask, mmask : initial input mask (given a priori), dark mask (based on nightly dark frames), master mask (made by combining imask and nightly dark, if exists)

