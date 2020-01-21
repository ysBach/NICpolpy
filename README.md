# NICpolpy
Nishi Harima Astronomical Observatory (NHAO)'s Near-Infrared Camera (NIC) Polarimetry photometry pipeline. Under-development by Yoonsoo P. Bach at Seoul National University, South Korea, since late 2019.



## 1. Installation
You will need Python **3.6+** (recommended: [Anaconda 3](https://www.anaconda.com/distribution/#download-section)).

<details><summary>For the <b>first</b> installation (click):</summary>
<p>
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
All the preprocessing intermediate results are stored, with appropriate header information.

The processing with ``NICpolpy``:

* **Memory ≲ 100 MB** for processing
* Speed:
  * If Fourier pattern subtraction is off: **Speed = 100 ± 10 FITS frames/second** (CPU load = **20-30 %**)
  * If Fourier pattern subtraction is on: **Speed = 36 ± 3 FITS frames/second** (CPU load = **35-55 %**)
* You don't even have to provide dark/flat (you can if you wish), because I inherited the flat(2018-05-07) and dark(2019-10-22) to the package itself, as well as bad pixel mask.
* Example code for image preprocessing & photometry : [here](example/preproc_and_phot.py).

(Tested on MBP 2018 15" 2.6 GHz i7 with Fourier pattern subtraction multithread number of pools = 5)


Also *please refer to the verbose output as well as FITS header* such as the one at the **end of this README**.



## A Note on Image Size
Some data from NHAO NIC is in 32-bit format, using twice the storage than required. You may use [this example code](example/convert_16bit.py) to **convert those into 16-bit** without losing any dynamic range.



## Results

The photometry was done by automatic aperture/annulus selection (example of h191108_0400):

![img](file:///Users/ysbach/Dropbox/github/NICpolpy/readme_images/ap_selection.png?lastModify=1579595802)

*left: o-ray, right: e-ray. The red is the elliptical aperture and the white is the elliptical annulus for sky. The FWHM is estimated from Source-Extractor-like extraction, and aperture radius = 2FWHM, annulus radii = (4FWHM, 6FWHM) for both x(major) and y(minor) axes of the source's shape.*



**Note the long description in the ``HISTORY``**. There is virtually no need to write any other lab-notes or the descriptions for what the codes do.

``k200110_0066.fits``, Vesta, ``EXPTIME = 2.0``, o-ray, Fourier pattern subtracted:

```
SIMPLE  =                    T / conforms to FITS standard                      
BITPIX  =                  -32 / array data type                                
NAXIS   =                    2 / number of array dimensions                     
NAXIS1  =                  141                                                  
NAXIS2  =                  421                                                  
COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy
COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H 
COMMENT ['PROCESS key can be B (bias), D (dark), F (flat), T (trim), W (WCS astr
COMMENT ometry), C(CRrej).']                                                    
COMMENT .................................................2020-01-21T08:05:10.734
HISTORY  Copy of image k200110_0066.fits flipped, rotated 180 degrees           
HISTORY From the user, gain = 9.4 [unit = electron / adu]                       
HISTORY ..................................(dt = 0.001 s) 2020-01-21T08:05:10.663
HISTORY From the user, rdnoise = 83.0 [unit = electron]                         
HISTORY ..................................(dt = 0.001 s) 2020-01-21T08:05:10.665
HISTORY Dark scaling True using EXPTIME                                         
HISTORY .................................................2020-01-21T08:05:10.770
HISTORY Dark subtracted (see DARKPATH)                                          
HISTORY ..................................(dt = 0.023 s) 2020-01-21T08:05:10.797
HISTORY Vertical pattern subtracted using ['[:, 50:150]', '[:, 874:974]'] by tak
HISTORY ing median with sigma-clipping in astropy (v 4.0), given {'sigma': 2, 'm
HISTORY axiters': 5}.                                                           
HISTORY ..................................(dt = 0.044 s) 2020-01-21T08:05:10.854
HISTORY Trimmed using [300:500, :]                                              
HISTORY ..................................(dt = 0.003 s) 2020-01-21T08:05:10.867
HISTORY Fourier frequencies obtained from [300:500, :] by sigma-clip with {'sigm
HISTORY a_lower': inf, 'sigma_upper': 3} minimum frequency = 0.01 and only upto 
HISTORY maximum 3 peaks. See FIT-AXIS, FIT-NF, FIT-Fiii.                        
HISTORY ..................................(dt = 0.021 s) 2020-01-21T08:05:10.884
HISTORY Fourier series fitted for x positions (in FITS format) [513:900], using 
HISTORY y sections (in FITS format) ['[10:280]', '[830:1010]'].                 
HISTORY ..................................(dt = 1.429 s) 2020-01-21T08:05:12.315
HISTORY The obtained Fourier pattern subtracted                                 
HISTORY ..................................(dt = 0.005 s) 2020-01-21T08:05:12.324
HISTORY Flat corrected (see FLATPATH)                                           
HISTORY ..................................(dt = 0.005 s) 2020-01-21T08:05:12.431
HISTORY Pixel error calculated for both o/e-rays                                
HISTORY ..................................(dt = 0.072 s) 2020-01-21T08:05:12.556
HISTORY Cosmic-Ray rejected by astroscrappy (v 1.0.8), with parameters: {'gain':
HISTORY  9.4, 'readnoise': 83, 'sigclip': 4.5, 'sigfrac': 5, 'objlim': 5, 'satle
HISTORY vel': 65535.0, 'pssl': 0.0, 'niter': 4, 'sepmed': True, 'cleantype': 'me
HISTORY dmask', 'fsmode': 'median', 'psfmodel': 'gauss', 'psffwhm': 2.5, 'psfsiz
HISTORY e': 7, 'psfk': None, 'psfbeta': 4.765}                                  
HISTORY ..................................(dt = 0.041 s) 2020-01-21T08:05:12.599
HISTORY Background estimated from sep (v 1.10.0) with {'maskthresh': 0.0, 'filte
HISTORY r_threshold': 0.0, 'box_size': (64, 64), 'filter_size': (12, 12)}.      
HISTORY ..................................(dt = 0.002 s) 2020-01-21T08:05:12.650
HISTORY Objects found from sep (v 1.10.0) with {'thresh': 1, 'minarea': 100, 'de
HISTORY blend_cont': 1, 'bezel_x': (40, 40), 'bezel_y': (200, 120)}.            
HISTORY ..................................(dt = 0.023 s) 2020-01-21T08:05:12.675
HISTORY Photometry done for elliptical aperture/annulus with f_ap = (2, 2), f_in
HISTORY  = (4.0, 4.0), f_out = (6.0, 6.0) for FWHM = (10.764, 9.973)            
HISTORY ..................................(dt = 0.058 s) 2020-01-21T08:05:12.757
OBSERVAT= 'NHAO    '           / Observatory                                    
LATITUDE= '+35:01:31'          / Latitude of the Site                           
LONGITUD= '134:20:08'          / Longitude of the Site                          
HEIGHT  =                  449 / [m] Altitude of the Site                       
TELESCOP= 'NAYUTA 2m'          / Telescope Name                                 
INSTRUME= 'NIC     '           / Instrument Name                                
TELINFO = '2020-01-10 22:29:41.796' / timestamp of Telescope.inf                
FOC-VAL =               -4.997 / Encoder value of the focus unit(mm)            
OBSERVER= 'MAINT   '           / Name of observers                              
DEC     = '+09:56:21.89'       / DEC of pointing(+/-DD:MM:SS.SS)                
DEC2000 = '+09:51:22.75'       / DEC(J2000) of pointing(+/-DD:MM:SS.SS)         
EQUINOX =               2020.0 / Standard FK5(year)                             
OBJECT  = 'Vesta   '           / Object Name                                    
RA      = '02:47:47.288'       / RA of telescope pointing                       
RA2000  = '02:46:42.596'       / RA(J2000) of telescope pointing(HH:MM:SS.SS)   
AIRMASS =                1.476 / Typical air mass during exposure               
AIRM-STR=                1.476 / Airmass at the observation start time          
SECZ    =                1.478 / SEC(Zenith Distance) at typical time           
ZD      =             47.41840 / Zenith distance at typical time(degree)        
AG-POS1 =                      / AG Reference Position X                        
AG-POS2 =                      / AG Reference Position Y                        
AG-PRB1 =                      / AG Probe Position X                            
AG-PRB2 =                      / AG Probe Position Y                            
ALTITUDE=             42.58160 / Altitude of telescope pointing(degree)         
AUTOGUID= 'OFF     '           / Auto Guide ON/OFF                              
AZIMUTH =             69.56264 / Azimuth of telescope pointing(degree)          
PA      =                  0.0 / Position angle                                 
INSROT  =               50.969 / Typical inst rot. Angle ar exp.(degree)        
IMGROT  =              134.995 / Angle of the Image Rotator(degree)             
M2-POS1 =               -1.200 / X-position of the M2(mm)                       
M2-POS2 =               -1.770 / Y-position of the M2(mm)                       
DOM-HUM =                 50.0 / Humidity measured in the dome                  
DOM-TMP =                 7.50 / Temperature measured in the dome(deg)          
OUT-HUM =                 77.0 / Humidity measured outside of the dome          
OUT-TMP =                 2.50 / Temperature measured outside the dome(deg)     
OUT-WND =                 1.20 / Wind velocity outside of the dome(m/s)         
WEATHER = '        '           / Weather condition                              
TEXINFO = 'Fri Jan 10 22:29:42 JST 2020' / timestamp of extra telescope info    
DF_COUNT=                    0 / [%] dome flat brightness                       
NICTMP1 =               +79.35 / [K] NIC temperature 1 J                        
NICTMP2 =               +75.61 / [K] NIC temperature 2 H                        
NICTMP3 =               +75.25 / [K] NIC temperature 3 K                        
NICTMP4 =               +77.54 / [K] NIC temperature 4 Opt bench                
NICTMP5 =               +52.83 / [K] NIC temeprature 5 Cold tip                 
NICHEAT =                +0000 / NIC heater output, 0 to 1000                   
OBS_CMD = 'gPLo    '           / NIC Obs command (TL/Lo/DL/PLo)                 
DATA-TYP= 'OBJECT  '           / Charactersitics of this data                   
MODE    =                    3 / Exposure mode                                  
EXP_J   =                    2 / [sec] J band exposure time                     
EXP_H   =                    2 / [sec] H band exposure time                     
EXP_K   =                    2 / [sec] K band exposure time                     
SHUTTER = 'pol     '           / shutter position, open/close/pol               
POL-AGL1=                 67.5 / [deg] waveplate rotation angle                 
WAVEPLAT= 'in      '           / waveplate status (in/out)                      
DATE    = '2020-01-10T13:29:44' / YYYY-mm-ddThh:mm:ss UTC at the sart of readout
DATE_UTC= '2020-01-10'         / YYYY-mm-dd UTC at the start of readout_x_xM.sh 
TIME_UTC= '13:29:44.751'       / hh:mm:ss UTC at the start of readout_x_xM.sh   
DATE_LT = '2020-01-10'         / YYYY-mm-dd JST at the start of readout_x_xM.sh 
TIME_LT = '22:29:44.751'       / hh:mm:ss JST at the start of readout_x_xM.sh   
DATE-OBS= '2019-10-21'         / ISO-8601 time of observation                   
UT-STR  = '13:29:46.751'       / Est. UTC at the start of exposure (first read) 
MJD-STR =          58858.56235 / Exp. start MJD, converted from Unix Time       
UT-END  = '13:29:50.901'       / Est. UTC at the end of exposure (final read)   
MJD-END =          58777.96051 / [d] MJD at end of observation                  
POINTINF= 'Fri Jan 10 22:29:39 JST 2020' / timestamp of pointing info           
RA_OFF  =           -45.960000 / [arcsec] Ra offset                             
DEC_OFF =           -24.769000 / [arcsec] DEC offset                            
DITH_NUM=                    0 / Total number of dithering                      
DITH_NTH=                    0 / N th exposure in dithering                     
DITH_RAD=                    0 / [arcsec] Dithering radius                      
FILTER  = 'K       '           / Filter Name                                    
FILTER00= 'K       '           / Filter Name                                    
EXPOS   =                    2 / [sec] Exposure Time                            
EXPTIME =                    2 / [sec] Exposure Time                            
FRAMEID = 'k200110_0066'       / Image sequential number                        
DET-ID  =                    3 / ID of the detector used for this data          
BIN-FCT1=                    1 / Binning factor of axis 1                       
BIN-FCT2=                    1 / Binning factor of axis 2                       
BUNIT   = 'adu     '           / Unit of original pixel value                   
COUNTER = '0066    '           / Image counter of the day, 1-indexing           
WCSAXES =                    2 / Number of coordinate axes                      
CRPIX1  =               -559.0 / Pixel coordinate of reference point            
CRPIX2  =               -344.0 / Pixel coordinate of reference point            
CDELT1  =                  1.0 / Coordinate increment at reference point        
CDELT2  =                  1.0 / Coordinate increment at reference point        
CRVAL1  =                  0.0 / Coordinate value at reference point            
CRVAL2  =                  0.0 / Coordinate value at reference point            
LATPOLE =                 90.0 / [deg] Native latitude of celestial pole        
MJD-OBS =              58777.0 / [d] MJD at start of observation                
MJD-OBS =              58858.0 / [d] MJD at start of observation                
GAIN    =                  9.4 / [electron / adu] Gain of the detector          
RDNOISE =                 83.0 / [electron] Readout noise of the detector       
PROCESS = 'DFC     '           / The processed history: see comment.            
CCDPROCV= '2.0.1   '           / ccdproc version used for processing.           
DARKPATH= '/Users/ysbach/anaconda3/lib/python3.7/site-packages/nicpolpy-0.1.de&'
CONTINUE  'v0-py3.7.egg/nicpolpy/dark/v_k191022_DARK_120.fits&'                 
CONTINUE  '' / Path to the used dark file                                       
FIT-AXIS= 'COL     '           / The direction to which Fourier series is fitted
FIT-NF  =                    3 / Number of frequencies used for fit.            
FIT-F001=          0.021484375 / [1/pix] The 001-th frequency                   
FIT-F002=          0.185546875 / [1/pix] The 002-th frequency                   
FIT-F003=             0.265625 / [1/pix] The 003-th frequency                   
HIERARCH TRIM_IMAGE = 'trimim  ' / Shortened name for ccdproc command           
TRIMIM  = 'ccd=<CCDData>, fits_section=[560:700, 345:765]'                      
LTV1    =                 -559                                                  
LTV2    =                 -344                                                  
LTM1_1  =                  1.0                                                  
LTM2_2  =                  1.0                                                  
OERAY   = 'o       '           / O-ray or E-ray. Either 'o' or 'e'.             
FLATPATH= '/Users/ysbach/anaconda3/lib/python3.7/site-packages/nicpolpy-0.1.de&'
CONTINUE  'v0-py3.7.egg/nicpolpy/flat/v_k180507_FLAT_all_o.fits&'               
CONTINUE  '' / Path to the used flat file                                       
NOBJ-SEP=                    1 / Number of objects found from SEP.              
DATE-END= '2019-10-21T23:03:08.064' / ISO-8601 time at end of observation       
END                                                                             
```