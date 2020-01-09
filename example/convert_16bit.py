from pathlib import Path

import ysfitsutilpy as yfu

# %%
top = Path("folder/where/your_data_is_stored")
out = Path("output/path")
allfits = list(top.glob("**/*.fits"))
# allfits = (top/"4_Vesta_20191218_NHAO_NIC/").glob('*.fits')
# %%
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
