from photutils.detection import DAOStarFinder
from .util import FIND_KEYS
from astropy.stats import sigma_clipped_stats

__all__ = ['findstar']


def findstar(ccd, oeray, fwhm=10, threshold=None, thresh_sigma=3,
             sigclipkw={'sigma': 3, 'maxiters': 5, 'std_ddof': 1}, mask=None):

    if oeray not in ['o', 'e']:
        raise ValueError("oeray must be either 'o' or 'e'.")

    if threshold is None:
        avg, med, std = sigma_clipped_stats(ccd.data, **sigclipkw)
        threshold = med + thresh_sigma*std

    finder = DAOStarFinder(threshold, fwhm=fwhm, **FIND_KEYS[oeray])
    return finder(ccd.data, mask=mask)
