import numpy as np


def _profile_moffat(r, flux=1, width=1, power=1):
    ''' The circular Moffat profile (Students' t-distribution variant)
    Paramters
    ---------
    r : float
        The distance from the profile center in pixel.
    flux : float, optional.
        The total flux.
    width, power : float, optional.
        The width and power parameters for the profile, respectively.
        ``power`` is the ``beta`` value in IRAF or ``alpha`` value in
        astropy. Thus, it must be a positive value, as the ``-power``
        will be used for the profile.
    '''
    const = (power - 1)/(np.pi*width**2)  # normalization constant
    return flux * const * (1 + (r/width)**2)**(-power)


def _profile_gauss(r, flux=1, sigma=1):
    ''' The circular Gaussian profile.
    Paramters
    ---------
    r : float
        The distance from the profile center in pixel.
    flux : float, optional.
        The total flux.
    sigma : float, optional.
        The standard deviation parameter for the Gaussian function.
    '''
    const = 1 / (2*np.pi*sigma**2)  # normalization constant
    return flux * const * np.exp(-r**2/(2*sigma**2))


def _profile_exp(r, flux=1, r0=1):
    ''' The exponential profile for DAOGROW.
    Paramters
    ---------
    r : float
        The distance from the profile center in pixel.
    flux : float, optional.
        The total flux.
    r0 : float, optional.
        The scaled length (``DR_i``) of the exponential profile.
    '''
    const = 1 / (2*np.pi*r0**2)  # normalization constant
    return flux * const * np.exp(-r/r0)


def _profile_moffat_i(r, flux, width=1, power=1):
    ''' The 2-pi-r integrated Moffat profile. See _profile_moffat.
    '''
    return flux*(1 - 1/(1 + r**2)**(power - 1))


def _profile_gauss_i(r, flux, sigma=1):
    ''' The 2-pi-r integrated Gauss profile. See _profile_gauss.
    '''
    return flux*(1 - np.exp(-r**2/(2*sigma**2)))


def _profile_exp_i(r, flux, r0=1):
    ''' The 2-pi-r integrated exponential profile. See _profile_exp.
    '''
    return flux*(1 - (1 + r/r0)*np.exp(-r/r0))


def profile(r, airmass, a, b, c, width=1, power=1, sigma=1, r0=1):
    ''' The azimuthally averaged DAOGROW stellar profile.
    Parameters
    ----------
    r : float
        The distance from the profile center in pixel.
    airmass : float
        The airmass of the i-th frame, ``X_i``.
    '''
    fraction = a + b*airmass
    moffat_i = _profile_moffat_i(r, flux=1, width=width, power=power)
    gauss_i = _profile_gauss_i(r, flux=1, sigma=sigma)
    exp_i = _profile_exp_i(r, flux=1, r0=r0)
    return fraction*moffat_i + (1 - fraction)*(c*gauss_i + (1 - c)*exp_i)


def dmag(r1, r2, airmass, a, b, c, width=1, power=1, sigma=1, r0=1):
    """
    Parameters
    ----------
    r1, r2 : float
        The inner and outer radii in pixel.
    airmass : float
        The airmass of the i-th frame, ``X_i``.
    """
    params = dict(a=a, b=b, c=c, width=width, power=power, sigma=sigma, r0=r0)
    prof1 = profile(r1, airmass, **params)
    prof2 = profile(r2, airmass, **params)
    delta_mag = -2.5*np.log10(prof2/prof1)  # = m1 - m2
    return delta_mag
