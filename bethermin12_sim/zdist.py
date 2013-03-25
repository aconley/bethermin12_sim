import numpy as np
import math

""" Redshift distribution for Bethermin et al. 2012 model.

This contains the comoving volume element and the phi_b(z) function.
"""

__all__ = ["zdist"]

class zdist:
    """ Volume element and z distribution"""

    def __init__(self, zmin:'Minimum z generated'=0.5,
                 zmax: 'Maximum z generated'=7.0,
                 Om0: 'Density parameter of ordinary matter'=0.315,
                 H0: 'Hubble constant in km/sec/Mpc'=67.7,
                 phib0: 'log10 number density at SFMF break'=-3.02,
                 gamma_sfmf: 'Evolution of density of SFMF at z>1'=0.4,
                 ninterp: 'Number of interpolation samples to use'=1000):
        from scipy.interpolate import interp1d
        from scipy.integrate import trapz
        from astropy.cosmology import FlatLambdaCDM

        self._zmin = float(zmin)
        self._zmax = float(zmax)
        if self._zmin == self._zmax:
            raise ValueError("No range between zmin and zmax")
        if self._zmin > self._zmax:
            self._zmin, self._zmax = self._zmax, self._zmin
        if self._zmin < 0.0:
            raise ValueError("zmin must be >= 0: %f" % self._zmin)
        self._Om0 = float(Om0)
        if self._Om0 <= 0.0:
            raise ValueError("Om0 must be positive: %f" % self._Om0)
        self._H0 = float(H0)
        if self._H0 <= 0.0:
            raise ValueError("H0 must be positive: %f" % self._H0)
        self._ninterp = int(ninterp)
        if self._ninterp <= 0:
            raise ValueError("Ninterp must be > 0: %d" % self._ninterp)
        self._phib0 = float(phib0)
        self._gamma_sfmf = float(gamma_sfmf)

        zvals = np.linspace(self._zmin, self._zmax, self._ninterp)

        # Cosmology bit
        c_over_H0 = 299792.458 / self._H0 # in Mpc
        cos = FlatLambdaCDM(H0=self._H0, Om0=self._Om0, Tcmb0=0.0,
                            Neff=0.0)
        # in comoving Mpc^3
        dvdzdomega = c_over_H0 * (1.0 + zvals)**2 *\
            cos.angular_diameter_distance(zvals)**2 /\
            np.sqrt((1.0 + zvals)**3 * self._Om0 - (1.0 - self._Om0))
        
        # Schecter evolution bit
        phi = self._phib0 * np.ones(self._ninterp, dtype=np.float64)
        wgt1 = np.nonzero(zvals > 1.0)[0]
        if len(wgt1) > 0:
            phi[wgt1] += self._gamma_sfmf * (1.0 - zvals[wgt1])
        
        # Combined
        comb = 10**phi * dvdzdomega

        # Needed to understand normalization
        self._dVdzdOmega = trapz(comb, x=zvals)

        # Form inverse cumulative array needed to generate samples
        cumsum = comb.cumsum()
        cumsum -= cumsum[0] # So that 0 corresponds to the bottom
        cumsum /= cumsum[-1] # Normalization -> 0-1 is full range
        self._interpolant = interp1d(cumsum, zvals, kind='cubic')

    def random(self, ngen: 'Number of samples to generate'):
        """ Generates z samples from redshift distribution"""
        return self._interpolant(np.random.rand(ngen)).astype(np.float32)

    @property
    def zmin(self):
        return self._zmin
    
    @property
    def zmax(self):
        return self._zmax

    @property
    def Om0(self):
        return self._Om0

    @property
    def H0(self):
        return self._H0

    @property
    def phib0(self):
        return self._phib0

    @property
    def gamma_sfmf(self):
        return self._gamma_sfmf

    @property
    def dVdzdOmega(self):
        """ This is actually dV / dz dOmega * phi(b)"""
        return self._dVdzdOmega
