import numpy as np
import math

""" Generates sources from the Bethermin et al. 2012 model"""

__all__ = ["gencat"]

from schecter import mass_schecter
from zdist import zdist

class gencat:
    """ Generates catalog sources from Bethermin et al. 2012 model"""
    
    def __init__(self, log10Mb: 'log 10 of Mb, in solar masses'=11.2,
                 alpha: 'power-law slope of low mass end'=1.3,
                 log10Mmin: 'log10 minimum mass, in solar masses'=9.5,
                 log10Mmax: 'log10 maximum mass, in solar masses'=12.75,
                 ninterpm: 'Number of mass interplation samples to use'=2000,
                 zmin:'Minimum z generated'=0.5,
                 zmax: 'Maximum z generated'=7.0,
                 Om0: 'Density parameter of ordinary matter'=0.315,
                 H0: 'Hubble constant in km/sec/Mpc'=67.7,
                 phib0: 'log10 number density at SFMF break'=-3.02,
                 gamma_sfmf: 'Evolution of density of SFMF at z>1'=0.4,
                 ninterpz: 'Number of interpolation samples to use in z'=1000,
                 rsb0: 'Relative amplitude of SB distribution'=0.012,
                 gammasb: 'Redshift evolution of SB amplitude'=1.0,
                 zsb: 'Redshift where SB fraction stops evolving'=1.0,
                 logsSFRM0: 'Base log10 specific star formation'=-10.2,
                 betaMS: 'Slope of sFRM-M relation'=-0.2,
                 zevo: 'Redshift where MS normalization stops changing'=2.5,
                 gammams: 'Redshift evolution of sSFR'=3.0,
                 bsb: 'Boost if sSFR for starbursts, in dex'=0.6,
                 sigmams: 'Width of MS log-normal distribution'=0.15,
                 sigmasb: 'Width of SB log-normal distribution'=0.2):

        self._rsb0 = float(rsb0)
        self._gammasb = float(gammasb)
        self._zsb = float(zsb)
        self._interp = int(ninterpz)
        self._logsSFRM0 = float(logsSFRM0)
        self._betaMS = float(betaMS)
        self._zevo = float(zevo)
        self._gammams = float(gammams)
        self._bsb = float(bsb)
        self._sigmams = float(sigmams)
        self._sigmasb = float(sigmasb)

        self._sch = mass_schecter(log10Mb, alpha, log10Mmin, log10Mmax,
                                  ninterpm)
        self._zdist = zdist(zmin, zmax, Om0, H0, phib0, gamma_sfmf, ninterpz)

        # Set number per sr
        self._npersr = self._zdist.dVdzdOmega * self._sch.dNdV

    @property
    def npersr(self):
        return self._npersr

    def random(self, ngen):
        """ Generates samples from the Bethermin 2012 model.

        Returns a tuple of (z, log10 M, is_starburst, log10 sSFR),
        each of which is a ngen element ndarray."""

        log10mass = self._sch.random(ngen)
        z = self._zdist.random(ngen)

        # Figure out if each source is a starburst
        rsb = (1.0 + self._zsb)**self._gammasb * np.ones(ngen)
        w = np.nonzero(z < self._zsb)[0]
        if len(w) > 0:
            rsb[w] = (1.0 + z[w])**self._gammasb
        rsb *= self._rsb0
        is_starburst = np.zeros(ngen, dtype=np.uint8)
        w = np.nonzero(np.random.rand(ngen) <= rsb / (1.0 + rsb))[0]
        if len(w) > 0:
            is_starburst[w] = 1
        del rsb

        # Figure out sSFR for each source.  These are gaussian -- just
        # times different numbers and means depending on whether they
        # are a SB, plus redshift, plus mass
        # Redshift evolution of MS value

        logsSFRM = self._logsSFRM0 + self._betaMS * (log10mass - 11.0)
        w = np.nonzero(z < self._zevo)[0]
        if len(w) > 0:
            logsSFRM[w] += self._gammams * np.log10(1.0 + z[w])
        w = np.nonzero(z >= self._zevo)[0]
        if len(w) > 0:
            logsSFRM[w] += self._gammams * math.log10(1.0 + self._zevo)

        # Setup sigmas as well
        sigmas = self._sigmams * np.ones(ngen)
        w = np.nonzero(is_starburst)[0]
        if len(w) > 0:
            logsSFRM[w] += self._bsb
            sigmas[w] = self._sigmasb

        # Actual generation of log sSFR
        log10sSFR = logsSFRM + sigmas * np.random.randn(ngen)
        del logsSFRM
        del sigmas
        del w

        return (z, log10mass, is_starburst, log10sSFR)

        
        
        
