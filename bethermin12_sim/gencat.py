import numpy as np
import math

""" Generates sources from the Bethermin et al. 2012 model"""

__all__ = ["gencat"]

from .schecter import mass_schecter
from .zdist import zdist
from .seds import sed_model

class gencat:
    """ Generates catalog sources from Bethermin et al. 2012 model"""
    
    def __init__(self, log10Mb: 'log 10 of Mb, in solar masses'=11.2,
                 alpha: 'power-law slope of low mass end'=1.3,
                 log10Mmin: 'log10 minimum mass, in solar masses'=8.5,
                 log10Mmax: 'log10 maximum mass, in solar masses'=12.75,
                 ninterpm: 'Number of mass interplation samples to use'=2000,
                 zmin:'Minimum z generated'=0.1,
                 zmax: 'Maximum z generated'=10.0,
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
                 sigmasb: 'Width of SB log-normal distribution'=0.2,
                 mnU_MS0: '<U>_{MS,0}'=4.0, 
                 gammaU_MS0: 'gamma_{MS,0}'=1.3,
                 z_UMS: 'z_{<U>MS}'=2.0,
                 mnU_SB0: '<U>_{MS,0}'=35.0, 
                 gammaU_SB0: 'gamma_{MS,0}'=0.4,
                 z_USB: 'z_{<U>SB}'=3.1,
                 scatU: 'Scatter in U in dex'=0.2,
                 ninterpdl: 'Number of interpolation points in dl'=200):

        self._zmin = float(zmin)
        self._zmax = float(zmax)
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
        self._scatU = float(scatU) #dex
        self._scatUe = math.log(10.0) * self._scatU #param for lognormal
        self._mnUMS = float(mnU_MS0)
        self._gammaUMS = float(gammaU_MS0)
        self._zUMS = float(z_UMS)
        self._mnUSB = float(mnU_SB0)
        self._gammaUSB = float(gammaU_SB0)
        self._zUSB = float(z_USB)
        self._Om0 = float(Om0)
        self._H0 = float(H0)

        if self._mnUMS <= 0:
            raise ValueError("mnU_MS0 must be positive, not %f" % self._mnUMS)
        if self._zUMS < 0:
            raise ValueError("z_UMS must be non-negative, not %f" % self._zUMS)
        if self._mnUSB <= 0:
            raise ValueError("mnU_SB0 must be positive, not %f" % self._mnUSB)
        if self._zUSB < 0:
            raise ValueError("z_USB must be non-negative, not %f" % self._zUSB)

        self._sch = mass_schecter(log10Mb, alpha, log10Mmin, log10Mmax,
                                  ninterpm)
        self._zdist = zdist(self._zmin, self._zmax, self._Om0, self._H0, 
                            phib0, gamma_sfmf, ninterpz)
        self._ms = sed_model(zmin=self._zmin, zmax=self._zmax, Om0=self._Om0,
                             H0=self._H0, ninterp=ninterpdl)

        # Set number per sr
        self._npersr = self._zdist.dVPhidzdOmega * self._sch.dNdV

    @property
    def npersr(self):
        return self._npersr

    def random(self, ngen, wave=None):
        """ Generates samples from the Bethermin 2012 model.

        Returns a tuple of (z, log10 M, is_starburst, log10 sSFR),
        each of which is a ngen element ndarray.  If wave is
        not None, will also generate flux densities for each source."""

        log10mass = self._sch.random(ngen)
        z = self._zdist.random(ngen)

        # Figure out if each source is a starburst
        rsb = (1.0 + self._zsb)**self._gammasb * np.ones(ngen)
        w = np.nonzero(z < self._zsb)[0]
        if len(w) > 0:
            rsb[w] = (1.0 + z[w])**self._gammasb
        rsb *= self._rsb0
        is_starburst = np.zeros(ngen, dtype=np.uint8)
        w = np.nonzero(np.random.rand(ngen) < rsb/(rsb + 1.0))[0]
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

        if not wave is None:
            nwave = len(wave)
            fluxes = np.empty((ngen, nwave), dtype=np.float32)
            kfac = math.log10(1.7e-10) # Kennicutt '98 conversion

            # Get log10 lir; note I'm assuming the r1500 business
            # only applies to SBs, not MSs
            log10lir = (log10mass + log10sSFR - kfac).astype(np.float32)

            # Do starbursts
            wsb = np.nonzero(is_starburst)[0]
            nsb = len(wsb)
            if nsb > 0:
                # Get the U (mean radiation field)
                u = 1.0 + z[wsb]
                np.copyto(u, self._zUSB, where=u > 1.0+self._zUSB)
                u **= self._gammaUSB
                u *= self._mnUSB
                # Add scatter to U
                if self._scatU > 0.0:
                    u *= np.random.lognormal(sigma=self._scatUe, size=(nsb))
                    
                # Deal with extinction effects on L_IR
                # coeff values are from eq 7 of B12 * 0.4 (from eq 8)
                pow_r1500 = 10**(1.628 * log10mass[wsb] - 15.728)
                fsf = pow_r1500 / (1.0 + pow_r1500)
                log10lir[wsb] += np.log10(fsf).astype(np.float32)

                for idx in range(nsb):
                    cidx = wsb[idx]
                    fluxes[cidx,:] =\
                        self._ms.get_fluxes(wave, z[cidx], u[idx], True,
                                            log10lir=log10lir[cidx])
            del wsb

            # Do MS
            wms = np.nonzero(~is_starburst)[0] # ~ is bitwise negation
            nms = len(wms)
            if nms > 0:
                u = 1.0 + z[wms]
                np.copyto(u, self._zUMS, where=u > 1.0+self._zUMS)
                u **= self._gammaUMS
                u *= self._mnUSB
                if self._scatU > 0.0:
                    u *= np.random.lognormal(sigma=self._scatUe, size=(nms))
                    
                for idx in range(nms):
                    cidx = wms[idx]
                    fluxes[cidx,:] =\
                        self._ms.get_fluxes(wave, z[cidx], u[idx], False,
                                            log10lir=log10lir[cidx])
            del wms
            return (z, log10mass, is_starburst, log10sSFR, log10lir, fluxes)
        else:
            return (z, log10mass, is_starburst, log10sSFR)

        
        
        
