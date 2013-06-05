import numpy as np
import math
import astropy.io.fits as fits
from astropy.nddata import convolve
from astropy.nddata.convolution.make_kernel import make_kernel


""" Generates simulated maps"""

__all__ = ["genmap"]

from .gencat import gencat

def get_beam(fwhm, pixscale, oversamp):
    """ Generate Gaussian kernel"""

    if fwhm <= 0:
        raise ValueError("Invalid (negative) FWHM")
    if pixscale <= 0:
        raise ValueError("Invalid (negative) pixel scale")
    if oversamp < 1:
        raise ValueError("Invalid (<1) oversampling")

    retext = round(fwhm * 5.0 / pixscale)
    if retext % 2 == 0:
        retext += 1

    bmsigma = fwhm / math.sqrt(8 * math.log(2))

    if oversamp == 1:
        # Easy case
        beam = make_kernel((retext, retext), bmsigma / pixscale, 
                           'gaussian').astype(np.float32)
        beam /= beam.max
    else:
        genext = retext * oversamp
        genpixscale = pixscale / oversamp
        gbeam = make_kernel((genext, genext), bmsigma / genpixscale, 
                           'gaussian').astype(np.float32)
        gbeam /= gbeam.max() # Normalize -before- rebinning

        # Rebinning -- tricky stuff!
        bmview = gbeam.reshape(retext, oversamp, retext, oversamp)
        beam = bmview.mean(axis=3).mean(axis=1)

    return beam

class genmap:
    """ Generates simulated maps from the Bethermin et al. 2012 model
    using a Gaussian beam"""

    def __init__(self, 
                 wave: "Wavelength of simulated bands, in um"=[250.0,350,500],
                 pixsize: "Pixel size, in arcsec"=[6.0, 8.33333, 12.0],
                 fwhm: "FWHM of beam, in arcsec"=[17.6, 23.9, 35.2],
                 gensize: "Number of sources to generate at a time"=100000,
                 bmoversamp: "Beam oversampling"=5,
                 log10Mb: 'log 10 of Mb, in solar masses'=11.2,
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
        
        self._wave = np.asarray(wave, dtype=np.float32)
        if self._wave.min() <= 0:
            raise ValueError("Non-positive wavelengths not supported")
        self._nbands = len(self._wave)
        
        if len(pixsize) != self._nbands:
            if len(pixsize) == 1:
                self._pixsize = np.asarray(pixsize[0], dtype=np.float32) *\
                    np.ones_like(self._wave)
            else:
                raise ValueError("Number of pixel sizes doesn't match number of wavelengths")
        else:
            self._pixsize=  np.asarray(pixsize, dtype=np.float32)
        if self._pixsize.min() <= 0:
            raise ValueError("Invalid (negative) pixel size")

        if len(fwhm) != self._nbands:
            if len(fwhm) == 1:
                self._fwhm = np.asarray(fwhm[0], dtype=np.float32) *\
                    np.ones_like(self._wave)
            else:
                raise ValueError("Number of FWHM doesn't match number of wavelengths")
        else:
            self._fwhm=  np.asarray(fwhm, dtype=np.float32)
        if self._fwhm.min() <= 0:
            raise ValueError("Invalid (negative) FWHM")
        if (self._fwhm / self._pixsize).min() < 1:
            raise ValueError("Some FWHM not properly sampled by pixel size")

        # If gensize is zero, we do the full set at once
        self._gensize = int(gensize)
        if self._gensize < 0:
            raise ValueError("Invalid (negative) gensize")

        self._bmoversamp = int(bmoversamp)
        if self._bmoversamp < 1:
            raise ValueError("Invalid (<1) beam oversampling")
        if self._bmoversamp % 2 == 0:
            raise ValueError("Invalid (even) beam oversampling %d" % self._bmoversamp)
            

        self._gencat = gencat(log10Mb, alpha, log10Mmin, log10Mmax, 
                              ninterpm, zmin, zmax, Om0, H0, phib0, 
                              gamma_sfmf, ninterpz, rsb0, gammasb, zsb, 
                              logsSFRM0, betaMS, zevo, gammams, bsb, 
                              sigmams, sigmasb, mnU_MS0, gammaU_MS0, 
                              z_UMS, mnU_SB0, gammaU_SB0, z_USB,
                              scatU, ninterpdl)
        self._npersr = self._gencat.npersr

    @property
    def npersr(self):
        return self._npersr

    @property
    def nbands(self):
        return self._nbands

    @property
    def wave(self):
        return self._wave

    @property
    def pixsize(self):
        return self._pixsize

    @property
    def fwhm(self):
        return self._fwhm

    @property
    def bmoversamp(self):
        return self._bmoversamp

    @property
    def gensize(self):
        return self._gensize

    def get_beam(self, idx):
        """ Gets the beam for the specified index"""
        return get_beam(self._fwhm[idx], self._pixsize[idx],
                        self._bmoversamp)

    def generate(self, area: "Area of generated maps, in deg^2",
                 sigma: "Map instrument noise, in Jy"=None,
                 verbose: "Print informational messages"=False):
        """ Generates simulated maps"""
        
        if area <= 0.0:
            raise ValueError("Invalid (non-positive) area")
        
        if sigma is None:
            int_sigma = np.zeros(self._nbands, dtype=np.float32)
        elif type(sigma) == list:
            if len(sigma) != self._nbands:
                if len(sigma) == 1:
                    int_sigma = sigma[0] * np.ones_like(self._wave)
                else:
                    raise ValueError("Number of sigmas doesn't match number"
                                     " of wavelengths")
            else:
                int_sigma = np.asarray(sigma, dtype=np.float32)
        elif type(sigma) == np.ndarray:
            if len(sigma) != self._nbands:
                if len(sigma) == 1:
                    int_sigma = sigma[0] * np.ones_like(self._wave)
                else:
                    raise ValueError("Number of sigmas doesn't match number"
                                     " of wavelengths")
            else:
                int_sigma = sigma.astype(np.float32, copy=False)
        else:
            int_sigma=  float(sigma) * np.ones_like(self._wave)

        if int_sigma.min() < 0:
            raise ValueError("Invalid (negative) instrument sigma")

        # Make the non-convolved images
        # The first step is to initialize the output maps.
        # Since we do the catalog in chunks (in typical applications
        # the catalog takes more memory than the maps), we must hold
        # all the maps in memory at once
        nextent = np.empty(self._nbands, dtype=np.int32)
        truearea = np.empty(self._nbands, dtype=np.float32)
        maps = []
        for i in range(self._nbands):
            pixarea = (self._pixsize[i] / 3600.0)**2
            nextent[i] = math.ceil(math.sqrt(area / pixarea))
            truearea[i] = nextent[i]**2 * pixarea
            maps.append(np.zeros((nextent[i], nextent[i]), 
                                 dtype=np.float32))
            
        # Figure out how many sources to make
        truearea = truearea.mean()
        nsources_base = self._npersr * (math.pi / 180.0)**2 * truearea
        nsources = np.random.poisson(lam=nsources_base)
        if verbose:
            print("True area: %0.2f [deg^2]" % truearea)
            print("Number of sources to generate: %d" % nsources)
            
        # We do this in chunks
        if self._gensize == 0:
            # One big chunk
            nchunks = 1
            chunks = np.array([nsources], dtype=np.int64)
        else:
            # Recall this is python 3 -- floating point division
            nchunks = math.ceil(nsources / self._gensize)
            chunks = self._gensize * np.ones(nchunks, dtype=np.int64)
            chunks[-1] = nsources - (nchunks - 1) * self._gensize
            assert chunks.sum() == nsources
            
        # Source generation loop
        nexgen = float(nextent[0])
        if verbose:
            print("Generating sources")
        for i, nsrc in enumerate(chunks):
            if verbose:
                print("  Doing chunk %d of %d" % (i+1, nchunks)) 
                
            # Get fluxes (in Jy)
            fluxes = self._gencat.random(nsrc, wave=self._wave)[-1]
                
            # Generate positions in base image, uniformly distributed
            # Note these are floating point
            xpos = nexgen * np.random.rand(nsrc)
            ypos = nexgen * np.random.rand(nsrc)
                
            # Add to first map without rescaling.
            # Note this has to happen in a for loop because multiple
            # sources can go in the -same- pixel
            cmap = maps[0]
            xf = np.floor(xpos)
            yf = np.floor(ypos)
            nx, ny = cmap.shape
            np.place(xf, xf > nx-1, nx-1)
            np.place(yf, yf > ny-1, ny-1)
            for cx, cy, cf in zip(xf, yf, fluxes[:,0]):
                cmap[cx, cy] += cf

            # Other bands, with pixel scale adjustment
            for mapidx in range(1, self._nbands):
                posrescale = self._pixsize[0] / self._pixsize[mapidx]
                xf = np.floor(posrescale * xpos)
                yf = np.floor(posrescale * ypos)
                cmap = maps[mapidx]
                nx, ny = cmap.shape
                np.place(xf, xf > nx-1, nx-1)
                np.place(yf, yf > ny-1, ny-1)
                for cx, cy, cf in zip(xf, yf, fluxes[:,mapidx]):
                    cmap[cx, cy] += cf
                    
            del fluxes, xpos, ypos, xf, yf

        # Now image details -- convolution, instrument noise
        for mapidx in range(self._nbands):
            if verbose:
                print("Preparing map for wavelength %5.1f um extent: %d x %d" %
                      (self._wave[mapidx], maps[mapidx].shape[0],
                       maps[mapidx].shape[1]))
                
            if verbose:
                print("  Convolving")
            beam = self.get_beam(mapidx)
            maps[mapidx] = convolve(maps[mapidx], beam, boundary='wrap')

            if int_sigma[mapidx] > 0:
                if verbose:
                    print("  Adding instrument noise: %0.4f [Jy]" %
                          int_sigma[mapidx])
                maps[mapidx] += np.random.normal(scale=int_sigma[mapidx],
                                                 size=maps[mapidx].shape)
                
        return maps
