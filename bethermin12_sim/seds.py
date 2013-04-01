import numpy as np
import math

""" Handles SED templates for Bethermin et al. 2012 model"""

__all__ = ["sed_model"]

class sed_model:
    """ Gets SEDs and fluxes from Magdis template models"""

    def __init__(self, Om0: 'Omega0' = 0.315,
                 H0: 'H0 [km/s/Mpc]'=67.7,
                 zmin: 'Minimum z supported'=0.5,
                 zmax: 'Maximum z supported'=7.0,
                 ninterp: 'Interplation points for luminosity distance'=100):

        from astropy.cosmology import FlatLambdaCDM
        import astropy.io.fits as fits
        from pkg_resources import resource_filename
        from scipy.interpolate import interp1d

        self._zmin = float(zmin)
        self._zmax = float(zmax)
        self._Om0 = float(Om0)
        self._H0 = float(H0)
        self._ninterp = int(ninterp)
        

        if self._zmin == self._zmax:
            raise ValueError("No range between zmin and zmax")
        if self._zmin > self._zmax:
            self._zmin, self._zmax = self._zmax, self._zmin
        if self._zmin < 0.0:
            raise ValueError("zmin must be >= 0: %f" % self._zmin)
        if self._Om0 <= 0.0:
            raise ValueError("Om0 must be positive: %f" % self._Om0)
        if self._H0 <= 0.0:
            raise ValueError("H0 must be positive: %f" % self._H0)
        if self._ninterp <= 0:
            raise ValueError("Ninterp must be > 0: %d" % self._ninterp)

        # Set up luminosity distance interpolant.  We actually
        # interpolate log((1+z) / (4 pi d_L^2)) in log(1+z)
        cos = FlatLambdaCDM(H0=self._H0, Om0=self._Om0, Tcmb0=0.0, Neff=0.0)
        zrange = np.linspace(self._zmin, self._zmax, self._ninterp)
        mpc_in_cm = 3.0857e24
        prefac = 1.0 / (4 * math.pi * mpc_in_cm**2)
        dlval = prefac * (1.0 + zrange) / cos.luminosity_distance(zrange)**2
        self._dlfac = interp1d(np.log(1 + zrange), np.log(dlval))

        # Read in the data products, and set up interpolations on them
        sb_tpl = resource_filename(__name__, 'resources/SED_sb.fits')
        hdu = fits.open(sb_tpl)
        dat = hdu['SEDS'].data
        hdu.close()
        self._sblam = dat['LAMBDA'][0]
        self._sbumean = dat['UMEAN'][0]
        arg = np.argsort(self._sbumean)
        self._sbumean = self._sbumean[arg]
        self._sbrange = np.array([self._sbumean[0], self._sbumean[-1]])
        self._sbseds = dat['SEDS'][0,:,:].transpose()[arg,:] # umean by lambda
        self._sbinterp = []
        for idx in range(len(self._sbumean)):
            self._sbinterp.append(interp1d(self._sblam, self._sbseds[idx,:]))

        ms_tpl = resource_filename(__name__, 'resources/SED_ms.fits')
        hdu = fits.open(ms_tpl)
        dat = hdu['SEDS'].data
        hdu.close()
        self._mslam = dat['LAMBDA'][0]
        self._msumean = dat['UMEAN'][0]
        arg = np.argsort(self._msumean)
        self._msumean = self._msumean[arg]
        self._msrange = np.array([self._msumean[0],self._msumean[-1]])
        self._msseds = dat['SEDS'][0,:,:].transpose()[arg,:] # umean by lambda
        self._msinterp = []
        for idx in range(len(self._msumean)):
            self._msinterp.append(interp1d(self._mslam, self._msseds[idx,:]))

    def get_sed(self, z, U, is_starburst, log10lir=1.0):
        """ Gets the combined SED in erg/s/cm^2/Hz"""

        zval = float(z)
        if zval > self._zmax or zval < self._zmin:
            raise ValueError("z out of supported range: %f" % zval)
        opz = 1.0 + zval        
        ldfac = 10**log10lir * np.exp(self._dlfac(np.log(opz)))

        if is_starburst:
            return (self._sblam/opz, 
                    ldfac * self._intsed1(U, self._sbumean, self._sbseds))
        else:
            return (self._mslam/opz, 
                    ldfac * self._intsed1(U, self._msumean, self._msseds))

    def _intsed1(self, U, uarr, seds):
        # uarr[idx] <= U < uarr[idx+1]
        idx = np.searchsorted(uarr, U, side='right') 
        if idx == 0:
            return seds[0, :]
        elif idx == len(uarr):
            return seds[-1,:]
        wt2 = (U - uarr[idx-1])/(uarr[idx] - uarr[idx-1])
        wt1 = 1.0 - wt2
        return wt1 * seds[idx-1,:] + wt2 * seds[idx,:]

    def get_fluxes(self, wave, z, U, is_starburst, log10lir=0.0):
        """ Gets the flux density at the observer frame wavelengths wave
        in Jy for a source with a given log10 L_IR"""

        zval = float(z)
        if zval > self._zmax or zval < self._zmin:
            raise ValueError("z out of supported range: %f" % zval)
        opz = 1.0 + zval
        ldfac = 10**(log10lir + 23.0) * \
            np.exp(self._dlfac(np.log(opz))) # 1e23 is to Jy

        if is_starburst:
            return  ldfac * self._intsed2(np.asarray(wave)/opz, U, 
                                          self._sbumean, self._sbinterp)
        else:
            return  ldfac * self._intsed2(np.asarray(wave)/opz, U, 
                                          self._msumean, self._msinterp)
        
    def _intsed2(self, wave, U, uarr, seds):
        # uarr[idx] <= U < uarr[idx+1]
        idx = np.searchsorted(uarr, U, side='right') 
        if idx == 0:
            return seds[0](wave)
        elif idx == len(uarr):
            return seds[idx-1](wave)
        wt2 = (U - uarr[idx-1])/(uarr[idx] - uarr[idx-1])
        wt1 = 1.0 - wt2
        return wt1 * seds[idx-1](wave) + wt2 * seds[idx](wave)
