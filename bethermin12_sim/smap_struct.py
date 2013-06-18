from __future__ import print_function

import math
import astropy.wcs as wcs
import astropy.io.fits as fits

""" Reads and writes SMAP fits structures"""

__all__ = ["smap_map"]

class smap_map:
    """ Represents SMAP map"""
    def __init__(self, filename=None):
        """ Basic constructor"""
        self._has_data = False
        self._has_exposure = False
        self._has_error = False
        self._has_mask = False
        if not filename is None:
            self.read(filename)

    def read(self, filename):
        """ Read from a FITS structure"""
        hdulist = fits.open(filename, uint=True)

        # Figure out what we have
        try:
            image_ext = hdulist.index_of('image')
        except KeyError:
            hdulist.close()
            raise IOError("File %s doesn't have primary image" % filename)
        
        self.image = hdulist[image_ext].data
        self.astrometry = wcs.WCS(hdulist[image_ext].header)
        if 'WAVELN' in hdulist[image_ext].header:
            self.wave = int(hdulist[image_ext].header['WAVELN'])
        elif hasattr(self, 'wave'):
            del self.wave
        if 'DESC' in hdulist[image_ext].header:
            self.bands = hdulist[image_ext].header['DESC']
        elif hasattr(self, 'bands'):
            del self.bands
        self.xsize = self.image.shape[0]
        self.ysize = self.image.shape[1]
        self._has_filter = False # Not supported yet
        # self.todmask = hdulist[image_ext].header['TOD_EXCLUDEMASK']

        # Set name
        namedict = {250: 'PSW', 350: 'PMW', 500: 'PLW'}
        if hasattr(self, 'wave') and self.wave in namedict:
            self.names = namedict[self.wave]
        elif hasattr(self, 'names'):
            del self.names

        # Set up pixel scale, assume square
        cd = self.astrometry.wcs.cd
        pixscale1 = abs(math.sqrt(cd[0, 0]**2 + cd[0,1]**2))
        pixscale2 = abs(math.sqrt(cd[1, 0]**2 + cd[1,1]**2))
        self.pixscale = 3600.0 * math.sqrt(pixscale1 * pixscale2)
        
        try:
            error_ext = hdulist.index_of('error')
            self._has_error = True
            self.error = hdulist[error_ext].data
        except KeyError:
            self._has_error = False
            if hasattr(self, 'error'):
                del self.error

        try:
            exposure_ext = hdulist.index_of('exposure')
            self._has_exposure = True
            self.exposure = hdulist[exposure_ext].data
        except KeyError:
            self._has_exposure = False
            if hasattr(self, 'exposure'):
                del self.exposure

        try:
            mask_ext = hdulist.index_of('mask')
            self._has_mask = True
            self.mask = hdulist[mask_ext].data
        except KeyError:
            self._has_mask = False
            if hasattr(self, 'mask'):
                del self.mask
            
        hdulist.close()
        self._has_data = True

    def write(self, filename):
        """ Write as FITS file"""
        
        if not self._has_data:
            raise IOError("Attempting to write empty smap map")

        # Set up the header
        head = self.astrometry.to_header()
        if hasattr(self, 'wave'):
            head['WAVELN'] = self.wave
        if hasattr(self, 'bands'):
            head['DESC'] = self.bands

        # Set up image
        hdulist = fits.HDUList(fits.PrimaryHDU())
        hdulist.append(fits.ImageHDU(data=self.image, 
                                     header=head, name='image'))
        if self._has_error:
            hdulist.append(fits.ImageHDU(data=self.error, 
                                         header=head, name='error'))
        if self._has_exposure:
            hdulist.append(fits.ImageHDU(data=self.exposure, 
                                         header=head, name='exposure'))
        if self._has_mask:
            hdulist.append(fits.ImageHDU(data=self.mask, uint=True,
                                         header=head, name='mask'))

        hdulist.writeto(filename)
        
    def create(self, image, pixscale, racen, deccen,  wave=None,
               bands=None, error=None, exposure=None, mask=None):
        """ Create an image from input data.

        Parameters
        ----------
        image: ndarray
          Map.  2D array
        
        wave: float
          Wavelength in microns of map

        pixscale: float
          Pixel scale in arcsec

        racen: float
          Central ra value

        deccen: float
          Central dec value

        bands: string
          Optional band description

        error: ndarray
          Error array

        exposure: ndarray
          Exposure array

        mask: ndarray
          Mask array
        """
        
        import numpy as np

        if not type(image) == np.ndarray:
            raise Exception("Input image not ndarray")
        if len(image.shape) != 2:
            raise Exception("Non-2D image frame")

        self._has_data = True
        self.image = image.astype(np.float32)
        self.xsize = image.shape[0]
        self.ysize = image.shape[1]

        self.pixscale = float(pixscale)
        self.astrometry = wcs.WCS(naxis=2)
        self.astrometry.wcs.crpix = [self.xsize/2, self.ysize/2]
        self.astrometry.wcs.cd = np.array([[-self.pixscale / 3600.0, 0],
                                           [0, self.pixscale / 3600.0]])
        self.astrometry.wcs.crval = np.array([racen, deccen],
                                             dtype=np.float64)
        self.astrometry.wcs.ctype = ['RA---TAN'.encode(), 
                                     'DEC--TAN'.encode()]

        if not wave is None:
            self.wave = int(wave)
            namedict = {250: 'PSW', 350: 'PMW', 500: 'PLW'}
            if self.wave in namedict:
                self.names = namedict[self.wave]
            elif hasattr(self, 'names'):
                del self.names
        else:
            if hasattr(self, 'wave'): del self.wave
            if hasattr(self, 'names'): del self.names
            
        if not bands is None:
            self.bands = str(bands)
        elif hasattr(self, 'bands'): 
            del self.bands

        if not error is None:
            if not type(error) == np.ndarray:
                raise ValueError("Input error array not ndarray")
            if len(error.shape) != 2:
                raise ValueError("Input error array not 2D")
            if error.shape[0] != self.image.shape[0] or \
                    error.shape[1] != self.image.shape[1]:
                raise ValueError("Input error array not same extent as image")
            self.error = error.astype(np.float32)
            self._has_error = True
        else:
            if hasattr(self, 'error'):
                del self.error
            self._has_error = False

        if not exposure is None:
            if not type(exposure) == np.ndarray:
                raise ValueError("Input exposure array not ndarray")
            if len(exposure.shape) != 2:
                raise ValueError("Input exposure array not 2D")
            if exposure.shape[0] != self.image.shape[0] or \
                    exposure.shape[1] != self.image.shape[1]:
                raise ValueError("Input exposure array not same "
                                 "extent as image")
            self.exposure = exposure.astype(np.float32)
            self._has_exposure = True
        else:
            if hasattr(self, 'exposure'):
                del self.exposure
            self._has_exposure = False

        if not mask is None:
            if not type(mask) == np.ndarray:
                raise ValueError("Input mask array not ndarray")
            if len(mask.shape) != 2:
                raise ValueError("Input mask array not 2D")
            if mask.shape[0] != self.image.shape[0] or \
                    mask.shape[1] != self.image.shape[1]:
                raise ValueError("Input mask array not same "
                                 "extent as image")
            self.mask = mask.astype(np.uint16)
            self._has_mask = True
        else:
            if hasattr(self, 'mask'):
                del self.mask
            self._has_mask = False
            
    @property
    def has_data(self):
        return self._has_data

    @property
    def has_error(self):
        return self._has_error

    @property
    def has_exposure(self):
        return self._has_exposure

    @property
    def has_mask(self):
        return self._has_mask

    def __str__(self):
        """ String representation"""
        if not self._has_data:
            return "Empty SMAP map"
        outstr = "SMAP map"
        if hasattr(self, 'wave'):
            outstr += " wavelength: %d [um]" % self.wave
        if hasattr(self, 'names'):
            outstr += " desc: %s" % self.names
        outstr += "\n RA: %0.4f [deg]  DEC: %0.4f [deg]" %\
            (self.astrometry.wcs.crval[0], self.astrometry.wcs.crval[1])
        outstr += " Pixscale: %0.2f [arcsec]" % self.pixscale
        outstr += "\n Map [%d x %d]" % (self.xsize, self.ysize)
        if self._has_error:
            outstr += "\n Error map [%d x %d]" % (self.xsize, self.ysize)
        if self._has_exposure:
            outstr += "\n Exposure map [%d x %d]" % (self.xsize, self.ysize)
        if self._has_mask:
            outstr += "\n Mask map [%d x %d]" % (self.xsize, self.ysize)

        return outstr
