import os

import numpy as np
from scipy import special
from astropy.io import fits

def diffraction_limit(wl, a_diameter, pixel_scale):
    '''
    -- Description:
    calculate the diffraction limit for a telescope
    match units between wl and a_diameter
    -- Returns:
    diffraction limit in units of pixels for a telescope
    -- Arguments:
              wl: wavelength of light
      a_diameter: diameter of the effective aperture of the telescope
     pixel_scale: px/mas
    '''

    j_zero = special.jn_zeros(1,1)[0]
    angle = np.arcsin(j_zero * wl / (np.pi*a_diameter)) # in units of rad
    mas = np.rad2deg(angle) * 3600 #  to arcsec
    pixels = mas / pixel_scale # to pixels
    return pixels

def airy(x, y, amplitude, x0, y0, airy_radius, obscuration):
    '''
    -- Description:
    radial airy distribution
    -- Returns:
    airy distribution with some effects due to obscuration at given x,y
    -- Arguments:
               x: x coord
               y: y coord
       amplitude: amplitude of the peak
              x0: x center of distribution
              y0: y center of distribution
     airy_radius: Width factor, usually half of diffraction limit value
     obscuration: Some fraction of obscuration of the primary mirror
    '''

    r = np.sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0))
    val = np.pi * r / (2*airy_radius)
    normal_airy = 2*special.j1(val)/val # airy from j1 bessel function
    obscured_airy = obscuration * 2*special.j1(obscuration*val)/val # if obscuration -> 0, then obscuration terms are zero
    return (amplitude / ((1-obscuration*obscuration)*(1-obscuration*obscuration))) * (normal_airy - obscured_airy)*(normal_airy - obscured_airy)

def lorentzian(x, y, amplitude, x0, y0, hwhm):
    '''
    -- Description:
    radial lorentzian distribution
    -- Returns:
    lorentzian probability distribution at given x,y
    -- Arguments:
               x: x coord
               y: y coord 
       amplitude: amplitude of the peak
              x0: x center of distribution
              y0: y center of distribution
            hwhm: half width half maximum, a width parameter
    '''

    r = np.sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0))
    return amplitude / (1 + (r*r)/(hwhm*hwhm) )

def generate_fake_psf(ix, iy, amplitude, noise_amp):
    '''
    -- Description:
    generates nominal fake PSF for Gemini Telescope, Lorentzian + Airy
    -- Returns:
    image of PSF in ix,iy
    -- Arguments:
              ix: x dimension of shape
              iy: y dimension of shape
       amplitude: height of peak of psf
       noise_amp: uniform random noise amplitude
    '''

    x0 = ix/2
    y0 = iy/2

    obscuration = 0 # fraction of light collection area obscured by secondary mirror
    hwhm = 4 # pixels

    # wavelength center of various bands in meters
    m_band_wl = 4.750e-6
    k_band_wl = 2.190e-6
    l_band_wl = 3.450e-6
    h_band_wl = 1.630e-6

    wavelength = m_band_wl

    a_diameter = 9.96 # keck effective aperture
    pixel_scale = 0.009942 # arcsec/pixel

    dl = diffraction_limit(wavelength, a_diameter, pixel_scale)
    airy_radius = dl/2

    L = lambda x, y : lorentzian(x, y, amplitude/2, x0, y0, hwhm)
    A = lambda x, y : airy(x, y, amplitude/2, x0, y0, airy_radius, obscuration)

    img = np.zeros((iy,ix))

    for i in range(ix):
        for j in range(iy):

            ii = i+0.5 # add 0.5 to get pixel center
            jj = j+0.5 # add 0.5 to get pixel center

            img[j,i] = L(ii,jj) + A(ii,jj)

    img += np.random.random(img.shape) * noise_amp

    return img


def write_image_to_fits(image, output_filepath, header=None):
    '''
    -- Description:
    Writes an image to fits file
    -- Returns:
    Nothing, but writes a file to output_filepath
    -- Arguments:
           image: 2d array image to write
 output_filepath: the full filepath to output image to
          header: Default None, if specified adds cards to header fits file
    '''

    hdu = fits.PrimaryHDU(image)
    if header is not None:
        for key in header.keys():
            hdu.header[key] = header[key]
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_filepath, overwrite=True)

def generate_check_image(image_num, image, dist_x, dist_y):
    '''
    -- Description:
    Helper function for writing check images
    -- Returns: 
    Nothing, but writes check_image_#.fits, check_image_#_dist_x.fits, check_image_#_dist_y.fits
    -- Arguments:
           image: 2d array image to write
       image_num: image index
          dist_x: distortion map in x, 2d array
          dist_y: distortion map in y, 2d array
    '''

    check_image_dir = 'check_images'
    os.makedirs(check_image_dir, exist_ok=True) 

    output_filepath = os.path.join(check_image_dir, 'check_image_{}.fits'.format(image_num))
    write_image_to_fits(image, output_filepath, {'ITIME': 1})
    print('Writing to {}...'.format(output_filepath))

    output_filepath = os.path.join(check_image_dir, 'check_image_{}_dist_x.fits'.format(image_num))
    write_image_to_fits(dist_x, output_filepath)
    print('Writing to {}...'.format(output_filepath))

    output_filepath = os.path.join(check_image_dir, 'check_image_{}_dist_y.fits'.format(image_num))
    write_image_to_fits(dist_y, output_filepath)
    print('Writing to {}...'.format(output_filepath))

def main():

    # single pixel, no shift
    image_num = 0
    ix = 16
    iy = 16
    image = np.zeros((ix,iy))
    image[int(ix/2),int(iy/2)] = 1
    dist_x = np.zeros((ix,iy))
    dist_y = np.zeros((ix,iy))
    generate_check_image(image_num, image, dist_x, dist_y)

    # single pixel, half pixel shift
    image_num += 1
    ix = 16
    iy = 16
    image = np.zeros((ix,iy))
    image[int(ix/2),int(iy/2)] = 1
    dist_x = np.ones((ix,iy))/2
    dist_y = np.ones((ix,iy))/2
    generate_check_image(image_num, image, dist_x, dist_y)

    # single pixel, quarter pixel shift
    image_num += 1
    ix = 16
    iy = 16
    image = np.zeros((ix,iy))
    image[int(ix/2),int(iy/2)] = 1
    dist_x = np.ones((ix,iy))/4
    dist_y = np.ones((ix,iy))/4
    generate_check_image(image_num, image, dist_x, dist_y)

    # single pixel, non simple pixel shift
    image_num += 1
    ix = 16
    iy = 16
    image = np.zeros((ix,iy))
    image[int(ix/2),int(iy/2)] = 1
    dist_x = np.ones((ix,iy))/1.2312
    dist_y = np.ones((ix,iy))/3.1413
    generate_check_image(image_num, image, dist_x, dist_y)

    # fake psf, half pixel shift
    image_num += 1
    ix = 2**5
    iy = 2**5
    amplitude = 1
    noise_amplitude = 0.1
    image = generate_fake_psf(ix, iy, amplitude, noise_amplitude)
    dist_x = np.ones((ix,iy))/2
    dist_y = np.ones((ix,iy))/2
    generate_check_image(image_num, image, dist_x, dist_y)

    # fake psf, quarter pixel shift
    image_num += 1
    ix = 2**5
    iy = 2**5
    amplitude = 1
    noise_amplitude = 0.1
    image = generate_fake_psf(ix, iy, amplitude, noise_amplitude)
    dist_x = np.ones((ix,iy))/4
    dist_y = np.ones((ix,iy))/4
    generate_check_image(image_num, image, dist_x, dist_y)

    # fake psf, no shift
    image_num += 1
    ix = 2**5
    iy = 2**5
    amplitude = 1
    noise_amplitude = 0.0
    image = generate_fake_psf(ix, iy, amplitude, noise_amplitude)
    dist_x = np.zeros((ix,iy))
    dist_y = np.zeros((ix,iy))
    generate_check_image(image_num, image, dist_x, dist_y)

if __name__=='__main__':
    main()
