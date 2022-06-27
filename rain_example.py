import os
import time

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from rain import rain

def distortion(index_x, index_y, x_dist, y_dist):
    '''
    -- Description:
    performs the distortion correction according to x_dist and y_dist
    assumes that x_dist and y_dist are both functions of x and y
    -- Returns:
    the distortion corrected pixel coordinates
    -- Arguments:
         index_x: the x index of the pixel
         index_y: the y index of the pixel
          x_dist: distortion map in x
          y_dist: distortion map in y
    '''

    # center on pixel
    x_actual = index_x+0.5
    y_actual = index_y+0.5
    return x_actual+x_dist[index_y,index_x], y_actual+y_dist[index_y,index_x]

def main():
    '''
    -- Description:
    general process is: determine the kernel you want to use, load in data and distortion map, generate an index_coords array of all possible coordinates, then rain the image
    -- Returns:
    Saves the result to file of raining check image 0.
    '''

    # Rain settings
    # nominal values for lanczos3
    n_cpu = 32 # number of processes
    kernel='lanczos3_lut' # kernel used in output pixels
    pixel_frac = 1 # pixel fraction, fraction of pixel side length
    n_div = 1 # number of divisions per pixel
    n_pad = 0 # number of pixels padding around the edges

    # sorting out path names
    data_dir = './check_images'
    image_filepath = os.path.join(data_dir, 'check_image_0.fits')
    dist_x_filepath = os.path.join(data_dir, 'check_image_0_dist_x.fits')
    dist_y_filepath = os.path.join(data_dir, 'check_image_0_dist_y.fits')
    basename = os.path.basename(image_filepath)
    output_filename = 'rain_lanc3_{}'.format(basename)
    output_path = os.path.join(data_dir, output_filename)
    print('Correcting {} -> {}'.format(image_filepath, output_path))

    # get the image data
    image = fits.getdata(image_filepath)

    # generate all combinations of the coordinates
    iy,ix = image.shape
    xs = np.arange(0,ix,1)
    ys = np.arange(0,iy,1)
    grid = np.meshgrid(xs, ys)
    index_coords = np.stack(grid).T.reshape(-1,2) # N^2 by 2 matrix of all the coordinates

    # no bad pixel map
    bad_pixel_map = None

    # read in the distortion correction data
    x_dist = fits.getdata(dist_x_filepath)
    y_dist = fits.getdata(dist_y_filepath)

    # apply distortion correction to new pixel center coordinates
    new_pc_coords = []
    for index_x,index_y in tqdm(index_coords, desc='Coords'):
        new_pc_coords.append(distortion(index_x, index_y, x_dist, y_dist))

    time_start = time.time()

    # perform rain
    wet_image, missed_pixels = rain(image, pixel_frac, new_pc_coords, n_div, kernel=kernel, bad_pixel_map=bad_pixel_map, parallel=True, n_cpu=n_cpu, n_pad=n_pad)

    time_end = time.time()
    time_diff = time_end - time_start

    print('Time               : {:.3f}'.format(time_diff))
    print('Original Image Sum : {:.6f}'.format(np.sum(image)))
    print('Wet Image Sum      : {:.6f}'.format(np.sum(wet_image)))
    print('Missed Pixels      : {:.6f}'.format(missed_pixels))
    print('Missed + Wet Image : {:.6f}'.format(missed_pixels+np.sum(wet_image)))

    # write rained image to file
    print('Writing to {}...'.format(output_path))
    hdu = fits.PrimaryHDU(wet_image)
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_path, overwrite=True)

if __name__ == '__main__':
    main()
