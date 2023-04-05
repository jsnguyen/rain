import os
import time
import glob

import numpy as np
import matplotlib.pyplot as plt
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

def crop_center(img, sx, sy):
    y,x = img.shape
    startx = x//2-(sx//2)
    starty = y//2-(sy//2)    
    return img[starty:starty+sy,startx:startx+sx]

def main():

    # standard way to run
    #n_cpu = os.cpu_count()
    n_cpu = 32

    #data_dir = '/Users/jsn/landing/data/2015Oct26/reduced/'
    data_dir = '/home/jsn/landing/data/2009Aug31/reduced/'

    kernel='lanczos3_lut'
    pixel_frac = 1 # pixel fraction, fraction of pixel side length
    n_div = 1 # number of divisions per pixel
    n_pad = 0

    search_dir = os.path.join(data_dir,'nosky_r_*.fits')
    for input_path in sorted(glob.glob(search_dir)):
        basename = os.path.basename(input_path)

        output_filename = 'rain_{}'.format(basename)
        output_path = os.path.join(data_dir, output_filename)

        print('Correcting {} -> {}'.format(input_path, output_path))

        image = fits.getdata(input_path)
        header = fits.getheader(input_path)

        iy,ix = image.shape
        xs = np.arange(0,ix,1)
        ys = np.arange(0,iy,1)
        grid = np.meshgrid(xs, ys)
        index_coords = np.stack(grid).T.reshape(-1,2)

        bad_pixel_map = fits.getdata('./masks/nirc2mask.fits')
        x_dist = fits.getdata('distortion/nirc2_distort_X_pre20150413_v1.fits')
        y_dist = fits.getdata('distortion/nirc2_distort_Y_pre20150413_v1.fits')

        x_dist = crop_center(x_dist, ix, iy)
        y_dist = crop_center(y_dist, ix, iy)

        new_pc_coords = []
        for index_x,index_y in tqdm(index_coords, desc='Generate New Coordinates'):
            new_pc_coords.append(distortion(index_x, index_y, x_dist, y_dist))

        time_start = time.time()

        wet_image, missed_pixels = rain(image, pixel_frac, new_pc_coords, n_div, kernel=kernel, n_pad=n_pad, bad_pixel_map=bad_pixel_map, parallel=True, n_cpu=n_cpu)

        time_end = time.time()
        time_diff = time_end - time_start

        print('Time               : {:.3f}'.format(time_diff))
        print('Original Image Sum : {:.6f}'.format(np.sum(image)))
        print('Wet Image Sum      : {:.6f}'.format(np.sum(wet_image)))
        print('Missed Pixels      : {:.6f}'.format(missed_pixels))
        print('Missed + Wet Image : {:.6f}'.format(missed_pixels+np.sum(wet_image)))

        print('Writing to {}...'.format(output_path))
        hdu = fits.PrimaryHDU(wet_image)
        hdul = fits.HDUList([hdu])
        hdul[0].header = header
        hdul.writeto(output_path, overwrite=True)

if __name__=='__main__':
    main()
