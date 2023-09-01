import os
import time
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm

from rain import rain

def distortion(index_x, index_y, x_dist, y_dist, shape):
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

    xpad_width = (1024-shape[1])//2
    ypad_width = (1024-shape[0])//2

    x_dist = x_dist[ypad_width:-ypad_width,:]
    x_dist = x_dist[:,xpad_width:-xpad_width]

    y_dist = y_dist[ypad_width:-ypad_width,:]
    y_dist = y_dist[:,xpad_width:-xpad_width]

    return x_actual+x_dist[index_y,index_x], y_actual+y_dist[index_y,index_x]

def do_rain(data_dir):

    # standard way to run
    #n_cpu = os.cpu_count()
    n_cpu = 32

    kernel='lanczos3_lut'
    pixel_frac = 1 # pixel fraction, fraction of pixel side length
    n_div = 1 # number of divisions per pixel
    n_pad = 0

    data_dir = Path(data_dir)
    for input_path in sorted(data_dir.glob('r_*.fits')):
        basename = Path(input_path).name

        output_folder = data_dir.parent / 'rain'
        output_folder.mkdir(parents=True, exist_ok=True)

        output_path =  output_folder / 'rain_{}'.format(basename)
        print('Correcting {} -> {}'.format(input_path, output_path))

        image = fits.getdata(input_path)
        header = fits.getheader(input_path)

        iy,ix = image.shape
        x_offset = int((1024 - ix)/2)
        y_offset = int((1024 - iy)/2)
        xs = np.arange(0,ix,1)
        ys = np.arange(0,iy,1)
        grid = np.meshgrid(xs, ys)
        index_coords = np.stack(grid).T.reshape(-1,2)

        #bad_pixel_map = fits.getdata('./masks/nirc2mask.fits')
        bad_pixel_map = None
        x_dist = fits.getdata('distortion/nirc2_distort_X_pre20150413_v2.fits')
        y_dist = fits.getdata('distortion/nirc2_distort_Y_pre20150413_v2.fits')
        new_pc_coords = []
        for index_x,index_y in tqdm(index_coords, desc='Generate New Coordinates'):
            new_pc_coords.append(distortion(index_x, index_y, x_dist, y_dist, image.shape))

        time_start = time.time()

        wet_image, missed_pixels = rain(image, pixel_frac, new_pc_coords, n_div, kernel=kernel, n_pad=n_pad, bad_pixel_map=bad_pixel_map, parallel=True, n_cpu=n_cpu)

        time_end = time.time()
        time_diff = time_end - time_start

        print('Time               : {:.3f}'.format(time_diff))
        print('Original Image Sum : {:.3f}'.format(np.sum(image)))
        print('Wet Image Sum      : {:.3f}'.format(np.sum(wet_image)))
        print('Missed Pixels      : {:.3f}'.format(missed_pixels))
        print('Missed + Wet Image : {:.3f}'.format(missed_pixels+np.sum(wet_image)))

        print('Writing to {}...'.format(output_path))
        hdu = fits.PrimaryHDU(wet_image)
        hdul = fits.HDUList([hdu])
        hdul[0].header = header
        hdul.writeto(output_path, overwrite=True)

if __name__=='__main__':
    do_rain('/home/jsn/landing/data/2023-08-01/reduced/')
