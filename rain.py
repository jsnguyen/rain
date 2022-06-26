from multiprocessing import Pool, Array
import multiprocessing

import numpy as np
from scipy import special
from tqdm import tqdm

def gaussian_2d(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    '''
    -- Description:
    simple 2d gaussian, NOT radial
    function of x and y -> f(x,y)
    -- Returns:
    2d gaussian at given x,y point
    -- Arguments:
     amplitude: peak height of the gaussian
            x0: x center
            y0: y center
       sigma_x: x axis standard deviation
       sigma_y: y axis standard deviation
    '''
    return amplitude * np.exp(-( (x-x0)*(x-x0) / (2*sigma_x*sigma_x) + (y-y0)*(y-y0) / (2*sigma_y*sigma_y) ) )


def gaussian_2d_norm(sigma_x, sigma_y):
    '''
    -- Description:
    normalized 2d gaussian norm
    -- Returns:
    normalization factor for 2d gaussian
    -- Arguments:
         sigma_x: x axis standard deviation
         sigma_y: y axis standard deviation
    '''

    return 2*np.pi*sigma_x*sigma_y

def gaussian_2d_lut(n_div, sigma, radius):
    '''
    -- Description:
    lookup table generator for 2d gaussian
    only need half of the distribution since it is symmetric
    note the extra factor of 2 in the denominator for a 2d gaussian
    -- Returns:
    array of 2d gaussian values
    -- Arguments:
           n_div: # of divisions for the kernel
           sigma: the sigma value of the gaussian
          radius: the length of the gaussian from the center to the edge
    '''

    #xs = np.linspace(-radius, radius, n_div)
    xs = np.linspace(0, radius, n_div)
    return np.exp(-(xs*xs)/(2*sigma*sigma))

def lanczos3(x, y, amplitude, x0, y0, radius):
    '''
    -- Description:
    lanczos3 window, but scaled so that the domain is from -radius to radius
    function of x and y -> f(x,y)
    note that numpy sinc is the normalized sinc function
    -- Arguments:
       amplitude: peak height of the kernel
              x0: x center
              y0: y center
    radius: half the width of the window
    '''

    # width of unscaled lanczos3 is 6 and the width of our rescaled window is 2*radius
    # therefore our scaling factor is 3 / radius
    # this scales x and y to fit within -3 to 3
    xoff = (x-x0) * 3 / radius
    yoff = (y-y0) * 3 / radius

    # a window from -3 to 3 for lanczos3
    if -3 <= xoff <= 3 and -3 <= yoff <= 3:
        return amplitude*np.sinc(xoff)*np.sinc(xoff/3)*np.sinc(yoff)*np.sinc(yoff/3)
    else:
        return 0

def lanczos3_norm():
    '''
    -- Description:
    coeff is the double integral of the lanczos3 function from -radius to radius
    -- Returns:
    the normalization factor for the lanczos3 window
    '''

    Si = lambda x : special.sici(x)[0] # the sine integral
    return 4/(np.pi*np.pi) * (Si(2*np.pi) - 2 * Si(4*np.pi)) * (Si(2*np.pi) - 2 * Si(4*np.pi)) # from wolfram alpha...

def lanczos3_lut(res):
    '''
    -- Description:
    lookup table generator for lanczos3 kernel
    only need half of the distribution since it is symmetric
    note the extra factor of 2 in the denominator for a 2d gaussian
    -- Returns:
    array of lanczos3 function
    -- Arguments:
           n_div: # of divisions for the kernel
    '''

    xs = np.linspace(0, 3, res)
    return np.sinc(xs)*np.sinc(xs/3)

def sprinkle(params):
    '''
    -- Description:
    sprinkle a single pixel at coordinates j,i (x,y) onto the image
    a single input pixel may affect multiple output image pixels
    -- Returns:
    if parallel, returns nothing since it modifies shared memory arrays
    if serial, returns the wet image and missed pixel values
    -- Arguments:
          params: list of tuple since Pool only takes one argument

              ix: the x dimension of the image
              iy: the y dimension of the image
        norm_val: the value of the pixel that we are sprinkling
    scaled_new_x: the new x coordinate of the pixel center
    scaled_new_y: the new y coordinate of the pixel center
           width: the width parameter specific to the kernel you are using
          kernel: either 'gaussian' or 'lanczos3'
    padded_shape: the new shape of the new image, with padding around the edges
           n_div: the number times to divide a given pixel
           n_cpu: the number of cores used, not used if serial
        parallel: True or False for parallelization
    '''

    ix, iy, norm_val, scaled_new_x, scaled_new_y, xmin, xmax, ymin, ymax, width, kernel, n_lut, pixel_size, padded_shape, n_div, n_cpu, parallel = params

    # define global variable of shared memory and get the cpu_id
    # only relevant for parallel processes
    if parallel:
        global global_wet_arr
        global global_missed_pixel_arr
        global global_lut
        cpu_id = ( (multiprocessing.current_process()._identity[0]) - 1 ) % n_cpu # -1 to index at 0
    else:
        # if serial run, make the initialize image here
        # otherwise, parallel runs used the shared memory global variables
        wet_image = np.zeros(padded_shape)
        missed_pixels = 0
        
    if not '_lut' in kernel:

        if kernel == 'gaussian':
            scaled_pixel_kernel = lambda x, y : gaussian_2d(x, y, 1, 0, 0, width, width)
        elif kernel == 'lanczos3':
            scaled_pixel_kernel = lambda x, y : lanczos3(x, y, 1, 0, 0, width)

        # main loop
        # calculate the kernel value at the center of the pixel and write it to the pixel
        for i in np.arange(xmin, xmax, 1, dtype=int):
            for j in np.arange(ymin, ymax, 1, dtype=int):

                # get pixel center
                pc_x = i + pixel_size/2
                pc_y = j + pixel_size/2

                # offset by new x and new y coordinate
                pc_x -= scaled_new_x
                pc_y -= scaled_new_y

                # rebin into closest pixel
                ii = int(i/n_div)
                jj = int(j/n_div)

                if parallel:
                    # make sure that the coordinates are not out of bounds
                    if ii < 0 or ii > ix-1 or jj < 0 or jj > iy-1:
                        global_missed_pixel_arr[cpu_id] += norm_val * scaled_pixel_kernel(pc_x, pc_y)
                    else:
                        global_wet_arr[cpu_id][jj,ii] += norm_val * scaled_pixel_kernel(pc_x, pc_y) # write to shared memory according to cpu_id

                else:
                    # serial
                    if ii < 0 or ii > ix-1 or jj < 0 or jj > iy-1:
                        missed_pixels += norm_val * scaled_pixel_kernel(pc_x, pc_y)
                    else:
                        wet_image[jj,ii] += norm_val * scaled_pixel_kernel(pc_x, pc_y)

    else:

        # jj,ii are the x,y pixel coordinates on the final image
        # j,i are the x,y coordinates for the lookup table
        for i in np.arange(xmin, xmax, 1, dtype=int):
            for j in np.arange(ymin, ymax, 1, dtype=int):

                # translate to pixel centers
                pc_x = i+pixel_size/2
                pc_y = j+pixel_size/2

                # offset to the new x/y coordinates
                pc_x -= scaled_new_x
                pc_y -= scaled_new_y

                # scale half the width to the length of the lookuptable
                pc_x *= n_lut/(width/2*n_div)
                pc_y *= n_lut/(width/2*n_div)

                # absolute value to flip the coordinate across y axis since distributions are symmetric
                pc_x = np.abs(pc_x)
                pc_y = np.abs(pc_y)
                
                # if not in lookup table range then skip it since it is not in function
                if pc_x < 0 or pc_x > n_lut-1 or pc_y < 0 or pc_y > n_lut-1:
                    continue

                # cast to integers
                pc_x = int(pc_x)
                pc_y = int(pc_y)

                # rebin into closest pixel
                ii = int(i/n_div)
                jj = int(j/n_div)

                if parallel:
                    # make sure that the coordinates are not out of bounds
                    if ii < 0 or ii > ix-1 or jj < 0 or jj > iy-1:
                        global_missed_pixel_arr[cpu_id] += norm_val * global_lut[pc_x] * global_lut[pc_y]
                    else:
                        global_wet_arr[cpu_id][jj,ii] += norm_val * global_lut[pc_x] * global_lut[pc_y]

                else:
                    # serial
                    if ii < 0 or ii > ix-1 or jj < 0 or jj > iy-1:
                        missed_pixel_arr[cpu_id] += norm_val * lut[pc_x] * lut[pc_y]
                    else:
                        wet_arr[cpu_id][jj,ii] += norm_val * lut[pc_x] * lut[pc_y]

    # parallel doesn't return anything, has the shared memory arrays instead
    if parallel:
        return 
    # serial returns the wet_image
    else:
        return wet_image, missed_pixels

def init_arr(arrs, missed_pixel_arr, lut, shape):
    '''
    -- Description:
    only used for parallelization initialization
    initialize each of the shared arrays by assigning the memory to a global variable
    -- Returns:
    nothing, initializes shared memory
    -- Arguments:
             arr: List of arrays for each cpu process, len(arr)==n_cpu
              ix: x image dimension
              iy: y image dimension
    '''

    global global_wet_arr
    global global_missed_pixel_arr
    global global_lut
    global_wet_arr =[]
    for el in arrs:
        global_wet_arr.append(np.frombuffer(el, dtype='float64').reshape(shape)) # hacky thing to make it so the array still behaves like a numpy array (but it isn't actually)

    global_missed_pixel_arr = missed_pixel_arr
    global_lut = lut

def rain(image, pixel_frac, new_pc_coords, n_div, n_sigma=4, n_pad=0, kernel='gaussian', n_lut=65536, pixel_size=1, bad_pixel_map=None, parallel=False, n_cpu=1):
    '''
     -- Description:
     main function for raining on an entire image
     photometry preserving algorithm for applying linear distortion corrections
     this function is parallelized, each process gets its own output image and recombines into one image at the end, no collisions possible if each process gets its own image
     requires (n_cpu * (ix+2*n_pad) * (iy+2*n_pad) * 64 bits) of memory to output the wet_image
     requires (n_cpu * (ix+2*n_pad)*n_div * (iy+2*n_pad)*n_div * 64 bits) of memory to output the wet_div_image
     -- Returns:
     the "rained" wet image and missed_pixel values
     -- Arguments:
                  image: the image we are raining
             pixel_frac: fractional length of the pixel
          new_pc_coords: list of new pixel CENTERS x',y' after distortion. (post-distortion coordinates) 
                  n_div: number of subpixels to divide original pixel
                n_sigma: number of sigma to evaluate gaussian over (only applies to gaussian kernel)
                  n_pad: number of extra rows to pad on final wet image
                 kernel: either 'gaussian' or 'lanczos3'
               parallel: use parallelization
    '''

    # image dimensions
    iy,ix = image.shape

    # generate all combinations of coordinates
    xs = np.arange(0,ix,1)
    ys = np.arange(0,iy,1)
    grid = np.meshgrid(xs, ys)
    index_coords = np.stack(grid).T.reshape(-1,2)

    # padded image dimensions
    padded_ix = ix+2*n_pad
    padded_iy = iy+2*n_pad
    padded_shape = (padded_iy, padded_ix)

    if kernel=='gaussian':

        fwhm_factor = 2*np.sqrt(2*np.log(2)) # FWHM to sigma
        scaled_sigma = pixel_size * pixel_frac * n_div / fwhm_factor # scale sigma to pixel size and FWHM
        norm = gaussian_2d_norm(scaled_sigma, scaled_sigma) # normalize so total volume of the 2d gaussian is the value of the pixel
        width = scaled_sigma
        sigma_range = n_sigma * scaled_sigma

        window = sigma_range

    elif kernel=='gaussian_lut':

        fwhm_factor = 2*np.sqrt(2*np.log(2)) # FWHM to sigma
        sigma = pixel_frac / fwhm_factor # we want the FWHM of the gaussian to be the fraction of the pixel
        radius = n_sigma*sigma # radius of the gaussian kernel, must truncate at some point
        width = 2*radius

        # get normalization factor
        norm = n_div*n_div*gaussian_2d_norm(sigma, sigma)

        scaled_sigma = sigma * n_div # scale sigma to pixel size and FWHM
        sigma_range = n_sigma * scaled_sigma

        window = sigma_range

        lut = gaussian_2d_lut(n_lut, sigma, radius)
    
    elif kernel=='lanczos3':

        radius = 3 * pixel_frac * n_div
        width = radius # width factor for lanczos3
        norm = n_div*n_div*lanczos3_norm()

        window = radius

    elif kernel=='lanczos3_lut':

        radius = 3*pixel_frac # lanczos3 is only defined from -3 to 3
        width = 2*radius # total width of the kernel

        # get normalization factor
        norm = n_div*n_div*lanczos3_norm()

        scaled_radius = radius*n_div

        window = scaled_radius

        lut = lanczos3_lut(n_lut)

    # get list of parameters to put into the pool
    # the only things that vary per run are the original pixel coordinates and the new pixel coordinates
    # precomputing the new coordinates is faster that computing on the fly
    params = []
    for (index_x, index_y), (new_x, new_y) in tqdm(zip(index_coords,new_pc_coords), desc='Params', total=len(index_coords)):

        # actual value of the pixel we're sprinkling
        val = image[index_y, index_x]

        # pixel center coordinates
        # add offset due to padding and scale by n_div
        scaled_new_x = (new_x+n_pad) * n_div
        scaled_new_y = (new_y+n_pad) * n_div

        xmin = np.floor(scaled_new_x - window)
        xmax = np.ceil(scaled_new_x + window)
        ymin = np.floor(scaled_new_y - window)
        ymax = np.ceil(scaled_new_y + window)

        # the normalized pixel value, multiply this factor for scaling the kernel
        norm_val = val / norm

        # if no bad pixel map we don't need to filter any parameters
        if bad_pixel_map is None:
            params.append((ix, iy, norm_val, scaled_new_x, scaled_new_y, xmin, xmax, ymin, ymax, width, kernel, n_lut, pixel_size, padded_shape, n_div, n_cpu, parallel))
        else:
            # skip the bad pixels and don't add them to the parameters list
            if not bad_pixel_map[index_y,index_x]:
                params.append((ix, iy, norm_val, scaled_new_x, scaled_new_y, xmin, xmax, ymin, ymax, width, kernel, n_lut, pixel_size, padded_shape, n_div, n_cpu, parallel))

    wet_image = np.zeros(padded_shape)
    if parallel:

        # if we operated on a single array, there is a possibility for thread collisions
        # if each process gets its own shared memory array, there is no possibility for collisions
        # there are n_cpu arrays to avoid collisions, this is also very fast since we don't need resource locks
        # this is not terribly memory-efficient, however
        # note that wet_arr is a local variable here
        # using shared memory Array type from multiprocessing
        wet_arr = []
        for el in range(n_cpu):
            wet_arr.append(Array('d', np.zeros(padded_shape).flatten(), lock=False))

        # also initialize an Array for the missed pixel values, one value for each CPU
        missed_pixel_arr = Array('d', np.zeros(n_cpu), lock=False)

        if '_lut' in kernel:
            lut = Array('d', lut, lock=False)
        else:
            lut = None

        # initialze the all the arrays
        # initialize map the memory of local wet_arr to a global variable in init_arr
        # cast to list to that the iterator actually runs
        # doesn't return anything, modifies the wet_arr variable
        # pool hands out chunks to each process and each process gets its own shared memory image that is shared between iterations
        with Pool(n_cpu, initializer=init_arr, initargs=(wet_arr, missed_pixel_arr, lut, padded_shape)) as pool:
            list(tqdm(pool.imap(sprinkle, params), total=len(params), desc='Rain  ')) # parallel runs use imap to work with tqdm

        # sum over all of the arrays from each of our processes to get our final wet_image
        for el in wet_arr:
            wet_image += np.frombuffer(el, dtype='float64').reshape(padded_shape) # still need to convert to a numpy array in the end

        # sum over all the missed pixels per process
        missed_pixels = np.sum(missed_pixel_arr)

    else:

        # serial runs
        # wet image is returned in serial runs
        for p in tqdm(params, total=len(params), desc='Rain  '):
            wi, mp = sprinkle(p)
            wet_image += wi
            missed_pixels += mp

    # if we have a bad pixel map, set the bad pixels back to the original value they were before we applied rain to the image
    if bad_pixel_map is not None:
        for (index_x,index_y) in tqdm(index_coords, desc='Reset bad pixels', total=len(index_coords)):
            if bad_pixel_map[index_y,index_x]:
                # new wet image coordinates are offset by n_pad since we added n_pad rows and columns around the image
                wet_image[index_y+n_pad,index_x+n_pad] = image[index_y,index_x]

    return wet_image, missed_pixels
