# Rain

Pure Python implementation of Drizzle. Used for photometry-preserving image distortion correction.

# Requirements

`numpy`, `scipy`, `tqdm`

## Rain

First thing you should look at is `rain_example.py`, this file shows how to set up and use Rain. Essentially you pass a distortion map and the corresponding image, and Rain applies the distortion map to the image. Docstrings are also implemented for most of the functions that describe their usage. The `rain.py` is also well commented but further documentation is a work in progress.

Make sure you generate check images:

```python3 generate_check_images.py```

Check images are just fake images to test the performance of Rain.

To test Rain on all of the check images:

```python3 rain_check_images.py```

## Misc

See `docs/Rain_Explanation.pdf` for a more technical explanation.

See `notebooks/rain_drizzle_comparison.ipynb` for a comparison between Rain and Drizzle.
