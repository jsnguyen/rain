# Rain

Pure Python implementation of Drizzle. Used for photometry preserving distortion correction.

# Requirements

```numpy```

```tqdm```

## Rain

`rain_example.py` shows how to set up and use Rain. Docstrings are also implemented for most of the functions that describe their usage.

First make sure you generate check images:

```python3 generate_check_images.py```

Test Rain on all of the check images:

```python3 rain_check_images.py```

## Misc

See `docs/Rain_Explanation.pdf` for a more technical explanation.

See `notebooks/rain_drizzle_comparison.ipynb` for a comparison between the two methods.
