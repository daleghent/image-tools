# Python Environment Setup for FITS Processing Tools

This guide helps you create a dedicated Python virtual environment to run the FITS calibration and processing scripts:

- `mkflat.py` – Master flat field generator
- `mkmaster.py` – Master bias and dark frame generator
- `imgproc.py` – Light image calibration, binning, alignment, and stacking

## Step-by-Step Instructions

### 1. Clone or Copy the Scripts

Ensure you have the three scripts available in your working directory.

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv fitsenv
source fitsenv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

This will install:

- `astropy`
- `ccdproc`
- `numpy`
- `scipy`
- `scikit-image`

### 4. Run the Scripts

Use the scripts as needed, for example:

```bash
mkmaster.py /path/to/darks ./output_calib
mkflat.py /path/to/flats ./output_flats -B master_bias.fits -D master_dark.fits
imgproc.py /path/to/lights ./output_lights -B ./output_calib -D ./output_calib -F ./output_flats --stack
```

### 5. Deactivate the Environment

When done, deactivate the environment:

```bash
deactivate
```

## License

MIT License © 2025 Dale Ghent <daleg@elemental.org>
