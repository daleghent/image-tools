# Master Flat Field Generator

This Python utility calibrates and combines flat field FITS images into a normalized master flat. It supports optional bias and dark frame subtraction and applies sigma clipping to reject outliers during combination.

## Features

- Loads flat FITS files from a directory
- Optional calibration using bias and/or dark frames
- Automatic extraction of filter and binning information
- Sigma-clipped averaging to combine calibrated flats
- Output master flat includes metadata about processing

## Usage

```bash
./mkflat.py <input_dir> [output_dir] [options]
```

### Positional Arguments

- `input_dir` – Directory containing uncalibrated flat FITS files.
- `output_dir` – Optional. Directory where the master flat will be saved (default: current directory).

### Optional Arguments

- `-l`, `--limit <N>` – Limit the number of flat files to use.
- `-B`, `--bias_file <file>` – Path to a master bias FITS file.
- `-D`, `--dark_file <file>` – Path to a master dark FITS file.
- `--sigma-low <value>` – Low sigma clipping threshold (default: 3.0).
- `--sigma-high <value>` – High sigma clipping threshold (default: 3.0).

## Output

The script produces a single master flat FITS file named:

```
masterFLAT_<FILTER>_<BINNING>.fits
```

Example: `masterFLAT_R_1x1.fits`

This file will contain keywords:

- `IMAGETYP = 'masterFLAT'`
- `NORM = True` (indicates normalization to mean=1.0)
- `NFRAMES` – number of input frames used
- `SIGMALO`, `SIGMAHI` – clipping thresholds used

## Requirements

- Python 3
- [astropy](https://www.astropy.org/)
- [ccdproc](https://ccdproc.readthedocs.io/)
- numpy

Install dependencies via pip:

```bash
pip install astropy ccdproc numpy
```

## License

MIT License © 2025 Dale Ghent <daleg@elemental.org>
