# FITS Image Calibration and Stacking Tool

This script performs full-frame calibration, binning, optional alignment, and optional stacking on astronomical FITS images using provided master calibration frames (bias, dark, flat). It is optimized for bulk processing with multiprocessing support.

## Features

- Supports bias, dark, and flat calibration using metadata-matched master frames
- Optional image alignment using phase cross-correlation
- Image binning by user-defined factor
- Optional temporal stacking of frames into grouped exposures
- Parallel processing using Python multiprocessing
- Output filenames include date, JD, object, and observer metadata

## Usage

```bash
imgproc.py <input_dir> [output_dir] [options]
```

### Positional Arguments

- `input_dir` – Directory containing raw FITS files.
- `output_dir` – Directory to save calibrated images (default: dynamically generated).

### Options

- `-B, --bias-dir <dir>` – Directory containing master bias frames.
- `-D, --dark-dir <dir>` – Directory containing master dark frames.
- `-F, --flat-dir <dir>` – Directory containing master flat frames.
- `-b, --bin-level <N>` – Binning level (default: 1).
- `-a, --align` – Enable alignment of stacked images.
- `-s, --stack` – Enable stacking of groups of images.
- `-S, --stack-threshold <N>` – Threshold number of frames before stacking is triggered (default: 150).
- `-o, --observer <ID>` – Observer code to include in output headers (default: DGAH).
- `-f, --filter <value>` – Only process images with this FILTER header value.
- `-c, --concurrency <N>` – Number of processes to use (default: CPU core count).
- `-v, --verbose` – Enable verbose diagnostic output.

## Output

Each calibrated (or stacked) FITS image will be written to the output directory with a filename like:

```
<OBJECT>_<OBSERVER>_<FILTER>_<DATE>_<JD>.fits
```

Example:
```
NGC1234_DGAH_V_20250501_2460457.12500000.fits
```

Each output FITS header includes:

- `BIASCAL`, `DARKCAL`, `FLATCAL` – Flags indicating calibration status
- `OBSERVER` – Observer code
- `DATE-AVG` – Average exposure time (for stacked frames)
- Updated binning metadata (`XBINNING`, `YBINNING`, etc.)

## Requirements

- Python 3
- `astropy`
- `ccdproc`
- `numpy`
- `scipy`
- `scikit-image`

Install dependencies:

```bash
pip install astropy ccdproc numpy scipy scikit-image
```

## License

MIT License © 2025 Dale Ghent <daleg@elemental.org>
