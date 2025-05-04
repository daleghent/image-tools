# Master BIAS and DARK Frame Generator

This Python script processes FITS calibration frames (BIAS and DARK) to produce sigma-clipped master calibration frames. It can group files based on metadata, subtract a master BIAS from DARK frames, and name output files with identifying metadata.

## Features

- Automatically groups FITS files by image type, temperature, binning, gain, and readout mode
- Optional bias subtraction for DARK frame calibration
- Sigma-clipped average combining
- Metadata-encoded output filenames for traceability
- Includes detailed statistics and diagnostic output

## Usage

```bash
./mkcalib.py <input_dir> [output_dir] [options]
```

### Positional Arguments

- `input_dir` – Directory containing raw BIAS and DARK FITS files.
- `output_dir` – Directory to write master calibration frames (default: current directory).

### Optional Arguments

- `--sigma-low` – Lower threshold for sigma clipping (default: 3.0)
- `--sigma-high` – Upper threshold for sigma clipping (default: 3.0)
- `--min-frames` – Minimum number of frames required to produce a master (default: 15)
- `--type` – Restrict processing to `BIAS`, `DARK`, or `ALL` (default: ALL)
- `-B`, `--bias-master` – Path to a master BIAS file for dark frame calibration

## Output

The script saves master calibration frames named using metadata from the input files, for example:

- `masterBIAS_1x1_T-20.00_G1.4_O100_HighGain.fits`
- `masterDARK_30.0_1x1_T-20.00_G1.4_O100_HighGain.fits`

Each master file includes FITS header keywords:

- `IMAGETYP` = 'masterBIAS' or 'masterDARK'
- `COMBINED` = True
- `NFRAMES` = number of input frames
- `SIGMALO`, `SIGMAHI` = sigma clipping thresholds

## Requirements

- Python 3
- astropy
- ccdproc
- numpy

Install dependencies:

```bash
pip install astropy ccdproc numpy
```

## License

MIT License © 2025 Dale Ghent <daleg@elemental.org>
