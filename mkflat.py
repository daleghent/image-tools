#!/bin/env python3

import os
import argparse
import warnings
import numpy as np
from pathlib import Path
from ccdproc import ImageFileCollection, CCDData, combine, subtract_bias, subtract_dark
from astropy import units as u
from astropy.io import fits
from astropy.stats import mad_std
from astropy.wcs import FITSFixedWarning

# Constants
default_sigma_high = 3.0
default_sigma_low = 3.0
FITS_EXTENSIONS = ('.fit', '.fits', '.fts', '.fits.fz')

# Suppress WCS-related warnings from FITS headers
warnings.simplefilter('ignore', category=FITSFixedWarning)

def main():
    parser = argparse.ArgumentParser(description="Calibrate and combine flat images to create a master flat.")
    parser.add_argument("input_dir", help="Directory containing the uncalibrated flat images.")
    parser.add_argument("output_dir", nargs="?", default=".", help="Directory to store the calibrated master flat (default: current directory).")
    parser.add_argument("-l", "--limit", type=int, help="Limit files used to this number (optional)")
    parser.add_argument("-D", "--dark_file", help="Path to the dark calibration file (optional).")
    parser.add_argument("-B", "--bias_file", help="Path to the bias calibration file (optional).")
    parser.add_argument("-s", "--scale", action="store_true", help="Enable scaling of each flat by its median before combination.")
    parser.add_argument("--sigma-low", type=float, default=default_sigma_low, help=f"Low threshold for sigma clipping (default: {default_sigma_low})")
    parser.add_argument("--sigma-high", type=float, default=default_sigma_high, help=f"High threshold for sigma clipping (default: {default_sigma_high})")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    file_limit = args.limit if args.limit else -1
    dark_file = Path(args.dark_file) if args.dark_file else None
    bias_file = Path(args.bias_file) if args.bias_file else None
    sigma_low = args.sigma_low
    sigma_high = args.sigma_high
    enable_scaling = args.scale

    if not input_dir.is_dir():
        raise ValueError(f"Input directory '{input_dir}' does not exist.")

    # Load bias frame if provided
    bias_ccd = None
    if bias_file:
        if not bias_file.is_file():
            raise ValueError(f"Bias file '{bias_file}' does not exist.")
        print(f"Reading bias file {bias_file.name}")
        bias_ccd = CCDData.read(bias_file, unit="adu")

    # Load dark frame if provided
    dark_ccd = None
    if dark_file:
        if not dark_file.is_file():
            raise ValueError(f"Dark file '{dark_file}' does not exist.")
        print(f"Reading dark file {dark_file.name}")
        dark_ccd = CCDData.read(dark_file, unit="adu")

    output_dir.mkdir(parents=True, exist_ok=True)

    flat_collection = ImageFileCollection(input_dir)
    flat_files = sorted([f for f in flat_collection.files if f.lower().endswith(FITS_EXTENSIONS)])

    calibrated_flats = []
    filter_name = None
    binning_level = None
    i = 0

    if file_limit > -1:
        print(f"Limiting stack to the first {file_limit} files")

    for flat_path in flat_files:
        if file_limit > -1 and i >= file_limit:
            break

        i += 1
        print(f"Reading flat file {flat_path}")
        try:
            flat_ccd = CCDData.read(input_dir / flat_path, unit="adu")
        except Exception as e:
            print(f"ERROR: Could not read {flat_path}: {e}")
            continue

        # Extract metadata for naming and consistency checking
        try:
            binX = int(flat_ccd.header["XBINNING"])
            binY = int(flat_ccd.header["YBINNING"])
        except KeyError:
            print(f"Skipping {flat_path}: missing XBINNING or YBINNING keyword")
            continue

        if binning_level is None:
            binning_level = f"{binX}x{binY}"

        this_filter = flat_ccd.header.get("FILTER", None)
        if filter_name is None:
            filter_name = this_filter
        elif this_filter != filter_name:
            print(f"WARNING: {flat_path} has different FILTER ({this_filter}) than others ({filter_name})")

        # Apply bias subtraction
        if bias_ccd is not None:
            flat_ccd = subtract_bias(flat_ccd, bias_ccd)

        # Apply dark subtraction
        if dark_ccd is not None:
            exposure_key = "EXPOSURE" if "EXPOSURE" in flat_ccd.header else "EXPTIME"
            try:
                flat_ccd = subtract_dark(flat_ccd, dark_ccd, exposure_time=exposure_key, exposure_unit=u.second)
            except Exception as e:
                print(f"WARNING: Dark subtraction failed on {flat_path}: {e}")
                continue

        calibrated_flats.append(flat_ccd)

    # Release memory used by calibration frames
    del bias_ccd
    del dark_ccd

    if not calibrated_flats:
        print("No valid flat frames were found after filtering. Exiting.")
        return

    flat_count = len(calibrated_flats)

    print(f"Combining {flat_count} calibrated flats using sigma clipping (low={sigma_low}, high={sigma_high})...")
    if enable_scaling:
        print("Scaling enabled: each flat will be normalized by its median before combination.")

    master_flat = combine(
        calibrated_flats,
        method="average",
        scale=(lambda a: 1 / np.median(a)) if enable_scaling else None,
        sigma_clip=True,
        sigma_clip_low_thresh=sigma_low,
        sigma_clip_high_thresh=sigma_high,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std,
        dtype=np.float32
    )

    del calibrated_flats

    # Normalize flat to mean = 1.0
    mean_val = np.mean(master_flat.data)
    if mean_val != 0:
        master_flat.data /= mean_val

    if filter_name is None:
        filter_name = "NOFILTER"
        master_flat.header["FILTER"] = (filter_name, "Name of filter")

    master_flat.header["IMAGETYP"] = ("masterFLAT", "Image type")
    master_flat.header["NORM"] = (True, "Flat field normalized to mean=1.0")
    master_flat.header["NFRAMES"] = (flat_count, "Number of input subframes")
    master_flat.header["SIGMALO"] = (sigma_low, "Sigma clipping low")
    master_flat.header["SIGMAHI"] = (sigma_high, "Sigma clipping high")
    master_flat.header["SCALEMED"] = (enable_scaling, "Input images scaled by median before combination")

    master_flat_name = f"masterFLAT_{filter_name}_{binning_level}.fits"
    master_flat_path = output_dir / master_flat_name

    master_flat.write(master_flat_path, overwrite=True, hdu_mask=None, hdu_uncertainty=None)
    print(f"Master flat saved to {master_flat_path}")

if __name__ == "__main__":
    main()
