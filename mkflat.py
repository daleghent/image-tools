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

warnings.simplefilter('ignore', category=FITSFixedWarning)

def main():
    parser = argparse.ArgumentParser(description="Calibrate and combine flat images to create a master flat.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing the uncalibrated flat images.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to store the calibrated master flat.")
    parser.add_argument("-l", "--limit", type=int, required=False, help="Limit files used to this number (optional)")
    parser.add_argument("-D", "--dark_file", required=False, help="Path to the dark calibration file (optional).")
    parser.add_argument("-B", "--bias_file", required=False, help="Path to the bias calibration file (optional).")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    file_limit = args.limit if args.limit else -1
    dark_file = Path(args.dark_file) if args.dark_file else None
    bias_file = Path(args.bias_file) if args.bias_file else None

    # Validate input directory
    if not input_dir.is_dir():
        raise ValueError(f"Input directory '{input_dir}' does not exist.")

    # Validate bias file if provided
    bias_ccd = None
    if bias_file:
        if not bias_file.is_file():
            raise ValueError(f"Bias file '{bias_file}' does not exist.")
        print(f"Reading bias file {Path(bias_file).name}")
        bias_ccd = CCDData.read(bias_file, unit="adu")

    # Validate dark file if provided
    dark_ccd = None
    if dark_file:
        if not dark_file.is_file():
            raise ValueError(f"Dark file '{dark_file}' does not exist.")
        print(f"Reading dark file {Path(dark_file).name}")
        dark_ccd = CCDData.read(dark_file, unit="adu")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create an ImageFileCollection for the input directory
    flat_collection = ImageFileCollection(input_dir, glob_include="*.fits")

    calibrated_flats = []
    filter_name = None
    binning_level = None
    i = 0

    if file_limit > -1:
        print(f"Limiting stack to the first {file_limit} files")

    for flat_path in flat_collection.files:
        if file_limit > -1 and i >= file_limit:
            break

        i += 1

        # Load the flat image
        print(f"Reading flat file {flat_path}")
        flat_ccd = CCDData.read(input_dir / flat_path, unit="adu")

        # Apply bias correction if provided
        if bias_ccd is not None:
            flat_ccd = subtract_bias(flat_ccd, bias_ccd)

        # Subtract the dark if provided
        if dark_ccd is not None:
            flat_ccd = subtract_dark(flat_ccd, dark_ccd, exposure_time="EXPOSURE", exposure_unit=u.second)

        # Add the calibrated flat frame data to the list of calibrated flats
        calibrated_flats.append(flat_ccd)

        # Extract metadata once for creating the master flat file name
        if filter_name is None:
            filter_name = flat_ccd.header.get("FILTER", "UNKNOWN-FILTER")
            if filter_name == "UNKNOWN-FILTER":
                print("FATAL: Could not find the FILTER keyword in the first flat frame!")
                exit(1)
        if binning_level is None:
            binX = flat_ccd.header.get("XBINNING", "UNKNOWN-BINX")
            binY = flat_ccd.header.get("YBINNING", "UNKNOWN-BINY")
            binning_level = f"{binX}x{binY}"

    # These don't need to occupy memory anymore
    bias_ccd = None
    dark_ccd = None

    # Combine the calibrated flats with sigma clipping
    print("Combining calibrated flats into a master flat...")
    master_flat = combine(
        calibrated_flats,
        method="average",
        dtype=np.float32
    )

    # Create the master flat filename
    master_flat_name = f"masterFLAT_{filter_name}_{binning_level}.fits"
    master_flat_path = output_dir / master_flat_name

    # Update the IMAGETYP keyword
    master_flat.header["IMAGETYP"] = "masterFLAT"

    # Save the master flat
    master_flat.write(master_flat_path, overwrite=True)
    print(f"Master flat saved to {master_flat_path}")

if __name__ == "__main__":
    main()
