#!/bin/env python3

import os
import argparse
import numpy as np
from astropy.nddata import CCDData
from skimage.measure import block_reduce
from astropy.io import fits

def bin_fits_image(input_path, output_path, bin_factor):
    """Bins a FITS image by the specified bin factor."""
    try:
        # Open the FITS file as CCDData
        with fits.open(input_path) as hdul:
            # Convert the primary HDU data to CCDData
            ccd = CCDData(hdul[0].data, unit="adu")

            # Perform binning using block_reduce
            binned_data = block_reduce(ccd.data, block_size=(bin_factor, bin_factor), func=np.mean)

            # Update the FITS header to reflect binning
            hdul[0].data = binned_data
            hdul[0].header['XBINNING'] = bin_factor
            hdul[0].header['YBINNING'] = bin_factor

            # Write the binned image to the output path
            hdul.writeto(output_path, overwrite=True)
            print(f"Binned image saved to {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Bin FITS images in a specified directory.")
    parser.add_argument(
        "-b", "--binning", type=int, required=True,
        help="Binning factor (e.g., 2 for 2x2 binning)."
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing input FITS files."
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to save binned FITS files."
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each FITS file in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".fits") or filename.endswith(".fit"):
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)

            print(f"Processing {filename}...")
            bin_fits_image(input_path, output_path, args.binning)

if __name__ == "__main__":
    main()
