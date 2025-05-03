#!/bin/env python3

import os
import sys
import argparse
import warnings
import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.wcs import FITSFixedWarning
from ccdproc import combine

# Suppress non-critical WCS warnings from Astropy
warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)

# Accepted FITS extensions
FITS_EXTENSIONS = ('.fit', '.fits', '.fts', '.fits.fz')

def create_master_frame(files, output_filename):
    """Combine BIAS or DARK FITS files into a master frame using sigma-clipped average."""
    ccd_list = []

    # Read all CCDData objects from the provided FITS files
    for file in files:
        print(f"Reading in {file}")
        try:
            ccd = CCDData.read(file, unit='adu')
            print(f"  Mean: {np.mean(ccd.data):.2f}, Min: {np.min(ccd.data)}, Max: {np.max(ccd.data)}\n")
            ccd_list.append(ccd)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # If no valid CCDs were read, skip processing
    if not ccd_list:
        print("No valid FITS files could be read. Skipping this group.")
        return

    # Warn user if too few images are being combined
    if len(ccd_list) < 3:
        print("Warning: Less than 3 frames provided; master may be noisy.")

    print(f"{len(ccd_list)} files successfully read.")

    # Perform sigma-clipped average stacking using MAD for dispersion
    master_frame = combine(
        ccd_list,
        method='average',
        dtype=np.float32,
        sigma_clip=True,
        sigma_clip_low_thresh=5,
        sigma_clip_high_thresh=5,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std  # Corrected: typo fixed here
    )

    # Tag header to indicate it's a combined master frame
    master_frame.meta['combined'] = True

    # Write result to disk, excluding mask and uncertainty HDUs
    master_frame.write(output_filename, overwrite=True, hdu_mask=None, hdu_uncertainty=None)

    # Output statistics of the master frame
    print(f"Master frame created: {output_filename}")
    print(f"Mean: {np.mean(master_frame.data):.2f}, Min: {np.min(master_frame.data):.2f}, Max: {np.max(master_frame.data):.2f}")

def organize_files_by_type(directory):
    """Organize files in a directory by their IMAGETYP keyword and relevant metadata."""
    file_groups = {}

    for filename in os.listdir(directory):
        # Process only recognized FITS extensions, case-insensitively
        if not filename.lower().endswith(FITS_EXTENSIONS):
            continue

        filepath = os.path.join(directory, filename)
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header

                # Get the image type from the header (e.g., BIAS, DARK)
                image_type = header.get('IMAGETYP', header.get('FRAMETYP', 'UNKNOWN')).strip().upper()

                # Extract binning and other metadata for grouping
                binning = f"{header.get('XBINNING', 1)}x{header.get('YBINNING', 1)}"
                set_temp = header.get('SET-TEMP', 0)
                exposure_time = header.get('EXPTIME', 0)

                # Determine grouping key by image type
                if image_type == 'DARK':
                    key = (image_type, exposure_time, set_temp, binning)
                elif image_type == 'BIAS':
                    key = (image_type, set_temp, binning)
                else:
                    continue  # Skip unsupported image types

                # Group files by their key
                file_groups.setdefault(key, []).append(filepath)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return file_groups

def generate_master_calibration_frames(input_directory, output_directory):
    """Generate master calibration frames from FITS files in the specified directory."""
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Organize files by image type and relevant grouping metadata
    file_groups = organize_files_by_type(input_directory)

    for key, files in file_groups.items():
        image_type = key[0]  # The first element of the key is the image type

        if image_type == 'BIAS':
            _, set_temp, binning = key
            # Output filename for BIAS frames includes binning
            output_filename = os.path.join(output_directory, f"masterBIAS_{binning}.fits")
        elif image_type == 'DARK':
            _, exposure_time, set_temp, binning = key
            # Output filename for DARK frames includes exposure and binning
            output_filename = os.path.join(output_directory, f"masterDARK_{exposure_time}_{set_temp}_{binning}.fits")
        else:
            continue  # Guard against unsupported types (should not happen)

        # Create the master frame for this group
        create_master_frame(files, output_filename)

def parse_arguments():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Combine BIAS and DARK FITS files into master calibration frames."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing raw calibration FITS files (BIAS, DARK)."
    )
    parser.add_argument(
        "output_dir",
        help="Directory where the master calibration frames will be saved."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse CLI arguments
    args = parse_arguments()

    # Begin master frame generation
    generate_master_calibration_frames(args.input_dir, args.output_dir)
