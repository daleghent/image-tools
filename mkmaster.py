#!/bin/env python3

import numpy as np
import os
import sys
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from astropy.stats import mad_std
from ccdproc import combine

def create_master_frame(files, output_filename):
    """Combine FITS files into a master frame using median stacking."""
    header = None

    for file in files:
        ccd_list = []
        try:
            ccd = CCDData.read(file, unit='adu')
            ccd_list.append(ccd)
        except Exception as e:
            print(f"Error reading {file}: {e}")

        if not ccd_list:
            print("No valid FITS files could be read.")
            return

    master_frame = combine(ccd_list, method='average', dtype=np.float32,
            sigma_clip=True, sigma_clip_low_thresh=3, sigma_clip_high_thresh=3,
            sigma_clip_func=np.ma.median, signma_clip_dev_func=mad_std)

    master_frame.write(output_filename, overwrite=True, hdu_mask=None, hdu_uncertainty=None)
    print(f"Master frame created: {output_filename}")

def organize_files_by_type(directory):
    """Organize files in a directory by their IMAGETYP keyword."""
    file_groups = {}

    for filename in os.listdir(directory):
        # Process only FITS files with specific extensions
        if filename.endswith(('.fit', '.fits', '.fts', '.fits.fz')):
            filepath = os.path.join(directory, filename)
            with fits.open(filepath) as hdul:
                header = hdul[0].header

                # Get the image type from the header (e.g., BIAS, DARK, FLAT)
                image_type = header.get('IMAGETYP', 'UNKNOWN').upper()

                # Extract additional information based on image type
                binning = f"{header.get('XBINNING', 1)}x{header.get('YBINNING', 1)}"  # Default binning is 1x1
                set_temp = header.get('SET-TEMP', 0)
                exposure_time = header.get('EXPTIME', 0)

                if image_type == 'DARK':
                    key = (image_type, exposure_time, set_temp, binning)
                elif image_type == 'FLAT':
                    filter_name = header.get('FILTER', 'UNKNOWN')  # Filter name for FLAT frames
                    key = (image_type, filter_name, binning)
                elif image_type == 'BIAS':
                    key = (image_type, set_temp, binning)
                else:
                    continue  # Skip files with unknown or unsupported IMAGETYP

                # Group files by their key
                if key not in file_groups:
                    file_groups[key] = []

                file_groups[key].append(filepath)

    return file_groups

def generate_master_calibration_frames(input_directory, output_directory):
    """Generate master calibration frames from FITS files in the specified directory."""
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Organize files by their type and relevant metadata
    file_groups = organize_files_by_type(input_directory)

    for key, files in file_groups.items():
        image_type = key[0]  # The first element of the key is the image type
        if image_type == 'BIAS':
            set_temp, binning = key[1], key[2]
            # Output filename for BIAS frames includes the binning
            output_filename = os.path.join(output_directory, f"masterBIAS_{binning}.fits")
        elif image_type == 'DARK':
            exposure_time, set_temp, binning = key[1], key[2], key[3]
            # Output filename for DARK frames includes exposure time and binning
            output_filename = os.path.join(output_directory, f"masterDARK_{exposure_time}_{set_temp}_{binning}.fits")
        elif image_type == 'FLAT':
            filter_name, binning = key[1], key[2]
            # Output filename for FLAT frames includes filter name and binning
            output_filename = os.path.join(output_directory, f"masterFLAT_{filter_name}_{binning}.fits")
        else:
            continue  # Skip unsupported image types

        # Create the master frame for the current group of files
        create_master_frame(files, output_filename)

if __name__ == "__main__":
    # Ensure proper usage with two arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_dir> <output_dir>")
        sys.exit(1)

    # Get input and output directories from command-line arguments
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Generate the master calibration frames
    generate_master_calibration_frames(input_dir, output_dir)

