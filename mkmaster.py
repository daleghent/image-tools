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

default_sigma_high = 3.0
default_sigma_low = 3.0
default_min_frames = 15

def create_master_frame(files, output_filename, sigma_low, sigma_high, min_frames, bias_master=None, image_type=None):
    """Combine BIAS or DARK FITS files into a master frame using sigma-clipped average."""
    ccd_list = []

    for file in files:
        print(f"Reading in {file}")
        try:
            ccd = CCDData.read(file, unit='adu')

            if bias_master is not None and image_type == 'DARK':
                ccd = ccd.subtract(bias_master)

            mean = np.mean(ccd.data)
            std = np.std(ccd.data)
            min_val = np.min(ccd.data)
            max_val = np.max(ccd.data)
            print(f"  Mean: {mean:.2f}, Min: {min_val}, Max: {max_val}, StdDev: {std:.2f}\n") 

            ccd_list.append(ccd)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    nframes = len(ccd_list)

    if nframes < min_frames:
        print(f"Skipped: Only {nframes} valid frames (minimum required: {min_frames})\n")
        return

    if nframes < 3:
        print("Warning: Less than 3 frames provided; master may be noisy.")

    print(f"{nframes} files successfully read.")
    print(f"Combining master {image_type} using sigma clipping: low={sigma_low}, high={sigma_high}")

    master_frame = combine(
        ccd_list,
        method='average',
        dtype=np.float32,
        sigma_clip=True,
        sigma_clip_low_thresh=sigma_low,
        sigma_clip_high_thresh=sigma_high,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=mad_std
    )

    master_frame.meta['IMAGETYP'] = (f"master{image_type}", "Image type")
    master_frame.meta['COMBINED'] = (True, "Image is a combination of subframes")
    master_frame.meta['NFRAMES'] = (nframes, "Number of input subframes")
    master_frame.meta['SIGMALO'] = (sigma_low, "Sigma clipping low")
    master_frame.meta['SIGMAHI'] = (sigma_high, "Sigma clipping high")

    master_frame.write(output_filename, overwrite=True, hdu_mask=None, hdu_uncertainty=None)

    mean = np.mean(master_frame.data)
    min_val = np.min(master_frame.data)
    max_val = np.max(master_frame.data)
    std = np.std(master_frame.data)
    print(f"Master frame created: {output_filename}")
    print(f"Mean: {mean:.2f}, Min: {min_val:.2f}, Max: {max_val:.2f}, StdDev: {std:.2f}\n")

def organize_files_by_type(directory):
    """Organize files in a directory by their IMAGETYP keyword and relevant metadata."""
    file_groups = {}

    for filename in os.listdir(directory):
        if not filename.lower().endswith(FITS_EXTENSIONS):
            continue

        filepath = os.path.join(directory, filename)
        try:
            with fits.open(filepath) as hdul:
                header = hdul[0].header
                image_type = header.get('IMAGETYP', header.get('FRAMETYP', 'UNKNOWN')).strip().upper()
                binning = f"{header.get('XBINNING', 1)}x{header.get('YBINNING', 1)}"
                set_temp = header.get('SET-TEMP', 0)
                exposure_time = header.get('EXPTIME', 0)

                if image_type == 'DARK':
                    key = (image_type, exposure_time, set_temp, binning)
                elif image_type == 'BIAS':
                    key = (image_type, set_temp, binning)
                else:
                    continue

                file_groups.setdefault(key, []).append(filepath)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return file_groups

def generate_master_calibration_frames(input_directory, output_directory, sigma_low, sigma_high, types, min_frames, bias_master_path):
    """Generate master calibration frames from FITS files in the specified directory."""
    os.makedirs(output_directory, exist_ok=True)
    file_groups = organize_files_by_type(input_directory)

    bias_master = None
    if bias_master_path:
        print(f"Loading master BIAS from {bias_master_path}")
        try:
            bias_master = CCDData.read(bias_master_path, unit='adu')
        except Exception as e:
            print(f"Error loading master BIAS: {e}")
            sys.exit(1)

    for key, files in file_groups.items():
        image_type = key[0]

        if image_type not in types:
            continue

        # Extract GAIN, OFFSET/BLKLEVEL, READOUTM from the first file
        try:
            with fits.open(files[0]) as hdul:
                header = hdul[0].header
                ccd_temp = header.get('CCD-TEMP', 0)
                gain = header.get('GAIN', 'unk')
                offset = header.get('OFFSET', header.get('BLKLEVEL', 'unk'))
                readout_mode = header.get('READOUTM', None)
        except Exception as e:
            print(f"Error reading metadata from {files[0]}: {e}")
            continue

        temp_str = f"T{ccd_temp:.2f}"
        gain_str = f"G{gain}"
        offset_str = f"O{offset}"
        suffix = f"{temp_str}_{gain_str}_{offset_str}"

        # Sanitize readout mode for filename
        if readout_mode is not None:
            readout_mode = str(readout_mode).strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
            suffix = f"{suffix}_{readout_mode}"

        if image_type == 'BIAS':
            _, set_temp, binning = key
            output_filename = os.path.join(output_directory, f"masterBIAS_{binning}_{suffix}.fits")
            create_master_frame(files, output_filename, sigma_low, sigma_high, min_frames, image_type='BIAS')
        elif image_type == 'DARK':
            _, exposure_time, set_temp, binning = key
            output_filename = os.path.join(output_directory, f"masterDARK_{exposure_time}_{binning}_{suffix}.fits")
            create_master_frame(files, output_filename, sigma_low, sigma_high, min_frames, bias_master, image_type='DARK')

def parse_arguments():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Combine BIAS and DARK FITS files into master calibration frames."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing FITS calibration subframes (BIAS, DARK)."
    )
    parser.add_argument(
        "output_dir", nargs='?', default='.',
        help="Directory where the master calibration frames will be saved (default: current directory)."
    )
    parser.add_argument(
        "--sigma-low", type=float, default=default_sigma_low,
        help=f"Lower threshold for sigma clipping (default: {default_sigma_low})"
    )
    parser.add_argument(
        "--sigma-high", type=float, default=default_sigma_high,
        help=f"Upper threshold for sigma clipping (default: {default_sigma_high})"
    )
    parser.add_argument(
        "--min-frames", type=int, default=default_min_frames,
        help=f"Minimum number of valid frames required to create a master frame (default: {default_min_frames})"
    )
    parser.add_argument(
        "--type", choices=['BIAS', 'DARK', 'ALL'], default='ALL',
        help="Restrict processing to BIAS, DARK, or ALL types (default: ALL)"
    )
    parser.add_argument(
        "-B", "--bias-master",
        help="Optional master BIAS frame to subtract from DARK frames before combining"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    types = ['BIAS', 'DARK'] if args.type == 'ALL' else [args.type]

    generate_master_calibration_frames(
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        sigma_low=args.sigma_low,
        sigma_high=args.sigma_high,
        types=types,
        min_frames=args.min_frames,
        bias_master_path=args.bias_master
    )
