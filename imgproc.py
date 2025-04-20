#!/bin/env python3

import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.nddata import CCDData
from astropy import units as u
from ccdproc import combine, subtract_bias, subtract_dark, flat_correct
from multiprocessing import Pool, Manager, cpu_count
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

default_bin = 1
default_stackthresh = 150
default_observer = "DGAH"
default_concurrency = cpu_count()

FITS_EXTENSIONS = ('.fits', '.fit', '.fts', '.ftz', '.fz', '.fits.gz')
groups = []

def calibrate_image(light_hdu, bias_dir, dark_dir, flat_dir, verbose):
    ccd = CCDData(light_hdu.data, meta=light_hdu.header, unit="adu")

    def get_frame_data(master_dir, keywords, header):
        if not master_dir:
            return None
        for file in os.listdir(master_dir):
            if file.endswith(FITS_EXTENSIONS):
                with fits.open(os.path.join(master_dir, file)) as hdul:
                    master_header = hdul[0].header
                    if all(header.get(key) == master_header.get(key) for key in keywords):
                        return CCDData(hdul[0].data, meta=hdul[0].header, unit="adu")
        return None

    master_bias = get_frame_data(bias_dir, ['SET-TEMP', 'GAIN', 'OFFSET', 'READOUTM', 'INSTRUME'], light_hdu.header)
    if bias_dir and master_bias is None:
        print("Error: No applicable bias calibration frame found in the specified directory.")
        exit(1)

    if master_bias is not None:
        ccd = subtract_bias(ccd, master_bias)

    master_dark = get_frame_data(dark_dir, ['EXPOSURE', 'SET-TEMP', 'GAIN', 'OFFSET', 'READOUTM', 'INSTRUME'], light_hdu.header)
    if dark_dir and master_dark is None:
        print("Error: No applicable dark calibration frame found in the specified directory.")
        exit(1)

    if master_dark is not None:
        ccd = subtract_dark(ccd, master_dark, exposure_time='EXPOSURE', exposure_unit=u.s)

    master_flat = get_frame_data(flat_dir, ['FILTER'], light_hdu.header)
    if flat_dir and master_flat is None:
        print("Error: No applicable flat calibration frame found in the specified directory.")
        exit(1)

    if master_flat is not None:
        ccd = flat_correct(ccd, master_flat)

    return ccd.data.astype(np.float32)

def align_images(reference_image, target_image, verbose):
    shift_result, error, diffphase = phase_cross_correlation(reference_image, target_image, upsample_factor=10)
    aligned_image = shift(target_image, shift_result)
    return aligned_image

def bin_image(data, bin_level, verbose):
    shape = (data.shape[0] // bin_level, bin_level, data.shape[1] // bin_level, bin_level)
    return data[:shape[0] * bin_level, :shape[2] * bin_level].reshape(shape).mean(axis=(1, 3))

def process_single_image(file_path, bias_dir, dark_dir, flat_dir, bin_level, observer, output_dir, object_name, verbose):
    with fits.open(file_path) as hdul:
        calibrated_data = calibrate_image(hdul[0], bias_dir, dark_dir, flat_dir, verbose)
        binned_data = bin_image(calibrated_data, bin_level, verbose) if bin_level > 1 else calibrated_data

        header = hdul[0].header
        header['XBINNING'] = bin_level
        header['XPIXSZ'] *= bin_level
        header['YBINNING'] = bin_level
        header['YPIXSZ'] *= bin_level
        header['OBSERVER'] = observer
        filter = header['FILTER']

        try:
            ts = Time(header['DATE-OBS'], format='isot', scale='utc')
        except Exception as e:
            print(f"Error parsing DATE-OBS in {file_path}: {e}")
            return

        ts_iso = ts.to_value('iso', subfmt='date')
        output_filename = f"{object_name}_{observer}_{filter}_{ts_iso.replace('-', '')}_{ts.jd:.3f}.fits"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Writing single file: {output_filename}")
        fits.writeto(output_path, binned_data.astype(np.float32), header, overwrite=True)

def process_group_stack(group_files, bias_dir, dark_dir, flat_dir, bin_level, observer, output_dir, object_name, verbose, align):
    calibrated_images = []
    timestamps = []

    for file_path in group_files:
        with fits.open(file_path) as hdul:
            calibrated_data = calibrate_image(hdul[0], bias_dir, dark_dir, flat_dir, verbose)          
            binned_data = bin_image(calibrated_data, bin_level, verbose) if bin_level > 1 else calibrated_data
            
            if align:
                if calibrated_images:
                    binned_data = align_images(calibrated_images[0], binned_data, verbose)  # Align to the first image

            calibrated_images.append(CCDData(binned_data, unit="adu"))

            try:
                timestamps.append(hdul[0].header['DATE-OBS'])
            except KeyError:
                print(f"Missing DATE-OBS in {file_path}, skipping.")
                continue

    # Use ccdproc.combine to stack the images
    stacked_ccd = combine(
        calibrated_images, 
        method='average',  # You can use 'average', 'median', or other methods
        sigma_clip=True,   # Enable sigma clipping for robust combination
        sigma_clip_low_thresh=3, 
        sigma_clip_high_thresh=3, 
        sigma_clip_func=np.ma.median, 
        sigma_clip_dev_func=np.std
    )

    if not timestamps:
        print("No valid timestamps found for stacking; skipping group.")
        return

    midpoint_timestamp = Time(timestamps, format='isot', scale='utc').mean()
    header = fits.getheader(group_files[0])
    header['DATE-AVG'] = midpoint_timestamp.isot
    header['OBSERVER'] = observer

    midpoint_iso = midpoint_timestamp.to_value('iso', subfmt='date')
    midpoint_jd = midpoint_timestamp.jd
    output_filename = f"{object_name}_{observer}_{midpoint_iso.replace('-', '')}_{midpoint_jd:.3f}.fits"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Writing stacked file: {output_filename}")
    fits.writeto(output_path, stacked_ccd.data.astype(np.float32), header, overwrite=True)

def process_group(group_files, bias_dir, dark_dir, flat_dir, bin_level, observer, output_dir, object_name, verbose, align):
    if len(group_files) == 1:
        process_single_image(group_files[0], bias_dir, dark_dir, flat_dir, bin_level, observer, output_dir, object_name, verbose)
    else:
        process_group_stack(group_files, bias_dir, dark_dir, flat_dir, bin_level, observer, output_dir, object_name, verbose, align)

def process_images(input_dir, output_dir, bin_level, stack, stack_height, bias_dir, dark_dir, flat_dir, observer, verbose, align):

    def get_obs_time_safe(path):
        try:
            return Time(fits.getheader(path)['DATE-OBS'], format='isot', scale='utc')
        except Exception as e:
            print(f"Skipping file {path} due to bad DATE-OBS: {e}")
            return Time('1900-01-01T00:00:00')  # Arbitrary early fallback

    file_paths = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(FITS_EXTENSIONS)],
        key=get_obs_time_safe
    )

    if not file_paths:
        print(f"No FITS files found in {input_dir}")
        return

    first_file_path = file_paths[0]
    with fits.open(first_file_path) as hdul:
        first_header = hdul[0].header
        object_name = first_header.get('OBJECT', 'UNKNOWN').replace(' ', '_')
        obs_date = Time(first_header['DATE-OBS']).to_value('iso', subfmt='date')
        default_output_dir = f"{object_name}_{obs_date}_{observer}_CAL"

    if output_dir is None:
        output_dir = default_output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    global groups

    if (stack):
        groups = [file_paths[i:i + stack_height] for i in range(0, len(file_paths), stack_height)]
        print(f"Calibrating {len(groups)} stacks @ {stack_height} images per stack.")
    else:
        groups = [[fp] for fp in file_paths]
        print(f"Calibrating {len(groups)} images.")

    pool_args = [(group, bias_dir, dark_dir, flat_dir, bin_level, observer, output_dir, object_name, verbose, align) for group in groups]

    print(f"Starting processing with concurrency: {concurrency}")
    with Pool(processes=concurrency) as pool:
        pool.starmap(process_group, pool_args)

def main():
    global concurrency

    parser = argparse.ArgumentParser(description="Calibrate, align, bin, and stack FITS images.")
    parser.add_argument('input_dir', type=str, help="Directory containing source FITS files.")
    parser.add_argument('output_dir', nargs='?', default=None, type=str, help="Directory for output FITS files. Default is '<object_name>_<DATE-OBS>_<observer_code>_STACKED'.")
    parser.add_argument('-a', '--align', action='store_true', help="Enable image alignment.")
    parser.add_argument('-b', '--bin-level', type=int, default=default_bin, help=f"Binning level (default: {default_bin}).")
    parser.add_argument('-c', '--concurrency', type=int, default=default_concurrency, help=f"Number of concurrent threads to use (default: {default_concurrency}).")
    parser.add_argument('-o', '--observer', type=str, default=default_observer, help=f"Observer code (default: {default_observer}).")
    parser.add_argument('-s', '--stack', action='store_true', help="Enable stacking.")
    parser.add_argument('-S', '--stack-threshold', type=int, default=default_stackthresh, help=f"Stacking threshold (default: {default_stackthresh}).")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output.")
    parser.add_argument('-B', '--bias-dir', type=str, help="Directory containing master bias frames.")
    parser.add_argument('-D', '--dark-dir', type=str, help="Directory containing master dark frames.")
    parser.add_argument('-F', '--flat-dir', type=str, help="Directory containing master flat frames.")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    bin_level = args.bin_level
    stack = args.stack
    stack_threshold = args.stack_threshold
    bias_dir = args.bias_dir
    dark_dir = args.dark_dir
    flat_dir = args.flat_dir
    observer = args.observer
    verbose = args.verbose
    concurrency = args.concurrency
    align = args.align

    num_files = len([f for f in os.listdir(input_dir) if f.lower().endswith(FITS_EXTENSIONS)])
    stack_height = 1 if num_files <= stack_threshold else round(num_files / stack_threshold)

    process_images(input_dir, output_dir, bin_level, stack, stack_height, bias_dir, dark_dir, flat_dir, observer, verbose, align)

if __name__ == "__main__":
    main()

