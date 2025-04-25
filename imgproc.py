#!/bin/env python3

import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.nddata import CCDData
from astropy import units as u
from ccdproc import combine, subtract_bias, subtract_dark, flat_correct
from multiprocessing import Pool, cpu_count
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

# Default settings
default_bin = 1
default_stackthresh = 150
default_observer = "DGAH"
default_concurrency = cpu_count()

# Global configuration variables
BIAS_DIR = None
DARK_DIR = None
FLAT_DIR = None
BIN_LEVEL = default_bin
OBSERVER = default_observer
VERBOSE = False
CONCURRENCY = default_concurrency
ALIGN = False

FITS_EXTENSIONS = ('.fits', '.fit', '.fts', '.ftz', '.fz', '.fits.gz')
groups = []


def calibrate_image(light_hdu):
    """Apply bias, dark, and flat corrections to a single HDU."""
    header = light_hdu.header
    ccd = CCDData(light_hdu.data, meta=header, unit="adu")

    def get_frame_data(master_dir, keywords):
        if not master_dir:
            return None
        for fname in os.listdir(master_dir):
            if fname.endswith(FITS_EXTENSIONS):
                with fits.open(os.path.join(master_dir, fname)) as hdul:
                    mh = hdul[0].header
                    if all(header.get(k) == mh.get(k) for k in keywords):
                        return CCDData(hdul[0].data, meta=mh, unit="adu")
        return None

    # Bias
    master_bias = get_frame_data(BIAS_DIR, ['SET-TEMP', 'GAIN', 'OFFSET', 'READOUTM', 'INSTRUME'])
    if BIAS_DIR and master_bias is None:
        print("Error: No applicable bias calibration frame found.")
        exit(1)
    if master_bias is not None:
        ccd = subtract_bias(ccd, master_bias)

    # Dark
    master_dark = get_frame_data(DARK_DIR, ['EXPOSURE', 'SET-TEMP', 'GAIN', 'OFFSET', 'READOUTM', 'INSTRUME'])
    if DARK_DIR and master_dark is None:
        print("Error: No applicable dark calibration frame found.")
        exit(1)
    if master_dark is not None:
        ccd = subtract_dark(ccd, master_dark, exposure_time='EXPOSURE', exposure_unit=u.s)

    # Flat
    master_flat = get_frame_data(FLAT_DIR, ['FILTER'])
    if FLAT_DIR and master_flat is None:
        print("Error: No applicable flat calibration frame found.")
        exit(1)
    if master_flat is not None:
        ccd = flat_correct(ccd, master_flat)

    return ccd.data.astype(np.float32)


def align_images(reference_image, target_image):
    shift_result, _, _ = phase_cross_correlation(reference_image, target_image, upsample_factor=10)
    return shift(target_image, shift_result)


def bin_image(data):
    """Bin image data by the global BIN_LEVEL."""
    if BIN_LEVEL <= 1:
        return data
    ny, nx = data.shape
    by = ny // BIN_LEVEL
    bx = nx // BIN_LEVEL
    data = data[:by * BIN_LEVEL, :bx * BIN_LEVEL]
    reshaped = data.reshape(by, BIN_LEVEL, bx, BIN_LEVEL)
    return reshaped.mean(axis=(1, 3))


def process_single_image(file_path, output_dir, object_name):
    with fits.open(file_path) as hdul:
        calibrated = calibrate_image(hdul[0])
        binned = bin_image(calibrated)
        header = hdul[0].header

        # Update header
        header['XBINNING'] = BIN_LEVEL
        header['XPIXSZ'] *= BIN_LEVEL
        header['YBINNING'] = BIN_LEVEL
        header['YPIXSZ'] *= BIN_LEVEL
        header['OBSERVER'] = OBSERVER
        filt = header.get('FILTER', 'UNKNOWN')

        # Timestamp handling
        try:
            ts = Time(header['DATE-OBS'], format='isot', scale='utc')
        except Exception as e:
            print(f"Error parsing DATE-OBS in {file_path}: {e}")
            return
        ts_iso = ts.to_value('iso', subfmt='date')
        output_fname = f"{object_name}_{OBSERVER}_{filt}_{ts_iso.replace('-', '')}_{ts.jd:.3f}.fits"
        out_path = os.path.join(output_dir, output_fname)

        print(f"Writing single file: {output_fname}")
        fits.writeto(out_path, binned.astype(np.float32), header, overwrite=True)


def process_group_stack(group_files, output_dir, object_name):
    calibrated_list = []
    times = []
    for fp in group_files:
        with fits.open(fp) as hdul:
            cal = calibrate_image(hdul[0])
            binned = bin_image(cal)
            if ALIGN and calibrated_list:
                binned = align_images(calibrated_list[0].data, binned)
            calibrated_list.append(CCDData(binned, unit="adu"))
            times.append(hdul[0].header.get('DATE-OBS'))

    if not times:
        print("No valid timestamps found; skipping stack.")
        return

    stacked = combine(
        calibrated_list,
        method='average',
        sigma_clip=True,
        sigma_clip_low_thresh=3,
        sigma_clip_high_thresh=3,
        sigma_clip_func=np.ma.median,
        sigma_clip_dev_func=np.std
    )

    midpoint = Time(times, format='isot', scale='utc').mean()
    hdr = fits.getheader(group_files[0])
    hdr['DATE-AVG'] = midpoint.isot
    hdr['OBSERVER'] = OBSERVER

    mid_iso = midpoint.to_value('iso', subfmt='date')
    mid_jd = midpoint.jd
    out_fname = f"{object_name}_{OBSERVER}_{mid_iso.replace('-', '')}_{mid_jd:.3f}.fits"
    out_path = os.path.join(output_dir, out_fname)

    print(f"Writing stacked file: {out_fname}")
    fits.writeto(out_path, stacked.data.astype(np.float32), hdr, overwrite=True)


def process_group(group_files, output_dir, object_name):
    if len(group_files) == 1:
        process_single_image(group_files[0], output_dir, object_name)
    else:
        process_group_stack(group_files, output_dir, object_name)


def process_images(input_dir, output_dir, stack, stack_height):
    # Sort FITS by observation time
    def safe_time(p):
        try:
            return Time(fits.getheader(p)['DATE-OBS'], format='isot', scale='utc')
        except Exception:
            print(f"Skipping {p} due to bad DATE-OBS")
            return Time('1900-01-01T00:00:00')

    fps = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(FITS_EXTENSIONS)],
        key=safe_time
    )

    if not fps:
        print(f"No FITS files found in {input_dir}")
        return

    # Determine object and default output
    with fits.open(fps[0]) as hdul:
        hdr0 = hdul[0].header
        obj = hdr0.get('OBJECT', 'UNKNOWN').replace(' ', '_')
        date_iso = Time(hdr0['DATE-OBS'], format='isot', scale='utc').to_value('iso', subfmt='date')
        default_out = f"{obj}_{date_iso}_{OBSERVER}_CAL"

    if output_dir is None:
        output_dir = default_out
    os.makedirs(output_dir, exist_ok=True)

    global groups
    if stack:
        groups = [fps[i:i + stack_height] for i in range(0, len(fps), stack_height)]
        print(f"Calibrating {len(groups)} stacks @ {stack_height} images per stack.")
    else:
        groups = [[fp] for fp in fps]
        print(f"Calibrating {len(groups)} individual images.")

    pool_args = [(grp, output_dir, obj) for grp in groups]
    print(f"Starting processing with concurrency: {CONCURRENCY}")
    with Pool(processes=CONCURRENCY) as pool:
        pool.starmap(process_group, pool_args)


def main():
    global BIAS_DIR, DARK_DIR, FLAT_DIR, BIN_LEVEL, OBSERVER, VERBOSE, CONCURRENCY, ALIGN

    parser = argparse.ArgumentParser(description="Calibrate, align, bin, and stack FITS images.")
    parser.add_argument('input_dir', type=str, help="Directory containing source FITS files.")
    parser.add_argument('output_dir', nargs='?', default=None, help="Directory for output FITS files.")
    parser.add_argument('-a', '--align', action='store_true', help="Enable image alignment.")
    parser.add_argument('-b', '--bin-level', type=int, default=default_bin, help=f"Binning level (default: {default_bin}).")
    parser.add_argument('-c', '--concurrency', type=int, default=default_concurrency, help=f"Number of processes (default: {default_concurrency}).")
    parser.add_argument('-o', '--observer', type=str, default=default_observer, help=f"Observer code (default: {default_observer}).")
    parser.add_argument('-s', '--stack', action='store_true', help="Enable stacking.")
    parser.add_argument('-S', '--stack-threshold', type=int, default=default_stackthresh, help=f"Stack threshold (default: {default_stackthresh}).")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose output.")
    parser.add_argument('-B', '--bias-dir', type=str, help="Directory containing master bias frames.")
    parser.add_argument('-D', '--dark-dir', type=str, help="Directory containing master dark frames.")
    parser.add_argument('-F', '--flat-dir', type=str, help="Directory containing master flat frames.")

    args = parser.parse_args()
    BIAS_DIR = args.bias_dir
    DARK_DIR = args.dark_dir
    FLAT_DIR = args.flat_dir
    BIN_LEVEL = args.bin_level
    OBSERVER = args.observer
    VERBOSE = args.verbose
    CONCURRENCY = args.concurrency
    ALIGN = args.align

    num_files = len([f for f in os.listdir(args.input_dir) if f.lower().endswith(FITS_EXTENSIONS)])
    stack_height = 1 if num_files <= args.stack_threshold else round(num_files / args.stack_threshold)

    process_images(args.input_dir, args.output_dir, args.stack, stack_height)


if __name__ == "__main__":
    main()

