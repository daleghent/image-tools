#!/bin/env python3

# MIT License
#
# Copyright 2025 Dale Ghent <daleg@elemental.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime

def get_fits_metadata(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

        if data is None or 'EXPTIME' not in header:
            return None

        saturated = np.any(data >= 65535)

        return {
            'exptime': header['EXPTIME'],
            'mean': np.mean(data),
            'camera': header.get('INSTRUME', '(Unknown Camera)'),
            'gain': header.get('GAIN', 'N/A'),
            'offset': header.get('OFFSET', 'N/A'),
            'readout': header.get('READOUTM', 'N/A'),
            'ccd-temp': header.get('CCD-TEMP', 'N/A'),
            'swcreate': header.get('SWCREATE', 'N/A'),
            'saturated': saturated
        }

from collections import defaultdict

def process_fits_directory(directory, adu_limit, min_exp=None, max_exp=None):
    exposure_times = []  # All exposure times (not averaged)
    
    grouped_means = defaultdict(list)  # exposure time -> list of mean values
    mean_values = []
    unsat_grouped_means = defaultdict(list)
    
    metadata_sample = None
    saturation_count = 0

    for filename in os.listdir(directory):
        if filename.lower().endswith((".fits", ".fit")):
            filepath = os.path.join(directory, filename)
            result = get_fits_metadata(filepath)
            if result:
                exp_time = result['exptime']
                if (min_exp is not None and exp_time < min_exp) or (max_exp is not None and exp_time > max_exp):
                    continue

                exposure_times.append(exp_time)
                mean_values.append(result['mean'])
                grouped_means[exp_time].append(result['mean'])

                if result['saturated']:
                    saturation_count += 1

                if result['mean'] <= adu_limit:
                    unsat_grouped_means[exp_time].append(result['mean'])

                if metadata_sample is None:
                    metadata_sample = {
                        'camera': result['camera'],
                        'gain': result['gain'],
                        'offset': result['offset'],
                        'readout': result['readout'],
                        'ccd-temp': result['ccd-temp'],
                        'swcreate': result['swcreate']
                    }

    # Average values for each unique exposure time
    unique_exptimes = sorted(grouped_means.keys())
    averaged_means = [np.mean(grouped_means[et]) for et in unique_exptimes]

    print("Samples per exposure time:")
    for et in unique_exptimes:
        print(f"  Exposure: {et:.2f}s | Samples: {len(grouped_means[et])}")

    # Average unsaturated values
    unsat_unique_exptimes = sorted(unsat_grouped_means.keys())
    unsat_averaged_means = [np.mean(unsat_grouped_means[et]) for et in unsat_unique_exptimes]

    return (np.array(unique_exptimes), np.array(averaged_means),
            np.array(unsat_unique_exptimes), np.array(unsat_averaged_means),
            metadata_sample, saturation_count)

def calculate_r_squared(y_actual, y_predicted):
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)

def plot_and_save_graph(exposure_times, mean_values, unsat_exptimes, unsat_means,
                        metadata, output_dir):
    camera = metadata.get('camera', 'Unknown_Camera')
    gain = metadata.get('gain', 'N/A')
    offset = metadata.get('offset', 'N/A')
    readout = metadata.get('readout', 'N/A')
    ccd_temp = metadata.get('ccd-temp', 'N/A')
    swcreate = metadata.get('swcreate', 'N/A')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Least-squares linear fit
    coeffs = np.polyfit(unsat_exptimes, unsat_means, 1)
    fit_line = np.poly1d(coeffs)
    fit_values = fit_line(unsat_exptimes)
    r_squared = calculate_r_squared(unsat_means, fit_values)

    print(f"Least-squares linear fit: Mean = {coeffs[0]:.3f} * Exposure + {coeffs[1]:.3f}")
    print(f"R² value: {r_squared:.6f}")

    # Deviation from linearity
    deviations = 100 * np.abs((unsat_means - fit_values) / fit_values)
    print("\nLinearity deviation (%):")
    for t, m, d in sorted(zip(unsat_exptimes, unsat_means, deviations)):
        print(f"  Exposure: {t:.2f}s | Mean: {m:.2f} | Deviation: {d:.2f}%")

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(exposure_times, mean_values, color='black', label="Saturated", s=10)
    plt.scatter(unsat_exptimes, unsat_means, color='blue', label="Mean ADU (Used in Fit)", s=10)
    plt.plot(sorted(exposure_times), fit_line(sorted(exposure_times)), 'r--', linewidth=1, label="Least-Squares Fit")
    # replaced by errorbar with stddev
    plt.xlabel("Exposure Time (s)")
    plt.ylabel("Mean ADU")
    plt.suptitle(f"{camera} Sensor Linearity", fontsize=12, y=0.96)
    plt.title(f"Gain: {gain} | Offset: {offset} | Temp: {ccd_temp}°C | Mode: {readout}", fontsize=10)
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.scatter([], [], color='darkgreen', s=10, label="% Deviation")
    plt.legend()

    # Annotate each point with its deviation
    for i, (t, m, d) in enumerate(zip(unsat_exptimes, unsat_means, deviations)):
      plt.annotate(f"{d:.1f}%", (t, m), textcoords="offset points", xytext=(0, 5),
      ha='center', fontsize=5, color='darkgreen')
      
    # R² text
    plt.text(0.05, 0.95, f"$R^2$ = {r_squared:.6f}",
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

    # Footer with timestamp and software
    plt.figtext(0.5, 0.01,
                f"Fit cutoff: {args.adu_limit} ADU | Created at {timestamp} | Imaging Software: {swcreate} | linearity.py by Dale Ghent",
                wrap=True, horizontalalignment='center', fontsize=6, style='italic')

    plt.tight_layout()

    # Save plot
    camera_safe = camera.replace(" ", "_").replace("/", "_")
    readout_safe = readout.replace(" ", "_").replace("/", "_")
    iso_timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_filename = f"{camera_safe}_G{gain}_O{offset}_{readout_safe}_Sensor_Linearity_{iso_timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and save linearity plot from FITS files with exposure time filtering.")
    parser.add_argument("directory", help="Directory containing FITS files.")
    parser.add_argument("-a", "--adu-limit", type=float, default=65000,
                        help="Max mean ADU value to include in linear fit (default: 65000)")

    parser.add_argument("--min-exp", type=float, default=None,
                        help="Minimum exposure time to include in the fit")
    parser.add_argument("--max-exp", type=float, default=None,
                        help="Maximum exposure time to include in the fit")

    args = parser.parse_args()

    (exposure_times, mean_values,
     unsat_exptimes, unsat_means,
     metadata,
     saturation_count) = process_fits_directory(args.directory, args.adu_limit, args.min_exp, args.max_exp)

    if exposure_times.size > 0 and mean_values.size > 0 and metadata:
        plot_and_save_graph(exposure_times, mean_values, unsat_exptimes,
                            unsat_means, metadata, args.directory)
    else:
        print("No valid FITS files with required metadata found.")

