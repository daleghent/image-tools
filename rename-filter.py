#!/bin/env python3

from astropy.io import fits
import argparse
import os

def update_filter_keyword(fits_file, new_filter, output_file=None):
    """
    Update the FILTER keyword in the FITS header.

    Parameters:
        fits_file (str): Path to the input FITS file.
        new_filter (str): New value for the FILTER keyword.
        output_file (str): Path to save the modified FITS file. If None, overwrites the original file.
    """
    # Check if file exists
    if not os.path.exists(fits_file):
        print(f"Error: File '{fits_file}' not found.")
        return

    # Open the FITS file
    with fits.open(fits_file, mode='update') as hdulist:
        # Access the primary header
        header = hdulist[0].header

        # Update the FILTER keyword
        old_filter = header.get('FILTER', 'Not Defined')
        header['FILTER'] = new_filter

        # Print the changes made
        print(f"Updated FILTER from '{old_filter}' to '{new_filter}' in {fits_file}")

        # Save changes to the original file or a new file
        if output_file:
            hdulist.writeto(output_file, overwrite=True)
            print(f"Changes saved to {output_file}")
        else:
            print(f"Changes saved to the original file: {fits_file}")

if __name__ == "__main__":
    # Argument parser for command line usage
    parser = argparse.ArgumentParser(description="Update the FILTER keyword in a FITS header.")
    parser.add_argument('fits_file', help="Path to the input FITS file")
    parser.add_argument('new_filter', help="New value for the FILTER keyword")
    parser.add_argument('--output', help="Path to save the modified FITS file (optional)", default=None)

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the function to update the FILTER keyword
    update_filter_keyword(args.fits_file, args.new_filter, args.output)
