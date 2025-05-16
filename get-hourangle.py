#!/usr/bin/env python3

import argparse
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation, Angle, AltAz, SkyCoord
import astropy.units as u
import numpy as np

def compute_hour_angle_and_airmass(fits_file):
    with fits.open(fits_file) as hdul:
        header = hdul[0].header

        # Extract required keywords
        date_obs = header.get('DATE-OBS')
        exptime = header.get('EXPTIME')
        ra = header.get('RA')
        dec = header.get('DEC')
        site_long = header.get('SITELONG')
        site_lat = header.get('SITELAT')

        if not date_obs or ra is None or dec is None or exptime is None or site_long is None or site_lat is None:
            raise ValueError("FITS header must contain DATE-OBS, EXPTIME, RA, DEC, SITELONG, and SITELAT.")

        # Coordinates
        ra_angle = Angle(ra, unit=u.deg)
        dec_angle = Angle(dec, unit=u.deg)
        coord = SkyCoord(ra=ra_angle, dec=dec_angle, frame='icrs')

        # Times
        start_time = Time(date_obs, format='isot', scale='utc')
        end_time = start_time + TimeDelta(float(exptime), format='sec')

        # Location
        location = EarthLocation(lat=float(site_lat) * u.deg, lon=float(site_long) * u.deg)

        # Hour angles
        lst_start = start_time.sidereal_time('apparent', longitude=location.lon)
        lst_end = end_time.sidereal_time('apparent', longitude=location.lon)

        ha_start = (lst_start - ra_angle).wrap_at(24 * u.hourangle)
        ha_end = (lst_end - ra_angle).wrap_at(24 * u.hourangle)

        # Altitude and airmass
        altaz_start = coord.transform_to(AltAz(obstime=start_time, location=location))
        altaz_end = coord.transform_to(AltAz(obstime=end_time, location=location))

        def estimate_airmass(altitude):
            if altitude.deg <= 0:
                return ">10"
            else:
                return f"{(1 / np.cos(np.deg2rad(90 - altitude.deg))):.2f}"

        airmass_start = estimate_airmass(altaz_start.alt)
        airmass_end = estimate_airmass(altaz_end.alt)

        return ha_start, ha_end, airmass_start, airmass_end

def main():
    parser = argparse.ArgumentParser(description="Compute hour angle and airmass at start and end of exposure from a FITS file.")
    parser.add_argument("fits_file", help="Path to the FITS file.")
    args = parser.parse_args()

    try:
        ha_start, ha_end, airmass_start, airmass_end = compute_hour_angle_and_airmass(args.fits_file)

        print(f"Start: HA = {ha_start.to_string(sep=':', unit=u.hourangle, precision=2)} "
              f"({ha_start.to_value(unit=u.hourangle):.6f} hr), Airmass = {airmass_start}")
        print(f"End  : HA = {ha_end.to_string(sep=':', unit=u.hourangle, precision=2)} "
              f"({ha_end.to_value(unit=u.hourangle):.6f} hr), Airmass = {airmass_end}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

