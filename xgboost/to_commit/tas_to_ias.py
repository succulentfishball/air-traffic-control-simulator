import pandas as pd
import numpy as np


def air_at_altitude(alt: int) -> tuple[float, float, float]:
    """
    :param alt: height above sea level in feet
    :return: (air density, air temp, air pressure) at given altitude
    Equations from https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    Equations apply for altitude < 36000 feet which should be applicable for most approach cases
    """
    h = alt * 0.3048 # height above sea level in metres
    temp = 15.04 - 0.00649 * h # estimated temp in celsius
    p = (((temp + 273.1) / 288.08) ** 5.256) * 101.29 # pressure in kPa
    r = p / ((temp + 273.1) * 0.2869) # air density in kg / m3
    return (r, temp, p)


def tas_to_eas(tas: float, r: float) -> float:
    """
    :param tas: true airspeed (knots)
    :param r: actual air density (kg / m3)
    :return: equivalent airspeed (knots)
    sea level air density taken as 1.225 kg / m3 (ISA)
    """
    return tas / ((1.225 / r) ** 0.5)


def tas_to_mach(tas: float, temp: float) -> float:
    """
    :param tas: true airspeed (knots)
    :param temp: outside air temperature (celsius)
    :return: Mach number
    Equations from https://aerotoolbox.com/airspeed-conversions/
    """
    a = (1.4 * 287.053 * (temp + 273.1)) ** 0.5 # sonic speed in m/s
    a = a * 1.944 # sonic speed in knots
    return tas / a


def eas_to_cas(eas: float, mach: float) -> float:
    """
    :param eas: equivalent airspeed (knots)
    :param mach: mach number
    :return: calibrated airspeed (knots)
    Equation from https://en.wikipedia.org/wiki/Equivalent_airspeed
    Equation accurate within 1% up to Mach 1.2 according to wikipedia
    """
    d = (eas / (661.47 * mach)) ** 2 # ratio of static pressure to standard sea level pressure p/p0
    return eas * (1 + 0.125 * (1 - d) * (mach ** 2) + 0.0046875 * (1 - 10 * d + 9 * (d ** 2)) * (mach ** 4))


def tas_to_cas(tas: float, alt: int) -> float:
    """
    :param tas: true airspeed (knot)
    :param alt: height above sea level (feet)
    :return: calibrated airspeed (knot)
    """
    r, t, p = air_at_altitude(alt) # density in kg/m3, temp in celsius, pressure in kPa
    eas = tas_to_eas(tas, r)
    mach = tas_to_mach(tas, t)
    return eas_to_cas(eas, mach)


def main():

    fpath = r"..\data\train\Track_21-11-2022_.csv"
    out_path = r"..\data\train\Track_21-11-2022_.csv"

    df = pd.read_csv(fpath)

    df["CAS"] = df.apply(lambda row: 
                         tas_to_cas(row["derived_speed"], row["altitude"]) 
                            if not np.isnan(row["derived_speed"]) else np.nan, axis=1)

    df.to_csv(out_path, index=(2 + 2 == 5))


if __name__ == "__main__":
    main()