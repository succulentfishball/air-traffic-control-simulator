import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pykalman import KalmanFilter
from tas_to_ias import tas_to_cas
# %matplotlib tk


# added by derek
def speed_smoothing(df: pd.DataFrame, speed_gap: int = 30, corner_freq: float = 0.01, corner_freq_heading: float = 0.1):
    """
    :function: Given raw track df (with only one flight), get interpolated lat, lon, alt and derived speed and heading
               Use lowpass filter (4th order Butterworth) to smooth out speed noise
    :param corner_freq: corner frequency to filter out high frequency noise for speed
    :param corner_freq_heading: same but for derived_heading column
    :param speed_gap: how many points to look back to when calculating speed
    :return: interpolated df with values every 1s for lat, lon, timestamps (seconds after midnight),
             altitude, derived_speed, derived_heading, CAS, cas_smooth
    """
    # interp_df = liner_interpolation1s(df)
    # interp_df = kalman_filter_latlon(interp_df)
    distance = np.sqrt(interp_df["latitude"].diff(speed_gap)**2 + interp_df["longitude"].diff(speed_gap)**2)*59.9 #convert to NM
    speed = distance/interp_df["event_timestamp"].diff(speed_gap)*3600.0 #convert to knots
    interp_df["derived_speed"] = speed.values # add derived_speed column
    calculate_heading(interp_df) # add derived_heading column
    interp_df["CAS"] = interp_df.apply(lambda row: # convert derived_speed to CAS
                                       tas_to_cas(row["derived_speed"], row["altitude"]) 
                                       if not np.isnan(row["derived_speed"]) else np.nan, axis=1)
    speed = interp_df["CAS"][~interp_df["CAS"].isna()]
    nyq = 0.5 # nyquist frequency is 0.5 Hz because interpolated df means sampling frequency is 1 Hz
    corner_freq = corner_freq / nyq # normalized corner frequency: divide by nyquist frequency
    b, a = signal.butter(4, corner_freq, btype="lowpass", analog=False)
    smooth_speed = signal.filtfilt(b, a, speed)
    interp_df["cas_smooth"] = interp_df["CAS"].copy()
    interp_df.loc[speed.index, "cas_smooth"] = smooth_speed
    b2, a2 = signal.butter(4, corner_freq_heading, btype="lowpass", analog=False)
    smooth_heading = signal.filtfilt(b2, a2, interp_df["derived_heading"])
    interp_df["heading_smooth"] = smooth_heading
    return interp_df


# added by derek
def calculate_heading(df, gap=3):
    """
    Given track data dataframe (assuming all rows in the df belong to the same flight),
    calculate heading based on vector from one point to the 3rd point away
    """
    # reference vector (North-up)
    j = np.array([0, 1])
    tmp = pd.DataFrame(zip(df["longitude"].diff(gap).shift(-gap).ffill(), df["latitude"].diff(gap).shift(-gap).ffill())).apply(lambda x: np.array([x[0], x[1]]), axis=1)
    
    def calculate_angle(v):
        # between 0 and pi
        theta = np.arccos(np.dot(j, v) / (np.linalg.norm(j) * np.linalg.norm(v)))
        if v[0] < 0:
            theta = 2 * np.pi - theta
        return theta / np.pi * 180

    df["derived_heading"] = tmp.apply(calculate_angle).values
    return df


def flight_lvs(df, win):  # flight level smoothing and use histogram to obtain flight level platforms
    fl_df = df.copy()
    fl_df["fl_smooth"] = signal.savgol_filter(df["altitude"], win, 5)
    fl_hist = np.histogram(fl_df['fl_smooth'].values,bins=40)
    fl_hist_df = pd.DataFrame({"count":fl_hist[0],"fl":fl_hist[1][:-1]})
    staying_fls = fl_hist_df[fl_hist_df["count"]>60]