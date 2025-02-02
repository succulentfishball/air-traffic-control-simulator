import numpy as np
import pandas as pd
import ruptures as rpt
import matplotlib.pyplot as plt
from TrackProcessingCodeBox import speed_smoothing


def find_flat_segments(df: pd.DataFrame, column: str = "altitude", gap: int = 10, thresh: int = 10, min_staying_period: int = 30):
    """
    :param df: dataframe with interpolated lat, lon, alt (s.t. data exists for each second) for a single flight id
    :param gap: delta for calculating gradient e.g. (current altitude - altitude from 10 seconds ago) / 10s
    :param thresh: threshold gradient to determine whether the change is steep enough
    :param column: data column e.g. altitude
    :param min_staying_period: the minimum period (seconds) at which the altitude / heading / speed must be relatively stable
                               in order for it to be considered a stable value
    """
    timestamps = df["event_timestamp"]
    data = df[column]

    # at each point, calculate difference between current value and value "gap" seconds ago
    delta_b = data.diff(gap).bfill()
    # also calculate difference between current value and value "gap" seconds ahead
    delta_f = data.diff(-gap).ffill()
    # calculate gradient
    delta_b /= gap
    delta_f /= gap

    # bool series to indicate whether the graph is "flat enough" at each timestep
    is_flat = (delta_b.abs() < thresh) & (delta_f.abs() < thresh)
    # get the timestamps at which the graph is "flat enough"
    flat_timestamps = timestamps[is_flat].to_numpy()
    # list of [start_time, end_time] for the flat regions
    # df["is_flat"] = is_flat * (data.max() * 1.1)
    flat_windows = []

    for i in range(len(flat_timestamps)):
        if i == 0:
            curr_window = [flat_timestamps[i]]
        else:
            if flat_timestamps[i] - flat_timestamps[i - 1] == 1:
                continue
            else:
                curr_window.append(flat_timestamps[i - 1])
                flat_windows.append(curr_window)
                curr_window = [flat_timestamps[i]]
    if len(curr_window) == 1:
        curr_window.append(flat_timestamps[-1])
        flat_windows.append(curr_window)
    
    flat_windows = [window for window in flat_windows if window[1] - window[0] >= min_staying_period]
    
    true_level = data.max() * 1.1

    def add_flat_column(timestamp):
        return true_level * any(window[0] <= timestamp <= window[1] for window in flat_windows)

    df["is_flat"] = df["event_timestamp"].apply(add_flat_column)

    return flat_windows


def test_find_flat_segments(): 

    df = pd.read_excel(r"..\data\test\test_tcp\test_21-11-2022_396067.xlsx")

    # find_flat_segments(df, "altitude", 10, 2.75, display=True)
    find_flat_segments(df, "heading_smooth", 30, 0.15)
    # find_flat_segments(df, "cas_smooth", 15, 0.35)


def batch_find_flat_headings():

    """
    Set path to track data csv and path to voice data excel
    For each flight id, detect the parts where the heading is relatively flat
    """

    track_path = r"..\data\filtered\CAT21_2022-11-21_edited_ts.csv"
    voice_path = r"..\data\train\xgboost_v1\Voice_21-11-2022_refined_timestamp.xlsx"
    out_path = r"..\data\test\test_tcp\test_21-11-2022.xlsx"
    tolerance = 10 # degrees

    track_df = pd.read_csv(track_path)
    voice_df = pd.read_excel(voice_path)

    flight_ids = voice_df["flight_id"].unique()
    out_df = None

    for flight_id in flight_ids:
        sub_track_df = track_df[track_df["flight_id"] == flight_id]
        interp_df = speed_smoothing(sub_track_df)
        interp_df["flight_id"] = flight_id
        interp_df["callsign"] = sub_track_df["callsign"].iloc[0]
        flat_windows = find_flat_segments(interp_df, "heading_smooth", 30, 0.15)
        # interp_df = interp_df.set_index("event_timestamp")

        staying_headings = [interp_df["heading_smooth"][interp_df["event_timestamp"] == (window[0] + window[1]) // 2].iloc[0] for window in flat_windows]
        # staying_headings = [interp_df.loc[(window[0] + window[1]) // 2, "heading_smooth"] for window in flat_windows]
        sub_voice_df = voice_df[voice_df["flight_id"] == flight_id]
        # list to indicate whether the heading change was instructed by ATCo or ownself change one
        is_instructed = []

        for i in range(len(flat_windows)):
            time = flat_windows[i][0]
            angle = staying_headings[i]
            got_ins = (sub_voice_df["start_time"] < time) & (sub_voice_df["next_heading"] > angle - tolerance) \
                                                          & (sub_voice_df["next_heading"] < angle + tolerance)
            ins_idx = sub_voice_df.index[got_ins].to_list()
            is_instructed.append(ins_idx)
        
        true_level = interp_df["heading_smooth"].max() * 1.1

        def add_flat_column(timestamp):
            return true_level * any(window[0] <= timestamp <= window[1] for window in flat_windows)
        
        flat_windows = [flat_windows[i] for i in range(len(flat_windows)) if len(is_instructed[i]) > 0]

        interp_df["has_cmd"] = interp_df["event_timestamp"].apply(add_flat_column)
        if out_df is None:
            out_df = interp_df
        else:
            out_df = pd.concat([out_df, interp_df], ignore_index=(1 + 1 == 2))
    
    out_df.to_excel(out_path, index=(2 + 2 == 5))



if __name__ == "__main__":
    pass