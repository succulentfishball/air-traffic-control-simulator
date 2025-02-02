import os
import datetime
import numpy as np
import pandas as pd
import shapely as shapely
import geopandas as gpd


def get_voice_locations(voice_paths, trajectory_path, flight_id):

    """
    Helper fucntion used to identify the approximate location at which voice cmmunication was communicated
    """

    if isinstance(voice_paths, pd.DataFrame):
        voice_df = voice_paths
    else:
        voice_df = None
        for voice_path in voice_paths:
            tmp = pd.read_excel(voice_path)
            if voice_df is None:
                voice_df = tmp
            else:
                voice_df = pd.concat([voice_df, tmp], ignore_index=(1+1==2))
            tmp = None

    if isinstance(trajectory_path, pd.DataFrame):
        loc_df = trajectory_path
    else:
        loc_df = pd.read_csv(trajectory_path)

    voice_df["flight_id"] = voice_df["flight_id"].fillna(-1)
    sub_voice_df = voice_df[voice_df["flight_id"] == flight_id]
    sub_loc_df = loc_df[loc_df["flight_id"] == flight_id]

    start_idx = 0

    def match_timing(row):
        nonlocal start_idx
        mean_time = (row["start_time"] + row["end_time"]) / 2.0
        for i in range(start_idx, len(sub_loc_df) - 1):
            if sub_loc_df.iloc[i]["event_timestamp"] <= mean_time <= sub_loc_df.iloc[i + 1]["event_timestamp"]:
                start_idx = i
                # how far along the time interval it is
                ratio = (mean_time - sub_loc_df.iloc[i]["event_timestamp"]) / (sub_loc_df.iloc[i + 1]["event_timestamp"] - sub_loc_df.iloc[i]["event_timestamp"])
                # use this to approximate the distance relative to the two consecutive points
                curr_point = np.array([sub_loc_df.iloc[i]["longitude"], sub_loc_df.iloc[i]["latitude"]])
                next_point = np.array([sub_loc_df.iloc[i + 1]["longitude"], sub_loc_df.iloc[i + 1]["latitude"]])
                dir_vec = next_point - curr_point
                voice_point = curr_point + ratio * dir_vec
                return shapely.Point(voice_point)

    sub_voice_df["geom"] = sub_voice_df.apply(match_timing, axis=1)
    # voice_df = voice_df.drop("callsign_prefix", axis=1).rename(columns={"Lines": "Message", 
    #                                                                     "suggested_callsign": "callsign",
    #                                                                     "start_time": "timestamp"})
    
    # voice_df["timestamp"] = voice_df["timestamp"].apply(lambda x: datetime.timedelta(seconds=x) + datetime.datetime(y, m, d))

    return sub_voice_df


def main():

    voice_paths = ["../data/2022Voice/APP/with_ids/er/new_timestamps/124.05mhz_23-11-2022_24-11-2022_0000utc_0000utc_export_0_019902208_025793920_done.xlsx",
                   ]
    trajectory_path = "../data/filtered/CAT21_2022-11-23_edited_ts_speed.csv"
    out_path = "../data/test/2022-11-23_SIA256_voice.json"
    flight_id = 408307

    voice_df = get_voice_locations(voice_paths, trajectory_path, flight_id)

    voice_df.to_excel("../data/test/2022-11-23_SIA256_voice.xlsx", index=False)

    # voice_df = gpd.GeoDataFrame(voice_df, geometry="geom", crs="EPSG:4326")
    # voice_df.to_file(out_path, driver="GeoJSON")


def voice_ids_to_geojson(in_dir, track_path, out_path, date, save_json=True):

    """
    Given folder of processed voice data, and a date,
    extract all the flight ids in the voice data for the date
    And extract all the voice points
    output into the kepler format json
    """

    d, m, y = map(int, date.split("-"))
    df = None

    for f in os.listdir(in_dir):
        if f.split("_")[1] == date:
            tmp = pd.read_excel(os.path.join(in_dir, f))
            if df is None:
                df = tmp
            else:
                df = pd.concat([df, tmp], ignore_index=True)
            tmp = None

    # get all flight ids present in all the voice data
    flight_ids = df["flight_id"].fillna(-1).unique()
    # filtering out -1 ensures that every line of dialogue can be matched to a point and time
    flight_ids = flight_ids[flight_ids != -1]

    track_df = pd.read_csv(track_path)

    out_df = None

    for flight_id in flight_ids:
        tmp = get_voice_locations(df, track_df, flight_id)
        if out_df is None:
            out_df = tmp
        else:
            out_df = pd.concat([out_df, tmp], ignore_index=True)
        tmp = None

    df = None
    track_df = None

    if save_json:

        out_df = out_df.drop("callsign_prefix", axis=1).rename(columns={"Lines": "Message", 
                                                                        "suggested_callsign": "callsign",
                                                                        "start_time": "timestamp"})
        
        out_df["timestamp"] = out_df["timestamp"].apply(lambda x: datetime.timedelta(seconds=x) + datetime.datetime(y, m, d))

        out_df = gpd.GeoDataFrame(out_df, geometry="geom", crs="EPSG:4326")
        out_df.to_file(out_path, driver="GeoJSON") 
    
    return out_df


if __name__ == "__main__":

    in_dir = r"..\data\2022Voice\ARR\new_timestamps"
    track_path = r"..\data\filtered\CAT21_2022-11-26_edited_ts.csv"
    out_path = r"..\data\profiling\arrival tracks and voice\2022-11-26_voice.json"
    date = "26-11-2022"

    voice_ids_to_geojson(in_dir, track_path, out_path, date, save_json=True)