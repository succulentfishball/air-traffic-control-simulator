import os
import ast
import math
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from tas_to_ias import tas_to_cas
from derive_trajectory_speed import haversine
from voice_to_geojson import voice_ids_to_geojson
from trajectory_to_geojson import process_trajectory_csv
from star_check import get_arrival_points, parse_points, get_next_point, is_on_star
from TrackProcessingCodeBox import speed_calculation, liner_interpolation1s, kalman_filter_latlon, speed_smoothing


def add_runway_dir():
    """
    Add the runway direction to the voice_data.xlsx
    Obtain runway direction from the flight plans summary csv
    """
    summary_csv = r"..\data\Flight Plan_20220901-20221130\flights_summary_221121_221126.csv"
    data_xlsx = r"..\data\profiling\voice_numbers\voice_data.xlsx"

    df = pd.read_csv(summary_csv)
    # map flight id to runway dir
    idx_to_dir = dict(zip(df["flight_id"], df["runway_direction"]))
    df = None
    
    df = pd.read_excel(data_xlsx)
    df["runway_direction"] = df["flight_id"].apply(lambda x: idx_to_dir[x] if x in idx_to_dir else np.nan)
    df.to_excel(data_xlsx, index=(2 + 2 == 5))


def visualise_runway_dir():
    """
    Sanity check: given the voice_data.xlsx with the runway directions,
    Get all the ARRIVALS passing through PASPU or LAVAX,
    and runway direction 02 or 20, to view on Kepler.gl
    """
    path = r"..\data\profiling\voice_numbers\voice_data.xlsx"
    track_dir = r"..\data\filtered"
    out_path = r"..\data\test\runway_20.json"
    entrances_of_interest = ["PASPU", "LAVAX"]
    runway_dir = 20

    df = pd.read_excel(path)
    # filter arrivals passing through paspu or lavax, and desired runway direction
    df = df[(df["ADES"] == "WSSS") & (df["tma_entrance"].isin(entrances_of_interest)) & (df["runway_direction"] == runway_dir)]

    out_df = None
    
    for date in df["date"].unique():
        track_path = os.path.join(track_dir, "CAT21_{}_edited_ts.csv".format("-".join(date.split("-")[::-1])))
        # get the flight ids on that date
        flight_ids = df["flight_id"][df["date"] == date].to_numpy()
        tmp = process_trajectory_csv(track_path, "X", flight_ids, save_json=False)
        if out_df is None:
            out_df = tmp
        else:
            out_df = pd.concat([out_df, tmp], ignore_index=(1+1==2))
        tmp = None
    
    out_df.to_file(out_path, driver="GeoJSON")


def extract_voice_data():
    """
    Save out the voice data for the desired flight ids and entry points into a separate file (one file per date)
    """
    db_path = r"..\data\profiling\voice_numbers\voice_data.xlsx"
    voice_dir = r"..\data\2022Voice\APP\with_ids\er\new_timestamps"
    out_dir = r"..\data\train"
    entrypoints = ["LAVAX", "PASPU"]

    db = pd.read_excel(db_path)
    # arrivals at wsss through lavax and paspu
    db = db[(db["ADES"] == "WSSS") & (db["tma_entrance"].isin(entrypoints))]
    dates = db["date"].unique()

    for date in dates:
        # flight ids for that date
        flight_ids = db["flight_id"][db["date"] == date]
        df = None
        for f in os.listdir(voice_dir):
            if f.split("_")[1] == date and not f.startswith("~"):
                tmp = pd.read_excel(os.path.join(voice_dir, f))
                # get only desired flight ids and valid timestamp
                tmp = tmp[(tmp["flight_id"].isin(flight_ids)) & (tmp["Matched"])]
                if df is None:
                    df = tmp
                else:
                    df = pd.concat([df, tmp], ignore_index=(1 + 1 == 2))
                tmp = None
        df.to_excel(os.path.join(out_dir, "{}.xlsx".format(date)), index=(2 + 2 == 5))


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


def track_data_add_heading():

    """
    Given the voice_data.xlsx and a date, extract the track data of the desired flight ids
    and add derived_heading column
    """

    track_path = r"..\data\filtered\CAT21_2022-11-26_edited_ts.csv"
    date = "26-11-2022"
    out_path = r"..\data\train\Track_26-11-2022_.csv"
    voice_path = r"..\data\profiling\voice_numbers\voice_data.xlsx"
    waypoints = ["LAVAX", "PASPU"]

    # df = pd.read_csv(out_path)
    
    # df = df[df["derived_heading"].isna()]
    # print(df)
    # return

    # get desired flight ids
    df = pd.read_excel(voice_path)
    flight_ids = df["flight_id"][(df["date"] == date) & (df["ADES"] == "WSSS") & (df["tma_entrance"].isin(waypoints))]
    df = None

    df = pd.read_csv(track_path)
    outdf = None

    for flight_id in flight_ids:
        # subset with desired flight id
        subdf = df[df["flight_id"] == flight_id]
        assert len(subdf) >= 2
        calculate_heading(subdf)
        # subdf["derived_heading"] = subdf["derived_heading"].bfill().ffill()
        if outdf is None:
            outdf = subdf
        else:
            outdf = pd.concat([outdf, subdf], ignore_index=(1 + 1 == 2))
    
    outdf.to_csv(out_path, index=(2 + 2 == 5))


def track_data_add_kf_speed():

    """
    Given the track data csv, use the kalman filter to derive speed and add the derived_speed column
    """

    track_path = r"..\data\train\Track_26-11-2022_.csv"

    df = pd.read_csv(track_path)
    outdf = None

    for flight_id in df["flight_id"].unique():
        subdf = df[df["flight_id"] == flight_id]
        assert len(subdf) > 30
        timings = pd.Series(np.arange(subdf["event_timestamp"].min(), subdf["event_timestamp"].max()))
        speed, _ = speed_calculation(subdf, gap=30)
        assert len(timings) == len(speed)
        # map the timestamp to the derived speed
        timing_speed_map = dict(zip(timings, speed))
        max_time = timings.max()
        # let the final timestamp speed be the same as the second last timestamp speed
        # because the kalman filter interpolate does not include the final timestamp
        timing_speed_map[max_time + 1] = timing_speed_map[max_time]
        subdf["derived_speed"] = subdf["event_timestamp"].apply(lambda x: timing_speed_map[x])
        if outdf is None:
            outdf = subdf
        else:
            outdf = pd.concat([outdf, subdf], ignore_index=(1 + 1 == 2))
        subdf = None

    outdf.to_csv(track_path)


def refine_timestamps():
    """
    Given the voice data excel, obtain refined timestamps for each lien of dialogue
    e.g. in the raw data, there can be 6 lines of dialogue all sharing the same start_time and end_time
    In that case divide that entire interval into 6 to obtain approximate higher time resolution
    """

    path = r"..\data\train\Voice_21-11-2022_.xlsx"
    out_path = r"..\data\train\Voice_21-11-2022_refined_timestamp.xlsx"

    df = pd.read_excel(path)
    # set of timestamps in the voice data in the form (start_time, end_time)
    intervals_seen = set()

    df["new_start_time"] = -1.5
    df["new_end_time"] = -1.5

    def update_timestamps(row):
        # for each row, first get the interval
        interval = (row["start_time"], row["end_time"])
        # if this interval has not been processed previously
        if interval not in intervals_seen:
            intervals_seen.add(interval)
            # get the length (duration) of the interval
            duration = row["end_time"] - row["start_time"]
            # boolean index of all rows in the dataframe which has the same start_time and end_time
            idxs = (df["start_time"] == row["start_time"]) & (df["end_time"] == row["end_time"])
            # loc indexes of all rows in the df which have the same start and end time
            idxs = df.index[idxs]
            # number of rows which have the same start_time and end_time
            num_rows = len(idxs)
            # approximate duration of each line
            subduration = duration / num_rows
            for i in range(num_rows):
                new_start_time = row["start_time"] + subduration * i
                new_end_time = new_start_time + subduration
                # update row in df
                df.at[idxs[i], "new_start_time"] = new_start_time
                df.at[idxs[i], "new_end_time"] = new_end_time

    df.apply(update_timestamps, axis=1)
    df["start_time"] = df["new_start_time"]
    df["end_time"] = df["new_end_time"]
    df = df.drop(columns=["new_start_time", "new_end_time"])
    df.to_excel(out_path, index=(2 + 2 == 5))


def chunk_data_lstm(interval=10, speed_gap=30):
    """
    Chunk voice data into 10s time steps to put into the model
    LSTM format
    """
    voice_path = r"..\data\train\Voice_21-11-2022_refined_timestamp.xlsx"
    track_path = r"..\data\train\Track_21-11-2022_.csv"
    out_path = r"..\data\train\train_data_lstm_v2\Voice_21-11-2022_train.xlsx"

    vdf = pd.read_excel(voice_path).drop(columns="complete")
    tdf = pd.read_csv(track_path)

    # -1 indicating no change
    speed_classes = [200, 220, 250, 280, -1]
    alt_classes = [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 13000, 14000, 15000, 16000, 17000, -1]

    out_df = {"callsign": [], "flight_id": [], "interval_start": [],
              "curr_lat": [], "curr_lon": [], "curr_cas": [], "curr_heading": [], "curr_alt": [],
              "rwy_dir": [], "star_traffic": [], "below_traffic": [], "on_star": [], "dist_to_wsss": [], "time_to_arrival": [],
              "next_speed": [], "next_heading": [], "next_alt": [], "waypoint": [], "is_immediate": [], "stay_on_route": [],
              "has_data": []}
    
    # midpoint of voice communication
    vdf["mid_time"] = vdf.apply(lambda row: (row["start_time"] + row["end_time"]) / 2, axis=1)

    # for each flight id in the voice data
    for flight_id in vdf["flight_id"].unique():

        # get interpolated lat, lon, alt, derived_heading, derived_speed and CAS
        # so that there is an approximate reading for every 1 second
        sub_tdf = tdf[tdf["flight_id"] == flight_id]
        tmp = liner_interpolation1s(sub_tdf)
        tmp = kalman_filter_latlon(tmp) # gives interpolated lat, lon and alt
        # add one more row equal to the last row
        tmp = pd.concat([tmp, tmp.iloc[[-1]]], ignore_index=True)
        distance = np.sqrt(tmp["latitude"].diff(speed_gap)**2 + tmp["longitude"].diff(speed_gap)**2)*59.9 #convert to NM
        speed = distance/tmp["event_timestamp"].diff(speed_gap)*3600.0 #convert to knots
        tmp["derived_speed"] = speed.values # add derived_speed column
        calculate_heading(tmp) # add derived_heading column
        tmp["CAS"] = tmp.apply(lambda row: # convert derived_speed to CAS
                               tas_to_cas(row["derived_speed"], row["altitude"]) 
                               if not np.isnan(row["derived_speed"]) else np.nan, axis=1)
        
        # start and end time of all voice communication
        idxs = vdf["flight_id"] == flight_id
        all_start = vdf["start_time"][idxs].min()
        all_end = vdf["end_time"][idxs].max()
        # loc indexes of the rows in the voice data with this flight id
        idxs = vdf.index[idxs]
        callsign = vdf.loc[idxs[0]]["suggested_callsign"]
        # ptr to access idxs[0], idxs[1] ... idxs[len(idxs) - 1] etc
        ptr = 0
        # number of intervals to divide the data into
        num_intervals = math.ceil((all_end - all_start) / interval)

        # last command from ATCo
        last_speed, last_heading, last_alt = -1, -1, -1

        for i in range(num_intervals):
            interval_start = all_start + i * interval
            interval_end = interval_start + interval
            out_df["callsign"].append(callsign)
            out_df["flight_id"].append(flight_id)
            out_df["interval_start"].append(interval_start)

            # midpoint of interval
            interval_mid = round((interval_end + interval_start) / 2)
            curr_feats = tmp[tmp["event_timestamp"] == interval_mid]
            # get the interpolated lat, lon, cas, heading, alt at the given timestamp
            out_df["curr_lat"].append(curr_feats.iloc[0]["latitude"])
            out_df["curr_lon"].append(curr_feats.iloc[0]["longitude"])
            out_df["curr_cas"].append(curr_feats.iloc[0]["CAS"])
            out_df["curr_heading"].append(curr_feats.iloc[0]["derived_heading"])
            out_df["curr_alt"].append(curr_feats.iloc[0]["altitude"])

            # append blanks for these for now because these can be added in post processing
            out_df["rwy_dir"].append(np.nan)
            out_df["star_traffic"].append(np.nan)
            out_df["below_traffic"].append(np.nan)
            out_df["on_star"].append(np.nan)
            out_df["dist_to_wsss"].append(np.nan)
            out_df["time_to_arrival"].append(np.nan)

            # voice data within specified interval
            chk = (vdf.loc[idxs]["mid_time"] >= interval_start) & (vdf.loc[idxs]["mid_time"] < interval_end)
            in_interval = idxs[chk]
            idxs = idxs[~chk]
            
            # all speed instructions in specified interval
            speed_ins = vdf.loc[in_interval]["next_speed"]
            heading_ins = vdf.loc[in_interval]["next_heading"]
            alt_ins = vdf.loc[in_interval]["next_alt"]
            other_ins = vdf.loc[in_interval]["condition"]
            # boolean to indicate if any instruction within this interval
            # if false, means this interval is a silent segment
            has_any_ins = False
            
            # loc indexes of valid speed instruction
            speed_idxs = in_interval[~speed_ins.isna()]
            # if 1 or more speed instructions given during this interval, take the latest one
            if len(speed_idxs) > 0:
                ins = speed_ins.loc[speed_idxs[-1]]
                if ins not in speed_classes: # if speed instruction not 200/220/250/280 see which is the closest one
                    diffs = [abs(x - ins) if x != -1 else 99999 for x in speed_classes]
                    ins = speed_classes[np.argmin(diffs)]
                out_df["next_speed"].append(ins)
                # set this speed instruction to be the last existing speed instruction given by ATCo
                last_speed = ins
                has_any_ins = True
            # if no speed instruction given at all, use the last one if have, or -1 to indicate no change
            else:
                if last_speed != -1:
                    out_df["next_speed"].append(last_speed)
                else:
                    out_df["next_speed"].append(-1)
            
            # similar for heading and alt
            alt_idxs = in_interval[~alt_ins.isna()]
            if len(alt_idxs) > 0:
                ins = alt_ins.loc[alt_idxs[-1]]
                if ins not in alt_classes:
                    diffs = [abs(x - ins) if x != -1 else 99999 for x in alt_classes]
                    ins = alt_classes[np.argmin(diffs)]
                out_df["next_alt"].append(ins)
                last_alt = ins
                has_any_ins = True
            else:
                if last_alt != -1:
                    out_df["next_alt"].append(last_alt)
                else:
                    out_df["next_alt"].append(-1)
            
            heading_idxs = in_interval[~heading_ins.isna()]
            if len(heading_idxs) > 0:
                out_df["next_heading"].append(heading_ins.loc[heading_idxs[-1]])
                last_heading = heading_ins.loc[heading_idxs[-1]]
                has_any_ins = True
            else:
                if last_heading != -1:
                    out_df["next_heading"].append(last_heading)
                else:
                    out_df["next_heading"].append(curr_feats.iloc[0]["derived_heading"])

            # other ins e.g. to dovan, after lavax etc
            other_ins = other_ins[~other_ins.isna()]
            to_waypoint, is_immediate, hold = False, True, False
            for ins in other_ins:
                has_any_ins = True
                if "to " in ins:
                    to_waypoint = True
                if "after " in ins:
                    is_immediate = False
                if "hold" in ins:
                    hold = True

            if to_waypoint:
                out_df["waypoint"].append(1)
            else:
                out_df["waypoint"].append(0)
            # if say "after waypoint", then is_immediate is 0
            # if never say anything, is_immediate is 1
            if is_immediate:
                out_df["is_immediate"].append(1)
            else:
                out_df["is_immediate"].append(0)
            # if say hold, then don't stay on route (0)
            # else if never say anything then stay on route (1)
            if hold:
                out_df["stay_on_route"].append(0)
            else:
                out_df["stay_on_route"].append(1)

            # mask to indicate whether this interval has any instructions in the first place
            # so model can learn to differentiate between voice segment and silent segment
            out_df["has_data"].append(int(has_any_ins))

            # if has no instruction at all in thsi interval (silent segment), set everything to be 0
            if not has_any_ins:
                for key in out_df.keys():
                    if key not in ["callsign", "flight_id", "interval_start"]:
                        out_df[key][-1] = 0


        assert len(idxs) == 0
  
    out_df = pd.DataFrame(out_df)
    out_df.to_excel(out_path, index=(1 + 1 == 3))


def check_flight_on_star():

    """
    Given voice_data.xlsx, add another column for the parsed flight route (ending part)
    e.g. toman karto kexas lavax ruvik dovan bipop
    """

    path = r"..\data\profiling\voice_numbers\voice_data.xlsx"
    stars_path = "../data/Flight Plan_20220901-20221130/bluesky_waypoints_and_stars/aip_enr_star_graph_bluesky 2_stars.xlsx"
    waypoints_path = "../data/Flight Plan_20220901-20221130/bluesky_waypoints_and_stars/aip_enr_star_graph_bluesky 2_nodes.xlsx"

    waypoints_df = pd.read_excel(waypoints_path)
    waypoints = dict(zip(waypoints_df["waypointId"], waypoints_df["geom"]))
    waypoints_df = None

    stars_df = pd.read_excel(stars_path)
    stars = dict(zip(stars_df["routeId"], stars_df["waypoints"]))
    stars_df = None

    df = pd.read_excel(path)

    df["parsed_route"] = df["flight_plan"].apply(lambda x: get_arrival_points(x, waypoints, stars))

    df.to_excel(path, index=False)


def add_other_columns_lstm():
    """
    Given the train.xlsx, add in the additional columns like rwy_dir, star_traffic etc
    LSTM data format
    """
    tma_encoding = {"LAVAX": 0, "PASPU": 1}
    rwy_encoding = {20: 0, 2: 1}
    wsss_coord = (103.99400278, 1.35018889)
    time_tol = 5 # seconds for air traffic calculation
    dist_tol = 30 # km for air traffic calculation

    train_path = r"..\data\train\xgboost_v1\Voice_21-11-2022_train.xlsx"
    db_path = r"..\data\profiling\voice_numbers\voice_data.xlsx"
    waypoints_path = "../data/Flight Plan_20220901-20221130/bluesky_waypoints_and_stars/aip_enr_star_graph_bluesky 2_nodes.xlsx"
    track_path = r"..\data\filtered\CAT21_2022-11-21_edited_ts.csv"

    # map waypoint name to lon lat wkt
    waypoints_df = pd.read_excel(waypoints_path)
    waypoints = dict(zip(waypoints_df["waypointId"], waypoints_df["geom"]))
    waypoints_df = None

    track_df = pd.read_csv(track_path)

    # map flight id to (tma entrance, rwy dir, parsed route, ATA)
    db_df = pd.read_excel(db_path)
    db_df = db_df[(db_df["ADES"] == "WSSS") & (db_df["tma_entrance"].isin(["LAVAX", "PASPU"]))]
    flightdata_map = {}
    def populate_flightdata_map(row):
        flightdata_map[row["flight_id"]] = (row["tma_entrance"], row["runway_direction"], 
                                            ast.literal_eval(row["parsed_route"]), row["ATA"])
    
    db_df.apply(populate_flightdata_map, axis=1)
    db_df = None

    df = pd.read_excel(train_path)
    # df = df[df["flight_id"] == 426967] # for testing, remove later
    
    def update_row(row):

        # if not row["has_data"]:
        #     return (0, 0, 0, 0, 0, 0, 0)

        flight_id = row["flight_id"]
        tma_entry, rwy_dir, star_points, ata = flightdata_map[flight_id]
        tma_entry = tma_encoding[tma_entry]
        rwy_dir = rwy_encoding[rwy_dir]
        star_coords = parse_points(star_points, waypoints)
        curr_point = shapely.Point([row["curr_lon"], row["curr_lat"]])
        # -1 is the rare case (unable to determine what is the next point on the star)
        # if got time go and visualise what are the cases which are -1
        next_point_idx = get_next_point(curr_point, star_coords)
        # on_route is True in the vast majority of cases (maybe no point including then)
        if len(star_coords) > 1:
            on_route = int(is_on_star(curr_point, star_coords))
        else:
            on_route = int(True)
        # dist to wsss in km
        dist_to_wsss = haversine(row["curr_lon"], row["curr_lat"], wsss_coord[0], wsss_coord[1]) / 1000

        timestamp = row["interval_start"]
        # filter the track df to include only the rows within the time range
        subdf = track_df[(track_df["event_timestamp"] > timestamp - time_tol) & (track_df["event_timestamp"] < timestamp + time_tol)]
        # get only the first row for each flight id
        subdf = subdf.groupby("flight_id").first()
        # create shapely point for each flight id
        subdf["geom"] = subdf.apply(lambda row: shapely.Point([row["longitude"], row["latitude"]]), axis=1)
        # make into geodataframe
        subdf = gpd.GeoDataFrame(subdf, crs="EPSG:4326", geometry="geom")

        # if the index of the next point on the star is -1, then just check traffic around curr point
        # TODO: maybe next point can be current point + certain distance in the current direction
        if next_point_idx == -1:
            next_point = curr_point
        else:
            next_point = star_coords[next_point_idx]
        
        # create circular buffer around the reference points
        next_point = next_point.buffer(dist_tol / 111.139)
        curr_point = curr_point.buffer(dist_tol / 111.139)

        next_point_traffic = subdf["geom"].apply(lambda x: x.intersects(next_point)).sum()
        curr_point_traffic = subdf["geom"].apply(lambda x: x.intersects(curr_point)).sum()

        date, time = ata.split()
        day, mo, yr = map(int, date.split("/"))
        h, m = map(int, time.split(":"))
        time_to_wsss = (h * 3600 + m * 60 - timestamp) / 60.0 # time to landing in minutes

        return (tma_entry, rwy_dir, next_point_traffic, curr_point_traffic, on_route, dist_to_wsss, time_to_wsss)


    df["tma_entry"], df["rwy_dir"], df["star_traffic"], df["below_traffic"], \
        df["on_star"], df["dist_to_wsss"], df["time_to_arrival"] = zip(*df.apply(update_row, axis=1))
    
    df.to_excel(train_path, index=(2 + 2 == 5))


def get_speeds():

    """
    Get all groundtruth speeds and altitudes in the voice data
    """

    in_dir = r"..\data\train"

    speeds = dict()
    alts = dict()
    curr = 0

    for f in os.listdir(in_dir):
        if f.endswith("_timestamp.xlsx"):
            df = pd.read_excel(os.path.join(in_dir, f))
            
            def update_dict(x):
                if not np.isnan(x):
                    d = speeds if curr == 0 else alts
                    if x not in d:
                        d[x] = 1
                    else:
                        d[x] += 1

            curr = 0
            df["next_speed"].apply(update_dict)
            curr = 1
            df["next_alt"].apply(update_dict)

    speeds = {k: v for k, v in sorted(speeds.items(), key=lambda x : x[1], reverse=True)}
    alts = {k: v for k, v in sorted(alts.items(), key=lambda x : x[1], reverse=True)}
    
    print(speeds)
    print(alts)


def get_atas():
    """
    Given vocie_data.xlsx, add another column for the ATA
    """
    path = r"..\data\profiling\voice_numbers\voice_data.xlsx"
    flightplans_path = r"..\data\Flight Plan_20220901-20221130\icao_routes\Flight Plan_subset.csv"

    df = pd.read_csv(flightplans_path)
    df = df[df["flight_id"] != -1]

    idx_ata_map = dict(zip(df["flight_id"], df["ATA"]))
    df = None

    df = pd.read_excel(path)

    df["ATA"] = df["flight_id"].apply(lambda x: idx_ata_map[x] if x in idx_ata_map else "")

    df.to_excel(path, index=False)


def add_additional_timesteps_no_voice():
    """
    Given the voice excel, for each flight id, first see what are the windows during which there is no voice data
    Then sample some points during those windows
    So that we can have some data points where there is no voice data
    min_silent_length: minimum silent duration such that we sample one or more points during that interval
    """
    voice_path = r"..\data\train\Voice_26-11-2022_refined_timestamp.xlsx"
    output_dir = r"..\data\train\xgboost_v1"
    min_silent_length = 20

    df = pd.read_excel(voice_path)
    # df = df[df["flight_id"] == 397222] # for testing, remove later

    out_df = None

    for flight_id in df["flight_id"].unique():
        subdf = df[df["flight_id"] == flight_id]
        # for each dialogue line, find the difference between the start_time and the previous dialogue's end_time
        silent_intervals = subdf["start_time"] - subdf["end_time"].shift()
        silent_intervals = silent_intervals[silent_intervals > min_silent_length]
        if len(silent_intervals) > 0:
            # get the total number of ATCo instructions for this flight
            num_ins = (subdf[["next_speed", "next_heading", "next_alt", "condition"]].isna().sum(axis=1) < 4).sum()
            # number of additional points without voice data to sample
            sample_size = max(0, round(np.random.normal(loc=num_ins, scale=0.1 * num_ins)))
            # the loc indexes of the silent intervals to sample additional points without voice data
            sample_idxs = silent_intervals.sample(n=sample_size, replace=True, weights=silent_intervals.to_numpy()).index
            # series of the number of times each loc idx appears in the sample
            sample_idxs = sample_idxs.value_counts()

            silent_rows = {"Lines": [], "Source": [], "start_time": [], "end_time": [], "callsign_prefix": [], 
                        "suggested_callsign": [], "flight_id": [], "next_speed": [], "next_heading": [],
                        "next_alt": [], "complete": [], "condition": [], "Matched": []}

            for i in range(len(sample_idxs)):
                # loc idx of which interval to sample from, and how many points we want to sample
                idx, num_samples = sample_idxs.index[i], sample_idxs.iloc[i]
                # the interval to sample from
                sample_end_time = subdf.loc[idx, "start_time"]
                sample_start_time = sample_end_time - silent_intervals.loc[idx]
                sample_times = np.random.uniform(low=sample_start_time, high=sample_end_time, size=num_samples)
                for sample_time in sample_times:
                    silent_rows["Lines"].append("Silent segment")
                    silent_rows["Source"].append("124_S")
                    silent_rows["start_time"].append(sample_time)
                    silent_rows["end_time"].append(sample_time)
                    silent_rows["callsign_prefix"].append("[]")
                    silent_rows["suggested_callsign"].append(subdf["suggested_callsign"].iloc[0])
                    silent_rows["flight_id"].append(flight_id)
                    silent_rows["next_speed"].append(np.nan)
                    silent_rows["next_heading"].append(np.nan)
                    silent_rows["next_alt"].append(np.nan)
                    silent_rows["complete"].append(np.nan)
                    silent_rows["condition"].append(np.nan)
                    silent_rows["Matched"].append(True)
                
            silent_rows = pd.DataFrame(silent_rows)
            subdf = pd.concat([subdf, silent_rows], ignore_index=True)
        
        if out_df is None:
            out_df = subdf
        else:
            out_df = pd.concat([out_df, subdf], ignore_index=True)
    
    out_df.to_excel(os.path.join(output_dir, os.path.basename(voice_path)), index=(2 + 2 == 5))


def voice_to_tree_format():
    """
    Convert the voice excel into the xgboost format and add additional columns like curr_lat, curr_lon etc
    We only care about (1) timestamps which specifically have a ATCo instruction
    and (2) those specially-chosen silent timestamps
    """
    in_path = r"..\data\train\xgboost_v1\Voice_26-11-2022_refined_timestamp.xlsx"
    track_path = r"..\data\train\Track_26-11-2022_.csv"
    out_path = r"..\data\train\xgboost_v1\Voice_26-11-2022_train.xlsx"

    # -1 indicating no change
    speed_classes = [200, 220, 250, 280, -1]
    alt_classes = [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 13000, 14000, 15000, 16000, 17000, -1]
    speed_gap = 30
    time_from_last_cmd_default = 150

    out_df = {"callsign": [], "flight_id": [], "interval_start": [], "time_from_last_cmd": [],
              "curr_lat": [], "curr_lon": [], "curr_cas": [], "curr_heading": [], "curr_alt": [],
              "lag30_lat": [], "lag30_lon": [], "lag30_cas": [], "lag30_heading": [], "lag30_alt": [],
              "rwy_dir": [], "tma_entry": [], "star_traffic": [], "below_traffic": [], "on_star": [], "dist_to_wsss": [], "time_to_arrival": [],
              "next_speed": [], "next_heading": [], "next_alt": [], "waypoint": [], "is_immediate": [], "stay_on_route": [],
              "has_cmd": []}

    vdf = pd.read_excel(in_path)
    tdf = pd.read_csv(track_path)

    for flight_id in vdf["flight_id"].unique():

        # for testing, remove later
        # if flight_id != 397222:
        #     continue

        # get interpolated lat, lon, alt, derived_heading, derived_speed and CAS
        # so that there is an approximate reading for every 1 second
        sub_tdf = tdf[tdf["flight_id"] == flight_id]
        interp_df = speed_smoothing(sub_tdf)
        interp_df = interp_df.set_index("event_timestamp")

        sub_vdf = vdf[vdf["flight_id"] == flight_id]
        callsign = sub_vdf["suggested_callsign"].iloc[0]
        # loc indexes of rows which specifically have instructions
        ins_idxs = sub_vdf.index[sub_vdf[["next_speed", "next_heading", "next_alt", "condition"]].isna().sum(axis=1) < 4]
        # timestamps of each ATCo command
        ins_times = sub_vdf.loc[ins_idxs, "start_time"].to_numpy()
        # sanity check: ensure instruction times sorted
        is_sorted = lambda arr: np.all(arr[:-1] <= arr[1:])
        assert is_sorted(ins_times)
    
        last_speed, last_heading, last_alt = -1, -1, -1
    
        for i in range(len(ins_idxs)):
            # populate some of the basic information into output dataframe
            idx = ins_idxs[i]
            out_df["callsign"].append(callsign)
            out_df["flight_id"].append(flight_id)
            timestamp = round(sub_vdf.loc[idx, "start_time"])
            out_df["interval_start"].append(timestamp)
            if i == 0:
                out_df["time_from_last_cmd"].append(time_from_last_cmd_default)
            else:
                out_df["time_from_last_cmd"].append(timestamp - sub_vdf.loc[ins_idxs[i - 1], "start_time"])
            
            # obtain the positional information from interp_df
            out_df["curr_lat"].append(interp_df.loc[timestamp, "latitude"])
            out_df["curr_lon"].append(interp_df.loc[timestamp, "longitude"])
            out_df["curr_cas"].append(interp_df.loc[timestamp, "cas_smooth"])
            out_df["curr_heading"].append(interp_df.loc[timestamp, "derived_heading"])
            out_df["curr_alt"].append(interp_df.loc[timestamp, "altitude"])

            out_df["lag30_lat"].append(interp_df.loc[timestamp - 30, "latitude"])
            out_df["lag30_lon"].append(interp_df.loc[timestamp - 30, "longitude"])
            out_df["lag30_cas"].append(interp_df.loc[timestamp - 30, "cas_smooth"])
            out_df["lag30_heading"].append(interp_df.loc[timestamp - 30, "derived_heading"])
            out_df["lag30_alt"].append(interp_df.loc[timestamp - 30, "altitude"])

            # append blanks for these for now because these can be added in post processing
            out_df["rwy_dir"].append(np.nan)
            out_df["tma_entry"].append(np.nan)
            out_df["star_traffic"].append(np.nan)
            out_df["below_traffic"].append(np.nan)
            out_df["on_star"].append(np.nan)
            out_df["dist_to_wsss"].append(np.nan)
            out_df["time_to_arrival"].append(np.nan)

            # for this given row, get the speed, head, alt, and other ins
            speed_ins, heading_ins, alt_ins, other_ins = sub_vdf.loc[idx, ["next_speed", "next_heading", "next_alt", "condition"]]
            
            # if no speed ins or alt ins given, use the last existing ins, or -1 if no instruction given at all
            if np.isnan(speed_ins):
                speed_ins = last_speed
            else:
                last_speed = speed_ins
            if np.isnan(alt_ins):
                alt_ins = last_alt
            else:
                last_alt = alt_ins

            # for heading, if no heading instruction given, then simply set it to -1 (no change)
            # instead of taking the last existing heading
            heading_ins = -1 if np.isnan(heading_ins) else heading_ins
            
            if speed_ins not in speed_classes: # if speed instruction not 200/220/250/280 see which is the closest one
                diffs = [abs(x - speed_ins) if x != -1 else 99999 for x in speed_classes]
                speed_ins = speed_classes[np.argmin(diffs)]
            if alt_ins not in alt_classes:
                diffs = [abs(x - alt_ins) if x != -1 else 99999 for x in alt_classes]
                alt_ins = alt_classes[np.argmin(diffs)]
            
            out_df["next_speed"].append(speed_ins)
            out_df["next_heading"].append(heading_ins)
            out_df["next_alt"].append(alt_ins)

            if isinstance(other_ins, str) and "to " in other_ins:
                out_df["waypoint"].append(1)
            else:
                out_df["waypoint"].append(0)
            if isinstance(other_ins, str) and "after " in other_ins:
                out_df["is_immediate"].append(0)
            else:
                out_df["is_immediate"].append(1)
            if isinstance(other_ins, str) and "hold" in other_ins:
                out_df["stay_on_route"].append(0)
            else:
                out_df["stay_on_route"].append(1)
            
            out_df["has_cmd"].append(1)

        # now also append the silent segments into the out df
        # loc indexes of the specially chosen silent points
        silent_idxs = sub_vdf.index[sub_vdf["Lines"] == "Silent segment"]
        
        # now do the same shit again
        for idx in silent_idxs:
            out_df["callsign"].append(callsign)
            out_df["flight_id"].append(flight_id)
            # timestamp of this silent row
            timestamp = round(sub_vdf.loc[idx, "start_time"])
            out_df["interval_start"].append(timestamp)
            # get time from last ATCo instruction
            i = np.searchsorted(ins_times, timestamp)
            if i == 0:
                out_df["time_from_last_cmd"].append(time_from_last_cmd_default)
            else:
                out_df["time_from_last_cmd"].append(timestamp - ins_times[i - 1])
            
            # obtain the positional information from interp_df
            out_df["curr_lat"].append(interp_df.loc[timestamp, "latitude"])
            out_df["curr_lon"].append(interp_df.loc[timestamp, "longitude"])
            out_df["curr_cas"].append(interp_df.loc[timestamp, "cas_smooth"])
            out_df["curr_heading"].append(interp_df.loc[timestamp, "derived_heading"])
            out_df["curr_alt"].append(interp_df.loc[timestamp, "altitude"])

            out_df["lag30_lat"].append(interp_df.loc[timestamp - 30, "latitude"])
            out_df["lag30_lon"].append(interp_df.loc[timestamp - 30, "longitude"])
            out_df["lag30_cas"].append(interp_df.loc[timestamp - 30, "cas_smooth"])
            out_df["lag30_heading"].append(interp_df.loc[timestamp - 30, "derived_heading"])
            out_df["lag30_alt"].append(interp_df.loc[timestamp - 30, "altitude"])

            # append blanks for these for now because these can be added in post processing
            out_df["rwy_dir"].append(np.nan)
            out_df["tma_entry"].append(np.nan)
            out_df["star_traffic"].append(np.nan)
            out_df["below_traffic"].append(np.nan)
            out_df["on_star"].append(np.nan)
            out_df["dist_to_wsss"].append(np.nan)
            out_df["time_to_arrival"].append(np.nan)

            # append nan for the next_speed etc because they are dont-cares
            out_df["next_speed"].append(np.nan)
            out_df["next_heading"].append(np.nan)
            out_df["next_alt"].append(np.nan)
            out_df["waypoint"].append(np.nan)
            out_df["is_immediate"].append(np.nan)
            out_df["stay_on_route"].append(np.nan)
            out_df["has_cmd"].append(0)
    
    out_df = pd.DataFrame(out_df)
    out_df.to_excel(out_path, index=(2 + 2 == 5))


if __name__ == "__main__":
    add_other_columns_lstm()