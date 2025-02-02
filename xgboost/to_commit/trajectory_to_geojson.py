import os
import math
import datetime
import numpy as np
import pandas as pd
import shapely as shapely
import geopandas as gpd
import matplotlib.pyplot as plt


# test script to convert a trajectory data point (lon, lat, heading) into an arrow
# so that when we visualise, it will not just be a point but an arrow showing which way the plane is moving
# visualization parameters
arrow_length_m = 50.0 # length of each arrow edge > in metres
min_centerline_length = 50.0 # min length of the horizontal part of the arrow ->
max_centerline_length = 250.0 # length scaled based on groundspeed
groundspeed_lower = 50.0 # anything at or below this will be the minimum arrow length
groundspeed_upper = 375.0 
alt_lower = 1000
alt_upper = 40000


def get_arrow_geom(lon, lat, heading, speed):
    """
    :param lon: float
    :param lat: float, together with lon form the center point of the arrow >
    :param heading: float heading between 0 and 360 (degrees)
    :param speed: float ground_speed which scales the length of the arrow ->
    :return: WKT string representing the arrow which is oriented based on heading and scaled based on ground speed
    """
    # basis lon/lat vectors
    i = np.array([1, 0])
    j = np.array([0, 1])
    theta = heading / 180.0 * math.pi
    # change of basis
    u = math.sin(theta) * i + math.cos(theta) * j
    v = -math.cos(theta) * i + math.sin(theta) * j
    # length of arrow edge in degrees
    l = arrow_length_m / 111139
    # scaling factor for u and v
    k = l / math.sqrt(2)
    # get the other 2 points to form the arrow >
    center = np.array([lon, lat])
    p1 = center - k * u - k * v
    p2 = center - k * u + k * v
    # get the point to form the centerline part ->
    if speed <= groundspeed_lower:
        centerline_l = min_centerline_length
    elif speed >= groundspeed_upper:
        centerline_l = max_centerline_length
    else:
        centerline_l = (speed - groundspeed_lower) / (groundspeed_upper - groundspeed_lower) * \
                       (max_centerline_length - min_centerline_length) + min_centerline_length
    centerline_l /= 111139
    p3 = center - centerline_l * u
    geom = shapely.MultiLineString([[p1, center, p2], [p3, center]])
    return geom


def get_color(alt):
    """
    :param alt: int altitude
    :return: [r, g, b] for the color of the arrow
    Higher altitude: whiter, lower altitude: bluer
    """
    if alt >= alt_upper:
        color = 255
    elif alt <= alt_lower:
        color = 0
    else:
        color = (alt - alt_lower) / (alt_upper - alt_lower) * 255
    return "[{}, {}, 255]".format(color, color)


def process_row(row):
    geometry = get_arrow_geom(row["curr_lon"], row["curr_lat"], row["curr_heading"], row["curr_cas"])
    color = get_color(row["curr_alt"])
    return (geometry, color)


def process_trajectory_csv(csv_path, out_path, flight_id=None, save_json=True):

    df = pd.read_csv(csv_path)
    # drop unnecessary columns
    df = df[["callsign", "event_timestamp", "latitude", "ground_speed",
             "longitude", "altitude", "flight_id", "derived_heading"]]
    
    # filter flight id
    if flight_id is not None and isinstance(flight_id, int):
        df = df[df["flight_id"] == flight_id]

    elif flight_id is not None and isinstance(flight_id, np.ndarray):
        df = df[df["flight_id"].isin(flight_id)]

    # populate geometry column as shapely object
    df["geometry"], df["lineColor"] = zip(*df.apply(process_row, axis=1))
    # remove original latitude and longitude columns so they won't get mistakenly read by kepler as points
    df = df.drop(["latitude", "longitude"], axis=1)
    df = df.rename(columns={"event_timestamp": "timestamp"})
    # y, m, d = map(int, os.path.splitext(os.path.basename(csv_path))[0].split("_")[1].split("-"))
    # df["timestamp"] = df["timestamp"].apply(lambda x: datetime.datetime(y, m, d) + datetime.timedelta(seconds=x))
    # convert to gdf
    df = gpd.GeoDataFrame(data=df, geometry="geometry", crs="EPSG:4326")
    if save_json:
        df.to_file(out_path, driver="GeoJSON")
    return df


def main():

    """
    Set path to the track data csv and output path for the geosjon
    optionally include flight_id if we only want to extract one flight id else it will convert everything
    output into the kepler json format
    """

    # csv_path = "../data/filtered/CAT21_2022-11-26_edited_ts.csv"
    csv_path = r"..\data\train\Track_24-11-2022_.csv"
    out_path = r"..\data\test\412844.json"
    flight_id = 412844

    process_trajectory_csv(csv_path, out_path, flight_id)


def voice_ids_to_geojson():

    """
    Given folder of processed voice data, and a date,
    extract all the flight ids in the voice data for the date
    And put all the tracks into a geosjon
    """

    in_dir = r"..\data\2022Voice\ARR\new_timestamps"
    track_path = r"..\data\filtered\CAT21_2022-11-26_edited_ts.csv"
    out_path = r"..\data\profiling\arrival tracks and voice\2022-11-26_tracks.json"
    date = "26-11-2022"

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
    flight_ids = flight_ids[flight_ids != -1]

    process_trajectory_csv(track_path, out_path, flight_ids)


if __name__ == "__main__":
    main()