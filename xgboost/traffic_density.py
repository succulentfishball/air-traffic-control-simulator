import numpy as np
import pandas as pd
import shapely
import geopandas as gpd


def main():

    csv_path = "../data/filtered/CAT21_2022-11-25_edited_ts.csv"
    timestamp = 36940
    flight_id = 420694
    time_window = 10 # seconds
    radius = 50 #km

    # get approximate location of the flight at that timestamp. Similar code to voice_to_geojson.py
    loc_df = pd.read_csv(csv_path)
    sub_df = loc_df[loc_df["flight_id"] == flight_id]
    for i in range(len(sub_df) - 1):
        if sub_df.iloc[i]["event_timestamp"] <= timestamp <= sub_df.iloc[i + 1]["event_timestamp"]:
            # how far along the time interval it is
            ratio = (timestamp - sub_df.iloc[i]["event_timestamp"]) / (sub_df.iloc[i + 1]["event_timestamp"] - sub_df.iloc[i]["event_timestamp"])
            # use this to approximate the distance relative to the two consecutive points
            curr_point = np.array([sub_df.iloc[i]["longitude"], sub_df.iloc[i]["latitude"]])
            next_point = np.array([sub_df.iloc[i + 1]["longitude"], sub_df.iloc[i + 1]["latitude"]])
            dir_vec = next_point - curr_point
            voice_point = curr_point + ratio * dir_vec
            break
    
    # get the rows in the trajectory csv which have timestamps close to the given timestamp
    loc_df = loc_df[(loc_df["event_timestamp"] > timestamp - time_window) & (loc_df["event_timestamp"] < timestamp + time_window)]

    # get only the first row for each flight id
    loc_df = loc_df.groupby("flight_id").first()

    # get circular buffer
    region = shapely.Point(voice_point).buffer(radius / 111.139)

    loc_df["geom"] = loc_df.apply(lambda row: shapely.Point([row["longitude"], row["latitude"]]), axis=1)
    loc_df = gpd.GeoDataFrame(loc_df, geometry="geom", crs="EPSG:4326")
    num_aircraft = loc_df["geom"].apply(lambda x: x.intersects(region)).value_counts().loc[True]
    print(num_aircraft)








if __name__ == "__main__":
    main()