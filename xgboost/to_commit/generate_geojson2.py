import pandas as pd
import geopandas as gpd
import datetime
from trajectory_to_geojson import get_arrow_geom, process_row


def main():

    path = r"..\data\train\train_data_lstm_v1\Voice_24-11-2022_train.xlsx"
    out_path = r"..\data\train\train_data_lstm_v1\24-11-2022_tmn11_actual.json"

    df = pd.read_excel(path)
    df = df[df["flight_id"] == 412844]

    df["curr_lon"], df["curr_lat"] = zip(*df.apply(lambda x: (x["curr_lon"] - 0.08008999999999844,
                                                             x["curr_lat"] + 0.024344699999999886), axis=1))

    df["geometry"], df["lineColor"] = zip(*df.apply(process_row, axis=1))
    df["timestamp"] = df["interval_start"].apply(lambda x: datetime.datetime(2022, 11, 21) + datetime.timedelta(seconds=x))

    # remove original latitude and longitude columns so they won't get mistakenly read by kepler as points
    df = df.drop(["curr_lat", "curr_lon", "interval_start"], axis=1)
    df = gpd.GeoDataFrame(data=df, geometry="geometry", crs="EPSG:4326")
    df.to_file(out_path, driver="GeoJSON")


if __name__ == "__main__":
    main()