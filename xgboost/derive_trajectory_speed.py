import pandas as pd
import os
from math import radians, cos, sin, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371000 # Radius of earth in meters
    return c * r


def main():

    in_dir = r"..\data\filtered"
    window = 25 # moving average window in seconds

    for f in os.listdir(in_dir):
        if f.endswith("_edited_ts.csv") and f == "CAT21_2022-11-21_edited_ts.csv":

            df = pd.read_csv(os.path.join(in_dir, f))

            # df = df[df["flight_id"] == 408307]

            out_df = None

            # for each unique flight id
            for flight_id in df["flight_id"].unique():
                # get the subset df with that flight id
                subdf = df[df["flight_id"] == flight_id]
                # if only one row, then not able to calculate speed meaningfully, just use ground_speed value
                if len(subdf) == 1:
                    subdf["derived_speed"] = subdf["ground_speed"].values
                # else, calculate speed
                else:

                    idx = 0

                    def calculate_speed(row):
                        """
                        Idea: for each row in the df, go upwards until we either reach the first row in the df
                        Or we go outside of the window range
                        Then calculate speed as the total distance between the two points divided by total time between the points
                        Special cases: for first row, use the first point and the next point to calculate speed
                        """
                        nonlocal idx

                        # if first row
                        if idx == 0:
                            lon1, lat1 = row["longitude"], row["latitude"]
                            lon2, lat2 = subdf.iloc[idx + 1]["longitude"], subdf.iloc[idx + 1]["latitude"]
                            timedelta = subdf.iloc[idx + 1]["event_timestamp"] - row["event_timestamp"]
                            speed = haversine(lon1, lat1, lon2, lat2) / timedelta # speed in m/s
                            speed *= 1.94384 # speed in knots
                            idx += 1
                            return speed
                        
                        else:
                            last_idx = 0
                            curr_timestamp = row["event_timestamp"]
                            for i in range(idx - 1, -1, -1):
                                prev_timestamp = subdf.iloc[i]["event_timestamp"]
                                if curr_timestamp - prev_timestamp > window:
                                    last_idx = i + 1
                                    break
                                if i == 0:
                                    last_idx = 0
                            if last_idx == idx:
                                last_idx = idx - 1
                            lon1, lat1 = row["longitude"], row["latitude"]
                            lon2, lat2 = subdf.iloc[last_idx]["longitude"], subdf.iloc[last_idx]["latitude"]
                            timedelta = row["event_timestamp"] - subdf.iloc[last_idx]["event_timestamp"]
                            speed = haversine(lon1, lat1, lon2, lat2) / timedelta
                            speed *= 1.94384
                            idx += 1
                            return speed
                
                    subdf["derived_speed"] = subdf.apply(calculate_speed, axis=1)
                
                if out_df is None:
                    out_df = subdf
                else:
                    out_df = pd.concat([out_df, subdf], ignore_index=True)
                
            out_df.to_csv(os.path.join(in_dir, os.path.splitext(f)[0] + "_speed.csv"), index=False)

            break


def old():

    in_dir = r"..\data\filtered"

    for f in os.listdir(in_dir):
        if f.endswith("_edited_ts.csv"):
            path = os.path.join(in_dir, f)

            df = pd.read_csv(path)
            df["derived_speed"] = df["ground_speed"].values

            # prev_callsign = ""
            # i = 0

            # def calculate_speed(row):
            #     nonlocal i
            #     # either first row in csv or when there is change in callsign
            #     if row["callsign"] != prev_callsign:
            #         prev_callsign = row["callsign"]
            #         # if the next row is a new callsign (i.e. for this flight there is only one row)
            #         if row["callsign"] != df.iloc[i + 1]["callsign"]:
            #             i += 1
            #             return
            #         # get the points to compare
            #         lon1, lat1 = row["longitude"], row["latitude"]
            #         lon2, lat2 = df.iloc[i + 1]["longitude"], df.iloc[i + 1]["latitude"]
            #         time_delta = df.iloc[i + 1]["event_timestamp"] - df.iloc[i]["event_timestamp"]
            #     # else, compare with row above
            #     lon1, lat1 = row["longitude"], row["latitude"]
            #     lon2, lat2 = df.iloc[i - 1]["longitude"], df.iloc[i - 1]["latitude"]
            #     time_delta = df.iloc[i]["event_timestamp"] - df.iloc[i - 1]["event_timestamp"]
            #     dist = haversine(lon1, lat1, lon2, lat2)
            #     speed = dist / time_delta # speed in metres / s
            #     speed *= 1.94384 # speed in knots
            #     df.iat[i, -1] = speed
            #     i += 1


            # df.apply(calculate_speed, axis=1)



            for idx in df["flight_id"].unique():
                subdf = df[df["flight_id"] == idx]
                if len(subdf) < 2:
                    continue
                index = subdf.index

                i = 0
                
                def calculate_speed(row):
                    nonlocal i
                    if i == 0:
                        lon1, lat1 = row["longitude"], row["latitude"]
                        lon2, lat2 = subdf.iloc[i + 1]["longitude"], subdf.iloc[i + 1]["latitude"]
                        time_delta = subdf.iloc[i + 1]["event_timestamp"] - subdf.iloc[i]["event_timestamp"]
                    else:
                        lon1, lat1 = row["longitude"], row["latitude"]
                        lon2, lat2 = subdf.iloc[i - 1]["longitude"], subdf.iloc[i - 1]["latitude"]
                        time_delta = subdf.iloc[i]["event_timestamp"] - subdf.iloc[i - 1]["event_timestamp"]
                    dist = haversine(lon1, lat1, lon2, lat2)
                    speed = dist / time_delta # speed in metres / s
                    speed *= 1.94384 # speed in knots
                    df.at[index[i], "derived_speed"] = speed
                    i += 1

                subdf.apply(calculate_speed, axis=1)
                
            df.to_csv(path, index=(2 + 2 == 5))


if __name__ == "__main__":
    main()