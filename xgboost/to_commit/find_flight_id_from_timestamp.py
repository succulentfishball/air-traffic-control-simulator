import pandas as pd
import numpy as np
import os


def main():

    """
    Set in_dir which is the directory containign the voice data excel files
    For each file, for each line of dialogue, assign the flight id if the timestamp is valid
    """

    in_dir = r"..\data\2022Voice\ARR\new_timestamps"
    cnt = 0

    for f in os.listdir(in_dir):
        if f.endswith(".xlsx"):
            in_path = os.path.join(in_dir, f)
            date = "-".join(f.split("_")[1].split("-")[::-1])
            ref_path = "../data/filtered/CAT21_{}_edited_ts.csv".format(date)

            in_df = pd.read_excel(in_path).fillna("")
            print(in_path)
            ref_df = pd.read_csv(ref_path)

            # first filter the ref csv to only include the flight trajectory in the desired timing
            # start and end times of the stm file in seconds after midnight
            start, end = map(lambda x: float(x[:-3] + "." + x[-3:]), in_path.split("_")[-3:-1])
            valid_flight_ids = ref_df[ref_df["event_timestamp"].apply(lambda x: start <= x <= end)]["flight_id"].unique()
            ref_df = ref_df[ref_df["flight_id"].isin(valid_flight_ids)]

            def find_flight_id(row):
                # callsign e.g. SIA177 -> 3-letter ICAO plus some numbers behind. Callsign finalised
                callsign = row["suggested_callsign"]
                # blank callsign -> blank flight id
                if callsign == "" or not row["Matched"]:
                    return ""
                # legit callsign -> find all flight ids of that call sign in the reference csv
                # ideally one call sign should only have one flight id
                subdf = ref_df[ref_df["callsign"] == callsign]
                flight_ids = subdf["flight_id"].unique()
                valid_idxs = []
                for i, flight_id in enumerate(flight_ids):
                    timings = subdf[subdf["flight_id"] == flight_id]["event_timestamp"]
                    # start time and end time of this flight id
                    start_time, end_time = timings.min(), timings.max()
                    if (start_time <= row["start_time"] <= end_time) and (start_time <= row["end_time"] <= end_time):
                        valid_idxs.append(i)
                flight_ids =  flight_ids[valid_idxs]
                assert len(flight_ids) <= 1
                return flight_ids[0] if len(flight_ids) == 1 else ""
                    
            in_df["flight_id"] = in_df.apply(find_flight_id, axis=1)
            in_df.to_excel(in_path, index=(2 + 2 == 5))
            cnt += 1
            print(cnt)


if __name__ == "__main__":
    main()