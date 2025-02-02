import math
import pytz
import time
import datetime
import numpy as np
import pandas as pd


# # gmt+8
# def get_trajectory_timestamp(ts):
#     date, time = ts.split()
#     year, month, day = map(int, date.split("-"))
#     h, m, s = map(int, time.split("+")[0].split(":"))
#     ts = datetime.datetime(year, month, day, h, m, s)
#     gmt8 = pytz.timezone("Asia/Singapore")
#     ts = gmt8.localize(ts)
#     return ts


# utc
def get_flightplan_timestamp(ts):
    date, time = ts.split()
    day, month, year = map(int, date.split("/"))
    h, m = map(int, time.split(":"))
    ts = datetime.datetime(year, month, day, h, m)
    # utc = pytz.timezone("UTC")
    # ts = utc.localize(ts)
    return ts


def main():

    flightplans_path = "../data/Flight Plan_20220901-20221130/icao_routes/Flight Plan_subset.csv"
    trajectory_path = "../data/filtered/CAT21_2022-11-27_edited_ts.csv"
    out_path = "../data/Flight Plan_20220901-20221130/icao_routes/flight_plans_CAT21_2022-11-27.xlsx"
    date = "2022-11-27"
    year, month, day = map(int, date.split("-"))

    fp_df = pd.read_csv(flightplans_path)
    tra_df = pd.read_csv(trajectory_path)

    flight_ids = tra_df["flight_id"].unique()

    out_df = {"flight_id": [], "idx": [], "ATD": [], "ATA": [], "ADEP": [], "ADES": [], "Callsign": [], "flight_plan": []}

    # for each id in the track data
    for idx in flight_ids:

        idxs = tra_df["flight_id"] == idx
        # timestamps for this flight id
        timings = tra_df["event_timestamp"][idxs]
        start_ts = timings.min()
        end_ts = timings.max()
        # midpoint timestamp for this flight in terms of seconds after utc midnight
        mid_ts = (end_ts - start_ts) / 2 + start_ts
        mid_ts = datetime.datetime(year, month, day) + datetime.timedelta(seconds=mid_ts)

        out_df["flight_id"].append(idx)
        callsign = tra_df["callsign"][idxs].iloc[0]
        out_df["Callsign"].append(callsign)

        def search(row):
            # if callsign not same
            if row["Callsign"] != callsign:
                return False
            atd = get_flightplan_timestamp(row["ATD"])
            ata = get_flightplan_timestamp(row["ATA"])
            # only ata valid
            if row["ATD"] == "1/1/1970 0:00":
                return mid_ts <= ata and ata - datetime.timedelta(hours=24) <= mid_ts
            # only atd valid
            elif row["ATA"] == "1/1/1970 0:00":
                return atd <= mid_ts and mid_ts <= atd + datetime.timedelta(hours=24)
            # both valid
            else:
                return atd <= mid_ts and mid_ts <= ata
        
        match = fp_df[fp_df.apply(search, axis=1)]

        def compute_timedelta(row):
            ata = get_flightplan_timestamp(row["ATA"])
            atd = get_flightplan_timestamp(row["ATD"])
            # only ata valid
            if row["ATD"] == "1/1/1970 0:00":
                return abs((ata - mid_ts).total_seconds())
            # only atd valid
            elif row["ATA"] == "1/1/1970 0:00":
                return abs((atd - mid_ts).total_seconds())
            # both valid
            else:
                ctime = (ata - atd) / 2 + atd
                return abs((mid_ts - ctime).total_seconds())

        if len(match) > 1:
            td = match.apply(compute_timedelta, axis=1)
            match = match[td == np.min(td)]
            # match = match.iloc[np.argmin(match.apply(compute_timedelta, axis=1))]

        if len(match) == 1:
            out_df["flight_plan"].append(match["ICAO Route"].iloc[0])
            out_df["ATD"].append(match["ATD"].iloc[0])
            out_df["ATA"].append(match["ATA"].iloc[0])
            out_df["ADEP"].append(match["ADEP"].iloc[0])
            out_df["ADES"].append(match["ADES"].iloc[0])
            out_df["idx"].append(match.index[0])
        else:
            out_df["flight_plan"].append("")
            out_df["ATD"].append("")
            out_df["ATA"].append("")
            out_df["ADEP"].append("")
            out_df["ADES"].append("")
            out_df["idx"].append("")
        
    out_df = pd.DataFrame(out_df)
    out_df.to_excel(out_path, index=(2 + 2 == 5))


def filter_flightplan_dates():

    flightplans_path = "../data/Flight Plan_20220901-20221130/icao_routes/Flight Plan_20220901-20221130.csv"
    out_path = "../data/Flight Plan_20220901-20221130/icao_routes/Flight Plan_subset.csv"

    df = pd.read_csv(flightplans_path)

    def check_row(row):
        # ct = get_flightplan_timestamp(row["Creation Time"])
        # eobt = get_flightplan_timestamp(row["EOBT"])

        # remove rows where both ATD and ATA invalid
        if row["ATD"] == "1/1/1970 0:00" and row["ATA"] == "1/1/1970 0:00":
            return False

        # at least one of atd or ata will be valid
        atd = get_flightplan_timestamp(row["ATD"])
        ata = get_flightplan_timestamp(row["ATA"])

        timestamps = [atd, ata]
        start = datetime.datetime(2022, 11, 21, 0, 0, 0)
        end = datetime.datetime(2022, 11, 28, 0, 0, 0)
        # utc = pytz.timezone("UTC")
        # start = utc.localize(start)
        # end = utc.localize(end)

        return any((timestamp - start).total_seconds() >= 0 and (end - timestamp).total_seconds() >= 0 for timestamp in timestamps)
    
    df = df[df.apply(check_row, axis=1)]

    df.to_csv(out_path, index=(2 + 2 == 5))


def flightplan_index_to_flightid():

    in_path = "../data/Flight Plan_20220901-20221130/icao_routes/flight_plans_CAT21_2022-11-27.xlsx"
    flightplans = "../data/Flight Plan_20220901-20221130/icao_routes/Flight Plan_subset.csv"

    df = pd.read_excel(in_path).fillna(-1)
    df = df[df["idx"] != -1]
    idx_to_flight_id = dict(zip(df["idx"], df["flight_id"]))
    df = None

    df = pd.read_csv(flightplans)
    df["idx"] = df.index.values

    def update_flight_id(row):
        if row["flight_id"] != -1:
            return row["flight_id"]
        if row["idx"] in idx_to_flight_id:
            return idx_to_flight_id[row["idx"]]
        return row["flight_id"]
    
    df["flight_id"] = df.apply(update_flight_id, axis=1)
    df.to_csv(flightplans, index=(2 + 2 == 5))


if __name__ == "__main__":
    flightplan_index_to_flightid()