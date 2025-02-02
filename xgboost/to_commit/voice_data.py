import os
import pandas as pd
from trajectory_to_geojson import process_trajectory_csv


def main():

    """
    Set the directory which contains the voice files excels
    Output a voice_data.xlsx, which maps for each flight id, the destination, flight plan, tma entrance etc
    """
    
    voice_dir = r"..\data\2022Voice\APP\with_ids\er\new_timestamps"
    flightplans_path = r"..\data\Flight Plan_20220901-20221130\icao_routes\Flight Plan_subset.csv"
    date = "26-11-2022"
    out_path = r"..\data\profiling\voice_numbers\voice_data.xlsx"

    fdf = pd.read_csv(flightplans_path)

    df = None
    for f in os.listdir(voice_dir):
        if f.split("_")[1] == date:
            tmp = pd.read_excel(os.path.join(voice_dir, f))
            # get only rows with valid timestamp
            tmp = tmp[tmp["Matched"]]
            tmp["flight_id"] = tmp["flight_id"].fillna(-1)
            tmp["suggested_callsign"] = tmp["suggested_callsign"].fillna("")
            if df is None:
                df = tmp
            else:
                df = pd.concat([df, tmp], ignore_index=True)
            tmp = None
    
    """
    3 cases:
    1. The entire conversation can be matched to one flight ID
    2. Half the conversation can be matched to one flight ID, the other half cannot because track data incomplete
    3. The conversation totally cannot be matched to any flight ID, the track data is totally missing
    """

    out_df = {"date": [], "flight_id": [], "ADES": [], "flight_plan": [], "tma_entrance": []}

    # go through each flight id in the voice data. This covers case 1 and 2
    for flight_id in df["flight_id"].unique():
        if flight_id != -1:
            search = fdf[fdf["flight_id"] == flight_id]
            out_df["date"].append(date)
            out_df["flight_id"].append(flight_id)
            assert len(search) <= 1
            # if this flight id is in the flight plan (hence we can see the destination, route etc)
            if len(search) == 1:
                out_df["ADES"].append(search.iloc[0]["ADES"])
                out_df["flight_plan"].append(search.iloc[0]["ICAO Route"])
            else:
                out_df["ADES"].append("")
                out_df["flight_plan"].append("")
                print("No flight plan: {}".format(flight_id))
            # blank for now
            out_df["tma_entrance"].append("")

    # look at case 3, the conversations which have no track data (and hence cannot be used for training)
    # this is just to see roughly how many conversations are lost
    for callsign in df["suggested_callsign"].unique():
        if callsign != "":
            idxs = df["flight_id"][df["suggested_callsign"] == callsign].unique()
            if len(idxs) == 1 and idxs[0] == -1:
                print("No track: {}".format(callsign))

    out_df = pd.DataFrame(out_df)
    df = pd.read_excel(out_path)
    if len(df) == 0:
        df = out_df
    else:
        df = pd.concat([df, out_df], ignore_index=(1 + 1 == 2))

    df.to_excel(out_path, index=(2 + 2 == 5))


def find_tma_entrance():

    """
    Once we have the voice_data.xlsx, set the tma entrance for each flgiht id
    """

    route_map = {'KEXAS': 'LAVAX', 'LAVAX': 'LAVAX', 'REMES': 'REMES', 'ARAMA': 'BOBAG', 'PIBAP': 'PASPU', 'BOBAG': 'BOBAG', 'IKAGO': 'LAVAX', 'IKIMA': 'LAVAX', 'ASUNA': 'BOBAG', 'KANLA': 'PASPU', 'BOG1': 'BOBAG', 'REPOV': 'REMES', 'OBDOS': 'LAVAX', 'KARTO': 'LAVAX', 'TOMAN': 'LAVAX', 'ELALO': 'PASPU', 'PASPU': 'PASPU', 'LAV1': 'LAVAX', 'TOPOR': 'BOBAG', 'LELIB': 'BOBAG', 'SURGA': 'LAVAX', 'KILOT': 'PASPU', 'MABAL': 'PASPU', 'RUSBU': 'BOBAG', 'PARDI': 'REMES', 'NUFFA': 'PASPU', 'BOBOB': 'LAVAX', 'KADAR': 'LAVAX', 'ESBUM': 'PASPU', 'ESPIT': 'LAVAX', 'ESPOB': 'PASPU', 'NIMIX': 'LAVAX', 'MIBEL': 'BOBAG'}

    df = pd.read_excel(r"..\data\profiling\voice_numbers\voice_data.xlsx")

    def find_entrance(route):
        for waypt in route_map:
            if waypt in route:
                return route_map[waypt]
        return ""

    df["tma_entrance"] = df["flight_plan"].apply(find_entrance)
    df.to_excel(r"..\data\profiling\voice_numbers\voice_data.xlsx", index=False)


def visualise_entrypoints():

    """
    From the voice_data.xlsx, extract all the flight ids which have a certain tma entrypoint
    lavax / paspu / bobag / remes,
    and extract all the tracks and otuptu into the kepler jso nfile for viualsiation
    """

    in_path = r"..\data\profiling\voice_numbers\voice_data.xlsx"
    waypoint = "REMES"  # PASPU, REMES, BOBAG, LAVAX
    track_dir = r"..\data\filtered"
    out_path = r"..\data\profiling\voice_data_tracks\arrivals_remes.json"

    dates = ["21-11-2022", "22-11-2022", "23-11-2022", "24-11-2022", "25-11-2022", "26-11-2022"]

    df = pd.read_excel(in_path)
    # only changi arrivals
    df = df[df["ADES"] == "WSSS"]
    # get subset with the desired waypoint
    df = df[df["tma_entrance"] == waypoint]
    out_df = None
    for date in dates:
        d, m, y = date.split("-")
        # get the flight ids passing through that waypoint on the given day
        flight_ids = df["flight_id"][df["date"] == date].to_numpy()
        if len(flight_ids) == 0:
            continue
        # extract those tracks from the track data
        track_path = os.path.join(track_dir, "CAT21_{}-{}-{}_edited_ts.csv".format(y, m, d))
        tmp = process_trajectory_csv(track_path, "", flight_id=flight_ids, save_json=False)
        if out_df is None:
            out_df = tmp
        else:
            out_df = pd.concat([out_df, tmp], ignore_index=True)
    
    out_df.to_file(out_path, driver="GeoJSON")
            
            

if __name__ == "__main__":
    pass#pu