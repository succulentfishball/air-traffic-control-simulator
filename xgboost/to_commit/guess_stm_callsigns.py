import pandas as pd
import os
import re


# parse the callsign if possible for each line of dialogue in the stm files
# save to excel file for each stm file


number_map = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "niner": "9",
    "nine": "9"
}

# map call sign to airline icao code e.g. {"speedbird": "BAW"}
callsign_map = {}


def main():

    icao_list = "../data/icao-airline-codes.xlsx"
    data_dir = "../data/2022Voice/ARR"

    icao_df = pd.read_excel(icao_list).fillna("")
    def populate_callsign_map(row):
        global callsign_map
        if row["Call sign"] != "" and row["ICAO"] != "":
            callsign_map[row["Call sign"].lower()] = row["ICAO"]
    icao_df.apply(populate_callsign_map, axis=1)

    # iterate over all stm files 
    for fname in os.listdir(data_dir):
        if os.path.splitext(fname)[-1].lower() == ".stm" and not os.path.exists(os.path.join(data_dir, os.path.splitext(fname)[0] + "_done.xlsx")):
            with open(os.path.join(data_dir, fname)) as f:
                # read into df
                df = pd.DataFrame({"Lines": f.readlines()})
                def process_line(line):
                    src, start, end = "", "", ""
                    if not line.startswith(";;"):
                        # get the dialogue content, removing leading / trailing whitespace and remove "er"s in transcription
                        src, start, end = line.split()[2:5]
                        line = line.split(">")[-1].lstrip().strip().strip("\n")#.replace(" er ", " ") don't replace er
                        if len(line) > 0:
                            line = " ".join(map(lambda x: number_map[x] if x in number_map else x, line.split()))
                    return (line, src, start, end)
                
                df["Lines"], df["Source"], df["start_time"], df["end_time"] = zip(*df["Lines"].apply(process_line))
                df = df[(df["Lines"].str.len() > 0) & ~(df["Lines"].str.startswith(";;"))]

                query = r"(?=({}))".format("|".join([r"\b{}\b".format(key) for key in callsign_map.keys()]))
                df["callsign_prefix"] = df["Lines"].str.findall(query)

                def process_callsign(row):
                    callsigns = list(set(row["callsign_prefix"])) # remove duplicates
                    content = row["Lines"]
                    for callsign in callsigns:
                        matches = re.finditer(callsign, content)
                        for m in matches:
                            ridx = m.end() # last index of match, non inclusive
                            suggestion = callsign_map[callsign]
                            for i in range(ridx, len(content)):
                                if content[i] == " ":
                                    continue
                                if content[i].isnumeric():
                                    suggestion += content[i]
                                else:
                                    break
                            if len(suggestion) >= 4: # if at least 1 numbers after icao callsign
                                return suggestion
                    return ""

                df["suggested_callsign"] = df.apply(process_callsign, axis=1)

                speakers = df["Source"].unique()
                for speaker in speakers:
                    if "P" in speaker: # pilot
                        idxs = df["Source"] == speaker
                        counts = df[idxs]["suggested_callsign"].value_counts().to_dict()
                        if "" in counts:
                            counts.pop("")
                        best_suggestion = max(counts, key=counts.get) if counts else ""
                        df.loc[idxs, "suggested_callsign"] = best_suggestion
                        
                df.to_excel(os.path.join(data_dir, os.path.splitext(fname)[0] + ".xlsx"), index=(2 + 2 == 5))


if __name__ == "__main__":
    main()