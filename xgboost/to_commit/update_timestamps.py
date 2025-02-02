import os
import pandas as pd


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


def main():

    """
    From the raw files (old stm file) to the updated timestamps file (with the silent segment restored),
    Update the timestamps in the excel files
    """

    in_dir = r"..\data\2022Voice\ARR"
    stm_dir = r"..\data\2022Voice\updated_timestamps"
    out_dir = r"..\data\2022Voice\ARR\new_timestamps"

    for x in os.listdir(in_dir):
        if x.endswith(".xlsx"):
            # new_stm = os.path.join(stm_dir, x.split("_done.xlsx")[0] + ".realtime_est.stm")
            new_stm = os.path.join(stm_dir, x.split(".xlsx")[0] + ".realtime_est.stm")
            old_xlsx = os.path.join(in_dir, x)
            out_path = os.path.join(out_dir, x)

            with open(new_stm) as f:
            
                df = pd.DataFrame({"Message": f.readlines()})

                def process_line(line):
                    content = line.split()
                    if len(content) == 2:
                        return (-1, -1, "")
                    start, end = map(float, content[:2])
                    message = " ".join(content[2:])
                    message = message.lower()
                    message = " ".join(map(lambda x: number_map[x] if x in number_map else x, message.split()))
                    return (start, end, message)

                df["start_time"], df["end_time"], df["Message"] = zip(*df["Message"].apply(process_line))
                df = df[df["Message"] != ""]
                
                old_df = pd.read_excel(old_xlsx)
                i, k = 0, 0
                old_df["Matched"] = False

                while i < len(old_df) and k < len(df):
                    msg = old_df.iloc[i]["Lines"].lower()
                    if msg == df.iloc[k]["Message"]:
                        old_df.iat[i, 2] = df.iloc[k]["start_time"]
                        old_df.iat[i, 3] = df.iloc[k]["end_time"]
                        old_df.iat[i, -1] = True
                        k += 1
                        i += 1
                        continue
                    else:
                        found = False
                        for j in range(i + 1, len(old_df)):
                            msg = msg + " " + old_df.iloc[j]["Lines"].lower()
                            if msg == df.iloc[k]["Message"]:
                                for shit in range(i, j + 1):
                                    old_df.iat[shit, 2] = df.iloc[k]["start_time"]
                                    old_df.iat[shit, 3] = df.iloc[k]["end_time"]
                                    old_df.iat[shit, -1] = True
                                found = True
                                k += 1
                                i = j + 1
                                break
                        if found:
                            continue
                        else:
                            i += 1

                old_df.to_excel(out_path, index=(2 + 2 == 5))             


def check():

    """
    See how many lines the updated txt file is missing
    """

    out_dir = r"..\data\2022Voice\ARR\new_timestamps"

    minimum = 0
    
    for f in os.listdir(out_dir):
        print(f)
        df = pd.read_excel(os.path.join(out_dir, f))
        for i in range(len(df)):
            if not df.iloc[i, -1]:
                x = i - len(df)
                print(x, end=" ")
                if x < minimum:
                    minimum = x
        print("\n")

    print(minimum)


if __name__ == "__main__":
    check()