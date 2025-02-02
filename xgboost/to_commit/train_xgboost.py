import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score


in_features = ['time_from_last_cmd',
               'curr_lat', 'curr_lon', 'curr_cas', 'curr_heading', 'curr_alt',
               'lag30_lat', 'lag30_lon', 'lag30_cas', 'lag30_heading', 'lag30_alt',
               'rwy_dir', 'tma_entry', 'star_traffic', 'below_traffic', 'on_star',
               'dist_to_wsss', 'time_to_arrival']
# in_features = ['curr_cas', 'star_traffic', 'below_traffic', 'time_to_arrival']
out_features = ['next_speed', 'next_heading', 'next_alt', 
                'waypoint', 'is_immediate', 'stay_on_route']

speed_classes = [200, 220, 250, 280, -1]
alt_classes = [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 13000, 14000, 15000, 16000, 17000, -1]


def load_data(df: pd.DataFrame, only_cmds: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    :param df: train.xlsx df
    :param only_cmds: only load rows with atco instructions, and load output features as out_feats
                      if False, load everything, but output feature is only "has_cmd"
    :return: tuple of input array, and labels array
             input array with shape (num_samples, num_features)
             if only predicting whether has voice (1 output feature), then out array has shape (num_samples,)
             else output array has shape (num_samples, num_output_features)
    """
    if not only_cmds:
        in_df = df[in_features].to_numpy()
        out_df = df["has_cmd"].to_numpy()
        return (in_df, out_df)
    # select only the rows with atco command
    df = df[df["has_cmd"] == 1]
    in_df = df[in_features]
    out_df = df[out_features]
    out_df["next_speed"] = out_df["next_speed"].apply(lambda x: speed_classes.index(x))
    out_df["next_heading"] = out_df["next_heading"].apply(lambda x: 0 if x == -1 else 1)
    out_df["next_alt"] = out_df["next_alt"].apply(lambda x: alt_classes.index(x))
    return (in_df.to_numpy(), out_df.to_numpy())
    
    
def train_predict_has_cmd(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    :param train_df: dataframe from train.xlsx
    :param val_df: same as above
    :function: train xgboost tree to just predict one output, to see whether there is voice command or not
               hence for load data, use all rows in the excel including the silent points (no voice)
    """
    train_x, train_y = load_data(train_df, only_cmds=False)
    val_x, val_y = load_data(val_df, only_cmds=False)
    model = xgb.XGBClassifier(objective="binary:logistic")
    model.fit(train_x, train_y)
    print(accuracy_score(model.predict(train_x), train_y))


def train_predict_cmds(train_df: pd.DataFrame, val_df: pd.DataFrame):
    train_x, train_y = load_data(train_df, only_cmds=True)
    val_x, val_y = load_data(val_df, only_cmds=True)
    
    # split labels into the different features because xgboost does not support multioutput for multiclass, only binary class
    train_y_speed = train_y[:, 0]
    train_y_heading = train_y[:, 1]
    train_y_alt = train_y[:, 2]
    train_y_others = train_y[:, 3:]

    val_y_speed = val_y[:, 0]
    val_y_heading = val_y[:, 1]
    val_y_alt = val_y[:, 2]
    val_y_others = val_y[:, 3:]

    speed_model = xgb.XGBClassifier(objective="multi:softmax", 
                                    num_class=len(speed_classes), 
                                    reg_lambda=10,
                                    reg_alpha=3,
                                    max_depth=3, 
                                    subsample=1,
                                    eval_metric=accuracy_score,
                                    )
    speed_model.fit(train_x, train_y_speed, eval_set=[(val_x, val_y_speed)])
    # print(accuracy_score(speed_model.predict(val_x), val_y_speed))

    alt_model = xgb.XGBClassifier(objective="multi:softmax", 
                                  num_class=len(alt_classes),
                                  )
    alt_model.fit(train_x, train_y_alt, eval_set=[(val_x, val_y_alt)])
    # print(accuracy_score(alt_model.predict(val_x), val_y_alt))

    pred_sped = [speed_classes[i] for i in speed_model.predict(val_x)]
    pred_alt = [alt_classes[i] for i in alt_model.predict(val_x)]
    df = pd.DataFrame({"pred_speed": pred_sped, "pred_alt": pred_alt})
    return df
    

def main(val_paths: list[str]):

    train_paths = [r"..\..\data\train\xgboost_v1\Voice_21-11-2022_train.xlsx",
                   r"..\..\data\train\xgboost_v1\Voice_22-11-2022_train.xlsx",
                   r"..\..\data\train\xgboost_v1\Voice_23-11-2022_train.xlsx",
                   r"..\..\data\train\xgboost_v1\Voice_24-11-2022_train.xlsx",
                   r"..\..\data\train\xgboost_v1\Voice_25-11-2022_train.xlsx"]
    # val_paths = [r"..\..\data\train\xgboost_v1\Voice_26-11-2022_train.xlsx"]
    out_path = r"..\..\data\train\xgboost_v1\Test_2025-02-02.xlsx"

    train_df = None
    for path in train_paths:
        tmp = pd.read_excel(path)
        if train_df is None:
            train_df = tmp
        else:
            train_df = pd.concat([train_df, tmp], ignore_index=(1 + 1 == 2))
    
    val_df = None
    for path in val_paths:
        tmp = pd.read_excel(path)
        if val_df is None:
            val_df = tmp
        else:
            val_df = pd.concat([val_df, tmp], ignore_index=(1 + 1 == 2))

    # train_predict_has_cmd(train_df, val_df)
    out_df = train_predict_cmds(train_df, val_df)
    out_df.to_excel(out_path, index=False)
    return out_df


if __name__ == "__main__":
    main()