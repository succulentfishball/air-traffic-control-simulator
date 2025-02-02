import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Masking, Input, Flatten
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import numpy as np


in_feats = ["curr_lat", "curr_lon", "curr_cas", "curr_heading", "curr_alt", "tma_entry", 
            "rwy_dir", "star_traffic", "below_traffic", "on_star", "dist_to_wsss", "time_to_arrival"]
# classification: speed, alt, waypoint, is_immediate, stay_on_route. Regression: heading
out_feats = ["next_heading", "next_speed", "next_alt", "waypoint", "is_immediate", "stay_on_route"]

# change in train_data_preprocess also
speed_classes = [200, 220, 250, 280, -1]
alt_classes = [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 13000, 14000, 15000, 16000, 17000, -1]


def prepare_dataset(df: pd.DataFrame):
    """
    From the train.xlsx, return the dataset array in the shape (num_flights, num_timesteps, num_features)
    num_flights is number of flight ids
    num_timesteps is just number of obversations
    num_features is the number of input features e.g. current speed, current heading etc
    """
    # get the maximum number of timesteps in any given flight
    num_timesteps = df["flight_id"].value_counts().max()
    df["on_star"] = df["on_star"].astype(np.intp)
    df["has_data"] = df["has_data"].astype(np.intp)

    # df["next_speed"] = df["next_speed"].apply(lambda x: speed_classes.index(x) if x != 0 else 0)
    # df["next_alt"] = df["next_alt"].apply(lambda x: alt_classes.index(x) if x != 0 else 0)

    in_arr = []
    out_arr = []

    for flight_id in df["flight_id"].unique():
        idxs = df["flight_id"] == flight_id
        # get the input features for this flight id in the shape (num_timestamps, num_in_features)
        x = df[in_feats][idxs].to_numpy()
        # pad it with zeros in front to make it the maximum number of timesteps
        # x = np.vstack((np.zeros((num_timesteps - len(x), x.shape[1])), x))
        y = df[out_feats][idxs].to_numpy()
        # same for outputs
        # y = np.vstack((np.zeros((num_timesteps - len(y), y.shape[1])), y))
        in_arr.append(x)
        out_arr.append(y)
        # break # testing, remove later
    
    in_arr = pad_sequences(in_arr, dtype="float64", value=0.0)
    out_arr = pad_sequences(out_arr, dtype="float64", value=0.0)

    return (in_arr, out_arr)
        

def train():

    train_paths = [r"..\data\train\train_data_lstm_v2\Voice_21-11-2022_train_.xlsx",
                   r"..\data\train\train_data_lstm_v2\Voice_22-11-2022_train.xlsx",
                   r"..\data\train\train_data_lstm_v2\Voice_23-11-2022_train.xlsx",
                   r"..\data\train\train_data_lstm_v2\Voice_24-11-2022_train_.xlsx",
                   r"..\data\train\train_data_lstm_v2\Voice_25-11-2022_train.xlsx",
                   r"..\data\train\train_data_lstm_v2\Voice_26-11-2022_train.xlsx"]
    out_path = r"..\data\train\test.keras"

    df = None
    for train_path in train_paths:
        tmp = pd.read_excel(train_path)
        if df is None:
            df = tmp
        else:
            df = pd.concat([df, tmp], ignore_index=True)
        tmp = None

    x, y = prepare_dataset(df)

    # TODO: make it properly
    train_x = x[:750, :, :]
    train_y = y[:750, :, :]
    val_x = x[750:, :, :]
    val_y = y[750:, :, :]

    scaler = MinMaxScaler()
    out_scaler = MinMaxScaler()

    # normalize
    n_train_samples, n_timesteps, n_features = train_x.shape
    train_x = train_x.reshape(-1, n_features)
    n_val_samples, n_timesteps, n_features = val_x.shape
    val_x = val_x.reshape(-1, n_features)
    scaler.fit(train_x)
    train_x = scaler.transform(train_x).reshape(n_train_samples, n_timesteps, n_features)
    val_x = scaler.transform(val_x).reshape(n_val_samples, n_timesteps, n_features)

    # output label for each feature
    train_y_heading = train_y[:, :, [0]]
    train_y_speed = train_y[:, :, 1]
    train_y_alt = train_y[:, :, 2]
    train_y_wypt = train_y[:, :, [3]]
    train_y_imm = train_y[:, :, [4]]
    train_y_route = train_y[:, :, [5]]
    val_y_heading = val_y[:, :, [0]]
    val_y_speed = val_y[:, :, 1]
    val_y_alt = val_y[:, :, 2]
    val_y_wypt = val_y[:, :, [3]]
    val_y_imm = val_y[:, :, [4]]
    val_y_route = val_y[:, :, [5]]
    
    n_train_y_samples, n_timesteps, n_out_features = train_y_heading.shape
    train_y_heading = train_y_heading.reshape(-1, n_out_features)
    n_val_y_samples, n_timesteps, n_out_features = val_y_heading.shape
    val_y_heading = val_y_heading.reshape(-1, n_out_features)
    out_scaler.fit(train_y_heading)
    train_y_heading = out_scaler.transform(train_y_heading).reshape(n_train_y_samples, n_timesteps, n_out_features)
    val_y_heading = out_scaler.transform(val_y_heading).reshape(n_val_y_samples, n_timesteps, n_out_features)

    input_layer = Input(shape=(train_x.shape[1], train_x.shape[2]))
    mask_out = Masking(mask_value=0.0)(input_layer)
    lstm_out = LSTM(128, return_sequences=True)(mask_out)
    lstm_out_2 = LSTM(64, return_sequences=True)(lstm_out)
    heading_out = Dense(1, activation="linear", name="heading_out")(lstm_out_2)
    speed_out = Dense(len(speed_classes), activation="softmax", name="speed_out")(lstm_out_2)
    alt_out = Dense(len(alt_classes), activation="softmax", name="alt_out")(lstm_out_2)
    wypt_out = Dense(1, activation="sigmoid", name="wypt_out")(lstm_out_2)
    imm_out = Dense(1, activation="sigmoid", name="imm_out")(lstm_out_2)
    route_out = Dense(1, activation="sigmoid", name="route_out")(lstm_out_2)

    model = Model(inputs=input_layer, outputs=[heading_out, speed_out, alt_out,
                                               wypt_out, imm_out, route_out])
    model.compile(optimizer="adam",
                  loss=["mean_squared_error", "sparse_categorical_crossentropy", "sparse_categorical_crossentropy",
                        "binary_crossentropy", "binary_crossentropy", "binary_crossentropy"],
                  metrics=["mae", "accuracy", "accuracy", "accuracy", "accuracy", "accuracy"]
    )
    
    model.summary()

    model.fit(
        train_x,
        [train_y_heading, train_y_speed, train_y_alt, train_y_wypt, train_y_imm, train_y_route],
        epochs=800,
        batch_size=8,
        validation_data=(val_x, [val_y_heading, val_y_speed, val_y_alt, val_y_wypt, val_y_imm, val_y_route]),
        verbose=2
    )

    model.save(out_path)


def test():

    model_path = r"C:\Users\Work and School\Documents\i2r\ra_2024\data\train\train_data_lstm_v2\test.keras"

    train_paths = [r"..\..\data\train\train_data_lstm_v2\Voice_21-11-2022_train_.xlsx",
                   r"..\..\data\train\train_data_lstm_v2\Voice_22-11-2022_train.xlsx",
                   r"..\..\data\train\train_data_lstm_v2\Voice_23-11-2022_train.xlsx",
                   r"..\..\data\train\train_data_lstm_v2\Voice_24-11-2022_train_.xlsx",
                   r"..\..\data\train\train_data_lstm_v2\Voice_25-11-2022_train.xlsx",
                   r"..\..\data\train\train_data_lstm_v2\Voice_26-11-2022_train.xlsx"]
    out_path = r"..\data\train\test.keras"

    df = None
    for train_path in train_paths:
        tmp = pd.read_excel(train_path)
        if df is None:
            df = tmp
        else:
            df = pd.concat([df, tmp], ignore_index=True)
        tmp = None

    x, y = prepare_dataset(df)

    # TODO: make it properly
    train_x = x[:750, :, :]
    train_y = y[:750, :, :]

    test_path = r"..\..\data\train\train_data_lstm_v2\Voice_24-11-2022_train_.xlsx"

    test_df = pd.read_excel(test_path)

    val_x, val_y = prepare_dataset(test_df)

    scaler = MinMaxScaler()

    # normalize
    n_train_samples, n_timesteps, n_features = train_x.shape
    train_x = train_x.reshape(-1, n_features)
    n_val_samples, n_timesteps, n_features = val_x.shape
    val_x = val_x.reshape(-1, n_features)
    scaler.fit(train_x)
    train_x = scaler.transform(train_x).reshape(n_train_samples, n_timesteps, n_features)
    val_x = scaler.transform(val_x).reshape(n_val_samples, n_timesteps, n_features)

    model = load_model(model_path)
    res = model.predict(val_x)

    predicted_heading = res[0]
    predicted_speed = res[1]
    predicted_alt = res[2]

    test_df = test_df[test_df["flight_id"] == 412844]
    
    predicted_heading = predicted_heading[0, -39:, :]
    predicted_speed = predicted_speed[0, -39:, :]
    predicted_alt = predicted_alt[0, -39:, :]

    curr_lon = test_df.iloc[0, 4]
    curr_lat = test_df.iloc[0, 3]
    curr_alt = test_df.iloc[0, 7]
    curr_heading = test_df.iloc[0, 6]
    curr_cas = test_df.iloc[0, 5]

    reconstructed_lon = []
    reconstructed_lat = []
    reconstructed_cas = []
    reconstructed_heading = []
    reconstructed_alt = []
    
    # for each possible timestep
    for i in range(predicted_speed.shape[0]):
        # get control input for heading
        curr_heading = predicted_heading[i, 0]
        # get new speed
        speed_idx = np.argmax(predicted_speed[i, :])
        if speed_idx != 4:
            curr_speed = speed_classes[speed_idx]
        # get new altitude
        alt_idx = np.argmax(predicted_alt[i, :])
        if alt_idx != 13:
            curr_alt = alt_classes[alt_idx]
        reconstructed_alt.append(curr_alt)
        reconstructed_heading.append(curr_heading)
        reconstructed_cas.append(curr_cas)
        theta = (90 - curr_heading) % 360
        curr_lon = curr_lon + curr_cas * np.cos(theta) / 1.944 / 111139 * 10
        curr_lat = curr_lat + curr_cas * np.sin(theta) / 1.944 / 111139 * 10
        reconstructed_lon.append(curr_lon)
        reconstructed_lat.append(curr_lat)
    
    df = {
        "curr_lon": reconstructed_lon,
        "curr_lat": reconstructed_lat,
        "curr_alt": reconstructed_alt,
        "curr_cas": reconstructed_cas,
        "curr_heading": reconstructed_heading
    }

    df = pd.DataFrame(df)
    import shapely.geometry
    df["geometry"] = df.apply(lambda x: shapely.geometry.Point([x["curr_lon"], x["curr_lat"]]), axis=1)
    import geopandas as gpd
    df = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    
    from trajectory_to_geojson import process_row
    import datetime
    df["geometry"], df["lineColor"] = zip(*df.apply(process_row, axis=1))
    df["timestamp"] = test_df["interval_start"].apply(lambda x: datetime.datetime(2024, 11, 24) + datetime.timedelta(seconds=x)).values
    # df = df.drop(["curr_lat", "curr_lon"], axis=1)
        
    df.to_excel(r"..\data\train\train_data_lstm_v2\2022-11-24_tmn11.xlsx", index=False)





2


if __name__ == "__main__":
    test()