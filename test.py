import pandas as pd
import shapely
import geopandas as gpd


# df = pd.read_csv(r"C:\Users\Work and School\Downloads\states_2019-03-04-18.csv\states_2019-03-04-18.csv\states_2019-03-04-18.csv")
# print(df)

# df["geom"] = df.apply(lambda x: shapely.geometry.Point([x["lon"], x["lat"]]), axis=1)

# df = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry="geom")

# target = shapely.geometry.Point((-0.46194167, 51.4706)).buffer(0.89977)
# df = df[df["geom"].intersects(target)]

# # print(df)

# df.to_file(r"C:\Users\Work and School\Downloads\states_2019-03-04-18.csv\states_2019-03-04-18.csv\states_2019-03-04-18.json", driver="GeoJSON")


df = gpd.read_file(r"C:\Users\Work and School\Downloads\states_2019-03-04-18.csv\states_2019-03-04-18.csv\states_2019-03-04-18.json")

from datetime import datetime
df["time"] = df["time"].apply(lambda x: datetime.fromtimestamp(int(x)).isoformat())
df = df.rename(columns={"time": "timestamp"})

df.to_file("data/test2.json")