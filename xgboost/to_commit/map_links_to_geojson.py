import json
import shapely
import pandas as pd
import geopandas as gpd


def load_json(path, **kwargs):
    with open(path, 'r') as f:
        data = json.load(f, **kwargs)
    return data


def load_waypoints():

    waypoints_df = pd.read_excel("../data/Flight Plan_20220901-20221130/aip_enr_star_graph_bluesky 2_nodes.xlsx")
    waypoints_df["geom"] = waypoints_df["geom"].apply(lambda x: shapely.wkt.loads(x))
    return dict(zip(waypoints_df["id"], waypoints_df["geom"])), dict(zip(waypoints_df["id"], waypoints_df["waypointId"]))


def main():

    """
    From the bluesky file, get all the airways and routes and output into the kepler json format
    """

    data = load_json("../data/Flight Plan_20220901-20221130/aip_enr_star_graph_bluesky 2.json")
    out_path = "../data/Flight Plan_20220901-20221130/aip_enr_star_graph_bluesky 2_stars.json"

    links = data["links"]
    
    int_df = {"EdgeIndex": [], "routeId": [], "latlon": [], "type": [], "source": [], "target": [], "key": []}
    out_df = {"routeId": [], "type": [], "key": [], "geom": [], "waypoints": []}

    waypoint_geoms, waypoint_names = load_waypoints()

    # populate intermediate dataframe
    for link in links:

        int_df["EdgeIndex"].append(link["EdgeIndex"])
        int_df["routeId"].append(link["routeId"])

        for i in range(len(link["latlon"])):
            link["latlon"][i] = link["latlon"][i][::-1]

        int_df["latlon"].append(link["latlon"])
        int_df["type"].append(link["type"])
        int_df["source"].append(link["source"])
        int_df["target"].append(link["target"])
        int_df["key"].append(link["key"])

    int_df = pd.DataFrame(int_df)
    
    # for each route, get the waypoint ids from start to end
    for route_id in int_df["routeId"].unique():

        adj_list = {}
        reverse_adj_list = {}
        sub_df = int_df[int_df["routeId"] == route_id]

        if sub_df.iloc[0]["type"] != "STAR":
            continue

        for i in range(len(sub_df)):
            row = sub_df.iloc[i]
            src = row["source"]
            dst = row["target"]

            if src not in adj_list:
                adj_list[src] = [dst]
            else:
                adj_list[src].append(dst)

            if dst not in adj_list:
                adj_list[dst] = []

            if src not in reverse_adj_list:
                reverse_adj_list[src] = []
            
            if dst not in reverse_adj_list:
                reverse_adj_list[dst] = [src]
            else:
                reverse_adj_list[dst].append(src)

        start = None
        for node in reverse_adj_list:
            if len(reverse_adj_list[node]) == 0:
                start = node
        assert start is not None

        path = [start]
        current = start
        while len(adj_list[current]) > 0:
            assert len(adj_list[current]) == 1
            path.append(adj_list[current][0])
            current = adj_list[current][0]
        
        path_geom = shapely.LineString([waypoint_geoms[x] for x in path])
        path_str = " ".join([waypoint_names[x] for x in path])

        out_df["routeId"].append(route_id)
        out_df["type"].append("STAR")
        out_df["key"].append(sub_df.iloc[0]["key"])
        out_df["geom"].append(path_geom)
        out_df["waypoints"].append(path_str)

    out_df = pd.DataFrame(out_df)
    out_df = gpd.GeoDataFrame(out_df, crs="EPSG:4326", geometry="geom")        
                
    out_df.to_file(out_path, driver="GeoJSON")
        


if __name__ == "__main__":
    main()