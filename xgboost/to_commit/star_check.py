import re
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt


"""
Given an ICAO route e.g. RUSBU1F RUSBU Y511 TOPOR A464 ARAMA BOG1 ARA1B
or e.g. SBR RAMPY M635 SURGA IKAGO 0040N10514E LAVAX RUVIK DOVAN BIPOP
assuming arrival at WSSS, create the arrival route (ending part of the flightplan) as a shapely LineString
Then given the current location of the aircraft, state whether it has reached the start of the arrival route,
and if so, whether it is currently on the route or deviated, and what is the next waypoint
*Not possible to construct the whole route because some of the airways / waypoints data is lacking
"""


tolerance = 30.0 # km


def get_arrival_points(icao_route, waypoints, stars):

    """
    given the icao_rotue as a string, return it as the list of waypoints e.g. ['dovan', 'bipop'] etc
    """

    # first check to see if there is any funny names like the LAV3, LAV1, PAS3, BOG3 etc
    segments = icao_route.split()
    for i in range(len(segments) - 1):
        if len(segments[i]) == 4 and segments[i][:3] in ["PAS", "LAV", "BOG"] and segments[i][3].isnumeric():
            # if the following id is not a star (just usual waypoint), then set it to DCT
            # otherwise if the next one is a star then ignore and continue, just use the star
            if segments[i + 1] not in stars and (segments[i + 1] in waypoints or len(re.findall("[0-9]+[N|S][0-9]+[E|W]", segments[i + 1]))):
                segments[i] = "DCT"
    icao_route = " ".join(segments)

    # replace the star name e.g. ARA1B with the waypoint names e.g. ARAMA BOBAG SAMKO BTM DOVAN BIPOP
    icao_route = " ".join([stars[x] if x in stars else x for x in icao_route.split()])
    segments = icao_route.split()
    # work backwards until there is a waypoint which we don't have data of
    start_idx = 0
    for i in range(len(segments) - 1, -1, -1):
        # if the segment not a recognised waypoint, could be either
        # (i) explicit coords e.g. 14N140E, (ii) unknown waypoint, (iii) awy id or (iv) DCT
        # for (i) and (iv) continue constructing the route

        if segments[i] not in waypoints:
            if segments[i] == "DCT":
                continue
            # if not DCT, try to see if it is a geographical coordinate e.g. 22N180E/M084F360
            match = re.findall("[0-9]+[N|S][0-9]+[E|W]", segments[i])
            if len(match):
                segments[i] = match[0]
            # if not a geographic coordinate i.e. awy id or unrecognised waypoint, stop constructing
            else:
                start_idx = i + 1
                break
    if i == 0:
        return [p for p in segments if p != "DCT"]
    if i + 1 < len(segments):
        return [p for p in segments[i + 1:] if p != "DCT"]
    else:
        return []


def parse_coord_to_point(coord):
    """
    :param coord: e.g. '0040N10514E'
    :return: shapely Point
    """
    lat, lon = re.findall("[0-9]+", coord)
    # degrees and minutes
    if len(lat) > 2:
        d = lat[:2]
        m = lat[2:]
        lat = int(d) + int(m) / 60.0
    # else just degrees, no minutes
    else:
        lat = int(lat)
    # same for lon, if degrees and minutes
    if len(lon) > 3:
        d = lon[:3]
        m = lon[3:]
        lon = int(d) + int(m) / 60.0
    # else only degrees
    else:
        lon = int(lon)
    if "S" in coord:
        lat *= -1
    if "W" in coord:
        lon *= -1
    return shapely.Point((lon, lat))


def parse_points(points, waypoints):
    """
    :param points: e.g. ['SURGA', 'IKAGO', '0040N10514E', 'LAVAX', 'RUVIK', 'DOVAN', 'BIPOP']
    :return: list of shapely Points
    """
    return [wkt.loads(waypoints[p]) if p in waypoints else parse_coord_to_point(p) for p in points]


def get_next_point(curr_point, points):
    """
    Given the current location of the aircraft and the arrival route,
    return the index of the next point on the route
    """
    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        p0 = np.array([p0.x, p0.y])
        p1 = np.array([p1.x, p1.y])
        # direction vector from p0 to p1
        d = p1 - p0
        x = np.array([curr_point.x, curr_point.y])
        k = np.dot(x - p0, d) / np.dot(d, d)
        f = p0 + k * d
        l = np.linalg.norm(x - f) * 111.139
        if l > tolerance:
            continue
        if i == 0:
            if k < 0:
                return 0
            elif 0 <= k < 1:
                return 1
            else:
                continue
        else:
            if 0 <= k < 1:
                return i + 1
    return -1


def is_on_star(curr_point, points):
    """
    Given the current location of the aircraft and the arrival route,
    return whether it is currently on the star or has deviated
    """
    route_linestr = shapely.LineString(points)
    route_buf = route_linestr.buffer(tolerance / 111.139)
    return route_buf.intersects(curr_point)


def visualise_flight_plans():

    """
    Get those flights which arrive at wsss and have no voice data and output into kepler format
    """

    stars_path = "../data/Flight Plan_20220901-20221130/bluesky_waypoints_and_stars/aip_enr_star_graph_bluesky 2_stars.xlsx"
    waypoints_path = "../data/Flight Plan_20220901-20221130/bluesky_waypoints_and_stars/aip_enr_star_graph_bluesky 2_nodes.xlsx"
    flightplans_path = "../data/Flight Plan_20220901-20221130/icao_routes/Flight Plan_subset.csv"
    out_path = "../data/profiling/flightplans_no_voice.json"

    waypoints_df = pd.read_excel(waypoints_path)
    waypoints = dict(zip(waypoints_df["waypointId"], waypoints_df["geom"]))
    waypoints_df = None

    stars_df = pd.read_excel(stars_path)
    stars = dict(zip(stars_df["routeId"], stars_df["waypoints"]))
    stars_df = None

    df = pd.read_csv(flightplans_path)
    # only those arriving at WSSS, and have no voice data
    df = df[(df["ADES"] == "WSSS") & (~df["has_voice"])]

    # count of how many routes cannot be parsed properly due to e.g. missing waypoint / route data
    stupid = 0

    def route_to_linestring(route):
        nonlocal stupid
        try:
            return shapely.LineString(parse_points(get_arrival_points(route, waypoints, stars), waypoints))
        except:
            stupid += 1
            return shapely.LineString([])
        

    df["arrival_route"] = df["ICAO Route"].apply(route_to_linestring)
    print(stupid)
    df = df[~df["arrival_route"].apply(lambda x: x.is_empty)]
    # drop unnecessary columns
    df = df[["Callsign", "ATD", "ATA", "ICAO Route", "flight_id", "arrival_route"]]
    df = gpd.GeoDataFrame(df, geometry="arrival_route", crs="EPSG:4326")
    df.to_file(out_path, driver="GeoJSON")


def main():

    """
    Example usage of the functions to see what is the next point on the star, and also whether it is on the star
    """

    stars_path = "../data/Flight Plan_20220901-20221130/bluesky_waypoints_and_stars/aip_enr_star_graph_bluesky 2_stars.xlsx"
    waypoints_path = "../data/Flight Plan_20220901-20221130/bluesky_waypoints_and_stars/aip_enr_star_graph_bluesky 2_nodes.xlsx"
    icao_route = "SBR RAMPY M635 SURGA IKAGO 0040N10514E LAVAX RUVIK DOVAN BIPOP"
    curr_point = shapely.Point((104.4547, 1.1143))

    waypoints_df = pd.read_excel(waypoints_path)
    waypoints = dict(zip(waypoints_df["waypointId"], waypoints_df["geom"]))
    waypoints_df = None

    stars_df = pd.read_excel(stars_path)
    stars = dict(zip(stars_df["routeId"], stars_df["waypoints"]))
    stars_df = None

    points_str = get_arrival_points(icao_route, waypoints, stars)
    points = parse_points(points_str, waypoints)

    next_idx = get_next_point(curr_point, points)
    print("Next Point: {}. {}".format(next_idx, points_str[next_idx]))

    print(is_on_star(curr_point, points))


if __name__ == "__main__":
    main()