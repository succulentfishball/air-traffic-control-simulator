import math
import pyomo.environ as pyo 
from typing import Optional
# from ConflictZones import ConflictZone
# from PlaneToPlaneIdDict import PlaneToPlaneIdDict
from Airplane import crash_threshold, hold, Airplane
import sys
from utils import Coordinate, haversine, Velocity, Action, Gate, LHR_COORDS





# Excluding acceleration and ascending - focused on landings
# class Action(Enum):
#     DECELERATE = 2 # speed -= 5
#     HOLD = 3 # Start hold pattern
#     STOP_HOLD = 4 # Stop hold pattern
#     RIGHT = 5 # direction.heading += 3
#     LEFT = 6 # direction.heading -= 3
#     DOWN = 8 # coordinate.altitude -= 5

class GameState:
  
    def __init__(self) -> None:
        self.airplanes = []
        # self.conflictZones: list[ConflictZone] = []

    def add_airplane(self, airplane):
        self.airplanes.append(airplane)
    
    def remove_airplane(self, airplane):
        self.airplanes.remove(airplane)
    
    def will_collide(self, airplane1, airplane2):
        horizontal_distance = haversine(airplane1.coordinates, airplane2.coordinates)
        vertical_distance = abs(airplane1.coordinates.altitude - airplane2.coordinates.altitude)
        result = horizontal_distance < crash_threshold + 5 and vertical_distance < crash_threshold + 5
        return result 
    
    # def reaches_airport(self, airplane, airport):
    #     x_distance = airplane.coordinates.x - airport.coordinates.x
    #     y_distance = airplane.coordinates.y - airport.coordinates.y
    #     z_distance = airplane.coordinates.z - airport.coordinates.z
    #     return (x_distance < x_threshold and y_distance < y_threshold and z_distance < z_threshold)

    def in_conflict_zone(self, airplane):
       coord = airplane.coordinates
       for conflictZone in self.conflictZones:
           if conflictZone.in_conflict_zone(coord):
               return True
       return False
    
    # def iter_state(self): # TODO: conflict zones will never be removed even if resolved
    #     self.airplanes.filter()
    #     # for each airplane, check if it shares a conflict zone with any other airplane.
    #     for i, airplane1 in enumerate(self.airplanes):
    #         for j, airplane2 in enumerate(self.airplanes):
    #             if i < j and self.will_collide(airplane1, airplane2):
    #                 conflictZone = ConflictZone(airplane1.planeId, airplane2.planeId)
    #                 for cZ in self.conflictZones:
    #                     if cZ.airplane1Id == airplane1.planeId or cZ.airplane2Id == airplane2.planeId:
    #                         # if any of the planes are already in (older) conflict zones, don't add the new conflict zone.
    #                          break
    #                     if cZ == conflictZone: # if conflictZone already exists between pair, don't add it again
    #                         break
    #                     else:
    #                         self.conflictZones.append(conflictZone)
    #                         airplane1.conflictZone = conflictZone
    #                         airplane2.conflictZone = conflictZone

    #     # for each airplane, update state
    #     self.airplanes.forEach(lambda airplane: airplane.update_state())
    #     # for each airplane, check if it has gone through entry point
    #     self.airplanes.filter(lambda airplane: not airplane.through_LHR_COORDS())
    #     # resolve conflicts
    #     self.resolve_conflicts()

    
# Cost function will decide every next move of the aircraft    
# def cost (airplane, next_coord: Coordinate, gs: GameState) -> int:
#     velocity = airplane.velocity
#     # the current velocity can be considered from current to entry. Would there be different velocities for current to next and next toentry?.
#     distance_to_LHR_COORDS = haversine(airplane.coordinates, LHR_COORDS)
#     distance_to_next_coord = haversine(airplane.coordinates, next_coord)
#     distance_from_next_to_entry = haversine(next_coord, LHR_COORDS)
#     if gs.in_conflict_zone(next_coord):
#         cost = sys.maxsize
#     else:
#         cost = velocity * distance_to_next_coord + velocity * distance_from_next_to_entry - velocity * distance_to_LHR_COORDS
#     return cost

# resolve conflicts
# def resolve_conflicts(self):
#     return NotImplementedError
# #     for cZ in self.conflictZones:

    

class Optimizer():
    
    def __init__(self, prediction_horizon: int, aircrafts: list, dt: int = 10):

        self.dt = dt
        aircrafts = [a for a in aircrafts if not a.through_entry_point()]
        
        # map each aircraft's index to the position of the center of the circle around which it is holding
        lon_points = {i + 1: None for i in range(len(aircrafts))}
        lat_points = {i + 1: None for i in range(len(aircrafts))}
        alt_points = {i + 1: None for i in range(len(aircrafts))}
        # radius of hold in metres
        holding_radius = 500
        # minimum horizontal separation in metres
        # self.min_horizontal_separation = 25000
        # minimum vertical separation in feet
        # self.min_vertical_separation = 1000
        
        model = pyo.ConcreteModel()
        
        model.aircraft = pyo.RangeSet(0, len(aircrafts))
        model.t = pyo.RangeSet(0, prediction_horizon)

        # control inputs. Assign a value for each aircraft, at each time in the prediction horizon
        # constrain acceleration to be between -3 and 0
        # meaning that the aircraft's speed can decrease by at most 3 knots in one second
        # and it cannot increase because it should not accelerate as it approaches the airport
        model.acceleration = pyo.Var(model.aircraft, model.t, bounds=(-3.0, 0.0))
        # constrain angular velocity to be between -0.5 and 0.5 degrees per second
        # so in one second, the aircraft's heading can change by at most 0.5 degrees per second
        model.angular_vel = pyo.Var(model.aircraft, model.t, bounds=(-0.5, 0.5))
        # constrain z_speed (rate of change of altitude) to be between -20 and 0 feet per second
        # so in one second, the aircraft's altitude can decrease by at most 20 feet
        # it should not be positive since the aircraft should not be increasing altitude as it is about to land
        model.z_speed = pyo.Var(model.aircraft, model.t, bounds=(-20.0, 0))
        # initialise one more control input which sets whether to make the aircraft hold
        # model.force_hold = pyo.Var(model.aircraft, model.t, within=pyo.Binary)

        # initialise states, which are affected by the control inputs
        # each state will have a value for each aircraft at each timestep
        # constrain longitude to be between +/- 180
        model.lon = pyo.Var(model.aircraft, model.t, bounds=(-180, 180))
        # constrain latitutde to be between +/- 90
        model.lat = pyo.Var(model.aircraft, model.t, bounds=(-90, 90))
        # we need another state to keep track of the aircraft's speed at current time instant
        # constrain speed to be between 200 knots and 280 knots
        model.speed = pyo.Var(model.aircraft, model.t, bounds=(0, 280))
        # constrain heading to be between 0 and 360
        model.heading = pyo.Var(model.aircraft, model.t, bounds=(-360, 720))
        # constrain altitude to be between 4000 and 30
        model.altitude = pyo.Var(model.aircraft, model.t, bounds=(4000, 30000))
        # a state to keep track of the progress of the hold, so once it is more than 1, it has completed the hold
        # model.hold_progress = pyo.Var(model.aircraft, model.t, bounds=(0, 2))
        model.goal_lon = pyo.Param(initialize=LHR_COORDS.long)
        model.goal_lat = pyo.Param(initialize=LHR_COORDS.lat)
        model.z = pyo.Var(model.aircraft, model.aircraft, model.t, within=pyo.Binary)
        self.m = 9999999

        model.acceleration_init = pyo.ConstraintList()
        model.z_speed_init = pyo.ConstraintList()
        model.lon_init = pyo.ConstraintList()
        model.lat_init = pyo.ConstraintList()
        model.speed_init = pyo.ConstraintList()
        model.heading_init = pyo.ConstraintList()
        model.altitude_init = pyo.ConstraintList()

        model.min_horizontal_separation = 25000
        model.min_vertical_separation = 1000

        # TODO: worry about it later
        for i, aircraft in enumerate(aircrafts):
            model.acceleration_init.add(expr=model.acceleration[i, 0] == 0)
            model.z_speed_init.add(expr=model.z_speed[i, 0] == 0)
            # model.force_hold_init = pyo.Constraint(expr=model.z_speed[i, 0] == aircraft.is_in_hold)
            model.lon_init.add(expr=model.lon[i, 0] == aircraft.coordinates.long)
            model.lat_init.add(expr=model.lat[i, 0] == aircraft.coordinates.lat)
            model.speed_init.add(expr=model.speed[i, 0] == aircraft.velocity.speed)
            model.heading_init.add(expr=model.heading[i, 0] == aircraft.velocity.heading)
            model.altitude_init.add(expr=model.altitude[i, 0] == aircraft.coordinates.altitude)
        
        self.model = model
        
    def lon_updates(self, model, i, t):
        if t == 0:
            return pyo.Constraint.Skip
        # x = pyo.value(model.force_hold[i, t-1])
        if 1:
            # return model.lon[i, t] == model.lon[i, t-1] + dt * ((1 - model.force_hold[i, t-1]) * model.speed[i, t-1] * pyo.cos(model.heading[i, t-1]))
            return model.lon[i, t] == model.lon[i, t-1] + self.dt * ((model.speed[i, t-1] / 1.944) / 111139) * pyo.cos(model.heading[i, t-1])
        # else:
            # point_on_circle = lon_points[i].pop(0)
            # return (model.lon[i, t] == point_on_circle[0])  
        


    def lat_updates(self, model, i, t):
        """
        :param i: index of aircraft
        :param t: index of time
        """
        if t == 0:
            return pyo.Constraint.Skip
        # if we are not currently in a hold, update lon and lat based on the x and y velocities
        # divide speed by 1.944 to convert it into metres per second
        # then divide it by a further 111139 to convert it into degrees per second
        # if not pyo.value(model.force_hold[i, t-1]):
        if 1 + 1 == 2:
            return model.lat[i, t] == model.lat[i, t-1] + self.dt * ((model.speed[i, t-1] / 1.944) / 111139) * pyo.sin(model.heading[i, t-1])
        # else if the aircraft is currently in a hold, update lon and lat based on movement in a circle
        # else:
        #     point_on_circle = lat_points[i].pop(0)
        #     return model.lat[i, t] == point_on_circle
        

    def alt_updates(self, model, i, t):
        """
        :param i: index of aircraft
        :param t: index of time
        """
        if t == 0:
            return pyo.Constraint.Skip
        # if we are not currently in a hold, update lon and lat based on the x and y velocities
        # divide speed by 1.944 to convert it into metres per second
        # then divide it by a further 111139 to convert it into degrees per second
        # if not pyo.value(model.force_hold[i, t-1]):
        if 2 + 2 == 4:
            return model.heading[i, t] == (model.heading[i, t-1] + self.dt * model.angular_vel[i, t-1])
        # else if the aircraft is currently in a hold, update lon and lat based on movement in a circle
        # else:
        #     point_on_circle = alt_points[i].pop(0)
        #     return model.heading[i, t] == point_on_circle[2]
        
        
    def speed_updates(self, model, i, t):
        if t == 0:
            return pyo.Constraint.Skip
        # if not in a hold, use v = u + at to update speed
        # if pyo.value(model.force_hold[i, t-1]):
        if 3 + 3 == 6:
            return (model.speed[i, t] == model.speed[i, t-1] + self.dt * model.acceleration[i, t-1])
        # else if in a hold, maintain constant speed
        # else:
            # return (model.speed[i, t] == model.speed[i, t-1])
        

    def altitude_update(self, model, i, t):
        if t == 0:
            return pyo.Constraint.Skip
        # if not in a hold, update based on z_speed
        # if not model.force_hold[i, t-1]:
        if 1 + 4 == 5:
            return (model.altitude[i, t] == model.altitude[i, t-1] + self.dt * model.z_speed[i, t-1])
        # else if in a hold, maintain altitude
        # else:
            # return (model.altitude[i, t] == model.altitude[i, t-1])
        
        

        # def hold_progress_update(model, i, t):
        #     if t == 0:
        #         return pyo.Constraint.Skip
        #     # if not in a hold, then this is a dont-care, keep it at 0
        #     if not model.force_hold[i, t-1]:
        #         return (model.hold_progress == 0)
        #     # else if it is in a hold
        #     else:
        #         # calculate angular velocity as linear velocity divided by the radius of the hold
                
        #         model.hold_progress += (((model.speed[i, t-1] / 1.944) / holding_radius) * dt) / (2 * 3.14159)
        
        # model.hold_progress_update = pyo.Constraint(model.aircraft, model.t, rule=hold_progress_update)

    def ifelse(self, model, i, j, t):
        return (model.min_horizontal_separation / 111139) ** 2  - ((model.lon[i, t] - model.lon[j, t]) ** 2 + (model.lat[i, t] - model.lat[j, t]) ** 2) <= self.m * model.z[i, j, t]

    def haversine2(model, i, j, t):
        R = 6371  # Earth's radius in km (use 3440 for nautical miles)
        
        lat1, lon1 = model.lat[i, t], model.lon[i, t]
        lat2, lon2 = model.lat[j, t], model.lon[j, t]

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Pyomo expressions for approximating trigonometric functions
        a = (pyo.sin(dlat / 2) ** 2) + pyo.cos(lat1) * pyo.cos(lat2) * (pyo.sin(dlon / 2) ** 2)
        c = 2 * pyo.atan(pyo.sqrt(a)/ pyo.sqrt(1 - a))

        return R * c  # Distance in km

    # def separation_rules(self, model, i, j, t):
    #     if i < j:
    #         # horizontal_dist = haversine2(model, i, j, t)
    #         horizontal_dist = (model.lon[i, t] - model.lon[j, t]) ** 2 + (model.lat[i, t] - model.lat[j, t]) ** 2
    #         vertical_dist = abs(model.altitude[i, t] - model.altitude[j, t])
    #         # abs_dist = ((model.lon[i, t] - model.lon[j, t]) ** 2 + (model.lat[i, t] - model.lat[j, t]) ** 2) ** 2 + abs(model.altitude[i, t] - model.altitude[j, t]) ** 2
    #         # c1 = horizontal_dist >= min_horizontal_separation
    #         # c2 = vertical_dist >= min_vertical_separation
    #         # return pyo.inequality(c1, True) or pyo.inequality(c2, True)
    #         # return (model.lon[i, t] - model.lon[j, t]) ** 2 + (model.lat[i, t] - model.lat[j, t]) ** 2 >= min_horizontal_separation or abs(model.altitude[i, t] - model.altitude[j, t]) >= min_vertical_separation
    #         # fucking_shit_1 = pyo.inequality(min_horizontal_separation, horizontal_dist)
    #         # fucking_shit_2 = pyo.inequality(min_vertical_separation, vertical_dist)
    #         # return fucking_shit_1 fucking_shit_
    #         return (horizontal_dist >= self.model.min_horizontal_separation).lor(vertical_dist >= self.model.min_vertical_separation)
    #         # return (horizontal_dist ** 2 + vertical_dist ** 2) < (min_horizontal_separation ** 2 + min_vertical_separation ** 2)
    #     return pyo.Constraint.Skip
        

    # def hold_control_rule(self, model, i, t):
    #     # if we weren't enforcing a hold in previous time step, ignore it now
    #     if t == 1 or 5 + 6 == 11:
    #         return pyo.Constraint.Skip
    #     # else if previously we had enforced a hold
    #     else:
    #         if len(self.lon_points[i]) == 0:
    #             return pyo.Constraint.Skip
    #         else:
    #             return (model.force_hold[i, t] == 1)

        
        
    def acc_rules(self, model, i, t):
        if t == 0:
            return pyo.Constraint.Skip
        # if not holding, no constraint
        # if not pyo.value(model.force_hold[i, t]) or 1:
        if 1:
            return pyo.Constraint.Skip
        # else if holding, ensure constraints set
        # else:
            # return model.acceleration[i, t] == model.acceleration[i, t-1]
        
            
    def ang_vel_rules(self, model, i, t):
        if t == 0:
            return pyo.Constraint.Skip
        # if not holding, no constraint
        # if not pyo.value(model.force_hold[i, t]):
        if 1:
            return pyo.Constraint.Skip
        # else if holding, ensure constraints set
        # else:
            # return model.angular_vel[i, t] == model.angular_vel[i, t-1]
        

    def z_speed_rules(self, model, i, t):
        if t == 0:
            return pyo.Constraint.Skip
        # if not holding, no constraint
        # if not pyo.value(model.force_hold[i, t]):
        if 1:
            return pyo.Constraint.Skip
        # else if holding, ensure constraints set
        # else:
            # return model.z_speed[i, t] == model.z_speed[i, t-1]
            

    def objective_function(self, model):
        cost = 0
        # for t in model.t:
        #     for i in model.aircraft:
        #         for j in model.aircraft:
        #             if i < j:
        #                 if (model.lon[i, t] - model.lon[j, t]) ** 2 + (model.lat[i, t] - model.lat[j, t]) ** 2:
        #                     cost += 999999
        #         c1 = ((model.lon[i, t] - model.goal_lon) ** 2 + (model.lat[i, t] - model.goal_lat) ** 2) / 50000
        #         # c2 = 0.2 * abs(model.speed[i, t] - 50) / 80
        #         # c3 = abs(model.altitude[i, t] - 4000) / 26000
        #         cost += (c1)
        #         # lon, lat = model.lon[i, t], model.lat[i, t]
        #         # # distance to dest in metres
        #         # dist = haversine(Coordinate(lon, lat, model.altitude[i, t]), LHR_COORDS) / 50000
        #         # delta_speed = abs(model.speed[i, t] - 200) / 80
        #         # delta_alt = abs(model.altitude[i, t] - 4000) / 26000
        #         # cost += (dist + delta_speed + delta_alt) * i
        return sum(
            model.z[i, j, t] * 999999 if i < j else 0
            for t in model.t for i in model.aircraft for j in model.aircraft) +\
            sum(
                (model.lon[i, t] - model.goal_lon) ** 2 + (model.lat[i, t] - model.goal_lat) ** 2 * ((i + 1) ** 2)
                for t in model.t for i in model.aircraft
            )
        return cost
      
        
    # def trigger_circular_hold(self, lon, lat, speed, heading, i):
    #     # list of lon, lat, heading on the entire circle
    #     list_points = hold(lon, lat, heading, speed)
    #     self.lon_points[i] = list(map(list_points, lambda x: x[0]))
    #     self.lat_points[i] = list(map(list_points, lambda x: x[1]))
    #     self.alt_points[i] = list(map(list_points, lambda x: x[2]))


        # def trigger_hold_rule(model, i, t):
        #     if pyo.value(model.force_hold[i, t]) == 1:
        #         trigger_circular_hold(
        #             pyo.value(model.lon[i, t]),
        #             pyo.value(model.lat[i, t]),
        #             pyo.value(model.speed[i, t]),
        #             pyo.value(model.heading[i, t]),
        #             i)
        #     return pyo.Constraint.Skip
        
        # model.trigger_hold_rule = pyo.Constraint(model.aircraft, model.t, rule=trigger_hold_rule)
    
        
    def execute(self):

        self.model.lon_updates = pyo.Constraint(self.model.aircraft, self.model.t, rule=self.lon_updates)
        self.model.lat_updates = pyo.Constraint(self.model.aircraft, self.model.t, rule=self.lat_updates)
        self.model.alt_updates = pyo.Constraint(self.model.aircraft, self.model.t, rule=self.alt_updates)
        self.model.speed_updates = pyo.Constraint(self.model.aircraft, self.model.t, rule=self.speed_updates)
        self.model.altitude_updates = pyo.Constraint(self.model.aircraft, self.model.t, rule=self.altitude_update)
        # self.model.separation_rules = pyo.Constraint(self.model.aircraft, self.model.aircraft, self.model.t, rule=self.separation_rules)
        # self.model.hold_control_rule = pyo.Constraint(self.model.aircraft, self.model.t, rule=self.hold_control_rule)
        self.model.ifelse = pyo.Constraint(self.model.aircraft, self.model.aircraft, self.model.t, rule=self.ifelse)
        self.model.acc_rules = pyo.Constraint(self.model.aircraft, self.model.t, rule=self.acc_rules)
        self.model.ang_vel_rules = pyo.Constraint(self.model.aircraft, self.model.t, rule=self.ang_vel_rules)
        self.model.z_speed_rules = pyo.Constraint(self.model.aircraft, self.model.t, rule=self.z_speed_rules)
        self.model.obj = pyo.Objective(rule=self.objective_function, sense=pyo.minimize)

        # Running the pyomo solver
        results = pyo.SolverFactory('ipopt').solve(self.model)
        results.write()
        
        print(self.model.lon)
        print(self.model.lat)

        lons = [self.model.lon[1, t]() for t in self.model.t]
        lats = [self.model.lat[1, t]() for t in self.model.t]
        lons2 = [self.model.lon[0, t]() for t in self.model.t]
        lats2 = [self.model.lat[0, t]() for t in self.model.t]
        headings = [self.model.heading[1, t]() for t in self.model.t]
        print(lons)
        print(lats)
        print(lons2)
        print(lats2)
        print([self.model.z[0, 0, t]() for t in self.model.t])
        # import shapely
        # import geopandas as gpd
        # gdf = {"fuck": [shapely.geometry.Point([lons[i], lats[i]]) for i in range(len(lons))]}
        # print(gdf)
        # gdf = gpd.GeoDataFrame(gdf, geometry="fuck", crs="EPSG:4326")
        # print(gdf)
        # import os
        # gdf.to_file(os.path.join(os.getcwd(), "test.json"), driver='GeoJSON')

        return self.model
    

a1 = Airplane(1,
              Coordinate(-0.1523287, 51.7110011, 15000),
              Velocity(220, 260))
# a2 = Airplane(2,
#               Coordinate(-0.0956895, 51.3769529, 16000),
#               Velocity(120, 260),
#               )
a2 = Airplane(2,
              Coordinate(-0.114430, 51.664308, 16000),
              Velocity(120, 260),
              )
airplanes = [a1, a2]

opt = Optimizer(
    prediction_horizon=60,
    aircrafts=airplanes,
    dt=10
)

opt.execute()