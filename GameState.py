import math
import pyomo.environ as pyo 
from typing import Optional
from ConflictZones import ConflictZone
from PlaneToPlaneIdDict import PlaneToPlaneIdDict

crash_threshold = 10
interval_length = 5

# Airport entry thresholds
x_threshold = 5
y_threshold = 5
z_threshold = 5

# Holding 
hold_radius = 5
circumference = 2 * math.pi * hold_radius

# Initialising Plane to PlaneID
planeDict: PlaneToPlaneIdDict = PlaneToPlaneIdDict()

class Coordinate:
    def __init__(self, lat, long, altitude) -> None:
        self.long = long
        self.lat = lat
        self.altitude = altitude

entry_point = Coordinate(0, 0, 200)


class Velocity: 
    def __init__(self, heading: int, speed) -> None:
        self.heading = heading # 0 is North
        self.speed = speed

class Action:
    def __init__(self, angular_velocity, horizontal_acceleration, vertical_velocity) -> None:
        self.angular_velocity = angular_velocity
        self.horizontal_acceleration = horizontal_acceleration
        self.vertical_velocity = vertical_velocity

class Airplane:
    def __init__(self, plane_id: int, coordinates: Coordinate, velocity: int, conflictZone: Optional[ConflictZone]) -> None:
        self.plane_id = plane_id
        self.coordinates = coordinates
        self.velocity = velocity
        self.is_in_hold = False  # whether aircraft is currently in the midst of holding pattern
        self.hold_progress = 0   # progress of the hold
        self.conflictZone = None
        planeDict.add_plane(plane_id, self)

    @property
    def coordinates(self):
        return self.coordinates
    
    @property
    def plane_id(self):
        return self.plane_id    
    
    @property
    def velocity(self):
        return self.velocity
    
    @property
    def is_in_hold(self):
        return self.is_in_hold
    
    @property
    def hold_progress(self):
        return self.hold_progress
    
    @property
    def conflictZone(self):
        return self.conflictZone

    def update_state(self, action):
        u = self.velocity.speed
        h = self.velocity.heading

        # match action:
        #     case Action.DECELERATE:
        #         self.velocity.speed -= 5
        #     case Action.HOLD:
        #         self.is_in_hold = True
        #         # TODO: logic for holding pattern & duration
        #     case Action.STOP_HOLD:
        #         self.is_in_hold = False
        #     case Action.RIGHT:
        #         self.velocity.direction += 3
        #     case Action.LEFT:
        #         self.velocity.direction -= 3
        #     case Action.DOWN:
        #         self.coordinates.altitude -= 5

        # # Calculating horizontal plane displacement, using average speed and heading
        # self.coordinates.lat += ((u + self.velocity.speed)/ 2) * math.cos((self.velocity.heading + h)/ 2)
        # self.coordinates.long += ((u + self.velocity.speed)/ 2) * math.sin((self.velocity.heading + h)/ 2)
        self.velocity.speed += action.horizontal_acceleration * interval_length
        self.velocity.heading += action.angular_velocity * interval_length
        self.velocity.heading %= 2 * math.pi

        self.coordinates.lat += ((u + self.velocity.speed)/ 2) * math.cos((self.velocity.heading + h)/ 2) * interval_length
        self.coordinates.long += ((u + self.velocity.speed)/ 2) * math.sin((self.velocity.heading + h)/ 2) * interval_length
        self.coordinates.altitude += action.vertical_velocity * interval_length

    
    def hold(self, airplane):
        w = (airplane.speed / 3600) * hold_radius # rad/s, assuming airplane.speed is nm/h and hold_radius is nm
        required_intervals = ((2 * math.pi) / w) / interval_length
        # hold is uninterrupted
        bearing_change_per_step = 360 / required_intervals
        arc_per_interval = circumference / required_intervals 
        angular_displacement_per_interval = arc_per_interval / hold_radius
        extra_incomplete_interval = math.ceil(required_intervals) - required_intervals
        hold_angular_displacements = []
        for i in range(1, floor(required_intervals) + 1):
            hold_angular_displacements.append(i * angular_displacement_per_interval)
        # at each step, the aangular displacement from the starting point of the hold 
            
        hold_steps = []

       
        
    def straight_path_bearing(self, airport):
        theta = math.atan((airport.coordinates.x - self.coordinates.x) / (airport.coordinates.y - self.coordinates.y))
        return 180 - theta
    # define a flag for holding, and percentage of holding completion

    def through_entry_point(self):
        return self.coordinates.lat == entry_point.lat and self.coordinates.long == entry_point.long and self.coordinates.altitude <= entry_point.altitude

class Gate:
    def __init__(self, radius, max_speed, max_altitude):
        self.radius = radius
        self.max_speed = max_speed
        self.max_altitude = max_altitude

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

    def add_airplane(self, airplane):
        self.airplanes.append(airplane)
    
    def remove_airplane(self, airplane):
        self.airplanes.remove(airplane)
    
    def will_collide(self, airplane1, airplane2):
        horizontal_distance = math.sqrt((airplane1.coordinates.long - airplane2.coordinates.long) ** 2 + (airplane1.coordinates.lat - airplane2.coordinates.lat) ** 2)
        vertical_distance = abs(airplane1.coordinates.altitude - airplane2.coordinates.altitude)
        return horizontal_distance < crash_threshold and vertical_distance < crash_threshold
    
    # def reaches_airport(self, airplane, airport):
    #     x_distance = airplane.coordinates.x - airport.coordinates.x
    #     y_distance = airplane.coordinates.y - airport.coordinates.y
    #     z_distance = airplane.coordinates.z - airport.coordinates.z
    #     return (x_distance < x_threshold and y_distance < y_threshold and z_distance < z_threshold)
    
    def iter_state(self):
        self.airplanes.filter()
        # for each airplane, update state
        self.airplanes.forEach(lambda airplane: airplane.update_state())
        # for each airplane, check if it has gone through entry point
        self.airplanes.filter(lambda airplane: not airplane.through_entry_point())


class Optimizer():
    
    def __init__(self, num_aircraft: int, prediction_horizon: int):
        model = pyo.ConcreteModel()
        
        model.aircraft = pyo.RangeSet(1, num_aircraft)
        model.t = pyo.RangeSet(1, prediction_horizon)

        # control inputs. Assign a value for each aircraft, at each time in the prediction horizon
        # constrain speed to be between 200 and 280 knots
        model.speed = pyo.Var(model.aircraft, model.t, within=pyo.Reals, bounds=(200, 280))
        # constrrain angular velocity to be 
        model.heading = pyo.Var(model.aircraft, model.t, within=pyo.Reals, bounds=())