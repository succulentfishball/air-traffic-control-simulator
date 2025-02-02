import math
# from GameState import planeDict, entry_point
from utils import Coordinate, Velocity, LHR_COORDS
# from ConflictZones import ConflictZone
from typing import Optional

crash_threshold = 10
interval_length = 5

# Airport entry thresholds
x_threshold = 5
y_threshold = 5
z_threshold = 5

class Airplane:
    def __init__(self, plane_id: int, coordinates: Coordinate, velocity: Velocity) -> None:
        self.plane_id = plane_id
        self.coordinates = coordinates
        self.velocity = velocity
        self.is_in_hold = False  # whether aircraft is currently in the midst of holding pattern
        self.hold_progress = 0   # progress of the hold
        # self.conflictZone: Optional[ConflictZone] = None
        # planeDict.add_plane(plane_id, self)

    # @property
    # def coordinates(self):
    #     return self.coordinates
    
    # @property
    # def plane_id(self):
    #     return self.plane_id    
    
    # @property
    # def velocity(self):
    #     return self.velocity
    
    # @property
    # def is_in_hold(self):
    #     return self.is_in_hold
    
    # @property
    # def hold_progress(self):
    #     return self.hold_progress
    
    # @property
    # def conflictZone(self):
        # return self.conflictZone

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


        self.coordinates.lat += ((u + self.velocity.speed)/ 2) * math.sin((self.coordinates.heading + h)/ 2) * interval_length
        self.coordinates.long += ((u + self.velocity.speed)/ 2) * math.cos((self.coordinates.heading + h)/ 2) * interval_length
        self.coordinates.altitude += action.vertical_velocity * interval_length

    def straight_path_bearing(self, airport):
        theta = math.atan((airport.coordinates.x - self.coordinates.x) / (airport.coordinates.y - self.coordinates.y))
        return 180 - theta
    # define a flag for holding, and percentage of holding completion

    def through_entry_point(self):
        return abs(self.coordinates.lat - LHR_COORDS.lat) < 0.009 and abs(self.coordinates.long - LHR_COORDS.long) < 0.009
        

# Outputs the states of a plane during holding 
# List(long, lat, bearing)
def hold(start_long, start_lat, start_bearing, initial_speed):
    
    # holding circle parameters
    hold_radius = 0.269978 #500m in nautical miles
    circumference = 2 * math.pi * hold_radius

    # initial_speed input is nm/h (knots)
    initial_speed /= 3600
    initial_long_vel = initial_speed  * math.cos(start_bearing * math.pi / 180)
    initial_lat_vel = initial_speed * math.sin(start_bearing * math.pi / 180)

    # hold steps
    w = initial_speed * hold_radius # rad/s, initial_speed is nm/s and hold_radius is nm
    required_intervals = ((2 * math.pi) / w) / interval_length # interval length is s
    incomplete_interval = math.ceil(required_intervals) - required_intervals
    complete_intervals = math.floor(required_intervals)

    bearing_change_per_step = 360 / required_intervals
    angular_displacement_change_per_interval = (circumference / required_intervals) / hold_radius

    # at each step, the angular displacement from the starting point of the hold 
    hold_angular_displacements = []
    for i in range(1, complete_intervals + 1):
        hold_angular_displacements.append(i * angular_displacement_change_per_interval)

    # at each step, the coordinates and bearing of the plane
    hold_steps = []
    for i in range(0, len(hold_angular_displacements)):
        long_dis = -1 * hold_radius + hold_radius * math.cos(hold_angular_displacements[i])
        lat_dis = hold_radius * math.sin(hold_angular_displacements[i])
        bearing = (start_bearing - (bearing_change_per_step * (i + 1))) % 360
        item = (start_long + ((long_dis / 1.944) / 111139), start_lat + ((lat_dis / 1.944) / 111139), bearing)
        hold_steps.append(item)

    # add last step, which comprises the incomplete interval
    if (incomplete_interval > 0.01):    
        duration = incomplete_interval * interval_length
        item = (start_long + (((duration * initial_long_vel) / 1.944) / 111139), start_lat + (((duration * initial_lat_vel) / 1.944) / 111139), start_bearing)
        hold_steps.append(item)

    return hold_steps