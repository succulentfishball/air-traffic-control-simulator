import math
# from ConflictZones import ConflictZone
# from PlaneToPlaneIdDict import PlaneToPlaneIdDict
# from Airplane import crash_threshold, hold, Airplane

# Initialising Plane to PlaneID
# planeDict: PlaneToPlaneIdDict = PlaneToPlaneIdDict()

class Coordinate:
    def __init__(self, long, lat, altitude) -> None:
        self.long = long
        self.lat = lat
        self.altitude = altitude
        
    # override '+' operator
    def __add__(self, other):
        return Coordinate(self.lat + other.lat, self.long + other.long, self.altitude + other.altitude)

    # override '/' operator
    def __truediv__(self, divisor: int):
        return Coordinate(self.lat / divisor, self.long / divisor, self.altitude / divisor)


LHR_COORDS = Coordinate(-0.46194167, 51.4706, 200)


# haversine formula to calculate distance between two coordinates in metres
def haversine(coord1: Coordinate, coord2: Coordinate):
   R = 6371e3  # Earth's radius in metres
  
   # Convert latitude and longitude to radians (precompute to avoid redundant calls)
   phi1, phi2 = math.radians(coord1.lat), math.radians(coord2.lat)
   delta_phi = math.radians(coord2.lat - coord1.lat)
   delta_lambda = math.radians(coord2.long - coord1.long)
  
   # Compute haversine formula
   sin_dphi2 = math.sin(delta_phi / 2)
   sin_dlambda2 = math.sin(delta_lambda / 2)
  
   a = sin_dphi2 * sin_dphi2 + math.cos(phi1) * math.cos(phi2) * sin_dlambda2 * sin_dlambda2
   return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class Velocity: 
    def __init__(self, heading: int, speed) -> None:
        self.heading = heading # 0 is East, Anticlockwise
        self.speed = speed

class Action:
    def __init__(self, angular_velocity, horizontal_acceleration, vertical_velocity) -> None:
        self.angular_velocity = angular_velocity
        self.horizontal_acceleration = horizontal_acceleration
        self.vertical_velocity = vertical_velocity

class Gate:
    def __init__(self, radius, max_speed, max_altitude):
        self.radius = radius
        self.max_speed = max_speed
        self.max_altitude = max_altitude