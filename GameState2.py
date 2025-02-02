import math
from typing import Optional
from Airplane import crash_threshold, hold, Airplane
import sys
from utils import Coordinate, haversine, Velocity, Action, Gate, LHR_COORDS
"""
A set partitioning model of a wedding seating problem
Adaptation where an initial solution is given to solvers: CPLEX_CMD, GUROBI_CMD, PULP_CBC_CMD

Authors: Stuart Mitchell 2009, Franco Peschiera 2019
"""

from pulp import *

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


def happiness(table):
    """
    Find the happiness of the table
    - by calculating the maximum distance between the letters
    """
    return abs(ord(table[0]) - ord(table[-1]))


# create list of all possible tables
possible_tables = [tuple(c) for c in pulp.allcombinations(guests, max_table_size)]

# create a binary variable to state that a table setting is used
x = pulp.LpVariable.dicts(
    "table", possible_tables, lowBound=0, upBound=1, cat=pulp.LpInteger
)

seating_model = pulp.LpProblem("Wedding Seating Model", pulp.LpMinimize)

seating_model += pulp.lpSum([happiness(table) * x[table] for table in possible_tables])

# specify the maximum number of tables
seating_model += (
    pulp.lpSum([x[table] for table in possible_tables]) <= max_tables,
    "Maximum_number_of_tables",
)

# A guest must seated at one and only one table
for guest in guests:
    seating_model += (
        pulp.lpSum([x[table] for table in possible_tables if guest in table]) == 1,
        f"Must_seat_{guest}",
    )

# I've taken the optimal solution from a previous solving. x is the variable dictionary.
solution = {
    ("M", "N"): 1.0,
    ("E", "F", "G"): 1.0,
    ("A", "B", "C", "D"): 1.0,
    ("I", "J", "K", "L"): 1.0,
    ("O", "P", "Q", "R"): 1.0,
}
for k, v in solution.items():
    x[k].setInitialValue(v)

solver = pulp.PULP_CBC_CMD(msg=True, warmStart=True)
# solver = pulp.CPLEX_CMD(msg=True, warmStart=True)
# solver = pulp.GUROBI_CMD(msg=True, warmStart=True)
# solver = pulp.CPLEX_PY(msg=True, warmStart=True)
# solver = pulp.GUROBI(msg=True, warmStart=True)
seating_model.solve(solver)


print(f"The chosen tables are out of a total of {len(possible_tables)}:")
for table in possible_tables:
    if x[table].value() == 1.0:
        print(table)