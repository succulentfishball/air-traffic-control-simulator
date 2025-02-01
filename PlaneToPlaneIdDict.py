from GameState import Airplane;
from typing import Dict, Int;

class PlaneToPlaneIdDict:
    def __init__(self):
        self.planeToPlaneIdDict: Dict[Int, Airplane] = {}
    
    def addPlane(self, plane: Airplane):
        if plane.plane_id in self.planeToPlaneIdDict:
            raise ValueError("Plane with id " + str(plane.plane_id) + " already exists in the dictionary")
        else: 
            self.planeToPlaneIdDict[plane.id] = plane
    
    def get_airplane(self, planeId: Int) -> Airplane:
        return self.planeToPlaneIdDict[planeId]