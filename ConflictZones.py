import math
from GameState import Coordinate, Airplane, enttr
from PlaneToPlaneIdDict import PlaneToPlaneIdDict

# put Optional[ConflictZone] as a variable inside of Airplane class later

class ConflictZone :
    next_id = 1

    def __init__(self, airplane1Id: int, airplane2Id: int):
        self.zone_id = ConflictZone.next_id
        ConflictZone.next_id += 1
        self.coordinatesOfCentre = (PlaneToPlaneIdDict[airplane1Id].coordinates + PlaneToPlaneIdDict[airplane2Id].coordinates) / 2
        self.radius = 10, # default radius
        self.airplane1Id= airplane1Id
        self.airplane2Id = airplane2Id
    
    @property
    def zone_id(self):
        return self.zone_id
            
    @zone_id.setter
    def zone_id(self, zone_id: int):
        if zone_id < 0:
            raise ValueError("Zone id must be a positive integer")
        self.zone_id = zone_id
    
    @property
    def coordinatesOfCentre(self):
        return self.coordinatesOfCentre
    
    @coordinates.setter
    def coordinatesOfCentre(self, coordinatesOfCentre: Coordinate):
        self.coordinatesOfCentre = coordinatesOfCentre

    @property
    def radius(self):
        return self.radius
    
    # no setter for radius as this should be a constant value. 

    @property
    def airplane1Id(self):
        return self.airplane1Id
    
    @airplane1Id.setter
    def airplane1Id(self, airplane1Id: int):
        self.airplane1Id = airplane1Id
    
    @property
    def airplane2Id(self):
        return self.airplane2Id
    
    @airplane2Id.setter 
    def airplane2Id(self, airplane2Id: Airplane):
        self.airplane2Id = airplane2Id
        




