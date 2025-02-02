from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
import math
from keplergl import KeplerGl
import pandas as pd
from Airplane import Airplane

app = FastAPI()

class TrajectoryData(BaseModel):
    # plane trajectory information and entry time
    # Assume planes appear at the border of the considered zone
    airplanes: List[Tuple[Airplane, float]]

@app.post("/main")
async def main(airplanes: TrajectoryData):
    # TODO: Run the main optimizer function over the Trajectory data.

    df = pd.DataFrame(data)
    map_1 = KeplerGl(height=600)
    map_1.add_data(data=df, name="Trajectory Data")
    map_1.save_to_html(file_name="atc_visualisation.html")
    return