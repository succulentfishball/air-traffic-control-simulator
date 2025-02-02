from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Tuple
import math
from keplergl import KeplerGl
import pandas as pd
from Airplane import Airplane   
from xgboost.to_commit import train_xgboost


app = FastAPI()


@app.post("/process")
async def main(filename: str):
    # TODO: Run the main optimizer function over the Trajectory data.

    df = train_xgboost.main([filename])
    
    map_1 = KeplerGl()
    map_1.add_data(data=df, name="Trajectory Data")
    map_1.save_to_html(file_name="atc_visualisation.html")
    return RedirectResponse(url=f"http://localhost:8000/atc_visualisation.html", status_code=302)