# ---- API Imports ----
from fastapi import FastAPI
from pydantic import BaseModel
import os

# ---- Function Imports ----
# This is needed as matplotlib by default tries to write to a config directory
# and Docker containers are mean and hate me :(
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib/"
from wkt_processing import process, get_wkt_extremes


app = FastAPI()


@app.get("/")
def root():
    """
    API root endpoint.
    """
    return {"Hello": "World"}


class ProcessWKTRequest(BaseModel):
    username: str
    password: str
    date_str: str
    wkt_points: list[list[tuple[float, float]]]


@app.post("process_wkt")
async def process_wkt(req: ProcessWKTRequest):
    original, results = process(
        req.username, req.password, req.wkt_points, req.date_str
    )

    return {"original": original, "results": results}


class WKTExtremes(BaseModel):
    wkt_points: list[list[tuple[float, float]]]


@app.post("/wkt_extremes")
async def wkt_extremes(data: WKTExtremes):
    """
    Get the extreme coordinates from multiple WKT polygons.
    """
    # Flatten the 2D list of WKT points and pass to the processing function
    return get_wkt_extremes([item for sublist in data.wkt_points for item in sublist])
