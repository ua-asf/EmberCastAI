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


class OriginalAndResults(BaseModel):
    original: list[int]
    results: list[int]
    dem: list[int]


@app.post("/process_wkt")
async def process_wkt(req: ProcessWKTRequest) -> OriginalAndResults:
    original, results, dem = process(
        req.username, req.password, req.wkt_points, req.date_str
    )

    return OriginalAndResults(original=original, results=results, dem=dem)


class WKTExtremes(BaseModel):
    wkt_points: list[list[tuple[float, float]]]


@app.post("/wkt_extremes")
async def wkt_extremes(data: WKTExtremes):
    """
    Get the extreme coordinates from multiple WKT polygons.
    """
    # Flatten the 2D list of WKT points and pass to the processing function
    return get_wkt_extremes([item for sublist in data.wkt_points for item in sublist])
