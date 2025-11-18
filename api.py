# ---- API Imports ----
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
import asyncio
import json
import os

# ---- Function Imports ----
# This is needed as matplotlib by default tries to write to a config directory
# and Docker containers are mean and hate me :(
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib/"
from wkt_processing import process, get_wkt_extremes


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    opentopo_key: str | None = None


class OriginalAndResults(BaseModel):
    dims: tuple[int, int]
    original: list[int]
    results: list[int]
    dem: list[int]


@app.post("/process_wkt")
async def process_wkt(req: ProcessWKTRequest) -> OriginalAndResults:
    dims, original, results, dem = await process(
        req.username, req.password, req.wkt_points, req.date_str, req.opentopo_key
    )

    return OriginalAndResults(dims=dims, original=original, results=results, dem=dem)


@app.post("/process_wkt_streaming")
async def process_wkt_streaming(req: ProcessWKTRequest) -> StreamingResponse:
    return StreamingResponse(
        streaming_wkt(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


async def streaming_wkt(req: ProcessWKTRequest) -> AsyncGenerator[str, None]:
    queue: asyncio.Queue = asyncio.Queue()
    processing_done = asyncio.Event()
    result_data = None
    error_data = None

    async def worker():
        nonlocal result_data, error_data
        try:
            dims, original, results, dem = await process(
                req.username,
                req.password,
                req.wkt_points,
                req.date_str,
                req.opentopo_key,
                queue=queue,
            )

            result_data = {
                "dims": dims,
                "original": original,
                "results": results,
                "dem": dem,
            }
        except Exception as e:
            error_data = {"error": str(e)}
        finally:
            processing_done.set()

    # Start worker
    task = asyncio.create_task(worker())

    try:
        # Stream progress updates
        while not processing_done.is_set() or not queue.empty():
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield f"{msg}\n\n"
                await asyncio.sleep(0)  # Force flush
            except asyncio.TimeoutError:
                await asyncio.sleep(0.01)
                continue

        # Send final result
        if error_data:
            yield f"error:{json.dumps(error_data)}\n\n"
        elif result_data:
            yield f"image:{json.dumps(result_data)}\n\n"

        await asyncio.sleep(0)

    finally:
        print("Cleaning up worker task...")
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class WKTExtremes(BaseModel):
    wkt_points: list[list[tuple[float, float]]]


@app.post("/wkt_extremes")
async def wkt_extremes(data: WKTExtremes):
    """
    Get the extreme coordinates from multiple WKT polygons.
    """
    # Flatten the 2D list of WKT points and pass to the processing function
    return get_wkt_extremes([item for sublist in data.wkt_points for item in sublist])
