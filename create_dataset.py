from wkt_processing import (
    get_drawn_wkt,
    get_wkt_extremes,
    get_hash,
    cut_into_squares,
)

from utils import get_fires

import os
import shutil

import threading


def create_dataset(
    username: str,
    password: str,
):
    fires = get_fires()

    # Create finalized dataset folder if it doesn't exist
    # The structure of this folder is:
    # finalized_dataset/data_point_N/[before.tiff, after.tiff]
    os.makedirs("finalized_dataset", exist_ok=True)

    data_point_counter = 0

    threads = []

    for name, days in fires.items():
        if len(days) < 2:
            continue

        # threads.append(
        #    threading.Thread(
        #        target=process_fire,
        #        args=(name, days, data_point_counter, username, password),
        #    )
        # )

        process_fire(name, days, data_point_counter, username, password)

        data_point_counter += len(days) - 1

    for thread in threads:
        thread.start()
        thread.join()


from datetime import datetime
from utils import date_format_str


def process_fire(
    name: str, days: list[str], data_point_counter: int, username: str, password: str
):
    print(f"Processing fire {name} with {len(days)} days of data")

    for i in range(len(days) - 1):
        day_one = days[i]
        day_two = days[i + 1]
        print(f"  Processing days {day_one} and {day_two}")

        # In this case, both names are the file path of the data folder.

        # --- Step 1: Acquire WKT Data ---
        day_one_wkts = get_wkt(day_one)
        day_two_wkts = get_wkt(day_two)

        # --- Step 2: Download and Geocode Data ---
        # Check if day_one and day_two have no GeoTIFF files
        # Get the Hashes for each day
        day_one_path = f"tmp/{get_hash(day_one_wkts)}"
        day_two_path = f"tmp/{get_hash(day_two_wkts)}"

        extremes = get_wkt_extremes(
            [point for wkt in day_one_wkts + day_two_wkts for point in wkt]
        )

        if datetime.strptime(day_one.split("/")[-1], date_format_str) < datetime(
            2014, 6, 14
        ):
            return

        # try:
        get_drawn_wkt(
            username=username,
            password=password,
            date_str=day_one.split("/")[-1],
            wkt_list=day_one_wkts,
            extremes=extremes,
            file_path=day_one_path,
        )

        get_drawn_wkt(
            username=username,
            password=password,
            date_str=day_two.split("/")[-1],
            wkt_list=day_two_wkts,
            extremes=extremes,
            file_path=day_two_path,
        )

        # --- Step 3: Move data to finalized_dataset ---
        os.makedirs(f"finalized_dataset/data_point_{data_point_counter}", exist_ok=True)

        shutil.copyfile(
            f"{day_one_path}/data/merged_wkt.tiff",
            f"finalized_dataset/data_point_{data_point_counter}/before.tiff",
        )
        shutil.copyfile(
            f"{day_two_path}/data/merged_wkt.tiff",
            f"finalized_dataset/data_point_{data_point_counter}/after.tiff",
        )
        # except Exception as e:
        #    print(f"    Error processing days {day_one} and {day_two}: {e}")
        #    continue

        # Iterate counter
        data_point_counter += 1


def get_wkt(path) -> list[list[tuple[float, float]]]:
    wkt_path = [file for file in os.listdir(path) if ".wkt" in file][0]
    wkt_strs = open(os.path.join(path, wkt_path), "r").read().strip()

    data = []

    for wkt in wkt_strs.splitlines():
        wkt_data = []

        # Strip POLYGON (( ))
        wkt = (
            wkt.strip()
            .replace("POLYGON ((", "")
            .lstrip(" ")
            .rstrip(" ")
            .replace("))", "")
        )

        for point in wkt.split(", "):
            lon, lat = point.strip().split(" ")
            wkt_data.append((float(lon), float(lat)))

        data.append(wkt_data)

    return data
