"""
tractor_wheel_polygons_FINAL_both_outputs_with_source.py
========================================================
Pure-Python tractor wheel polygon builder.

Writes BOTH:
1. GeoPackage (.gpkg)
2. Shapefile (.shp)

Also copies selected source attributes from the input GPKG rows
and adds a polygon source field to the output attribute table.

Install:
    pip install geopandas shapely tqdm fiona pyogrio pandas numpy

Run:
    python tractor_wheel_polygons_FINAL_both_outputs_with_source.py
"""

import glob
import math
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION – EDIT THIS BLOCK
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FOLDER       = r"C:\Users\imankarami\Data\GPS-points_of_tractors\INPUT\gross"
OUTPUT_FOLDER      = r"C:\Users\imankarami\Data\GPS-points_of_tractors\OUTPUT\gross_may_sep\1"

OUTPUT_BASENAME    = "3rd_tractor_may_sep_polygonsCTF"

# CHANGE THIS TEXT WHEN YOU CHANGE THE POLYGON TYPE / INPUT SET

POLYGON_SOURCE     = "3rd_tractor"

AUSSENBREITE       = 2.7   #GROSS 2.7   KLEIN 2.55 third 2.4
REIFENBREITE       = 0.7 #GROSS 0.7  KLEIN 0.65   third 0.6
OUTPUT_CRS         = 32633

COG_COL            = "CourseOverGround"
SPEED_COL          = "GroundBasedMachineSpeed"
DIRECTION_COL      = "GroundBasedMachineDirection"

MIN_SPEED_KMH      = 0.05

# optional: copy these input columns if they exist
OPTIONAL_INPUT_COLUMNS = [
    "PositionDate",
    "PositionTime",
    "WheelBasedMachineSpeed",
    "WheelBasedMachineDistance",
    "GroundBasedMachineSpeed",
    "GroundBasedMachineDirection",
    "RearHitchPosition",
    "MachineSelectedSpeed",
    "MachineSelectedDistance",
    "SpeedOverGround",
    "EngFuelRate",
    "EngSpeed",
    "SID",
]

# ─────────────────────────────────────────────────────────────────────────────


def rotate_point_cw(px, py, cx, cy, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = px - cx
    dy = py - cy
    rx = cx + cos_a * dx + sin_a * dy
    ry = cy - sin_a * dx + cos_a * dy
    return rx, ry


def compute_rotated_corners(x0, y0, cog_radians):
    li1x = x0 - (AUSSENBREITE / 2)
    li2x = li1x + REIFENBREITE
    re2x = x0 + (AUSSENBREITE / 2)
    re1x = re2x - REIFENBREITE

    angle_deg = cog_radians * 180.0 / math.pi

    p1 = rotate_point_cw(li1x, y0, x0, y0, angle_deg)  # left outer
    p2 = rotate_point_cw(li2x, y0, x0, y0, angle_deg)  # left inner
    p3 = rotate_point_cw(re1x, y0, x0, y0, angle_deg)  # right inner
    p4 = rotate_point_cw(re2x, y0, x0, y0, angle_deg)  # right outer

    return p1, p2, p3, p4


def cog_to_bearing_deg(cog_radians):
    deg = math.degrees(cog_radians) % 360.0
    return round(deg, 4)


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value):
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def safe_str(value):
    try:
        if pd.isna(value):
            return None
        return str(value)
    except Exception:
        return None


def parse_date_from_filename(filename):
    base = os.path.basename(str(filename))
    m = re.match(r"^(\d{2})(\d{2})(\d{2})_", base)
    if not m:
        return None, None

    yy, mm, dd = m.groups()
    iso_date = f"20{yy}-{mm}-{dd}"
    display_date = f"{dd}.{mm}.20{yy}"
    return iso_date, display_date


def choose_route_date(row):
    if "PositionDate" in row.index and pd.notna(row["PositionDate"]):
        try:
            dt = pd.to_datetime(row["PositionDate"], errors="coerce")
            if pd.notna(dt):
                return dt.strftime("%Y-%m-%d"), dt.strftime("%d.%m.%Y")
        except Exception:
            pass

    if "sourcefile" in row.index:
        return parse_date_from_filename(row["sourcefile"])

    return None, None


def load_all_gpkgs(input_folder):
    gpkg_files = sorted(glob.glob(os.path.join(input_folder, "*.gpkg")))
    if not gpkg_files:
        raise FileNotFoundError(f"No .gpkg files found in: {input_folder}")

    print(f"Loading {len(gpkg_files)} GPKG file(s)...")

    parts = []
    for f in gpkg_files:
        gdf = gpd.read_file(f)
        gdf["sourcefile"] = os.path.basename(f)
        parts.append(gdf)

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    print(f"Loaded {len(combined):,} total rows. CRS = {combined.crs}")
    return combined


def build_polygon_output(mow):
    mowscaled = mow.to_crs(epsg=OUTPUT_CRS)
    machinework = mowscaled.copy()

    machinework["li1x"] = machinework.geometry.x - (AUSSENBREITE / 2)
    machinework["li1y"] = machinework.geometry.y
    machinework["li2x"] = machinework["li1x"] + REIFENBREITE
    machinework["li2y"] = machinework.geometry.y
    machinework["re2x"] = machinework.geometry.x + (AUSSENBREITE / 2)
    machinework["re2y"] = machinework.geometry.y
    machinework["re1x"] = machinework["re2x"] - REIFENBREITE
    machinework["re1y"] = machinework.geometry.y

    first_row = machinework.iloc[0]
    old_points = compute_rotated_corners(
        first_row.geometry.x,
        first_row.geometry.y,
        first_row[COG_COL]
    )

    out_rows = []
    poly_count = 0

    for index, row in tqdm(machinework.iterrows(), total=len(machinework), desc="Building polygons"):
        x0 = row.geometry.x
        y0 = row.geometry.y
        cog = row[COG_COL]

        if pd.isna(cog) or pd.isna(x0) or pd.isna(y0):
            continue

        p1, p2, p3, p4 = compute_rotated_corners(x0, y0, cog)

        if any(math.isnan(v) for pt in (p1, p2, p3, p4) for v in pt):
            continue

        speed_ms = row[SPEED_COL]
        if pd.isna(speed_ms):
            old_points = (p1, p2, p3, p4)
            continue

        direction_deg = cog_to_bearing_deg(cog)
        cog_rad_val = round(float(cog), 6)
        speed_kmh = round(float(speed_ms) * 3.6, 3)
        mach_dir = safe_float(row[DIRECTION_COL]) if DIRECTION_COL in row.index else None
        route_date_iso, route_date_display = choose_route_date(row)

        base_attrs = {
            "PolySrc": POLYGON_SOURCE,   # <---- THIS IS THE NEW SOURCE FIELD
            "Wheel": None,
            "Direction": direction_deg,
            "COG_rad": cog_rad_val,
            "Speed_kmh": speed_kmh,
            "MachDir": mach_dir,
            "RouteDate": route_date_display,
            "RouteISO": route_date_iso,
            "PosDate": safe_str(row["PositionDate"]) if "PositionDate" in row.index else None,
            "PosTime": safe_str(row["PositionTime"]) if "PositionTime" in row.index else None,
            "SrcFile": safe_str(row["sourcefile"]) if "sourcefile" in row.index else None,
            "WhBSpd": safe_float(row["WheelBasedMachineSpeed"]) if "WheelBasedMachineSpeed" in row.index else None,
            "WhBDist": safe_float(row["WheelBasedMachineDistance"]) if "WheelBasedMachineDistance" in row.index else None,
            "GrBSpd": safe_float(row["GroundBasedMachineSpeed"]) if "GroundBasedMachineSpeed" in row.index else None,
            "EngFuel": safe_float(row["EngFuelRate"]) if "EngFuelRate" in row.index else None,
            "RearHitPos": safe_float(row["RearHitchPosition"]) if "RearHitchPosition" in row.index else None,
            "MacSelSpd": safe_float(row["MachineSelectedSpeed"]) if "MachineSelectedSpeed" in row.index else None,
            "MacSelDst": safe_float(row["MachineSelectedDistance"]) if "MachineSelectedDistance" in row.index else None,
            "SID": safe_int(row["SID"]) if "SID" in row.index else None,
            "CourseOG": safe_float(row["CourseOverGround"]) if "CourseOverGround" in row.index else None,
            "SpeedOG": safe_float(row["SpeedOverGround"]) if "SpeedOverGround" in row.index else None,
            "EngSpeed": safe_float(row["EngSpeed"]) if "EngSpeed" in row.index else None,
            "PtIndex": int(index),
        }

        if speed_ms * 3.6 >= MIN_SPEED_KMH:
            left_coords = [
                (p1[0], p1[1]),
                (p2[0], p2[1]),
                (old_points[1][0], old_points[1][1]),
                (old_points[0][0], old_points[0][1]),
                (p1[0], p1[1]),
            ]

            right_coords = [
                (p3[0], p3[1]),
                (p4[0], p4[1]),
                (old_points[3][0], old_points[3][1]),
                (old_points[2][0], old_points[2][1]),
                (p3[0], p3[1]),
            ]

            left_poly = Polygon(left_coords)
            right_poly = Polygon(right_coords)

            if not left_poly.is_valid:
                left_poly = left_poly.buffer(0)
            if not right_poly.is_valid:
                right_poly = right_poly.buffer(0)

            if not left_poly.is_empty and left_poly.area > 0:
                left_row = dict(base_attrs)
                left_row["Wheel"] = "left"
                left_row["geometry"] = left_poly
                out_rows.append(left_row)

            if not right_poly.is_empty and right_poly.area > 0:
                right_row = dict(base_attrs)
                right_row["Wheel"] = "right"
                right_row["geometry"] = right_poly
                out_rows.append(right_row)

            poly_count += 2

        old_points = (p1, p2, p3, p4)

    if not out_rows:
        raise RuntimeError("No polygons were generated.")

    polygons_gdf = gpd.GeoDataFrame(out_rows, crs=f"EPSG:{OUTPUT_CRS}")
    return polygons_gdf, poly_count


def save_outputs(polygons_gdf):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    gpkg_path = os.path.join(OUTPUT_FOLDER, OUTPUT_BASENAME + ".gpkg")
    shp_path = os.path.join(OUTPUT_FOLDER, OUTPUT_BASENAME + ".shp")

    polygons_gdf.to_file(gpkg_path, driver="GPKG")
    polygons_gdf.to_file(shp_path, driver="ESRI Shapefile")

    return gpkg_path, shp_path


def main():
    mow = load_all_gpkgs(INPUT_FOLDER)
    polygons_gdf, poly_count = build_polygon_output(mow)
    gpkg_path, shp_path = save_outputs(polygons_gdf)

    print()
    print(f"Done. {poly_count} polygons written.")
    print(f"GPKG output : {gpkg_path}")
    print(f"SHP output  : {shp_path}")
    print()
    print("Output fields:")
    print("PolySrc, Wheel, Direction, COG_rad, Speed_kmh, MachDir,")
    print("RouteDate, RouteISO, PosDate, PosTime, SrcFile,")
    print("WhBSpd, WhBDist, GrBSpd, EngFuel, RearHitPos,")
    print("MacSelSpd, MacSelDst, SID, CourseOG, SpeedOG, EngSpeed, PtIndex")


if __name__ == "__main__":
    main()