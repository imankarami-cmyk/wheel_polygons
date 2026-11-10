"""
tractor_wheel_polygons_FINAL.py
================================
Pure-Python tractor wheel polygon builder.
Supports TWO tractor data formats automatically:

FORMAT A – "gross/3rd tractor" (original format)
  - Has: GroundBasedMachineSpeed, GroundBasedMachineDirection, CourseOverGround
  - CourseOverGround is stored in RADIANS (navigation/compass convention)
  - Rotation: angle_deg = COG_radians * 180/pi   → pass directly to CW rotate
  - Speed column: GroundBasedMachineSpeed (m/s)

FORMAT B – "klein/rudi" (this tractor, new format)
  - Has: Latitude, Longitude, Altitude, Date, Time, DateTime, TimeDiff
  - NO GroundBasedMachineSpeed / GroundBasedMachineDirection
  - NO pre-existing CourseOverGround (or it may be present as computed value)
  - CourseOverGround is COMPUTED from GPS positions using math atan2(dx, dy)
    which gives a MATH angle (from East-axis, CCW positive) in radians [-pi, pi]
  - Rotation: angle_deg = 90 - math.degrees(COG_math_rad)
    (converts math angle to compass/QGIS clockwise angle)
  - Speed is COMPUTED from GPS positions + DateTime differences (m/s)
  - Both COG and speed are smoothed with a rolling median/mean window

Writes BOTH:
  1. GeoPackage (.gpkg)
  2. Shapefile  (.shp)

Install:
    pip install geopandas shapely tqdm pandas numpy

Run:
    python tractor_wheel_polygons_FINAL.py
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

# =============================================================================
#  CONFIGURATION – EDIT THIS BLOCK
# =============================================================================

INPUT_FOLDER    = r"C:\Users\imankarami\Data\GPS-points_of_tractors\INPUT\3rd_tractor"
OUTPUT_FOLDER   = r"C:\Users\imankarami\Data\GPS-points_of_tractors\OUTPUT\3rd_tractor"
OUTPUT_BASENAME = "third_polygonsCTF"
POLYGON_SOURCE  = "third"           # label stored in PolySrc field

# ── Tractor dimensions ────────────────────────────────────────────────────────
# gross tractor:   AUSSENBREITE=2.7,  REIFENBREITE=0.700
# klein/rudi:      AUSSENBREITE=1.73+0.48=2.21,  REIFENBREITE=0.48
AUSSENBREITE = 2.21    # total outer width [m]
REIFENBREITE = 0.48    # tyre width        [m]

OUTPUT_CRS   = 32633   # EPSG projected CRS

# ── Speed filter ──────────────────────────────────────────────────────────────
MIN_SPEED_KMH = 0.05   # polygons only written when speed >= this value

# ── COG smoothing window (samples) ────────────────────────────────────────────
# Used only for FORMAT B (computed COG).  Odd number recommended.
SMOOTH_WIN = 7

# ── Max realistic speed cap (FORMAT B only) ───────────────────────────────────
MAX_SPEED_MPS = 15.0   # ~54 km/h; rows above this get speed set to NaN

# =============================================================================
#  DO NOT EDIT BELOW THIS LINE UNLESS YOU KNOW WHAT YOU ARE DOING
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
#  ROTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def rotate_point_cw(px, py, cx, cy, angle_deg):
    """
    Rotate (px,py) around (cx,cy) CLOCKWISE by angle_deg degrees.
    Exact equivalent of QgsGeometry.rotate(angle_deg, QgsPointXY(cx,cy)).

    CW rotation matrix:
        x' =  cos θ · dx + sin θ · dy
        y' = −sin θ · dx + cos θ · dy
    """
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    dx, dy = px - cx, py - cy
    return cx + cos_a * dx + sin_a * dy, cy - sin_a * dx + cos_a * dy


def compute_rotated_corners(x0, y0, angle_deg_cw):
    """
    Given GPS centre (x0,y0) and a CLOCKWISE angle in degrees,
    return the four rotated tyre-edge corner points.

    Unrotated layout (tractor moving up along Y):
        li1  li2 · · re1  re2
        |←tw→|  gap  |←tw→|
        |←── AUSSENBREITE ──→|

    Index:  0=li1(left outer)  1=li2(left inner)
            2=re1(right inner) 3=re2(right outer)
    """
    li1x = x0 - AUSSENBREITE / 2
    li2x = li1x + REIFENBREITE
    re2x = x0 + AUSSENBREITE / 2
    re1x = re2x - REIFENBREITE

    p1 = rotate_point_cw(li1x, y0, x0, y0, angle_deg_cw)
    p2 = rotate_point_cw(li2x, y0, x0, y0, angle_deg_cw)
    p3 = rotate_point_cw(re1x, y0, x0, y0, angle_deg_cw)
    p4 = rotate_point_cw(re2x, y0, x0, y0, angle_deg_cw)
    return p1, p2, p3, p4


# ─────────────────────────────────────────────────────────────────────────────
#  FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(gdf):
    """
    Returns 'A' (gross/original) or 'B' (klein/rudi computed-COG).

    FORMAT A has GroundBasedMachineSpeed in the columns.
    FORMAT B does not – it has raw GPS columns (DateTime, Latitude, etc.)
    """
    cols = set(gdf.columns)
    if "GroundBasedMachineSpeed" in cols:
        return "A"
    else:
        return "B"


# ─────────────────────────────────────────────────────────────────────────────
#  FORMAT B – COMPUTE COG AND SPEED FROM GPS POSITIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_cog_and_speed_format_b(machinework):
    """
    For FORMAT B (klein tractor):

    DIFFERENCE 1 – CourseOverGround is NOT in the data.
      It is computed from consecutive GPS positions using atan2(dx, dy).
      This gives a MATH angle:  0 = East, CCW positive, range [-pi, pi].

    DIFFERENCE 2 – Speed is NOT in the data.
      It is computed from distance between consecutive points divided by
      the time difference from the DateTime column.

    DIFFERENCE 3 – Both COG and speed are smoothed with a rolling window
      to reduce GPS noise.

    DIFFERENCE 4 – The QGIS rotation angle formula is different:
      FORMAT A:  angle_deg = COG_radians * 180/pi          (navigation COG)
      FORMAT B:  angle_deg = 90 - math.degrees(COG_math)   (math COG → compass CW)

    Returns the machinework DataFrame with new columns:
      gpsspeed_smoothed_mps       – smoothed GPS speed in m/s
      cog_angle_deg_cw            – clockwise rotation angle in degrees
                                    ready to pass to rotate_point_cw()
    """
    mw = machinework.copy()

    # Ensure DateTime is parsed
    mw["DateTime"] = pd.to_datetime(mw["DateTime"], utc=True, errors="coerce")

    mw["_x"] = mw.geometry.x
    mw["_y"] = mw.geometry.y

    # ── Forward differences ───────────────────────────────────────────────────
    dx_f = mw["_x"].shift(-1) - mw["_x"]
    dy_f = mw["_y"].shift(-1) - mw["_y"]
    dt_f = (mw["DateTime"].shift(-1) - mw["DateTime"]).dt.total_seconds()

    dist_f        = np.hypot(dx_f, dy_f)
    speed_fwd     = (dist_f / dt_f).where((dt_f > 0) & np.isfinite(dt_f), np.nan)

    # ── Centered differences ─────────────────────────────────────────────────
    dx_c = mw["_x"].shift(-1) - mw["_x"].shift(1)
    dy_c = mw["_y"].shift(-1) - mw["_y"].shift(1)
    dt_c = (mw["DateTime"].shift(-1) - mw["DateTime"].shift(1)).dt.total_seconds()

    dist_c        = np.hypot(dx_c, dy_c)
    speed_centered = (dist_c / dt_c).where((dt_c > 0) & np.isfinite(dt_c), np.nan)

    # Prefer centered, fall back to forward
    speed_raw = speed_centered.fillna(speed_fwd)

    # Smooth speed
    mw["gpsspeed_smoothed_mps"] = (
        speed_raw.rolling(SMOOTH_WIN, center=True, min_periods=1).median()
    )

    # Cap unrealistic speeds
    mw.loc[mw["gpsspeed_smoothed_mps"] > MAX_SPEED_MPS, "gpsspeed_smoothed_mps"] = np.nan

    # ── Course over ground (math convention) ─────────────────────────────────
    # Forward raw: atan2(dx, dy) → MATH angle from East-axis, CCW, [-pi, pi]
    cog_fwd      = np.arctan2(dy_f, dx_f)
    cog_centered = np.arctan2(dy_c, dx_c)
    cog_raw      = pd.Series(cog_centered, index=mw.index).fillna(
                       pd.Series(cog_fwd, index=mw.index))

    # Smooth via sin/cos rolling mean (handles wrap-around at ±pi)
    s_mean = np.sin(cog_raw).rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
    c_mean = np.cos(cog_raw).rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
    cog_smoothed = np.arctan2(s_mean, c_mean)   # radians in [-pi, pi]

    # ── CRITICAL DIFFERENCE: convert math angle → QGIS clockwise angle ───────
    #
    #   FORMAT A (gross):   angle_deg_cw = COG_rad * 180/pi
    #     COG is already a compass/navigation angle (0=North, CW).
    #     Multiplying by 180/pi just converts radians to degrees.
    #
    #   FORMAT B (klein):   angle_deg_cw = 90 - math.degrees(COG_math)
    #     COG was computed with atan2(dx, dy) which gives a MATH angle
    #     (0=East, CCW positive).  To convert to compass CW degrees:
    #         compass_deg = 90 - math_deg
    #     This is exactly what the QGIS notebook does on line:
    #         angle_deg = 90 - row['rotation_deg']
    #     where rotation_deg = degrees(CourseOverGround_smoothed).
    #
    mw["cog_angle_deg_cw"] = 90.0 - np.degrees(cog_smoothed)

    # Also store smoothed COG for the output attribute table
    mw["cog_smoothed_rad"] = (cog_smoothed + 2 * np.pi) % (2 * np.pi)

    return mw


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_all_gpkgs(input_folder):
    files = sorted(glob.glob(os.path.join(input_folder, "*.gpkg")))
    if not files:
        raise FileNotFoundError(f"No .gpkg files found in: {input_folder}")
    print(f"Loading {len(files)} GPKG file(s)...")
    parts = []
    for f in files:
        gdf = gpd.read_file(f)
        gdf["sourcefile"] = os.path.basename(f)
        parts.append(gdf)
    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    print(f"  → {len(combined):,} rows,  CRS = {combined.crs}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def safe_float(v):
    try:
        return None if pd.isna(v) else float(v)
    except Exception:
        return None

def safe_int(v):
    try:
        return None if pd.isna(v) else int(v)
    except Exception:
        return None

def safe_str(v):
    try:
        return None if pd.isna(v) else str(v)
    except Exception:
        return None

def parse_date_from_filename(filename):
    base = os.path.basename(str(filename))
    m = re.match(r"^(\d{2})(\d{2})(\d{2})_", base)
    if not m:
        return None, None
    yy, mm, dd = m.groups()
    return f"20{yy}-{mm}-{dd}", f"{dd}.{mm}.20{yy}"

def choose_route_date(row, fmt):
    """Pick the best available date depending on data format."""
    # FORMAT A has PositionDate
    if fmt == "A" and "PositionDate" in row.index and pd.notna(row["PositionDate"]):
        try:
            dt = pd.to_datetime(row["PositionDate"], errors="coerce")
            if pd.notna(dt):
                return dt.strftime("%Y-%m-%d"), dt.strftime("%d.%m.%Y")
        except Exception:
            pass
    # FORMAT B has DateTime
    if fmt == "B" and "DateTime" in row.index and pd.notna(row["DateTime"]):
        try:
            dt = pd.to_datetime(row["DateTime"], errors="coerce", utc=True)
            if pd.notna(dt):
                return dt.strftime("%Y-%m-%d"), dt.strftime("%d.%m.%Y")
        except Exception:
            pass
    # Fall back to filename date
    if "sourcefile" in row.index:
        return parse_date_from_filename(row["sourcefile"])
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
#  POLYGON BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_polygon_output(mow):
    fmt = detect_format(mow)
    print(f"  → Detected data FORMAT: {fmt}  "
          f"({'gross/original – pre-existing COG+speed' if fmt == 'A' else 'klein/rudi – COG and speed computed from GPS'})")

    # ── Project ───────────────────────────────────────────────────────────────
    mowscaled   = mow.to_crs(epsg=OUTPUT_CRS)
    machinework = mowscaled.copy()

    # ── Unrotated corner columns (same for both formats) ──────────────────────
    machinework["li1x"] = machinework.geometry.x - AUSSENBREITE / 2
    machinework["li1y"] = machinework.geometry.y
    machinework["li2x"] = machinework["li1x"] + REIFENBREITE
    machinework["li2y"] = machinework.geometry.y
    machinework["re2x"] = machinework.geometry.x + AUSSENBREITE / 2
    machinework["re2y"] = machinework.geometry.y
    machinework["re1x"] = machinework["re2x"] - REIFENBREITE
    machinework["re1y"] = machinework.geometry.y

    # ── FORMAT-SPECIFIC preparation ───────────────────────────────────────────
    if fmt == "B":
        print("  Computing COG and speed from GPS positions + DateTime …")
        machinework = compute_cog_and_speed_format_b(machinework)

    # ── Initialise old_points from first row ──────────────────────────────────
    first = machinework.iloc[0]
    if fmt == "A":
        init_angle = float(first["CourseOverGround"]) * 180.0 / math.pi
    else:
        init_angle = float(first["cog_angle_deg_cw"]) if pd.notna(first["cog_angle_deg_cw"]) else 0.0

    old_points = compute_rotated_corners(first.geometry.x, first.geometry.y, init_angle)

    # ── Main loop ─────────────────────────────────────────────────────────────
    out_rows   = []
    poly_count = 0

    for index, row in tqdm(machinework.iterrows(),
                           total=len(machinework),
                           desc="Building polygons"):

        # ── Geometry validity ─────────────────────────────────────────────────
        geom = row.get("geometry", None)
        if geom is None or geom.is_empty:
            continue
        if getattr(geom, "geom_type", "") != "Point":
            try:
                geom = geom.centroid
            except Exception:
                continue

        x0, y0 = geom.x, geom.y
        if not (np.isfinite(x0) and np.isfinite(y0)):
            continue

        # ── Get rotation angle and speed depending on format ──────────────────
        if fmt == "A":
            # FORMAT A: COG is navigation radians → degrees directly
            cog_raw = row.get("CourseOverGround", np.nan)
            if pd.isna(cog_raw):
                continue
            angle_deg_cw = float(cog_raw) * 180.0 / math.pi

            speed_ms = row.get("GroundBasedMachineSpeed", np.nan)
            if pd.isna(speed_ms):
                old_points = compute_rotated_corners(x0, y0, angle_deg_cw)
                continue
            speed_kmh = round(float(speed_ms) * 3.6, 3)

            # Attribute values for FORMAT A
            cog_for_attr = round(float(cog_raw), 6)
            direction_deg = round(math.degrees(float(cog_raw)) % 360.0, 4)
            mach_dir  = safe_float(row.get("GroundBasedMachineDirection"))
            date_iso, date_display = choose_route_date(row, fmt)
            extra_attrs = {
                "PosDate" : safe_str(row.get("PositionDate")),
                "PosTime" : safe_str(row.get("PositionTime")),
                "WhBSpd"  : safe_float(row.get("WheelBasedMachineSpeed")),
                "WhBDist" : safe_float(row.get("WheelBasedMachineDistance")),
                "GrBSpd"  : safe_float(row.get("GroundBasedMachineSpeed")),
                "EngFuel" : safe_float(row.get("EngFuelRate")),
                "RearHitPos": safe_float(row.get("RearHitchPosition")),
                "MacSelSpd" : safe_float(row.get("MachineSelectedSpeed")),
                "MacSelDst" : safe_float(row.get("MachineSelectedDistance")),
                "SID"     : safe_int(row.get("SID")),
                "CourseOG": safe_float(row.get("CourseOverGround")),
                "SpeedOG" : safe_float(row.get("SpeedOverGround")),
                "EngSpeed": safe_float(row.get("EngSpeed")),
            }

        else:
            # FORMAT B: angle already converted to CW degrees in compute step
            angle_deg_cw = row.get("cog_angle_deg_cw", np.nan)
            if pd.isna(angle_deg_cw):
                continue
            angle_deg_cw = float(angle_deg_cw)

            speed_ms = row.get("gpsspeed_smoothed_mps", np.nan)
            if pd.isna(speed_ms):
                old_points = compute_rotated_corners(x0, y0, angle_deg_cw)
                continue
            speed_kmh = round(float(speed_ms) * 3.6, 3)

            # Attribute values for FORMAT B
            cog_for_attr  = round(float(row.get("cog_smoothed_rad", 0.0)), 6)
            direction_deg = round(float(row.get("cog_smoothed_rad", 0.0)) * 180.0 / math.pi % 360.0, 4)
            mach_dir  = None   # not available in FORMAT B
            date_iso, date_display = choose_route_date(row, fmt)
            extra_attrs = {
                "PosDate"   : safe_str(row.get("Date")),
                "PosTime"   : safe_str(row.get("Time")),
                "WhBSpd"    : None,
                "WhBDist"   : None,
                "GrBSpd"    : None,
                "EngFuel"   : None,
                "RearHitPos": None,
                "MacSelSpd" : None,
                "MacSelDst" : None,
                "SID"       : None,
                "CourseOG"  : cog_for_attr,
                "SpeedOG"   : None,
                "EngSpeed"  : None,
            }

        # ── Compute rotated corners ───────────────────────────────────────────
        p1, p2, p3, p4 = compute_rotated_corners(x0, y0, angle_deg_cw)

        if any(not np.isfinite(v) for pt in (p1, p2, p3, p4) for v in pt):
            old_points = (p1, p2, p3, p4)
            continue

        # ── Build polygons if moving ──────────────────────────────────────────
        if speed_kmh >= MIN_SPEED_KMH:

            left_coords = [
                (p1[0],            p1[1]           ),
                (p2[0],            p2[1]           ),
                (old_points[1][0], old_points[1][1]),
                (old_points[0][0], old_points[0][1]),
                (p1[0],            p1[1]           ),   # close ring
            ]
            right_coords = [
                (p3[0],            p3[1]           ),
                (p4[0],            p4[1]           ),
                (old_points[3][0], old_points[3][1]),
                (old_points[2][0], old_points[2][1]),
                (p3[0],            p3[1]           ),   # close ring
            ]

            left_poly  = Polygon(left_coords)
            right_poly = Polygon(right_coords)

            if not left_poly.is_valid:
                left_poly  = left_poly.buffer(0)
            if not right_poly.is_valid:
                right_poly = right_poly.buffer(0)

            base = {
                "PolySrc"   : POLYGON_SOURCE,
                "Direction" : direction_deg,
                "COG_rad"   : cog_for_attr,
                "Speed_kmh" : speed_kmh,
                "MachDir"   : mach_dir,
                "RouteDate" : date_display,
                "RouteISO"  : date_iso,
                "SrcFile"   : safe_str(row.get("sourcefile")),
                "PtIndex"   : int(index),
                **extra_attrs,
            }

            if not left_poly.is_empty and left_poly.area > 0:
                out_rows.append({**base, "Wheel": "left",  "geometry": left_poly})
            if not right_poly.is_empty and right_poly.area > 0:
                out_rows.append({**base, "Wheel": "right", "geometry": right_poly})

            poly_count += 2

        # Always update old_points (same as QGIS notebook)
        old_points = (p1, p2, p3, p4)

    if not out_rows:
        raise RuntimeError("No polygons were generated. Check your input data and column names.")

    polygons_gdf = gpd.GeoDataFrame(out_rows, crs=f"EPSG:{OUTPUT_CRS}")
    return polygons_gdf, poly_count, fmt


# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(polygons_gdf):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    gpkg_path = os.path.join(OUTPUT_FOLDER, OUTPUT_BASENAME + ".gpkg")
    shp_path  = os.path.join(OUTPUT_FOLDER, OUTPUT_BASENAME + ".shp")
    polygons_gdf.to_file(gpkg_path, driver="GPKG")
    polygons_gdf.to_file(shp_path,  driver="ESRI Shapefile")
    return gpkg_path, shp_path


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    mow = load_all_gpkgs(INPUT_FOLDER)
    polygons_gdf, poly_count, fmt = build_polygon_output(mow)
    gpkg_path, shp_path = save_outputs(polygons_gdf)

    print()
    print(f"✔  Done.  {poly_count} polygons written  (FORMAT {fmt}).")
    print(f"   GPKG : {gpkg_path}")
    print(f"   SHP  : {shp_path}")
    print()
    print("Output fields:")
    print("  PolySrc   – polygon source label (set in POLYGON_SOURCE)")
    print("  Wheel     – 'left' or 'right'")
    print("  Direction – heading degrees 0-360")
    print("  COG_rad   – course over ground in radians")
    print("  Speed_kmh – speed in km/h")
    print("  MachDir   – GroundBasedMachineDirection (FORMAT A only)")
    print("  RouteDate – date as DD.MM.YYYY")
    print("  RouteISO  – date as YYYY-MM-DD")
    print("  SrcFile   – source filename")
    print("  PtIndex   – row index in input data")
    print("  PosDate/PosTime, WhBSpd, WhBDist, GrBSpd, EngFuel,")
    print("  RearHitPos, MacSelSpd, MacSelDst, SID, CourseOG,")
    print("  SpeedOG, EngSpeed  (FORMAT A only; None for FORMAT B)")


if __name__ == "__main__":
    main()
