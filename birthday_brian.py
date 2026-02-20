#!/usr/bin/env python3
"""
Happy Birthday Brian McNoldy!

A hurricane track art tribute to Brian McNoldy, legendary hurricane researcher
at the University of Miami's Rosenstiel School (RSMAS).

This script creates a visualization using real Atlantic hurricane tracks
to celebrate Brian's birthday in true tropical cyclone style.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import requests

# Famous Atlantic hurricanes to feature (selected for their memorable tracks)
LEGENDARY_HURRICANES = [
    {"name": "ANDREW", "year": 1992, "note": "Cat 5, devastated S. Florida"},
    {"name": "KATRINA", "year": 2005, "note": "Cat 5, Gulf Coast catastrophe"},
    {"name": "WILMA", "year": 2005, "note": "Most intense Atlantic hurricane"},
    {"name": "IVAN", "year": 2004, "note": "Cat 5, remarkable loop"},
    {"name": "IRMA", "year": 2017, "note": "Cat 5, longest Cat 5 duration"},
    {"name": "MARIA", "year": 2017, "note": "Cat 5, Puerto Rico"},
    {"name": "DORIAN", "year": 2019, "note": "Cat 5, stalled over Bahamas"},
    {"name": "MICHAEL", "year": 2018, "note": "Cat 5, Florida Panhandle"},
]

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
)

HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"


def fetch_hurdat2_tracks() -> dict[str, list[dict]]:
    """Fetch hurricane tracks from HURDAT2 for the legendary storms."""
    print("Fetching HURDAT2 data from NOAA NHC...")
    try:
        resp = requests.get(HURDAT2_URL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Could not fetch HURDAT2: {e}")
        return {}

    lines = resp.text.strip().split("\n")
    storms = {}
    current_storm = None
    current_name = None
    current_year = None

    for line in lines:
        parts = [p.strip() for p in line.split(",")]

        # Header line: AL092005, KATRINA, 34,
        if len(parts) >= 3 and parts[0].startswith(("AL", "EP")):
            current_storm = parts[0]
            current_name = parts[1]
            try:
                current_year = int(current_storm[4:8])
            except (ValueError, IndexError):
                current_year = None
            storms[f"{current_name}_{current_year}"] = {
                "name": current_name,
                "year": current_year,
                "track": [],
            }
        elif current_storm and len(parts) >= 7:
            # Data line: 20050823, 1800, , HU, 26.1N, 89.2W, 175, ...
            try:
                lat_str = parts[4]
                lon_str = parts[5]
                wind_str = parts[6] if len(parts) > 6 else "0"

                lat = float(lat_str.replace("N", "").replace("S", ""))
                if "S" in lat_str:
                    lat = -lat
                lon = float(lon_str.replace("W", "").replace("E", ""))
                if "W" in lon_str:
                    lon = -lon
                wind = int(wind_str) if wind_str.strip().isdigit() else 0

                storms[f"{current_name}_{current_year}"]["track"].append({
                    "lat": lat,
                    "lon": lon,
                    "wind_kt": wind,
                })
            except (ValueError, IndexError):
                continue

    return storms


def get_saffir_simpson_color(wind_kt: float) -> str:
    """Return color based on Saffir-Simpson scale."""
    if wind_kt >= 137:
        return "#FF0000"  # Cat 5 - Red
    elif wind_kt >= 113:
        return "#FF6600"  # Cat 4 - Orange-Red
    elif wind_kt >= 96:
        return "#FFA500"  # Cat 3 - Orange
    elif wind_kt >= 83:
        return "#FFD700"  # Cat 2 - Gold
    elif wind_kt >= 64:
        return "#FFFF00"  # Cat 1 - Yellow
    elif wind_kt >= 34:
        return "#00FF00"  # TS - Green
    else:
        return "#0000FF"  # TD - Blue


def create_birthday_visualization(storms_data: dict, output_path: Path) -> str:
    """Create the birthday track art visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Circle
    import matplotlib.patheffects as path_effects

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False
        print("Note: Cartopy not available, using basic plot")

    # Set up the figure
    fig = plt.figure(figsize=(16, 12))

    if has_cartopy:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([-100, -20, 5, 50])

        # Add map features
        ax.add_feature(cfeature.LAND, facecolor="#E8E8E8", alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor="#B8D4E8", alpha=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="#666666")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":", color="#888888")
        ax.add_feature(cfeature.STATES, linewidth=0.2, color="#AAAAAA")
        ax.gridlines(draw_labels=True, alpha=0.3, linestyle="--")
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-100, -20)
        ax.set_ylim(5, 50)
        ax.set_facecolor("#B8D4E8")

    # Plot each legendary hurricane
    plotted_storms = []
    for storm_info in LEGENDARY_HURRICANES:
        key = f"{storm_info['name']}_{storm_info['year']}"
        if key not in storms_data:
            continue

        storm = storms_data[key]
        track = storm["track"]
        if len(track) < 3:
            continue

        lats = [p["lat"] for p in track]
        lons = [p["lon"] for p in track]
        winds = [p["wind_kt"] for p in track]

        # Plot track segments colored by intensity
        transform_kw = {"transform": ccrs.PlateCarree()} if has_cartopy else {}
        for i in range(1, len(lats)):
            color = get_saffir_simpson_color(winds[i])
            ax.plot(
                [lons[i-1], lons[i]], [lats[i-1], lats[i]],
                color=color, linewidth=3, alpha=0.8,
                solid_capstyle="round",
                **transform_kw
            )

        # Plot intensity points
        for i, (la, lo, w) in enumerate(zip(lats, lons, winds)):
            if i % 4 == 0:  # Every 4th point to avoid clutter
                color = get_saffir_simpson_color(w)
                size = 15 + w * 0.15
                ax.scatter(
                    lo, la, c=color, s=size,
                    edgecolors="white", linewidth=0.5, zorder=5,
                    alpha=0.9, **transform_kw
                )

        # Label the storm at peak intensity
        peak_idx = np.argmax(winds)
        label_x, label_y = lons[peak_idx], lats[peak_idx]

        txt = ax.text(
            label_x, label_y + 1.5,
            f"{storm['name']}\n({storm['year']})",
            fontsize=8, fontweight="bold",
            ha="center", va="bottom",
            color="#333333",
            **transform_kw
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground="white"),
            path_effects.Normal()
        ])

        plotted_storms.append(storm_info)

    # Add the birthday message with hurricane styling
    title_text = ax.text(
        0.5, 0.95,
        "Happy Birthday Brian McNoldy!",
        transform=ax.transAxes,
        fontsize=28, fontweight="bold",
        ha="center", va="top",
        color="#1a5276",
        family="sans-serif"
    )
    title_text.set_path_effects([
        path_effects.Stroke(linewidth=4, foreground="white"),
        path_effects.Normal()
    ])

    subtitle = ax.text(
        0.5, 0.89,
        "A tribute in hurricane tracks from your friends at Worldsphere",
        transform=ax.transAxes,
        fontsize=14, style="italic",
        ha="center", va="top",
        color="#2874a6"
    )
    subtitle.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground="white"),
        path_effects.Normal()
    ])

    # Add a fun meteorological message
    quote = ax.text(
        0.5, 0.06,
        '"Every hurricane has a story, and you\'ve helped tell them all."',
        transform=ax.transAxes,
        fontsize=12, style="italic",
        ha="center", va="bottom",
        color="#555555"
    )
    quote.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground="white"),
        path_effects.Normal()
    ])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#0000FF", label="TD (<34 kt)"),
        Patch(facecolor="#00FF00", label="TS (34-63 kt)"),
        Patch(facecolor="#FFFF00", label="Cat 1 (64-82 kt)"),
        Patch(facecolor="#FFD700", label="Cat 2 (83-95 kt)"),
        Patch(facecolor="#FFA500", label="Cat 3 (96-112 kt)"),
        Patch(facecolor="#FF6600", label="Cat 4 (113-136 kt)"),
        Patch(facecolor="#FF0000", label="Cat 5 (137+ kt)"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower left",
        fontsize=9, framealpha=0.9,
        title="Saffir-Simpson Scale"
    )

    # Add storm list annotation
    storm_list_text = "Featured Hurricanes:\n" + "\n".join(
        f"  {s['name']} ({s['year']})" for s in plotted_storms
    )
    ax.text(
        0.98, 0.02, storm_list_text,
        transform=ax.transAxes,
        fontsize=8, ha="right", va="bottom",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Save
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return str(output_path)


def main():
    """Generate the birthday visualization for Brian McNoldy."""
    print("=" * 60)
    print("  Happy Birthday Brian McNoldy!")
    print("  Hurricane Track Art Generator")
    print("=" * 60)
    print()

    # Fetch hurricane data
    storms_data = fetch_hurdat2_tracks()

    if not storms_data:
        print("Could not fetch hurricane data. Using sample tracks...")
        # Create sample data if network fails
        storms_data = create_sample_tracks()

    # Find how many legendary storms we have
    found = sum(1 for s in LEGENDARY_HURRICANES
                if f"{s['name']}_{s['year']}" in storms_data)
    print(f"Found {found}/{len(LEGENDARY_HURRICANES)} legendary hurricanes")

    # Generate the visualization
    output_dir = Path(__file__).parent / "climate_agent" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "happy_birthday_brian.png"

    print(f"Creating birthday visualization...")
    result = create_birthday_visualization(storms_data, output_path)

    print()
    print("=" * 60)
    print(f"  Birthday card saved to: {result}")
    print("=" * 60)
    print()
    print("Share this with Brian to celebrate his birthday!")
    print("From all of us who love hurricanes and the science behind them.")

    return result


def create_sample_tracks() -> dict:
    """Create sample hurricane tracks if network is unavailable."""
    # Representative tracks for demonstration
    return {
        "ANDREW_1992": {
            "name": "ANDREW",
            "year": 1992,
            "track": [
                {"lat": 13.0, "lon": -42.0, "wind_kt": 35},
                {"lat": 15.0, "lon": -52.0, "wind_kt": 50},
                {"lat": 18.0, "lon": -62.0, "wind_kt": 75},
                {"lat": 23.0, "lon": -72.0, "wind_kt": 110},
                {"lat": 25.5, "lon": -80.0, "wind_kt": 145},
                {"lat": 27.0, "lon": -85.0, "wind_kt": 125},
                {"lat": 29.5, "lon": -91.0, "wind_kt": 115},
            ]
        },
        "KATRINA_2005": {
            "name": "KATRINA",
            "year": 2005,
            "track": [
                {"lat": 23.5, "lon": -75.5, "wind_kt": 35},
                {"lat": 24.5, "lon": -78.0, "wind_kt": 50},
                {"lat": 26.0, "lon": -80.0, "wind_kt": 70},
                {"lat": 26.0, "lon": -85.5, "wind_kt": 100},
                {"lat": 26.0, "lon": -88.0, "wind_kt": 150},
                {"lat": 29.5, "lon": -89.5, "wind_kt": 125},
                {"lat": 34.0, "lon": -87.0, "wind_kt": 45},
            ]
        },
        "IRMA_2017": {
            "name": "IRMA",
            "year": 2017,
            "track": [
                {"lat": 16.0, "lon": -30.0, "wind_kt": 50},
                {"lat": 17.0, "lon": -45.0, "wind_kt": 100},
                {"lat": 18.0, "lon": -58.0, "wind_kt": 150},
                {"lat": 18.5, "lon": -64.0, "wind_kt": 160},
                {"lat": 21.5, "lon": -74.0, "wind_kt": 155},
                {"lat": 25.0, "lon": -80.5, "wind_kt": 115},
                {"lat": 30.0, "lon": -83.0, "wind_kt": 65},
            ]
        },
    }


if __name__ == "__main__":
    main()
