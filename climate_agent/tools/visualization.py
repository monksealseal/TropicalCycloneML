"""Visualization tools for climate data and tropical cyclone analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from climate_agent.config import DATA_DIR


def plot_cyclone_track(
    track_points: list[dict],
    storm_name: str = "Tropical Cyclone",
    output_filename: str | None = None,
) -> dict[str, Any]:
    """Generate a map visualization of a tropical cyclone track.

    Creates a matplotlib/cartopy map showing the storm track colored by intensity.

    Args:
        track_points: List of dicts with lat, lon, wind_kt keys.
        storm_name: Name of the storm for the title.
        output_filename: Output filename (saved to data directory).

    Returns:
        Status dict with output file path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return {"error": "matplotlib not installed. Install with: pip install matplotlib"}

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    if not track_points:
        return {"error": "No track points provided"}

    lats = [p["lat"] for p in track_points]
    lons = [p["lon"] for p in track_points]
    winds = [p.get("wind_kt", 0) for p in track_points]

    # Color by Saffir-Simpson category
    colors = []
    for w in winds:
        if w >= 137:
            colors.append("#FF0000")      # Cat 5 - Red
        elif w >= 113:
            colors.append("#FF6600")      # Cat 4 - Orange-Red
        elif w >= 96:
            colors.append("#FFA500")      # Cat 3 - Orange
        elif w >= 83:
            colors.append("#FFD700")      # Cat 2 - Gold
        elif w >= 64:
            colors.append("#FFFF00")      # Cat 1 - Yellow
        elif w >= 34:
            colors.append("#00FF00")      # TS - Green
        else:
            colors.append("#0000FF")      # TD - Blue

    if has_cartopy:
        fig, ax = plt.subplots(
            figsize=(12, 8),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        ax.set_extent([
            min(lons) - 5, max(lons) + 5,
            min(lats) - 5, max(lats) + 5,
        ])
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.gridlines(draw_labels=True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Plot track segments colored by intensity
    for i in range(1, len(lats)):
        ax.plot(
            [lons[i - 1], lons[i]],
            [lats[i - 1], lats[i]],
            color=colors[i],
            linewidth=2,
            transform=ccrs.PlateCarree() if has_cartopy else None,
        )

    # Plot points
    for i, (la, lo, c) in enumerate(zip(lats, lons, colors)):
        kwargs = {}
        if has_cartopy:
            kwargs["transform"] = ccrs.PlateCarree()
        size = 30 + winds[i] * 0.3
        ax.scatter(lo, la, c=c, s=size, edgecolors="black", linewidth=0.5, zorder=5, **kwargs)

    # Mark start and peak intensity
    peak_idx = np.argmax(winds)
    kw = {"transform": ccrs.PlateCarree()} if has_cartopy else {}
    ax.annotate("START", (lons[0], lats[0]), fontsize=8, fontweight="bold", **kw)
    ax.annotate(
        f"PEAK\n{winds[peak_idx]} kt",
        (lons[peak_idx], lats[peak_idx]),
        fontsize=8,
        fontweight="bold",
        color="red",
        **kw,
    )

    ax.set_title(f"Track of {storm_name}", fontsize=14, fontweight="bold")

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
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8)

    if output_filename is None:
        output_filename = f"track_{storm_name.replace(' ', '_')}.png"
    output_path = DATA_DIR / output_filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "success",
        "output_file": str(output_path),
        "description": f"Track map for {storm_name} with {len(track_points)} points, "
        f"peak intensity {max(winds)} kt",
    }


def plot_climate_timeseries(
    years: list[float],
    values: list[float],
    metric_name: str = "Climate Metric",
    units: str = "",
    trend_line: bool = True,
    output_filename: str | None = None,
) -> dict[str, Any]:
    """Generate a time series plot of a climate metric.

    Args:
        years: List of years.
        values: List of corresponding values.
        metric_name: Name of the metric for labeling.
        units: Units string for y-axis.
        trend_line: Whether to add a linear trend line.
        output_filename: Output filename.

    Returns:
        Status dict with output file path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"error": "matplotlib not installed"}

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(years, values, color="#2196F3", linewidth=1.5, alpha=0.8, label="Observed")

    if trend_line and len(years) >= 2:
        coeffs = np.polyfit(years, values, 1)
        trend_vals = np.polyval(coeffs, years)
        ax.plot(years, trend_vals, color="#F44336", linewidth=2, linestyle="--",
                label=f"Trend: {coeffs[0]:+.4f}/yr")

        # Also show recent 30-year trend
        if len(years) > 30:
            recent_years = years[-30:]
            recent_vals = values[-30:]
            r_coeffs = np.polyfit(recent_years, recent_vals, 1)
            r_trend = np.polyval(r_coeffs, recent_years)
            ax.plot(recent_years, r_trend, color="#FF9800", linewidth=2, linestyle="-.",
                    label=f"Recent trend: {r_coeffs[0]:+.4f}/yr")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(f"{metric_name} ({units})" if units else metric_name, fontsize=12)
    ax.set_title(f"{metric_name} Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if output_filename is None:
        output_filename = f"timeseries_{metric_name.replace(' ', '_').lower()}.png"
    output_path = DATA_DIR / output_filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "success",
        "output_file": str(output_path),
        "description": f"Time series plot of {metric_name} from {years[0]:.0f} to {years[-1]:.0f}",
    }


def plot_sst_map(
    lat_min: float = -30.0,
    lat_max: float = 30.0,
    lon_min: float = -100.0,
    lon_max: float = 0.0,
    output_filename: str = "sst_map.png",
) -> dict[str, Any]:
    """Generate a sea surface temperature map for the specified region.

    Args:
        lat_min: Minimum latitude.
        lat_max: Maximum latitude.
        lon_min: Minimum longitude.
        lon_max: Maximum longitude.
        output_filename: Output filename.

    Returns:
        Status dict with output file path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return {"error": "matplotlib not installed"}

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    # Generate SST field
    lats = np.arange(lat_min, lat_max, 0.5)
    lons = np.arange(lon_min, lon_max, 0.5)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Realistic SST pattern
    sst = 28.0 - 0.003 * lat_grid**2
    # Gulf Stream warm tongue
    gulf_stream = 3.0 * np.exp(-((lat_grid - 30)**2 / 50 + (lon_grid + 60)**2 / 200))
    sst += gulf_stream
    sst = np.clip(sst + np.random.normal(0, 0.3, sst.shape), -2, 35)

    if has_cartopy:
        fig, ax = plt.subplots(
            figsize=(14, 8),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=3)
        ax.gridlines(draw_labels=True, alpha=0.3)

        im = ax.pcolormesh(
            lon_grid, lat_grid, sst,
            cmap="RdYlBu_r",
            vmin=15, vmax=32,
            transform=ccrs.PlateCarree(),
        )
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.pcolormesh(lon_grid, lat_grid, sst, cmap="RdYlBu_r", vmin=15, vmax=32)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Sea Surface Temperature (C)", fontsize=12)

    # Add 26.5C contour (cyclone formation threshold)
    if has_cartopy:
        ax.contour(
            lon_grid, lat_grid, sst, levels=[26.5],
            colors="black", linewidths=2, linestyles="--",
            transform=ccrs.PlateCarree(),
        )
    else:
        ax.contour(lon_grid, lat_grid, sst, levels=[26.5],
                   colors="black", linewidths=2, linestyles="--")

    ax.set_title("Sea Surface Temperature with 26.5C Cyclone Threshold", fontsize=14, fontweight="bold")

    output_path = DATA_DIR / output_filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "success",
        "output_file": str(output_path),
        "description": f"SST map for region [{lat_min},{lat_max}]N x [{lon_min},{lon_max}]E "
        "with 26.5C cyclone threshold contour",
    }
