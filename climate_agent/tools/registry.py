"""Tool registry: maps tool definitions to implementations for the Claude agent."""

from __future__ import annotations

import json
from typing import Any

from climate_agent.tools.data_fetch import (
    fetch_active_cyclones,
    fetch_atmospheric_conditions,
    fetch_climate_indicators,
    fetch_hurricane_history,
    fetch_sea_surface_temperature,
)
from climate_agent.tools.analysis import (
    analyze_climate_trend,
    analyze_cyclone_season,
    analyze_storm_track,
    compute_carbon_budget,
    compute_potential_intensity,
)
from climate_agent.tools.forecasting import (
    assess_climate_risk,
    forecast_cyclone_intensity,
    forecast_cyclone_track,
)
from climate_agent.tools.visualization import (
    plot_climate_timeseries,
    plot_cyclone_track,
    plot_sst_map,
)

# Claude tool definitions (Anthropic API format)
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # === Data Fetching Tools ===
    {
        "name": "fetch_active_cyclones",
        "description": (
            "Fetch currently active tropical cyclones worldwide from NOAA NHC. "
            "Returns name, position, intensity, and movement for each active storm."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "fetch_hurricane_history",
        "description": (
            "Fetch historical tropical cyclone records from the IBTrACS database. "
            "Filter by ocean basin, year range, or storm name. Returns storm tracks, "
            "peak intensities, and summary statistics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "basin": {
                    "type": "string",
                    "description": "Ocean basin: 'NA' (North Atlantic), 'EP' (East Pacific), "
                    "'WP' (West Pacific), 'NI' (North Indian), 'SI' (South Indian), "
                    "'SP' (South Pacific), 'SA' (South Atlantic), or 'ALL'.",
                    "default": "ALL",
                },
                "year_start": {
                    "type": "integer",
                    "description": "Start year for filtering (inclusive).",
                },
                "year_end": {
                    "type": "integer",
                    "description": "End year for filtering (inclusive).",
                },
                "name": {
                    "type": "string",
                    "description": "Storm name to search for (case-insensitive partial match).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "fetch_sea_surface_temperature",
        "description": (
            "Fetch sea surface temperature (SST) data for a specified region. "
            "Returns SST statistics, area above the 26.5C cyclone formation threshold, "
            "and cyclone formation potential assessment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat_min": {"type": "number", "description": "Minimum latitude (-90 to 90).", "default": -90},
                "lat_max": {"type": "number", "description": "Maximum latitude (-90 to 90).", "default": 90},
                "lon_min": {"type": "number", "description": "Minimum longitude (-180 to 180).", "default": -180},
                "lon_max": {"type": "number", "description": "Maximum longitude (-180 to 180).", "default": 180},
            },
            "required": [],
        },
    },
    {
        "name": "fetch_atmospheric_conditions",
        "description": (
            "Fetch atmospheric conditions at a specific location relevant to tropical "
            "cyclone formation: wind shear, relative humidity, CAPE, sea-level pressure, "
            "and Coriolis parameter. Includes a cyclone genesis favorability assessment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude (-90 to 90)."},
                "lon": {"type": "number", "description": "Longitude (-180 to 180)."},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "fetch_climate_indicators",
        "description": (
            "Fetch current values and trends for major global climate change indicators: "
            "CO2 concentration, global temperature anomaly, sea level rise, Arctic sea ice, "
            "and tropical cyclone intensity trends."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # === Analysis Tools ===
    {
        "name": "analyze_cyclone_season",
        "description": (
            "Analyze a tropical cyclone season: computes Accumulated Cyclone Energy (ACE), "
            "storm counts by category, and compares to historical averages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "storm_data": {
                    "type": "array",
                    "description": "List of storm dicts with max_wind_kt, min_pressure_mb, and track.",
                    "items": {"type": "object"},
                },
                "basin": {
                    "type": "string",
                    "description": "Basin identifier (e.g., 'NA').",
                    "default": "NA",
                },
            },
            "required": ["storm_data"],
        },
    },
    {
        "name": "compute_potential_intensity",
        "description": (
            "Compute the theoretical Maximum Potential Intensity (MPI) of a tropical cyclone "
            "given sea surface temperature and atmospheric conditions using Emanuel (1988) theory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sst_celsius": {"type": "number", "description": "Sea surface temperature in Celsius."},
                "outflow_temp_celsius": {
                    "type": "number",
                    "description": "Outflow temperature at tropopause (Celsius).",
                    "default": -70.0,
                },
                "surface_pressure_hpa": {
                    "type": "number",
                    "description": "Environmental sea-level pressure (hPa).",
                    "default": 1015.0,
                },
            },
            "required": ["sst_celsius"],
        },
    },
    {
        "name": "analyze_climate_trend",
        "description": (
            "Analyze long-term trends for a climate metric: global temperature, CO2, "
            "sea level, Arctic ice, or cyclone intensity. Returns trend rates, "
            "acceleration, and linear projections."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "Metric to analyze: 'global_temperature', 'co2', 'sea_level', "
                    "'arctic_ice', or 'cyclone_intensity'.",
                    "enum": ["global_temperature", "co2", "sea_level", "arctic_ice", "cyclone_intensity"],
                },
                "start_year": {"type": "integer", "description": "Start year.", "default": 1880},
                "end_year": {"type": "integer", "description": "End year.", "default": 2025},
            },
            "required": ["metric"],
        },
    },
    {
        "name": "compute_carbon_budget",
        "description": (
            "Compute the remaining global carbon budget for a warming target (1.5C or 2.0C). "
            "Shows how many years of current emissions remain, and required reduction rates "
            "for net-zero pathways."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_warming_celsius": {
                    "type": "number",
                    "description": "Target warming limit (e.g., 1.5 or 2.0).",
                    "default": 1.5,
                },
                "current_annual_emissions_gtco2": {
                    "type": "number",
                    "description": "Current annual global CO2 emissions (GtCO2/yr).",
                    "default": 40.0,
                },
            },
            "required": [],
        },
    },
    {
        "name": "analyze_storm_track",
        "description": (
            "Analyze a tropical cyclone track: compute distance traveled, forward speed, "
            "intensification rate, recurvature detection, and rapid intensification assessment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "track_points": {
                    "type": "array",
                    "description": "List of track point dicts with lat, lon, wind_kt, time.",
                    "items": {"type": "object"},
                },
            },
            "required": ["track_points"],
        },
    },
    # === Forecasting Tools ===
    {
        "name": "forecast_cyclone_intensity",
        "description": (
            "Forecast tropical cyclone intensity using a SHIPS-like statistical-dynamical model. "
            "Predicts wind speed and pressure at multiple lead times (12h, 24h, 48h, 72h, 120h). "
            "Includes rapid intensification probability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "current_wind_kt": {"type": "number", "description": "Current max sustained winds (kt)."},
                "current_pressure_hpa": {"type": "number", "description": "Current min central pressure (hPa)."},
                "lat": {"type": "number", "description": "Current latitude."},
                "lon": {"type": "number", "description": "Current longitude."},
                "sst_celsius": {"type": "number", "description": "SST at current location (Celsius)."},
                "wind_shear_ms": {"type": "number", "description": "200-850 hPa wind shear (m/s)."},
                "forecast_hours": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Forecast lead times in hours.",
                },
            },
            "required": ["current_wind_kt", "current_pressure_hpa", "lat", "lon", "sst_celsius", "wind_shear_ms"],
        },
    },
    {
        "name": "forecast_cyclone_track",
        "description": (
            "Forecast tropical cyclone track using a beta-advection model with "
            "climatological recurvature. Returns predicted positions, headings, "
            "and uncertainty cones at each forecast time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Current latitude."},
                "lon": {"type": "number", "description": "Current longitude."},
                "current_heading_deg": {
                    "type": "number",
                    "description": "Current heading (0=N, 90=E, 180=S, 270=W).",
                    "default": 315,
                },
                "current_speed_kmh": {
                    "type": "number",
                    "description": "Current forward speed (km/h).",
                    "default": 20,
                },
                "forecast_hours": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Forecast lead times in hours.",
                },
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "assess_climate_risk",
        "description": (
            "Assess climate change risks for a specific location under different IPCC SSP "
            "scenarios. Evaluates tropical cyclone, sea level rise, extreme heat, and "
            "precipitation risks with adaptation priorities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location_lat": {"type": "number", "description": "Latitude of location."},
                "location_lon": {"type": "number", "description": "Longitude of location."},
                "scenario": {
                    "type": "string",
                    "description": "IPCC SSP scenario.",
                    "enum": ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"],
                    "default": "SSP2-4.5",
                },
                "time_horizon": {
                    "type": "integer",
                    "description": "Target year for assessment.",
                    "default": 2050,
                },
            },
            "required": ["location_lat", "location_lon"],
        },
    },
    # === Visualization Tools ===
    {
        "name": "plot_cyclone_track",
        "description": (
            "Generate a map visualization of a tropical cyclone track colored by intensity. "
            "Saves to PNG file. Requires track_points with lat, lon, and wind_kt."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "track_points": {
                    "type": "array",
                    "description": "List of track point dicts with lat, lon, wind_kt.",
                    "items": {"type": "object"},
                },
                "storm_name": {"type": "string", "description": "Name of the storm.", "default": "Tropical Cyclone"},
                "output_filename": {"type": "string", "description": "Output PNG filename."},
            },
            "required": ["track_points"],
        },
    },
    {
        "name": "plot_climate_timeseries",
        "description": (
            "Generate a time series plot of a climate metric with trend lines. "
            "Saves to PNG file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "years": {"type": "array", "items": {"type": "number"}, "description": "List of years."},
                "values": {"type": "array", "items": {"type": "number"}, "description": "Corresponding values."},
                "metric_name": {"type": "string", "description": "Name of the metric."},
                "units": {"type": "string", "description": "Units for y-axis."},
                "trend_line": {"type": "boolean", "description": "Add trend line.", "default": True},
                "output_filename": {"type": "string", "description": "Output PNG filename."},
            },
            "required": ["years", "values", "metric_name"],
        },
    },
    {
        "name": "plot_sst_map",
        "description": (
            "Generate a sea surface temperature map with the 26.5C cyclone formation "
            "threshold contour. Saves to PNG file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat_min": {"type": "number", "default": -30},
                "lat_max": {"type": "number", "default": 30},
                "lon_min": {"type": "number", "default": -100},
                "lon_max": {"type": "number", "default": 0},
                "output_filename": {"type": "string", "default": "sst_map.png"},
            },
            "required": [],
        },
    },
]

# Function dispatch table
_DISPATCH: dict[str, Any] = {
    "fetch_active_cyclones": fetch_active_cyclones,
    "fetch_hurricane_history": fetch_hurricane_history,
    "fetch_sea_surface_temperature": fetch_sea_surface_temperature,
    "fetch_atmospheric_conditions": fetch_atmospheric_conditions,
    "fetch_climate_indicators": fetch_climate_indicators,
    "analyze_cyclone_season": analyze_cyclone_season,
    "compute_potential_intensity": compute_potential_intensity,
    "analyze_climate_trend": analyze_climate_trend,
    "compute_carbon_budget": compute_carbon_budget,
    "analyze_storm_track": analyze_storm_track,
    "forecast_cyclone_intensity": forecast_cyclone_intensity,
    "forecast_cyclone_track": forecast_cyclone_track,
    "assess_climate_risk": assess_climate_risk,
    "plot_cyclone_track": plot_cyclone_track,
    "plot_climate_timeseries": plot_climate_timeseries,
    "plot_sst_map": plot_sst_map,
}


def dispatch_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Dispatch a tool call to the appropriate function and return JSON result.

    Args:
        tool_name: Name of the tool to invoke.
        tool_input: Dictionary of tool arguments.

    Returns:
        JSON string of the tool result.
    """
    func = _DISPATCH.get(tool_name)
    if func is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = func(**tool_input)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool '{tool_name}' failed: {type(e).__name__}: {e}"})
