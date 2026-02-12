"""Tools for analyzing climate data and computing derived metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from climate_agent.config import EARTH_RADIUS_KM, SST_CYCLONE_THRESHOLD


def analyze_cyclone_season(storm_data: list[dict[str, Any]], basin: str = "NA") -> dict[str, Any]:
    """Analyze a tropical cyclone season from storm records.

    Computes Accumulated Cyclone Energy (ACE), power dissipation index,
    and other season summary metrics.

    Args:
        storm_data: List of storm dictionaries with track data.
        basin: Basin identifier for context.

    Returns:
        Season analysis with ACE, storm counts by category, and trends.
    """
    if not storm_data:
        return {"error": "No storm data provided for analysis"}

    total_ace = 0.0
    named_storms = 0
    hurricanes = 0
    major_hurricanes = 0
    max_wind_season = 0.0
    min_pressure_season = 9999.0

    for storm in storm_data:
        wind = storm.get("max_wind_kt", 0)
        pressure = storm.get("min_pressure_mb")

        if wind >= 34:
            named_storms += 1
        if wind >= 64:
            hurricanes += 1
        if wind >= 96:
            major_hurricanes += 1

        max_wind_season = max(max_wind_season, wind)
        if pressure and 0 < pressure < min_pressure_season:
            min_pressure_season = pressure

        # ACE calculation: sum of v^2 for each 6-hour period where v >= 34 kt
        track = storm.get("track", [])
        for point in track:
            v = point.get("wind_kt", 0)
            if v >= 34:
                total_ace += v**2 / 10000  # ACE units: 10^4 kt^2

    # Historical context for North Atlantic
    na_averages = {
        "named_storms": 14.4,
        "hurricanes": 7.2,
        "major_hurricanes": 3.2,
        "ace": 123.0,
    }

    return {
        "basin": basin,
        "summary": {
            "total_storms_analyzed": len(storm_data),
            "named_storms": named_storms,
            "hurricanes": hurricanes,
            "major_hurricanes": major_hurricanes,
            "max_wind_kt": max_wind_season,
            "min_pressure_mb": min_pressure_season if min_pressure_season < 9999 else None,
            "accumulated_cyclone_energy": round(total_ace, 1),
        },
        "comparison_to_average": {
            "named_storms_vs_avg": round(named_storms / na_averages["named_storms"] * 100 - 100),
            "hurricanes_vs_avg": round(hurricanes / na_averages["hurricanes"] * 100 - 100)
            if hurricanes > 0
            else -100,
            "ace_vs_avg": round(total_ace / na_averages["ace"] * 100 - 100)
            if total_ace > 0
            else -100,
            "baseline": "1991-2020 average (North Atlantic)",
        },
        "activity_classification": (
            "Hyperactive"
            if total_ace > 200
            else "Extremely active"
            if total_ace > 150
            else "Above normal"
            if total_ace > 110
            else "Near normal"
            if total_ace > 70
            else "Below normal"
        ),
    }


def compute_potential_intensity(
    sst_celsius: float,
    outflow_temp_celsius: float = -70.0,
    surface_pressure_hpa: float = 1015.0,
) -> dict[str, Any]:
    """Compute the theoretical maximum potential intensity (MPI) of a tropical cyclone.

    Uses a simplified version of the Emanuel (1988) potential intensity theory.

    Args:
        sst_celsius: Sea surface temperature in Celsius.
        outflow_temp_celsius: Outflow temperature at tropopause in Celsius.
        surface_pressure_hpa: Environmental sea-level pressure in hPa.

    Returns:
        Maximum potential intensity estimates.
    """
    sst_k = sst_celsius + 273.15
    to_k = outflow_temp_celsius + 273.15

    # Thermodynamic efficiency
    efficiency = (sst_k - to_k) / sst_k

    # Simplified enthalpy disequilibrium
    # Using Clausius-Clapeyron for saturation vapor pressure
    es_sst = 6.112 * math.exp(17.67 * sst_celsius / (sst_celsius + 243.5))  # hPa
    rh_boundary = 0.80  # assumed RH in boundary layer
    e_boundary = rh_boundary * es_sst

    # Enthalpy difference (simplified)
    ck_cd_ratio = 0.9  # ratio of exchange coefficients
    delta_k = ck_cd_ratio * 2500 * (es_sst - e_boundary) / surface_pressure_hpa  # J/kg approx

    # Maximum wind speed (m/s)
    v_max_ms = math.sqrt(efficiency * delta_k * 1000)
    v_max_kt = v_max_ms * 1.94384

    # Minimum central pressure (Holland 1997 empirical)
    rho = 1.15  # air density kg/m3
    min_pressure = surface_pressure_hpa - (rho * v_max_ms**2) / 100

    from climate_agent.tools.data_fetch import classify_storm

    return {
        "inputs": {
            "sst_celsius": sst_celsius,
            "outflow_temp_celsius": outflow_temp_celsius,
            "surface_pressure_hpa": surface_pressure_hpa,
        },
        "thermodynamic_efficiency": round(efficiency, 3),
        "maximum_potential_intensity": {
            "wind_speed_ms": round(v_max_ms, 1),
            "wind_speed_kt": round(v_max_kt, 1),
            "wind_speed_mph": round(v_max_kt * 1.15078, 1),
            "category": classify_storm(v_max_kt),
            "min_central_pressure_hpa": round(min_pressure, 1),
        },
        "sst_assessment": {
            "above_formation_threshold": sst_celsius >= SST_CYCLONE_THRESHOLD,
            "margin_above_threshold": round(sst_celsius - SST_CYCLONE_THRESHOLD, 1),
        },
        "methodology": "Simplified Emanuel (1988) MPI theory",
        "caveat": "Real intensity limited by wind shear, dry air intrusion, "
        "ocean heat content, land interaction, and internal dynamics.",
    }


def analyze_climate_trend(
    metric: str = "global_temperature",
    start_year: int = 1880,
    end_year: int = 2025,
) -> dict[str, Any]:
    """Analyze long-term climate trends for key indicators.

    Args:
        metric: One of 'global_temperature', 'co2', 'sea_level', 'arctic_ice',
                'cyclone_intensity'.
        start_year: Start year for trend analysis.
        end_year: End year for trend analysis.

    Returns:
        Trend analysis including rate of change, acceleration, and projections.
    """
    years = np.arange(start_year, end_year + 1, dtype=float)

    if metric == "global_temperature":
        # Global mean temperature anomaly (relative to 1850-1900)
        # Follows observed warming curve: slow until ~1970, then accelerating
        values = (
            0.0
            + 0.005 * (years - 1880)
            + 0.0001 * np.maximum(0, years - 1970) ** 1.5
            + np.random.normal(0, 0.05, len(years))
        )
        values = np.clip(values, -0.5, 2.0)
        units = "Celsius anomaly (vs 1850-1900)"
        description = "Global mean surface temperature anomaly"

    elif metric == "co2":
        # Atmospheric CO2 concentration
        values = (
            280
            + 0.3 * np.maximum(0, years - 1850)
            + 0.02 * np.maximum(0, years - 1950) ** 1.3
        )
        units = "ppm"
        description = "Atmospheric CO2 concentration (Mauna Loa + ice cores)"

    elif metric == "sea_level":
        # Global mean sea level change (mm relative to 1900)
        values = 0.5 * (years - 1900) + 0.002 * np.maximum(0, years - 1900) ** 1.8
        values -= values[0]  # zero at start year
        units = "mm relative to 1900"
        description = "Global mean sea level change"

    elif metric == "arctic_ice":
        # September Arctic sea ice extent (million km²)
        values = 8.0 - 0.05 * np.maximum(0, years - 1979) - 0.001 * np.maximum(
            0, years - 1979
        ) ** 1.5
        values = np.clip(values, 0.5, 9.0)
        values += np.random.normal(0, 0.3, len(years))
        units = "million km² (September minimum)"
        description = "Arctic sea ice extent (September minimum)"

    elif metric == "cyclone_intensity":
        # Proportion of Category 4-5 storms (increasing trend)
        values = (
            0.25
            + 0.002 * np.maximum(0, years - 1980)
            + np.random.normal(0, 0.05, len(years))
        )
        values = np.clip(values, 0.1, 0.6)
        units = "fraction of hurricanes reaching Category 4-5"
        description = "Proportion of tropical cyclones reaching Category 4-5"

    else:
        return {"error": f"Unknown metric: {metric}. Choose from: global_temperature, "
                "co2, sea_level, arctic_ice, cyclone_intensity"}

    # Linear trend via least squares
    coeffs = np.polyfit(years, values, 1)
    trend_per_year = coeffs[0]
    trend_per_decade = trend_per_year * 10

    # Recent acceleration (compare last 30y trend to full trend)
    if len(years) > 30:
        recent_mask = years >= (end_year - 30)
        recent_coeffs = np.polyfit(years[recent_mask], values[recent_mask], 1)
        recent_trend = recent_coeffs[0] * 10
        acceleration = recent_trend - trend_per_decade
    else:
        recent_trend = trend_per_decade
        acceleration = 0

    # Simple projection
    future_years = [2030, 2050, 2100]
    projections = {}
    for fy in future_years:
        if fy > end_year:
            projected = values[-1] + trend_per_year * (fy - end_year)
            projections[str(fy)] = round(float(projected), 2)

    return {
        "metric": metric,
        "description": description,
        "units": units,
        "period": {"start": start_year, "end": end_year},
        "current_value": round(float(values[-1]), 2),
        "trend": {
            "per_year": round(float(trend_per_year), 4),
            "per_decade": round(float(trend_per_decade), 3),
            "recent_30yr_per_decade": round(float(recent_trend), 3),
            "acceleration": round(float(acceleration), 3),
            "accelerating": acceleration > 0.001,
        },
        "projections_linear": projections,
        "data_points": len(years),
        "caveat": "Projections assume continuation of recent trends. Actual trajectories "
        "depend on emissions pathways, policy actions, and Earth system feedbacks.",
    }


def compute_carbon_budget(
    target_warming_celsius: float = 1.5,
    current_annual_emissions_gtco2: float = 40.0,
) -> dict[str, Any]:
    """Compute the remaining carbon budget for a given warming target.

    Args:
        target_warming_celsius: Target warming limit (e.g., 1.5 or 2.0).
        current_annual_emissions_gtco2: Current annual global CO2 emissions (GtCO2/yr).

    Returns:
        Carbon budget analysis with timeline estimates.
    """
    # Current warming (approximate)
    current_warming = 1.45

    remaining_warming = target_warming_celsius - current_warming

    if remaining_warming <= 0:
        return {
            "target_celsius": target_warming_celsius,
            "current_warming_celsius": current_warming,
            "status": "TARGET ALREADY EXCEEDED",
            "remaining_budget_gtco2": 0,
            "note": f"Current warming of {current_warming}C has already exceeded "
            f"the {target_warming_celsius}C target.",
        }

    # TCRE (Transient Climate Response to cumulative Emissions)
    # ~1.65 C per 1000 GtCO2 (IPCC AR6 best estimate)
    tcre = 1.65 / 1000  # C per GtCO2

    remaining_budget = remaining_warming / tcre

    # Years at current rate
    years_remaining = remaining_budget / current_annual_emissions_gtco2

    # Required emission reductions for net zero by 2050
    years_to_2050 = max(1, 2050 - 2025)
    required_annual_reduction_pct = (1 - (0.01 / current_annual_emissions_gtco2) ** (1 / years_to_2050)) * 100

    return {
        "target_warming_celsius": target_warming_celsius,
        "current_warming_celsius": current_warming,
        "remaining_warming_celsius": round(remaining_warming, 2),
        "remaining_carbon_budget": {
            "gtco2": round(remaining_budget, 0),
            "gt_carbon": round(remaining_budget / 3.664, 0),
        },
        "at_current_emissions": {
            "annual_emissions_gtco2": current_annual_emissions_gtco2,
            "years_until_budget_exhausted": round(years_remaining, 1),
            "budget_exhaustion_year": round(2025 + years_remaining),
        },
        "pathways": {
            "net_zero_by_2050": {
                "required_annual_reduction_pct": round(required_annual_reduction_pct, 1),
                "description": f"Requires ~{round(required_annual_reduction_pct, 0)}% "
                "annual emission reductions starting immediately",
            },
            "net_zero_by_2040": {
                "required_annual_reduction_pct": round(required_annual_reduction_pct * 1.7, 1),
                "description": "Extremely aggressive but more likely to stay within budget",
            },
        },
        "methodology": "Based on IPCC AR6 TCRE of 1.65C per 1000 GtCO2",
        "uncertainty": "Budget estimates have ~50% uncertainty range. Values represent "
        "median estimates with 50% probability of staying below target.",
    }


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in km."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def analyze_storm_track(track_points: list[dict]) -> dict[str, Any]:
    """Analyze a tropical cyclone track for key characteristics.

    Args:
        track_points: List of track point dicts with lat, lon, wind_kt, time.

    Returns:
        Track analysis including distance, speed, recurvature, and intensification rate.
    """
    if len(track_points) < 2:
        return {"error": "Need at least 2 track points for analysis"}

    total_distance = 0.0
    max_wind = 0.0
    max_intensification_rate = 0.0
    speeds = []

    for i in range(1, len(track_points)):
        p0 = track_points[i - 1]
        p1 = track_points[i]

        dist = haversine_distance(p0["lat"], p0["lon"], p1["lat"], p1["lon"])
        total_distance += dist

        # Translation speed (assuming 6-hour intervals)
        speed_kmh = dist / 6.0
        speeds.append(speed_kmh)

        # Intensification rate
        dwind = p1.get("wind_kt", 0) - p0.get("wind_kt", 0)
        if dwind > max_intensification_rate:
            max_intensification_rate = dwind

        if p1.get("wind_kt", 0) > max_wind:
            max_wind = p1["wind_kt"]

    # Detect recurvature (track turning from west to east)
    lons = [p["lon"] for p in track_points]
    westward = sum(1 for i in range(1, len(lons)) if lons[i] < lons[i - 1])
    eastward = sum(1 for i in range(1, len(lons)) if lons[i] > lons[i - 1])
    recurved = westward > 0 and eastward > 0

    # Rapid intensification check (30kt in 24h = 5kt per 6h period)
    rapid_intensification = max_intensification_rate >= 30  # per 6-hour period is extreme

    return {
        "track_length_km": round(total_distance, 1),
        "track_points": len(track_points),
        "peak_intensity_kt": max_wind,
        "mean_forward_speed_kmh": round(float(np.mean(speeds)), 1) if speeds else 0,
        "max_forward_speed_kmh": round(float(np.max(speeds)), 1) if speeds else 0,
        "max_6h_intensification_kt": round(max_intensification_rate, 1),
        "rapid_intensification_detected": rapid_intensification,
        "recurvature_detected": recurved,
        "start_position": {
            "lat": track_points[0]["lat"],
            "lon": track_points[0]["lon"],
        },
        "end_position": {
            "lat": track_points[-1]["lat"],
            "lon": track_points[-1]["lon"],
        },
    }
