"""ML-based forecasting tools for tropical cyclone prediction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from climate_agent.config import DEFAULT_FORECAST_HOURS
from climate_agent.tools.data_fetch import classify_storm


def forecast_cyclone_intensity(
    current_wind_kt: float,
    current_pressure_hpa: float,
    lat: float,
    lon: float,
    sst_celsius: float,
    wind_shear_ms: float,
    forecast_hours: list[int] | None = None,
) -> dict[str, Any]:
    """Forecast tropical cyclone intensity using a statistical-dynamical model.

    Uses a simplified SHIPS-like (Statistical Hurricane Intensity Prediction Scheme)
    approach combining environmental predictors with current storm state.

    Args:
        current_wind_kt: Current maximum sustained winds in knots.
        current_pressure_hpa: Current minimum central pressure in hPa.
        lat: Current latitude.
        lon: Current longitude.
        sst_celsius: Sea surface temperature at current location (Celsius).
        wind_shear_ms: 200-850 hPa wind shear magnitude (m/s).
        forecast_hours: List of forecast lead times in hours.

    Returns:
        Intensity forecast at each lead time.
    """
    if forecast_hours is None:
        forecast_hours = DEFAULT_FORECAST_HOURS

    # SHIPS-like predictors
    sst_potential = max(0, sst_celsius - 26.5) * 8.0  # SST surplus contribution
    shear_penalty = -2.5 * max(0, wind_shear_ms - 5.0)  # High shear weakens storms
    latitude_factor = -0.5 * max(0, abs(lat) - 25)  # Higher latitude = cooler water

    # Maximum Potential Intensity (simplified)
    mpi_kt = min(165, 50 + sst_potential * 5)

    # Intensity tendency per 12 hours
    intensity_gap = mpi_kt - current_wind_kt
    base_tendency = 0.08 * intensity_gap + shear_penalty + latitude_factor

    forecasts = []
    wind = current_wind_kt
    pressure = current_pressure_hpa

    for hours in sorted(forecast_hours):
        steps = hours / 12.0
        # Apply tendency with diminishing returns and noise
        delta_wind = base_tendency * steps * (1 - 0.3 * steps / 10)
        forecast_wind = max(15, min(mpi_kt, wind + delta_wind))

        # Pressure-wind relationship (Knaff & Zehr 2007 simplified)
        forecast_pressure = 1015 - (forecast_wind / 165) ** 2 * 100

        # Uncertainty grows with lead time
        uncertainty_kt = 5 + 2.5 * (hours / 12)

        forecasts.append({
            "forecast_hour": hours,
            "wind_kt": round(forecast_wind, 0),
            "wind_mph": round(forecast_wind * 1.15078, 0),
            "pressure_hpa": round(forecast_pressure, 0),
            "category": classify_storm(forecast_wind),
            "uncertainty_kt": round(uncertainty_kt, 0),
            "wind_range_kt": [
                round(max(15, forecast_wind - uncertainty_kt), 0),
                round(min(185, forecast_wind + uncertainty_kt), 0),
            ],
        })

    # Rapid intensification probability
    ri_prob = _estimate_ri_probability(current_wind_kt, sst_celsius, wind_shear_ms, lat)

    return {
        "initial_conditions": {
            "wind_kt": current_wind_kt,
            "pressure_hpa": current_pressure_hpa,
            "lat": lat,
            "lon": lon,
            "sst_celsius": sst_celsius,
            "wind_shear_ms": wind_shear_ms,
        },
        "maximum_potential_intensity_kt": round(mpi_kt, 0),
        "forecasts": forecasts,
        "rapid_intensification": {
            "probability_24h": round(ri_prob, 2),
            "threshold": "30 kt increase in 24 hours",
            "assessment": (
                "High RI risk" if ri_prob > 0.4
                else "Moderate RI risk" if ri_prob > 0.2
                else "Low RI risk"
            ),
        },
        "model": "Statistical-dynamical (SHIPS-like)",
        "disclaimer": "This is a simplified forecast model for demonstration. "
        "Operational forecasts should use NHC official products.",
    }


def _estimate_ri_probability(
    wind_kt: float, sst: float, shear: float, lat: float
) -> float:
    """Estimate probability of rapid intensification in next 24 hours."""
    # Logistic regression-like model based on RI predictors
    score = 0.0
    score += 0.15 * max(0, sst - 28.0)  # Warm SSTs favor RI
    score -= 0.08 * max(0, shear - 5.0)  # High shear inhibits RI
    score += 0.01 * max(0, 100 - wind_kt)  # Weaker storms have more room to intensify
    score -= 0.02 * max(0, abs(lat) - 20)  # Higher latitude inhibits RI
    # Sigmoid transformation
    prob = 1.0 / (1.0 + math.exp(-score))
    return min(0.85, max(0.02, prob))


def forecast_cyclone_track(
    lat: float,
    lon: float,
    current_heading_deg: float = 315.0,
    current_speed_kmh: float = 20.0,
    forecast_hours: list[int] | None = None,
) -> dict[str, Any]:
    """Forecast tropical cyclone track using a simplified beta-advection model.

    Simulates storm motion as a combination of steering flow and beta drift,
    with recurvature at higher latitudes.

    Args:
        lat: Current latitude.
        lon: Current longitude.
        current_heading_deg: Current heading in degrees (0=N, 90=E, 180=S, 270=W).
        current_speed_kmh: Current forward speed in km/h.
        forecast_hours: List of forecast lead times.

    Returns:
        Track forecast at each lead time.
    """
    if forecast_hours is None:
        forecast_hours = DEFAULT_FORECAST_HOURS

    forecasts = []
    curr_lat = lat
    curr_lon = lon
    heading = math.radians(current_heading_deg)
    speed = current_speed_kmh

    # Time step: 6 hours
    dt_hours = 6.0

    for target_hour in sorted(forecast_hours):
        # Integrate forward to target hour
        while True:
            current_time = len(forecasts) * dt_hours if forecasts else 0
            if current_time >= target_hour:
                break

            # Beta drift: poleward and westward component
            beta_lat = 0.3  # deg per 6h poleward drift
            beta_lon = -0.2  # deg per 6h westward drift (beta gyre)

            # Recurvature: storms tend to turn NE at higher latitudes
            if abs(curr_lat) > 25:
                recurvature_factor = 0.05 * (abs(curr_lat) - 25)
                heading += recurvature_factor * 0.1  # Turn eastward
                speed *= 1.02  # Accelerate during recurvature

            # Steering flow displacement
            dlat = (speed * dt_hours / 111.0) * math.cos(heading) + beta_lat * 0.1
            dlon = (speed * dt_hours / (111.0 * math.cos(math.radians(curr_lat)))) * math.sin(
                heading
            ) + beta_lon * 0.1

            curr_lat += dlat
            curr_lon += dlon

            # Uncertainty cone radius (grows with time)
            hours_out = current_time + dt_hours
            uncertainty_km = 50 + 15 * hours_out  # ~15 km/h growth

        forecasts.append({
            "forecast_hour": target_hour,
            "lat": round(curr_lat, 2),
            "lon": round(curr_lon, 2),
            "uncertainty_radius_km": round(50 + 15 * target_hour, 0),
            "heading_deg": round(math.degrees(heading) % 360, 1),
            "speed_kmh": round(speed, 1),
        })

    return {
        "initial_position": {"lat": lat, "lon": lon},
        "initial_motion": {
            "heading_deg": current_heading_deg,
            "speed_kmh": current_speed_kmh,
        },
        "track_forecast": forecasts,
        "cone_of_uncertainty": {
            "description": "The forecast cone represents the probable track of the "
            "storm center. The actual track can fall anywhere within the cone.",
            "historical_accuracy": {
                "24h_mean_error_km": 80,
                "48h_mean_error_km": 150,
                "72h_mean_error_km": 230,
                "120h_mean_error_km": 380,
            },
        },
        "model": "Simplified beta-advection with climatological recurvature",
        "disclaimer": "This is a demonstration model. Use NHC official forecasts "
        "for actual tropical cyclone track guidance.",
    }


def assess_climate_risk(
    location_lat: float,
    location_lon: float,
    scenario: str = "SSP2-4.5",
    time_horizon: int = 2050,
) -> dict[str, Any]:
    """Assess climate change risks for a specific location.

    Evaluates multiple climate hazards including tropical cyclones, sea level rise,
    extreme heat, and flooding risk under different emission scenarios.

    Args:
        location_lat: Latitude of location.
        location_lon: Longitude of location.
        scenario: IPCC SSP scenario ('SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5').
        time_horizon: Target year for risk assessment.

    Returns:
        Multi-hazard climate risk assessment.
    """
    scenario_warming = {
        "SSP1-2.6": {"2030": 1.5, "2050": 1.7, "2100": 1.8},
        "SSP2-4.5": {"2030": 1.5, "2050": 2.0, "2100": 2.7},
        "SSP3-7.0": {"2030": 1.5, "2050": 2.1, "2100": 3.6},
        "SSP5-8.5": {"2030": 1.6, "2050": 2.4, "2100": 4.4},
    }

    if scenario not in scenario_warming:
        return {"error": f"Unknown scenario: {scenario}. Use SSP1-2.6, SSP2-4.5, SSP3-7.0, or SSP5-8.5"}

    # Get warming for closest decade
    decade_key = str(min(2100, max(2030, (time_horizon // 10) * 10)))
    if decade_key not in scenario_warming[scenario]:
        decade_key = "2050"
    warming = scenario_warming[scenario][decade_key]

    abs_lat = abs(location_lat)
    is_coastal = True  # Simplified; in production would use GIS coastline data
    is_tropical = abs_lat < 30
    is_small_island = abs_lat < 25 and is_coastal

    # Tropical cyclone risk
    tc_risk = "Low"
    if is_tropical and abs_lat > 5:
        tc_risk = "High" if abs_lat < 25 else "Moderate"

    # Sea level rise projection (mm)
    slr_base = {"SSP1-2.6": 300, "SSP2-4.5": 500, "SSP3-7.0": 700, "SSP5-8.5": 1000}
    slr_2100 = slr_base.get(scenario, 500)
    years_from_now = time_horizon - 2025
    slr_projected = slr_2100 * (years_from_now / 75) ** 1.2  # Accelerating

    # Extreme heat days (>35C)
    heat_days_base = max(0, 30 - abs_lat) * 2  # More in tropics
    heat_days_projected = heat_days_base + warming * 15

    # Precipitation change
    precip_change_pct = warming * (5 if is_tropical else -3)

    risks = []
    if tc_risk in ("High", "Moderate"):
        risks.append({
            "hazard": "Tropical Cyclones",
            "risk_level": tc_risk,
            "projection": f"Proportion of Category 4-5 storms projected to increase "
            f"by {round(warming * 10)}% under {scenario}",
            "impact": "Stronger peak winds, heavier rainfall, higher storm surge",
        })

    risks.append({
        "hazard": "Sea Level Rise",
        "risk_level": "Critical" if slr_projected > 500 else "High" if slr_projected > 300 else "Moderate",
        "projection": f"{round(slr_projected)} mm rise by {time_horizon}",
        "impact": "Coastal flooding, saltwater intrusion, erosion",
    })

    risks.append({
        "hazard": "Extreme Heat",
        "risk_level": "High" if heat_days_projected > 50 else "Moderate" if heat_days_projected > 20 else "Low",
        "projection": f"{round(heat_days_projected)} days above 35C per year by {time_horizon}",
        "impact": "Heat stress, agricultural losses, energy demand increase",
    })

    risks.append({
        "hazard": "Precipitation Changes",
        "risk_level": "Moderate",
        "projection": f"{precip_change_pct:+.0f}% change in annual precipitation",
        "impact": "Flooding risk (tropics) or drought risk (subtropics)",
    })

    overall_risk = "Critical" if any(r["risk_level"] == "Critical" for r in risks) else \
                   "High" if any(r["risk_level"] == "High" for r in risks) else "Moderate"

    return {
        "location": {"lat": location_lat, "lon": location_lon},
        "scenario": scenario,
        "time_horizon": time_horizon,
        "global_warming_celsius": warming,
        "overall_risk_level": overall_risk,
        "hazard_assessments": risks,
        "adaptation_priorities": _get_adaptation_priorities(risks),
        "methodology": "Based on IPCC AR6 WG1/WG2 projections and CMIP6 ensemble",
    }


def _get_adaptation_priorities(risks: list[dict]) -> list[str]:
    """Generate adaptation priorities based on identified risks."""
    priorities = []
    for risk in risks:
        if risk["risk_level"] in ("High", "Critical"):
            if "Cyclone" in risk["hazard"]:
                priorities.append("Strengthen building codes and early warning systems")
                priorities.append("Invest in coastal protection infrastructure")
            elif "Sea Level" in risk["hazard"]:
                priorities.append("Develop managed retreat plans for low-lying areas")
                priorities.append("Upgrade stormwater and flood management systems")
            elif "Heat" in risk["hazard"]:
                priorities.append("Expand urban green infrastructure and cooling centers")
                priorities.append("Reform outdoor labor policies for extreme heat")
            elif "Precipitation" in risk["hazard"]:
                priorities.append("Improve water storage and drought resilience")
    if not priorities:
        priorities.append("Continue monitoring and maintain current adaptation measures")
    return priorities
