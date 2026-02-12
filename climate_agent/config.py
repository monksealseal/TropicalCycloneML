"""Configuration constants for the climate agent."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data"

# NOAA API endpoints
NOAA_HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
NOAA_ACTIVE_CYCLONES_URL = "https://www.nhc.noaa.gov/CurrentSummaries.json"
NOAA_GFS_BASE_URL = "https://nomads.ncep.noaa.gov/dods/gfs_0p25"
NOAA_SST_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"
NOAA_OISST_OPENDAP = "https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR"

# IBTrACS (International Best Track Archive for Climate Stewardship)
IBTRACS_CSV_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
IBTRACS_LAST3_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.last3years.list.v04r01.csv"

# Global carbon data
GLOBAL_CARBON_BUDGET_URL = "https://globalcarbonbudgetdata.org/latest-data.html"

# Physical constants
EARTH_RADIUS_KM = 6371.0
CORIOLIS_PARAM_EQUATOR = 0.0  # f = 2*omega*sin(lat)
OMEGA_EARTH = 7.2921e-5  # rad/s

# Saffir-Simpson Hurricane Wind Scale (knots)
SAFFIR_SIMPSON = {
    "Tropical Depression": (0, 33),
    "Tropical Storm": (34, 63),
    "Category 1": (64, 82),
    "Category 2": (83, 95),
    "Category 3": (96, 112),
    "Category 4": (113, 136),
    "Category 5": (137, float("inf")),
}

# Sea Surface Temperature thresholds (Celsius)
SST_CYCLONE_THRESHOLD = 26.5  # Minimum SST for tropical cyclone formation
SST_RAPID_INTENSIFICATION = 28.5  # SST associated with rapid intensification

# Model parameters
DEFAULT_MODEL_INPUT_SIZE = 256
DEFAULT_FORECAST_HOURS = [12, 24, 48, 72, 120]

# Agent configuration
AGENT_MODEL = "claude-sonnet-4-20250514"
MAX_AGENT_TURNS = 50
AGENT_SYSTEM_PROMPT = """You are an autonomous climate science agent specializing in tropical \
cyclone analysis and climate change research. You have access to tools for fetching real-world \
climate data, running ML-based forecasts, analyzing atmospheric conditions, and generating \
visualizations.

Your capabilities include:
- Fetching and parsing NOAA hurricane track data (HURDAT2, IBTrACS)
- Analyzing sea surface temperatures and their relationship to cyclone intensity
- Computing atmospheric metrics (wind shear, CAPE, relative humidity)
- Running ML models for cyclone intensity and track prediction
- Analyzing long-term climate trends (SST warming, cyclone frequency changes)
- Computing carbon emission estimates and climate impact metrics
- Generating maps and time-series visualizations of climate data

When given a task, break it down into steps and use the available tools methodically. \
Always cite data sources and explain your reasoning. If data is unavailable, explain what \
would be needed and provide the best analysis possible with available information.

You are thorough, scientifically rigorous, and transparent about uncertainties."""
