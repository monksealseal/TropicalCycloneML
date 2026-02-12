"""
Configuration for government data sources and HuggingFace endpoints.
"""

# ---------------------------------------------------------------------------
# HuggingFace – monkseal555 account
# ---------------------------------------------------------------------------
HF_USER = "monkseal555"

# Known models / spaces (discovered via README + HF search)
HF_MODELS = {
    "autotrain-hurricane3": f"{HF_USER}/autotrain-hurricane3-1415853436",
    "weatherflow": f"{HF_USER}/weatherflow-Flowing-0230a87e",
}

HF_SPACES = {
    "testing": f"{HF_USER}/testing",
    "testing2": f"{HF_USER}/testing2",
}

HF_INFERENCE_API_URL = "https://api-inference.huggingface.co/models/{model_id}"
HF_SPACES_API_URL = "https://{space_id}.hf.space/api/predict"
HF_API_URL = "https://huggingface.co/api"

# ---------------------------------------------------------------------------
# NHC / NOAA – Real-time government data
# ---------------------------------------------------------------------------
NHC_CURRENT_STORMS_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"

NHC_ATCF_BASE = "https://ftp.nhc.noaa.gov/atcf"
NHC_ATCF_BEST_TRACK_URL = f"{NHC_ATCF_BASE}/btk/"          # B-deck files
NHC_ATCF_AID_URL = f"{NHC_ATCF_BASE}/aid_public/"           # A-deck files
NHC_ATCF_FIX_URL = f"{NHC_ATCF_BASE}/fix/"                  # F-deck files

NHC_GIS_URL = "https://www.nhc.noaa.gov/gis/"
NHC_RSS_ATL_URL = "https://www.nhc.noaa.gov/nhc_at5.xml"
NHC_RSS_EPAC_URL = "https://www.nhc.noaa.gov/nhc_ep5.xml"
NHC_RSS_CPAC_URL = "https://www.nhc.noaa.gov/nhc_cp5.xml"

# ArcGIS REST endpoints (support JSON / GeoJSON queries)
ARCGIS_ATL_ACTIVE = (
    "https://idpgis.ncep.noaa.gov/arcgis/rest/services/"
    "NWS_Forecasts_Guidance_Warnings/NHC_Atl_trop_cyclones_active/MapServer"
)
ARCGIS_EPAC_ACTIVE = (
    "https://idpgis.ncep.noaa.gov/arcgis/rest/services/"
    "NWS_Forecasts_Guidance_Warnings/NHC_E_Pac_trop_cyclones_active/MapServer"
)
ARCGIS_TROPICAL_WEATHER = (
    "https://mapservices.weather.noaa.gov/tropical/rest/services/"
    "tropical/NHC_tropical_weather/MapServer"
)

# Layer indices commonly used in the ArcGIS MapServer
ARCGIS_LAYERS = {
    "forecast_track": 0,
    "forecast_point": 1,
    "forecast_cone": 2,
    "watches_warnings": 3,
    "wind_radii_34kt": 4,
    "wind_radii_50kt": 5,
    "wind_radii_64kt": 6,
    "past_track": 7,
    "past_points": 8,
}

# ---------------------------------------------------------------------------
# IBTrACS – International Best Track Archive (updated ~3×/week)
# ---------------------------------------------------------------------------
IBTRACS_BASE = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv"
IBTRACS_LAST3_URL = f"{IBTRACS_BASE}/ibtracs.last3years.list.v04r01.csv"
IBTRACS_ACTIVE_URL = f"{IBTRACS_BASE}/ibtracs.ACTIVE.list.v04r01.csv"
IBTRACS_ALL_URL = f"{IBTRACS_BASE}/ibtracs.ALL.list.v04r01.csv"

# ---------------------------------------------------------------------------
# HURDAT2 – Historical hurricane database
# ---------------------------------------------------------------------------
HURDAT2_ATL_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"
HURDAT2_EPAC_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2024-031725.txt"

# ---------------------------------------------------------------------------
# JTWC – Joint Typhoon Warning Center
# ---------------------------------------------------------------------------
JTWC_RSS_URL = "https://www.metoc.navy.mil/jtwc/rss/jtwc.rss"
JTWC_BEST_TRACK_URL = "https://www.metoc.navy.mil/jtwc/products/best-tracks/"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_BASIN = "AL"  # Atlantic
USER_AGENT = "TropicalCycloneML-HFLiveTool/0.1"
