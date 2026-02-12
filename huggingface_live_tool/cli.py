#!/usr/bin/env python3
"""
Command-line interface for the HuggingFace Live Data Tool.

Usage examples
--------------
# List active tropical cyclones
python -m huggingface_live_tool storms

# Get a full real-time snapshot (all government sources)
python -m huggingface_live_tool snapshot --basin AL

# List all HuggingFace resources for monkseal555
python -m huggingface_live_tool hf-list

# Run inference on active storms through a HuggingFace model
python -m huggingface_live_tool infer --model weatherflow --basin AL

# Run inference using ATCF best-track data for a specific storm
python -m huggingface_live_tool infer-atcf --storm al052024 --model weatherflow

# Run inference using IBTrACS data
python -m huggingface_live_tool infer-ibtracs --storm MILTON --model weatherflow

# Send live data to a Gradio Space
python -m huggingface_live_tool space --space testing --basin AL

# Fetch NHC forecast track GeoJSON
python -m huggingface_live_tool forecast-track --basin AL

# Fetch NHC advisories (RSS)
python -m huggingface_live_tool advisories --basin AL

# Monitor mode: poll and run inference every 5 minutes
python -m huggingface_live_tool monitor --model weatherflow --interval 300

# Full analysis report
python -m huggingface_live_tool full-analysis --model weatherflow --basin AL
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from . import government_data as gov
from . import huggingface_client as hf
from . import pipeline
from .config import HF_MODELS, HF_SPACES, HF_USER


def _json_out(data: Any) -> None:
    """Pretty-print data as JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


def cmd_storms(args: argparse.Namespace) -> None:
    """List currently active tropical cyclones."""
    storms = gov.list_active_storms()
    if not storms:
        print("No active tropical cyclones reported by NHC.")
        return
    _json_out(storms)


def cmd_snapshot(args: argparse.Namespace) -> None:
    """Fetch a combined real-time data snapshot."""
    _json_out(gov.get_realtime_snapshot(args.basin))


def cmd_forecast_track(args: argparse.Namespace) -> None:
    """Fetch NHC ArcGIS forecast track GeoJSON."""
    _json_out(gov.fetch_forecast_track(args.basin))


def cmd_forecast_cone(args: argparse.Namespace) -> None:
    """Fetch NHC ArcGIS forecast cone GeoJSON."""
    _json_out(gov.fetch_forecast_cone(args.basin))


def cmd_wind_radii(args: argparse.Namespace) -> None:
    """Fetch wind radii GeoJSON."""
    _json_out(gov.fetch_wind_radii(args.basin, args.threshold))


def cmd_advisories(args: argparse.Namespace) -> None:
    """Fetch NHC RSS advisories."""
    _json_out(gov.fetch_nhc_rss(args.basin))


def cmd_atcf_list(args: argparse.Namespace) -> None:
    """List available ATCF best-track files."""
    files = gov.list_atcf_best_track_files()
    for f in files:
        print(f)
    print(f"\n{len(files)} files available")


def cmd_atcf_track(args: argparse.Namespace) -> None:
    """Fetch ATCF best-track data for a storm."""
    _json_out(gov.fetch_atcf_best_track(args.storm))


def cmd_ibtracs(args: argparse.Namespace) -> None:
    """Fetch IBTrACS data."""
    rows = gov.fetch_ibtracs(dataset=args.dataset, max_rows=args.limit)
    if args.storm:
        rows = [r for r in rows if args.storm.upper() in (r.get("NAME") or "").upper()]
    _json_out(rows)
    print(f"\n{len(rows)} records", file=sys.stderr)


def cmd_hurdat2(args: argparse.Namespace) -> None:
    """Fetch HURDAT2 data (historical)."""
    storms = gov.fetch_hurdat2(args.basin)
    if args.storm:
        storms = [s for s in storms if args.storm.upper() in s.get("name", "").upper()]
    _json_out(storms[-args.limit:] if args.limit else storms)
    print(f"\n{len(storms)} storms total", file=sys.stderr)


def cmd_hf_list(args: argparse.Namespace) -> None:
    """List all HuggingFace resources for monkseal555."""
    user = args.user or HF_USER
    result = hf.discover_all(user)

    print(f"=== HuggingFace resources for {user} ===\n")

    print("Models:")
    for m in result.get("models", []):
        print(f"  - {m.get('modelId', m.get('id', '?'))}")
    if not result.get("models"):
        print("  (none found)")

    print("\nSpaces:")
    for s in result.get("spaces", []):
        print(f"  - {s.get('id', '?')}")
    if not result.get("spaces"):
        print("  (none found)")

    print("\nDatasets:")
    for d in result.get("datasets", []):
        print(f"  - {d.get('id', '?')}")
    if not result.get("datasets"):
        print("  (none found)")

    print("\nKnown model aliases:")
    for alias, full_id in HF_MODELS.items():
        print(f"  {alias} → {full_id}")

    print("\nKnown space aliases:")
    for alias, full_id in HF_SPACES.items():
        print(f"  {alias} → {full_id}")


def cmd_hf_model_info(args: argparse.Namespace) -> None:
    """Show metadata for a HuggingFace model."""
    model_id = HF_MODELS.get(args.model, args.model)
    _json_out(hf.get_model_info(model_id))


def cmd_infer(args: argparse.Namespace) -> None:
    """Run live inference on active storms."""
    _json_out(pipeline.run_live_inference(
        model_id=args.model,
        basin=args.basin,
        token=args.token,
    ))


def cmd_infer_forecast(args: argparse.Namespace) -> None:
    """Run inference on NHC forecast track data."""
    _json_out(pipeline.run_forecast_inference(
        model_id=args.model,
        basin=args.basin,
        token=args.token,
    ))


def cmd_infer_atcf(args: argparse.Namespace) -> None:
    """Run inference on ATCF best-track data for a specific storm."""
    _json_out(pipeline.run_atcf_inference(
        storm_id=args.storm,
        model_id=args.model,
        token=args.token,
    ))


def cmd_infer_ibtracs(args: argparse.Namespace) -> None:
    """Run inference on IBTrACS data."""
    _json_out(pipeline.run_ibtracs_inference(
        model_id=args.model,
        dataset=args.dataset,
        storm_name=args.storm,
        max_rows=args.limit,
        token=args.token,
    ))


def cmd_space(args: argparse.Namespace) -> None:
    """Send live data to a Gradio Space."""
    _json_out(pipeline.run_space_with_live_data(
        space_id=args.space,
        basin=args.basin,
        token=args.token,
    ))


def cmd_monitor(args: argparse.Namespace) -> None:
    """Monitor mode: poll and run inference periodically."""
    pipeline.monitor(
        model_id=args.model,
        basin=args.basin,
        interval_seconds=args.interval,
        max_iterations=args.iterations,
        token=args.token,
    )


def cmd_full_analysis(args: argparse.Namespace) -> None:
    """Run a comprehensive analysis with all data sources."""
    _json_out(pipeline.full_analysis(
        model_id=args.model,
        basin=args.basin,
        token=args.token,
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="huggingface_live_tool",
        description=(
            "Fetch real-time tropical cyclone data from government sources "
            "and run inference through HuggingFace models/spaces (monkseal555)."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--token", default=None,
        help="HuggingFace API token (or set $HF_TOKEN)",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # -- Government data commands --
    sub.add_parser("storms", help="List active tropical cyclones (NHC)")

    p = sub.add_parser("snapshot", help="Real-time data snapshot")
    p.add_argument("--basin", default="AL", help="Basin: AL, EP, CP")

    p = sub.add_parser("forecast-track", help="NHC forecast track GeoJSON")
    p.add_argument("--basin", default="AL")

    p = sub.add_parser("forecast-cone", help="NHC forecast cone GeoJSON")
    p.add_argument("--basin", default="AL")

    p = sub.add_parser("wind-radii", help="NHC wind radii GeoJSON")
    p.add_argument("--basin", default="AL")
    p.add_argument("--threshold", type=int, default=34, choices=[34, 50, 64])

    p = sub.add_parser("advisories", help="NHC RSS advisories")
    p.add_argument("--basin", default="AL")

    sub.add_parser("atcf-list", help="List ATCF best-track files")

    p = sub.add_parser("atcf-track", help="Fetch ATCF best-track for a storm")
    p.add_argument("--storm", required=True, help="Storm ID (e.g. al052024)")

    p = sub.add_parser("ibtracs", help="Fetch IBTrACS data")
    p.add_argument("--dataset", default="last3years", choices=["last3years", "active", "all"])
    p.add_argument("--storm", default=None, help="Filter by storm name")
    p.add_argument("--limit", type=int, default=100)

    p = sub.add_parser("hurdat2", help="Fetch HURDAT2 historical data")
    p.add_argument("--basin", default="AL")
    p.add_argument("--storm", default=None, help="Filter by storm name")
    p.add_argument("--limit", type=int, default=10)

    # -- HuggingFace commands --
    p = sub.add_parser("hf-list", help="List HuggingFace resources")
    p.add_argument("--user", default=None)

    p = sub.add_parser("hf-model-info", help="Show model metadata")
    p.add_argument("--model", required=True, help="Model ID or alias")

    # -- Inference commands --
    p = sub.add_parser("infer", help="Run live inference on active storms")
    p.add_argument("--model", default="weatherflow")
    p.add_argument("--basin", default="AL")

    p = sub.add_parser("infer-forecast", help="Run inference on forecast track")
    p.add_argument("--model", default="weatherflow")
    p.add_argument("--basin", default="AL")

    p = sub.add_parser("infer-atcf", help="Run inference on ATCF best-track")
    p.add_argument("--storm", required=True, help="Storm ID (e.g. al052024)")
    p.add_argument("--model", default="weatherflow")

    p = sub.add_parser("infer-ibtracs", help="Run inference on IBTrACS data")
    p.add_argument("--model", default="weatherflow")
    p.add_argument("--dataset", default="last3years")
    p.add_argument("--storm", default=None, help="Filter by storm name")
    p.add_argument("--limit", type=int, default=500)

    p = sub.add_parser("space", help="Send live data to a Gradio Space")
    p.add_argument("--space", default="testing")
    p.add_argument("--basin", default="AL")

    p = sub.add_parser("monitor", help="Monitor mode: poll & infer periodically")
    p.add_argument("--model", default="weatherflow")
    p.add_argument("--basin", default="AL")
    p.add_argument("--interval", type=int, default=300, help="Poll interval in seconds")
    p.add_argument("--iterations", type=int, default=None, help="Max iterations (None=forever)")

    p = sub.add_parser("full-analysis", help="Comprehensive analysis report")
    p.add_argument("--model", default="weatherflow")
    p.add_argument("--basin", default="AL")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    commands = {
        "storms": cmd_storms,
        "snapshot": cmd_snapshot,
        "forecast-track": cmd_forecast_track,
        "forecast-cone": cmd_forecast_cone,
        "wind-radii": cmd_wind_radii,
        "advisories": cmd_advisories,
        "atcf-list": cmd_atcf_list,
        "atcf-track": cmd_atcf_track,
        "ibtracs": cmd_ibtracs,
        "hurdat2": cmd_hurdat2,
        "hf-list": cmd_hf_list,
        "hf-model-info": cmd_hf_model_info,
        "infer": cmd_infer,
        "infer-forecast": cmd_infer_forecast,
        "infer-atcf": cmd_infer_atcf,
        "infer-ibtracs": cmd_infer_ibtracs,
        "space": cmd_space,
        "monitor": cmd_monitor,
        "full-analysis": cmd_full_analysis,
    }

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
