#!/usr/bin/env python3
"""
OncologyML — One command to run everything.

Usage:
    python run.py              # Run analysis + print report
    python run.py --web        # Launch web dashboard at http://localhost:8000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="OncologyML — Cancer Research ML Agent",
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Launch the web dashboard instead of CLI analysis",
    )
    args = parser.parse_args()

    if args.web:
        print()
        print("  OncologyML Web Dashboard")
        print("  Open http://localhost:8000 in your browser")
        print("  Press Ctrl+C to stop")
        print()
        import uvicorn
        from cancer_agent.service.app import create_app
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        from cancer_agent.agent import CancerResearchAgent
        agent = CancerResearchAgent()
        agent.run()


if __name__ == "__main__":
    main()
