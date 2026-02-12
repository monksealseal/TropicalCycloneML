#!/usr/bin/env python3
"""Entry point for the Climate Analysis Agent.

Usage:
    # Interactive mode
    python run_agent.py

    # Single query
    python run_agent.py --query "What are the current global climate indicators?"

    # With options
    python run_agent.py --model claude-sonnet-4-20250514 --quiet --query "Analyze Hurricane Ian"

Requires ANTHROPIC_API_KEY environment variable.
"""

from climate_agent.__main__ import main

if __name__ == "__main__":
    main()
