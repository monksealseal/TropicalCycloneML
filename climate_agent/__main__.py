"""CLI entry point for the climate agent.

Usage:
    # Interactive mode (REPL)
    python -m climate_agent

    # Single query mode
    python -m climate_agent --query "What are the current climate indicators?"

    # Specify model
    python -m climate_agent --model claude-sonnet-4-20250514

    # Non-verbose mode
    python -m climate_agent --quiet
"""

from __future__ import annotations

import argparse
import sys


def print_banner() -> None:
    """Print the climate agent banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                  CLIMATE ANALYSIS AGENT                     ║
║         Autonomous Claude Agent for Climate Science         ║
║                                                             ║
║  Capabilities:                                              ║
║  • Tropical cyclone tracking, forecasting & analysis        ║
║  • Sea surface temperature & atmospheric analysis           ║
║  • Climate change indicators & trend analysis               ║
║  • Carbon budget & emission pathway modeling                ║
║  • Climate risk assessment under IPCC scenarios             ║
║  • Visualization: storm tracks, SST maps, time series       ║
║                                                             ║
║  Type 'quit' or 'exit' to end the session.                  ║
║  Type 'reset' to clear conversation history.                ║
║  Type 'stats' to see session statistics.                    ║
║  Type 'tools' to list available tools.                      ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def list_tools() -> None:
    """Print available tools."""
    from climate_agent.tools import TOOL_DEFINITIONS

    print("\nAvailable tools:")
    print("-" * 60)
    for tool in TOOL_DEFINITIONS:
        print(f"  {tool['name']}")
        desc = tool["description"]
        # Wrap description
        words = desc.split()
        line = "    "
        for word in words:
            if len(line) + len(word) + 1 > 70:
                print(line)
                line = "    " + word
            else:
                line += " " + word if line.strip() else "    " + word
        print(line)
        print()


def run_interactive(agent: "ClimateAgent") -> None:
    """Run the agent in interactive REPL mode."""
    print_banner()

    while True:
        try:
            user_input = input("\n🌍 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("Session reset. Conversation history cleared.")
            continue

        if user_input.lower() == "stats":
            stats = agent.get_session_stats()
            print(f"\nSession Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            continue

        if user_input.lower() == "tools":
            list_tools()
            continue

        try:
            response = agent.run(user_input)
            print(f"\n🤖 Agent:\n{response}")
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {e}", file=sys.stderr)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Claude agent for climate analysis and tropical cyclone forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m climate_agent
  python -m climate_agent --query "Analyze the 2023 Atlantic hurricane season"
  python -m climate_agent --query "What is the remaining carbon budget for 1.5C?"
  python -m climate_agent --query "Assess climate risk for Miami under SSP5-8.5"
  python -m climate_agent --query "Forecast intensity for a storm at 20N 60W with 80kt winds"
        """,
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to run (non-interactive mode).",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Claude model to use (default: claude-sonnet-4-20250514).",
    )
    parser.add_argument(
        "--max-turns", "-t",
        type=int,
        default=None,
        help="Maximum agent reasoning turns (default: 50).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress agent status messages.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools and exit.",
    )

    args = parser.parse_args()

    if args.list_tools:
        list_tools()
        return

    from climate_agent.agent import ClimateAgent

    try:
        agent = ClimateAgent(
            model=args.model,
            max_turns=args.max_turns,
            verbose=not args.quiet,
        )
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.query:
        # Single query mode
        try:
            response = agent.run(args.query)
            print(response)
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Interactive mode
        run_interactive(agent)


if __name__ == "__main__":
    main()
