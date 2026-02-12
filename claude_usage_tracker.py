#!/usr/bin/env python3
"""
Claude Code Usage Tracker
=========================
A comprehensive CLI tool that parses Claude Code session logs and provides
high-level insights into what the AI has been doing across your projects.

Usage:
    python3 claude_usage_tracker.py                  # Full summary of all sessions
    python3 claude_usage_tracker.py --sessions       # List all sessions
    python3 claude_usage_tracker.py --session <id>   # Deep dive into one session
    python3 claude_usage_tracker.py --timeline       # Chronological activity feed
    python3 claude_usage_tracker.py --tools          # Tool usage breakdown
    python3 claude_usage_tracker.py --costs          # Token usage & cost estimates
    python3 claude_usage_tracker.py --files          # Files touched across sessions
    python3 claude_usage_tracker.py --html report.html  # Generate HTML dashboard
    python3 claude_usage_tracker.py --watch          # Live-tail the current session
"""

import json
import os
import sys
import argparse
import glob
import time
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# ─── ANSI color helpers ──────────────────────────────────────────────────────

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bg_blue": "\033[44m",
    "bg_magenta": "\033[45m",
    "bg_cyan": "\033[46m",
}

NO_COLOR = os.environ.get("NO_COLOR") is not None


def c(color: str, text: str) -> str:
    """Colorize text if terminal supports it."""
    if NO_COLOR or not sys.stdout.isatty():
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def bold(text: str) -> str:
    return c("bold", text)


def dim(text: str) -> str:
    return c("dim", text)


def header_box(title: str, width: int = 70) -> str:
    """Create a boxed header."""
    top = f"╔{'═' * (width - 2)}╗"
    mid = f"║ {title:^{width - 4}} ║"
    bot = f"╚{'═' * (width - 2)}╝"
    return c("bright_cyan", f"{top}\n{mid}\n{bot}")


def section_header(title: str) -> str:
    """Create a section divider."""
    line = "─" * 60
    return f"\n{c('bright_yellow', line)}\n{c('bright_yellow', f'  {title}')}\n{c('bright_yellow', line)}"


def bar_chart(label: str, value: int, max_value: int, width: int = 30, color: str = "bright_green") -> str:
    """Render a horizontal bar chart line."""
    if max_value == 0:
        filled = 0
    else:
        filled = int((value / max_value) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:<20s} {c(color, bar)} {value}"


# ─── JSONL Parsing ───────────────────────────────────────────────────────────

CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"

# Approximate cost per token (USD) - Claude Opus pricing as of 2025
COST_PER_INPUT_TOKEN = 15.0 / 1_000_000    # $15 per 1M input tokens
COST_PER_OUTPUT_TOKEN = 75.0 / 1_000_000   # $75 per 1M output tokens
COST_PER_CACHE_WRITE = 18.75 / 1_000_000   # $18.75 per 1M cache write tokens
COST_PER_CACHE_READ = 1.50 / 1_000_000     # $1.50 per 1M cache read tokens

# Tool categories for grouping
TOOL_CATEGORIES = {
    "File Reading": ["Read", "Glob", "Grep"],
    "File Writing": ["Edit", "Write", "NotebookEdit"],
    "Execution": ["Bash"],
    "Web & Search": ["WebFetch", "WebSearch"],
    "AI Agents": ["Task"],
    "Planning": ["TodoWrite", "ExitPlanMode"],
    "User Interaction": ["AskUserQuestion", "Skill"],
}

TOOL_ICONS = {
    "Read": "📖",
    "Glob": "🔍",
    "Grep": "🔎",
    "Edit": "✏️ ",
    "Write": "📝",
    "NotebookEdit": "📓",
    "Bash": "💻",
    "WebFetch": "🌐",
    "WebSearch": "🔍",
    "Task": "🤖",
    "TodoWrite": "📋",
    "ExitPlanMode": "📐",
    "AskUserQuestion": "❓",
    "Skill": "⚡",
}


def find_all_sessions() -> list[dict]:
    """Discover all session JSONL files across all projects."""
    sessions = []
    if not PROJECTS_DIR.exists():
        return sessions

    for project_dir in PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name.replace("-", "/").lstrip("/")
        for jsonl_file in project_dir.glob("*.jsonl"):
            # Skip subagent logs at the top level listing
            if "subagents" in str(jsonl_file):
                continue
            session_id = jsonl_file.stem
            sessions.append({
                "project": project_name,
                "session_id": session_id,
                "file": jsonl_file,
                "subagent_dir": project_dir / session_id / "subagents",
            })
    return sessions


def parse_session(session_file: Path, include_subagents: bool = True) -> dict:
    """Parse a single session JSONL file into structured data."""
    records = []
    with open(session_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Extract metadata from first user message
    metadata = {}
    for rec in records:
        if rec.get("type") == "user" and rec.get("message", {}).get("role") == "user":
            metadata["session_id"] = rec.get("sessionId", "")
            metadata["cwd"] = rec.get("cwd", "")
            metadata["version"] = rec.get("version", "")
            metadata["git_branch"] = rec.get("gitBranch", "")
            break

    # Parse all records
    tool_uses = []
    user_messages = []
    assistant_texts = []
    token_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
    }
    timestamps = []
    files_read = []
    files_written = []
    files_edited = []
    bash_commands = []
    web_fetches = []
    searches = []
    agent_tasks = []

    for rec in records:
        ts = rec.get("timestamp")
        if ts:
            timestamps.append(ts)

        msg = rec.get("message", {})
        content = msg.get("content", "")
        role = msg.get("role", "")

        # Token usage accumulation
        usage = msg.get("usage", {})
        if usage:
            token_usage["input_tokens"] += usage.get("input_tokens", 0)
            token_usage["output_tokens"] += usage.get("output_tokens", 0)
            token_usage["cache_creation_tokens"] += usage.get("cache_creation_input_tokens", 0)
            token_usage["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0)

        # User messages
        if role == "user" and isinstance(content, str) and content.strip():
            user_messages.append({
                "text": content,
                "timestamp": ts,
            })

        # Assistant content
        if role == "assistant" and isinstance(content, list):
            for block in content:
                block_type = block.get("type", "")

                if block_type == "text" and block.get("text", "").strip():
                    assistant_texts.append({
                        "text": block["text"],
                        "timestamp": ts,
                    })

                elif block_type == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input", {})
                    tool_id = block.get("id", "")

                    tool_entry = {
                        "name": tool_name,
                        "id": tool_id,
                        "timestamp": ts,
                        "input": tool_input,
                    }
                    tool_uses.append(tool_entry)

                    # Categorize specific tool actions
                    if tool_name == "Read":
                        fp = tool_input.get("file_path", "")
                        if fp:
                            files_read.append(fp)

                    elif tool_name in ("Write", "NotebookEdit"):
                        fp = tool_input.get("file_path", "") or tool_input.get("notebook_path", "")
                        if fp:
                            files_written.append(fp)

                    elif tool_name == "Edit":
                        fp = tool_input.get("file_path", "")
                        if fp:
                            files_edited.append(fp)

                    elif tool_name == "Bash":
                        cmd = tool_input.get("command", "")
                        desc = tool_input.get("description", "")
                        bash_commands.append({
                            "command": cmd,
                            "description": desc,
                            "timestamp": ts,
                        })

                    elif tool_name == "WebFetch":
                        url = tool_input.get("url", "")
                        web_fetches.append(url)

                    elif tool_name == "WebSearch":
                        query = tool_input.get("query", "")
                        searches.append(query)

                    elif tool_name in ("Glob", "Grep"):
                        pattern = tool_input.get("pattern", "")
                        searches.append(f"[{tool_name}] {pattern}")

                    elif tool_name == "Task":
                        desc = tool_input.get("description", "")
                        agent_type = tool_input.get("subagent_type", "")
                        agent_tasks.append({
                            "description": desc,
                            "subagent_type": agent_type,
                            "timestamp": ts,
                        })

    # Parse timestamps
    start_time = None
    end_time = None
    if timestamps:
        parsed_times = []
        for ts in timestamps:
            try:
                parsed_times.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except (ValueError, AttributeError):
                pass
        if parsed_times:
            start_time = min(parsed_times)
            end_time = max(parsed_times)

    # Count subagent activity
    subagent_count = 0
    subagent_dir = session_file.parent / session_file.stem / "subagents"
    if include_subagents and subagent_dir.exists():
        subagent_files = list(subagent_dir.glob("*.jsonl"))
        subagent_count = len(subagent_files)
        for sa_file in subagent_files:
            sa_data = parse_session(sa_file, include_subagents=False)
            # Merge subagent stats
            tool_uses.extend(sa_data["tool_uses"])
            files_read.extend(sa_data["files_read"])
            files_written.extend(sa_data["files_written"])
            files_edited.extend(sa_data["files_edited"])
            bash_commands.extend(sa_data["bash_commands"])
            searches.extend(sa_data["searches"])
            for k in token_usage:
                token_usage[k] += sa_data["token_usage"].get(k, 0)

    # Compute cost estimate
    cost = (
        token_usage["input_tokens"] * COST_PER_INPUT_TOKEN
        + token_usage["output_tokens"] * COST_PER_OUTPUT_TOKEN
        + token_usage["cache_creation_tokens"] * COST_PER_CACHE_WRITE
        + token_usage["cache_read_tokens"] * COST_PER_CACHE_READ
    )

    return {
        "metadata": metadata,
        "tool_uses": tool_uses,
        "user_messages": user_messages,
        "assistant_texts": assistant_texts,
        "token_usage": token_usage,
        "estimated_cost": cost,
        "start_time": start_time,
        "end_time": end_time,
        "duration": (end_time - start_time) if start_time and end_time else None,
        "files_read": files_read,
        "files_written": files_written,
        "files_edited": files_edited,
        "bash_commands": bash_commands,
        "web_fetches": web_fetches,
        "searches": searches,
        "agent_tasks": agent_tasks,
        "subagent_count": subagent_count,
        "record_count": len(records),
    }


# ─── Display Functions ───────────────────────────────────────────────────────

def format_duration(td: timedelta | None) -> str:
    if td is None:
        return "unknown"
    total_seconds = int(td.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        m, s = divmod(total_seconds, 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(total_seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


def format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_cost(cost: float) -> str:
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def shorten_path(path: str, max_len: int = 50) -> str:
    if len(path) <= max_len:
        return path
    parts = Path(path).parts
    if len(parts) <= 2:
        return path
    return f".../{'/'.join(parts[-2:])}"


def display_overview(all_sessions: list[dict]) -> None:
    """Show the main dashboard overview."""
    print(header_box("CLAUDE CODE USAGE TRACKER"))
    print()

    total_sessions = len(all_sessions)
    total_tools = sum(len(s["tool_uses"]) for s in all_sessions)
    total_input = sum(s["token_usage"]["input_tokens"] for s in all_sessions)
    total_output = sum(s["token_usage"]["output_tokens"] for s in all_sessions)
    total_cost = sum(s["estimated_cost"] for s in all_sessions)
    total_files_read = sum(len(s["files_read"]) for s in all_sessions)
    total_files_written = sum(len(s["files_written"]) for s in all_sessions)
    total_files_edited = sum(len(s["files_edited"]) for s in all_sessions)
    total_commands = sum(len(s["bash_commands"]) for s in all_sessions)
    total_subagents = sum(s["subagent_count"] for s in all_sessions)

    # Aggregate durations
    total_duration = timedelta()
    for s in all_sessions:
        if s["duration"]:
            total_duration += s["duration"]

    # Key metrics
    print(section_header("OVERVIEW"))
    print()
    metrics = [
        ("Sessions", str(total_sessions), "bright_cyan"),
        ("Total Duration", format_duration(total_duration), "bright_magenta"),
        ("Tool Invocations", str(total_tools), "bright_green"),
        ("Files Read", str(total_files_read), "blue"),
        ("Files Written", str(total_files_written), "bright_yellow"),
        ("Files Edited", str(total_files_edited), "yellow"),
        ("Bash Commands", str(total_commands), "bright_red"),
        ("Sub-agents Spawned", str(total_subagents), "magenta"),
        ("Input Tokens", format_tokens(total_input), "cyan"),
        ("Output Tokens", format_tokens(total_output), "green"),
        ("Est. Cost", format_cost(total_cost), "bright_yellow"),
    ]
    max_label = max(len(m[0]) for m in metrics)
    for label, value, color in metrics:
        print(f"  {c('dim', label + ':'):<{max_label + 15}}  {c(color, bold(value))}")

    # Tool usage breakdown
    print(section_header("TOOL USAGE BREAKDOWN"))
    print()
    tool_counter = Counter()
    for s in all_sessions:
        for tu in s["tool_uses"]:
            tool_counter[tu["name"]] += 1

    if tool_counter:
        max_count = max(tool_counter.values())
        for tool_name, count in tool_counter.most_common():
            icon = TOOL_ICONS.get(tool_name, "🔧")
            print(f"  {icon} {bar_chart(tool_name, count, max_count, width=25)}")
    else:
        print(dim("  No tool usage recorded."))

    # Tool category breakdown
    print(section_header("ACTIVITY CATEGORIES"))
    print()
    category_counts = {}
    for cat_name, tool_names in TOOL_CATEGORIES.items():
        count = sum(tool_counter.get(t, 0) for t in tool_names)
        if count > 0:
            category_counts[cat_name] = count

    if category_counts:
        max_cat = max(category_counts.values())
        cat_colors = {
            "File Reading": "blue",
            "File Writing": "bright_yellow",
            "Execution": "bright_red",
            "Web & Search": "bright_cyan",
            "AI Agents": "bright_magenta",
            "Planning": "green",
            "User Interaction": "yellow",
        }
        for cat_name, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            color = cat_colors.get(cat_name, "white")
            print(bar_chart(cat_name, count, max_cat, width=30, color=color))

    # Projects
    print(section_header("PROJECTS"))
    print()
    projects = defaultdict(int)
    for s in all_sessions:
        proj = s.get("_project", "unknown")
        projects[proj] += 1
    for proj, count in sorted(projects.items(), key=lambda x: -x[1]):
        print(f"  {c('bright_cyan', proj)}: {count} session(s)")

    # Most touched files
    print(section_header("MOST ACTIVE FILES"))
    print()
    file_counter = Counter()
    for s in all_sessions:
        for fp in s["files_read"]:
            file_counter[fp] += 1
        for fp in s["files_written"]:
            file_counter[fp] += 2  # Weight writes higher
        for fp in s["files_edited"]:
            file_counter[fp] += 2

    if file_counter:
        for fp, count in file_counter.most_common(15):
            print(f"  {c('dim', f'({count:>3}x)')} {c('bright_green', shorten_path(fp))}")
    else:
        print(dim("  No file activity recorded."))

    print()


def display_sessions(all_sessions: list[dict]) -> None:
    """List all sessions with summary info."""
    print(header_box("ALL SESSIONS"))
    print()

    for i, s in enumerate(all_sessions, 1):
        meta = s["metadata"]
        sid = meta.get("session_id", "unknown")[:12]
        branch = meta.get("git_branch", "")
        version = meta.get("version", "")
        cwd = meta.get("cwd", "")

        start = s["start_time"].strftime("%Y-%m-%d %H:%M") if s["start_time"] else "unknown"
        dur = format_duration(s["duration"])
        n_tools = len(s["tool_uses"])

        # First user message as summary
        first_msg = ""
        if s["user_messages"]:
            first_msg = s["user_messages"][0]["text"][:80]

        print(f"  {c('bright_cyan', bold(f'Session {i}'))} {dim(f'({sid}...)')}")
        print(f"    {c('dim', 'Started:')}  {start}   {c('dim', 'Duration:')} {dur}")
        print(f"    {c('dim', 'Branch:')}   {c('yellow', branch)}")
        print(f"    {c('dim', 'Tools:')}    {n_tools}   {c('dim', 'Cost:')} {format_cost(s['estimated_cost'])}")
        if first_msg:
            print(f"    {c('dim', 'Task:')}     {c('white', first_msg)}")
        print()


def display_session_detail(session: dict) -> None:
    """Deep dive into a single session."""
    meta = session["metadata"]
    print(header_box(f"SESSION: {meta.get('session_id', 'unknown')[:16]}..."))
    print()

    # Metadata
    for key in ["session_id", "cwd", "version", "git_branch"]:
        val = meta.get(key, "")
        if val:
            print(f"  {c('dim', f'{key}:')} {val}")
    print()

    if session["start_time"]:
        print(f"  {c('dim', 'Started:')} {session['start_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if session["end_time"]:
        print(f"  {c('dim', 'Ended:')}   {session['end_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if session["duration"]:
        print(f"  {c('dim', 'Duration:')} {format_duration(session['duration'])}")
    print()

    # User requests
    print(section_header("USER REQUESTS"))
    for um in session["user_messages"]:
        text = um["text"][:200]
        print(f"\n  {c('bright_cyan', '▸')} {text}")
    print()

    # Tool timeline
    print(section_header("TOOL ACTIVITY TIMELINE"))
    print()
    for tu in session["tool_uses"]:
        icon = TOOL_ICONS.get(tu["name"], "🔧")
        ts_str = ""
        if tu["timestamp"]:
            try:
                ts = datetime.fromisoformat(tu["timestamp"].replace("Z", "+00:00"))
                ts_str = ts.strftime("%H:%M:%S")
            except (ValueError, AttributeError):
                pass

        detail = ""
        inp = tu["input"]
        if tu["name"] == "Bash":
            cmd = inp.get("command", "")[:70]
            detail = c("dim", cmd)
        elif tu["name"] == "Read":
            detail = c("green", shorten_path(inp.get("file_path", "")))
        elif tu["name"] in ("Edit", "Write"):
            detail = c("yellow", shorten_path(inp.get("file_path", "")))
        elif tu["name"] == "Glob":
            detail = c("cyan", inp.get("pattern", ""))
        elif tu["name"] == "Grep":
            detail = c("cyan", inp.get("pattern", ""))
        elif tu["name"] == "Task":
            detail = c("magenta", inp.get("description", ""))
        elif tu["name"] == "WebFetch":
            detail = c("blue", inp.get("url", "")[:60])
        elif tu["name"] == "WebSearch":
            detail = c("blue", inp.get("query", ""))
        elif tu["name"] == "TodoWrite":
            todos = inp.get("todos", [])
            detail = c("green", f"{len(todos)} todo(s)")

        print(f"  {c('dim', ts_str)}  {icon} {c('bright_green', tu['name'])} {detail}")

    # Token usage
    print(section_header("TOKEN USAGE"))
    print()
    tu = session["token_usage"]
    print(f"  Input tokens:         {format_tokens(tu['input_tokens'])}")
    print(f"  Output tokens:        {format_tokens(tu['output_tokens'])}")
    print(f"  Cache write tokens:   {format_tokens(tu['cache_creation_tokens'])}")
    print(f"  Cache read tokens:    {format_tokens(tu['cache_read_tokens'])}")
    cost_str = format_cost(session["estimated_cost"])
    print(f"  {c('bright_yellow', bold('Estimated cost:     ' + cost_str))}")
    print()


def display_timeline(all_sessions: list[dict]) -> None:
    """Show chronological activity feed across all sessions."""
    print(header_box("ACTIVITY TIMELINE"))
    print()

    # Collect all events with timestamps
    events = []
    for s in all_sessions:
        sid = s["metadata"].get("session_id", "")[:8]
        for um in s["user_messages"]:
            if um["timestamp"]:
                events.append({
                    "timestamp": um["timestamp"],
                    "type": "user_request",
                    "session": sid,
                    "detail": um["text"][:100],
                })
        for tu in s["tool_uses"]:
            if tu["timestamp"]:
                detail = ""
                inp = tu["input"]
                if tu["name"] == "Bash":
                    detail = inp.get("description", "") or inp.get("command", "")[:60]
                elif tu["name"] in ("Read", "Edit", "Write"):
                    detail = shorten_path(inp.get("file_path", ""))
                elif tu["name"] == "Task":
                    detail = inp.get("description", "")
                elif tu["name"] == "Grep":
                    detail = f'pattern: {inp.get("pattern", "")}'
                elif tu["name"] == "Glob":
                    detail = f'pattern: {inp.get("pattern", "")}'
                else:
                    detail = str(inp)[:60]

                events.append({
                    "timestamp": tu["timestamp"],
                    "type": "tool",
                    "tool": tu["name"],
                    "session": sid,
                    "detail": detail,
                })

    # Sort by timestamp
    events.sort(key=lambda e: e["timestamp"])

    current_date = ""
    for ev in events:
        try:
            ts = datetime.fromisoformat(ev["timestamp"].replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue

        date_str = ts.strftime("%Y-%m-%d")
        if date_str != current_date:
            current_date = date_str
            print(f"\n  {c('bright_cyan', bold(f'── {date_str} ──'))}")

        time_str = ts.strftime("%H:%M:%S")

        if ev["type"] == "user_request":
            print(f"  {c('dim', time_str)}  {c('bright_yellow', '👤 REQUEST:')} {ev['detail']}")
        else:
            icon = TOOL_ICONS.get(ev.get("tool", ""), "🔧")
            tool = ev.get("tool", "")
            print(f"  {c('dim', time_str)}  {icon} {c('green', tool):20s} {c('dim', ev['detail'])}")

    print()


def display_tools(all_sessions: list[dict]) -> None:
    """Show detailed tool usage statistics."""
    print(header_box("TOOL USAGE ANALYSIS"))
    print()

    tool_counter = Counter()
    tool_details = defaultdict(list)

    for s in all_sessions:
        for tu in s["tool_uses"]:
            tool_counter[tu["name"]] += 1
            tool_details[tu["name"]].append(tu)

    if not tool_counter:
        print(dim("  No tool usage recorded."))
        return

    max_count = max(tool_counter.values())
    total = sum(tool_counter.values())

    print(section_header("TOOL FREQUENCY"))
    print()
    for tool, count in tool_counter.most_common():
        pct = (count / total) * 100
        icon = TOOL_ICONS.get(tool, "🔧")
        print(f"  {icon} {bar_chart(tool, count, max_count, width=25)} ({pct:.1f}%)")

    # Bash command analysis
    if "Bash" in tool_details:
        print(section_header("TOP BASH COMMANDS"))
        print()
        cmd_counter = Counter()
        for tu in tool_details["Bash"]:
            cmd = tu["input"].get("command", "").split()[0] if tu["input"].get("command", "") else "empty"
            cmd_counter[cmd] += 1
        for cmd, count in cmd_counter.most_common(10):
            print(f"  {c('dim', f'{count:>3}x')} {c('bright_red', cmd)}")

    # Search patterns
    if "Grep" in tool_details or "Glob" in tool_details:
        print(section_header("SEARCH PATTERNS"))
        print()
        for tu in (tool_details.get("Grep", []) + tool_details.get("Glob", []))[:15]:
            pattern = tu["input"].get("pattern", "")
            print(f"  {c('cyan', '▸')} {pattern}")

    print()


def display_costs(all_sessions: list[dict]) -> None:
    """Show token usage and cost estimates."""
    print(header_box("TOKEN USAGE & COST ESTIMATES"))
    print()

    total_input = 0
    total_output = 0
    total_cache_write = 0
    total_cache_read = 0
    total_cost = 0.0

    print(f"  {'Session':<14s} {'Input':>10s} {'Output':>10s} {'Cache W':>10s} {'Cache R':>10s} {'Cost':>10s}")
    print(f"  {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")

    for s in all_sessions:
        tu = s["token_usage"]
        sid = s["metadata"].get("session_id", "unknown")[:12]
        total_input += tu["input_tokens"]
        total_output += tu["output_tokens"]
        total_cache_write += tu["cache_creation_tokens"]
        total_cache_read += tu["cache_read_tokens"]
        total_cost += s["estimated_cost"]

        print(
            f"  {sid:<14s} "
            f"{format_tokens(tu['input_tokens']):>10s} "
            f"{format_tokens(tu['output_tokens']):>10s} "
            f"{format_tokens(tu['cache_creation_tokens']):>10s} "
            f"{format_tokens(tu['cache_read_tokens']):>10s} "
            f"{c('bright_yellow', format_cost(s['estimated_cost'])):>20s}"
        )

    print(f"  {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")
    print(
        f"  {c('bold', 'TOTAL'):<14s} "
        f"{c('bold', format_tokens(total_input)):>10s} "
        f"{c('bold', format_tokens(total_output)):>10s} "
        f"{c('bold', format_tokens(total_cache_write)):>10s} "
        f"{c('bold', format_tokens(total_cache_read)):>10s} "
        f"{c('bright_yellow', bold(format_cost(total_cost))):>20s}"
    )

    print(f"\n  {c('dim', 'Pricing: Opus input $15/1M, output $75/1M, cache write $18.75/1M, cache read $1.50/1M')}")
    print()


def display_files(all_sessions: list[dict]) -> None:
    """Show all files touched across sessions."""
    print(header_box("FILES TOUCHED"))
    print()

    read_counter = Counter()
    write_counter = Counter()
    edit_counter = Counter()

    for s in all_sessions:
        for fp in s["files_read"]:
            read_counter[fp] += 1
        for fp in s["files_written"]:
            write_counter[fp] += 1
        for fp in s["files_edited"]:
            edit_counter[fp] += 1

    if read_counter:
        print(section_header("FILES READ"))
        print()
        for fp, count in read_counter.most_common(20):
            print(f"  {c('dim', f'{count:>3}x')} 📖 {c('green', shorten_path(fp))}")

    if write_counter:
        print(section_header("FILES CREATED / WRITTEN"))
        print()
        for fp, count in write_counter.most_common(20):
            print(f"  {c('dim', f'{count:>3}x')} 📝 {c('yellow', shorten_path(fp))}")

    if edit_counter:
        print(section_header("FILES EDITED"))
        print()
        for fp, count in edit_counter.most_common(20):
            print(f"  {c('dim', f'{count:>3}x')} ✏️  {c('bright_yellow', shorten_path(fp))}")

    if not read_counter and not write_counter and not edit_counter:
        print(dim("  No file activity recorded."))

    print()


def display_watch(session_file: Path) -> None:
    """Live-tail the most recent session log."""
    print(header_box("LIVE SESSION MONITOR"))
    print(f"\n  Watching: {c('cyan', str(session_file))}")
    print(f"  {c('dim', 'Press Ctrl+C to stop')}\n")

    seen_lines = 0
    try:
        while True:
            with open(session_file) as f:
                lines = f.readlines()

            new_lines = lines[seen_lines:]
            seen_lines = len(lines)

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts = rec.get("timestamp", "")
                msg = rec.get("message", {})
                content = msg.get("content", "")
                role = msg.get("role", "")
                rec_type = rec.get("type", "")

                ts_str = ""
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        ts_str = dt.strftime("%H:%M:%S")
                    except (ValueError, AttributeError):
                        pass

                if rec_type == "user" and isinstance(content, str) and content.strip():
                    print(f"  {c('dim', ts_str)}  {c('bright_yellow', '👤')} {content[:120]}")
                elif role == "assistant" and isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_use":
                            tool = block.get("name", "")
                            icon = TOOL_ICONS.get(tool, "🔧")
                            inp = block.get("input", {})
                            detail = ""
                            if tool == "Bash":
                                detail = inp.get("description", "") or inp.get("command", "")[:50]
                            elif tool in ("Read", "Edit", "Write"):
                                detail = shorten_path(inp.get("file_path", ""))
                            elif tool == "Task":
                                detail = inp.get("description", "")
                            print(f"  {c('dim', ts_str)}  {icon} {c('green', tool)} {c('dim', detail)}")
                        elif block.get("type") == "text" and block.get("text", "").strip():
                            text = block["text"].strip()[:120]
                            print(f"  {c('dim', ts_str)}  {c('white', '💬')} {text}")

            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n  {c('dim', 'Stopped watching.')}")


# ─── HTML Report ─────────────────────────────────────────────────────────────

def generate_html_report(all_sessions: list[dict], output_path: str) -> None:
    """Generate a self-contained HTML dashboard report."""

    total_sessions = len(all_sessions)
    total_tools = sum(len(s["tool_uses"]) for s in all_sessions)
    total_input = sum(s["token_usage"]["input_tokens"] for s in all_sessions)
    total_output = sum(s["token_usage"]["output_tokens"] for s in all_sessions)
    total_cost = sum(s["estimated_cost"] for s in all_sessions)
    total_files_read = sum(len(s["files_read"]) for s in all_sessions)
    total_files_written = sum(len(s["files_written"]) for s in all_sessions)
    total_files_edited = sum(len(s["files_edited"]) for s in all_sessions)
    total_commands = sum(len(s["bash_commands"]) for s in all_sessions)

    total_duration = timedelta()
    for s in all_sessions:
        if s["duration"]:
            total_duration += s["duration"]

    # Tool breakdown data
    tool_counter = Counter()
    for s in all_sessions:
        for tu in s["tool_uses"]:
            tool_counter[tu["name"]] += 1

    tool_labels = json.dumps([t for t, _ in tool_counter.most_common()])
    tool_values = json.dumps([v for _, v in tool_counter.most_common()])

    # Category data
    category_data = {}
    for cat_name, tool_names in TOOL_CATEGORIES.items():
        count = sum(tool_counter.get(t, 0) for t in tool_names)
        if count > 0:
            category_data[cat_name] = count
    cat_labels = json.dumps(list(category_data.keys()))
    cat_values = json.dumps(list(category_data.values()))

    # File activity
    file_counter = Counter()
    for s in all_sessions:
        for fp in s["files_read"]:
            file_counter[shorten_path(fp)] += 1
        for fp in s["files_written"]:
            file_counter[shorten_path(fp)] += 2
        for fp in s["files_edited"]:
            file_counter[shorten_path(fp)] += 2
    top_files_html = ""
    for fp, count in file_counter.most_common(15):
        top_files_html += f'<div class="file-row"><span class="file-count">{count}x</span> <span class="file-path">{fp}</span></div>\n'

    # Session rows
    session_rows_html = ""
    for s in all_sessions:
        meta = s["metadata"]
        sid = meta.get("session_id", "unknown")[:12]
        start = s["start_time"].strftime("%Y-%m-%d %H:%M") if s["start_time"] else "-"
        dur = format_duration(s["duration"])
        n_tools = len(s["tool_uses"])
        cost = format_cost(s["estimated_cost"])
        task = ""
        if s["user_messages"]:
            task = s["user_messages"][0]["text"][:80].replace("<", "&lt;").replace(">", "&gt;")
        branch = meta.get("git_branch", "")
        session_rows_html += f"""
        <tr>
            <td class="mono">{sid}...</td>
            <td>{start}</td>
            <td>{dur}</td>
            <td>{n_tools}</td>
            <td class="cost">{cost}</td>
            <td class="branch">{branch}</td>
            <td class="task">{task}</td>
        </tr>"""

    # Timeline events (last 100)
    events = []
    for s in all_sessions:
        sid = s["metadata"].get("session_id", "")[:8]
        for tu in s["tool_uses"]:
            if tu["timestamp"]:
                detail = ""
                inp = tu["input"]
                if tu["name"] == "Bash":
                    detail = (inp.get("description", "") or inp.get("command", "")[:60]).replace("<", "&lt;")
                elif tu["name"] in ("Read", "Edit", "Write"):
                    detail = shorten_path(inp.get("file_path", ""))
                elif tu["name"] == "Task":
                    detail = inp.get("description", "")
                else:
                    detail = ""
                events.append((tu["timestamp"], tu["name"], detail))
    events.sort(key=lambda e: e[0], reverse=True)
    timeline_html = ""
    for ts, tool, detail in events[:100]:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            ts_str = ts
        icon = TOOL_ICONS.get(tool, "🔧")
        timeline_html += f'<div class="event"><span class="ts">{ts_str}</span> {icon} <span class="tool">{tool}</span> <span class="detail">{detail}</span></div>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Claude Code Usage Report</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --purple: #bc8cff; --cyan: #39d353;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; padding: 24px; }}
  h1 {{ color: var(--accent); font-size: 24px; margin-bottom: 8px; }}
  h2 {{ color: var(--accent); font-size: 18px; margin: 24px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
  .subtitle {{ color: var(--text-dim); margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .card .label {{ color: var(--text-dim); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .card .value {{ font-size: 28px; font-weight: 700; margin-top: 4px; }}
  .card .value.cyan {{ color: var(--cyan); }}
  .card .value.green {{ color: var(--green); }}
  .card .value.yellow {{ color: var(--yellow); }}
  .card .value.purple {{ color: var(--purple); }}
  .card .value.accent {{ color: var(--accent); }}
  .card .value.red {{ color: var(--red); }}
  .chart-container {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .bar-row {{ display: flex; align-items: center; margin: 6px 0; }}
  .bar-label {{ width: 140px; font-size: 13px; color: var(--text-dim); flex-shrink: 0; }}
  .bar-track {{ flex: 1; height: 22px; background: var(--bg); border-radius: 4px; overflow: hidden; margin: 0 10px; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  .bar-value {{ width: 50px; font-size: 13px; text-align: right; color: var(--text); }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
  th {{ background: var(--bg); color: var(--accent); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-top: 1px solid var(--border); font-size: 13px; }}
  tr:hover {{ background: rgba(88, 166, 255, 0.04); }}
  .mono {{ font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; }}
  .cost {{ color: var(--yellow); font-weight: 600; }}
  .branch {{ color: var(--purple); font-family: 'SF Mono', monospace; font-size: 12px; }}
  .task {{ color: var(--text-dim); max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .file-row {{ display: flex; padding: 4px 0; font-size: 13px; }}
  .file-count {{ color: var(--text-dim); width: 40px; text-align: right; margin-right: 8px; }}
  .file-path {{ color: var(--green); font-family: 'SF Mono', monospace; font-size: 12px; }}
  .event {{ padding: 4px 0; font-size: 13px; border-bottom: 1px solid var(--border); }}
  .event .ts {{ color: var(--text-dim); font-family: monospace; font-size: 11px; }}
  .event .tool {{ color: var(--green); font-weight: 600; }}
  .event .detail {{ color: var(--text-dim); }}
  .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 768px) {{ .columns {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Claude Code Usage Report</h1>
<p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="grid">
  <div class="card"><div class="label">Sessions</div><div class="value accent">{total_sessions}</div></div>
  <div class="card"><div class="label">Duration</div><div class="value purple">{format_duration(total_duration)}</div></div>
  <div class="card"><div class="label">Tool Calls</div><div class="value green">{total_tools}</div></div>
  <div class="card"><div class="label">Files Read</div><div class="value cyan">{total_files_read}</div></div>
  <div class="card"><div class="label">Files Written</div><div class="value yellow">{total_files_written}</div></div>
  <div class="card"><div class="label">Files Edited</div><div class="value yellow">{total_files_edited}</div></div>
  <div class="card"><div class="label">Bash Commands</div><div class="value red">{total_commands}</div></div>
  <div class="card"><div class="label">Est. Cost</div><div class="value yellow">{format_cost(total_cost)}</div></div>
  <div class="card"><div class="label">Input Tokens</div><div class="value cyan">{format_tokens(total_input)}</div></div>
  <div class="card"><div class="label">Output Tokens</div><div class="value green">{format_tokens(total_output)}</div></div>
</div>

<div class="columns">
<div>
<h2>Tool Usage</h2>
<div class="chart-container" id="tool-chart"></div>
</div>
<div>
<h2>Activity Categories</h2>
<div class="chart-container" id="cat-chart"></div>
</div>
</div>

<h2>Sessions</h2>
<table>
<tr><th>ID</th><th>Started</th><th>Duration</th><th>Tools</th><th>Cost</th><th>Branch</th><th>Task</th></tr>
{session_rows_html}
</table>

<div class="columns">
<div>
<h2>Most Active Files</h2>
<div class="chart-container">{top_files_html}</div>
</div>
<div>
<h2>Recent Activity</h2>
<div class="chart-container" style="max-height: 400px; overflow-y: auto;">{timeline_html}</div>
</div>
</div>

<script>
// Render bar charts from data
function renderBars(containerId, labels, values, colors) {{
  const container = document.getElementById(containerId);
  const max = Math.max(...values);
  const defaultColors = ['#58a6ff','#3fb950','#d29922','#f85149','#bc8cff','#39d353','#f0883e','#a5d6ff'];
  let html = '';
  for (let i = 0; i < labels.length; i++) {{
    const pct = max > 0 ? (values[i] / max * 100) : 0;
    const color = (colors && colors[i]) || defaultColors[i % defaultColors.length];
    html += `<div class="bar-row">
      <span class="bar-label">${{labels[i]}}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div>
      <span class="bar-value">${{values[i]}}</span>
    </div>`;
  }}
  container.innerHTML = html;
}}

renderBars('tool-chart', {tool_labels}, {tool_values});
renderBars('cat-chart', {cat_labels}, {cat_values},
  ['#58a6ff','#d29922','#f85149','#39d353','#bc8cff','#3fb950','#f0883e']);
</script>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"  {c('bright_green', '✓')} HTML report saved to {c('cyan', output_path)}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Claude Code Usage Tracker - See what the AI has been doing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 claude_usage_tracker.py                  Full dashboard overview
  python3 claude_usage_tracker.py --sessions       List all sessions
  python3 claude_usage_tracker.py --session abc12  Detail for a specific session
  python3 claude_usage_tracker.py --timeline       Chronological activity feed
  python3 claude_usage_tracker.py --tools          Tool usage breakdown
  python3 claude_usage_tracker.py --costs          Token usage & cost estimates
  python3 claude_usage_tracker.py --files          Files touched across sessions
  python3 claude_usage_tracker.py --html out.html  Generate HTML dashboard
  python3 claude_usage_tracker.py --watch          Live-tail current session
        """,
    )

    parser.add_argument("--sessions", action="store_true", help="List all sessions")
    parser.add_argument("--session", type=str, help="Show detail for a specific session ID (prefix match)")
    parser.add_argument("--timeline", action="store_true", help="Show chronological activity feed")
    parser.add_argument("--tools", action="store_true", help="Show tool usage breakdown")
    parser.add_argument("--costs", action="store_true", help="Show token usage and cost estimates")
    parser.add_argument("--files", action="store_true", help="Show files touched across sessions")
    parser.add_argument("--html", type=str, metavar="FILE", help="Generate HTML dashboard report")
    parser.add_argument("--watch", action="store_true", help="Live-tail the most recent session")
    parser.add_argument("--claude-dir", type=str, default=None, help="Override ~/.claude directory path")

    args = parser.parse_args()

    global CLAUDE_DIR, PROJECTS_DIR
    if args.claude_dir:
        CLAUDE_DIR = Path(args.claude_dir)
        PROJECTS_DIR = CLAUDE_DIR / "projects"

    if not PROJECTS_DIR.exists():
        print(c("red", f"Error: No Claude Code projects directory found at {PROJECTS_DIR}"))
        print(c("dim", "Make sure you've used Claude Code at least once."))
        sys.exit(1)

    # Discover sessions
    sessions_info = find_all_sessions()
    if not sessions_info:
        print(c("yellow", "No sessions found."))
        sys.exit(0)

    # Watch mode (special: just tail the latest session)
    if args.watch:
        latest = max(sessions_info, key=lambda s: s["file"].stat().st_mtime)
        display_watch(latest["file"])
        return

    # Parse all sessions
    all_sessions = []
    for si in sessions_info:
        parsed = parse_session(si["file"])
        parsed["_project"] = si["project"]
        all_sessions.append(parsed)

    # Sort by start time
    all_sessions.sort(key=lambda s: s["start_time"] or datetime.min.replace(tzinfo=timezone.utc))

    # Dispatch to requested view
    if args.session:
        # Find matching session
        match = None
        for s in all_sessions:
            sid = s["metadata"].get("session_id", "")
            if sid.startswith(args.session):
                match = s
                break
        if match:
            display_session_detail(match)
        else:
            print(c("red", f"No session found matching '{args.session}'"))
            sys.exit(1)
    elif args.sessions:
        display_sessions(all_sessions)
    elif args.timeline:
        display_timeline(all_sessions)
    elif args.tools:
        display_tools(all_sessions)
    elif args.costs:
        display_costs(all_sessions)
    elif args.files:
        display_files(all_sessions)
    elif args.html:
        generate_html_report(all_sessions, args.html)
    else:
        # Default: full overview
        display_overview(all_sessions)


if __name__ == "__main__":
    main()
