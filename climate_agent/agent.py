"""Autonomous climate agent powered by Claude.

Implements an agentic loop where Claude receives a task, decides which climate
tools to call, processes the results, and iterates until the task is complete.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from anthropic import Anthropic

from climate_agent.config import AGENT_MODEL, AGENT_SYSTEM_PROMPT, MAX_AGENT_TURNS
from climate_agent.tools import TOOL_DEFINITIONS, dispatch_tool


class ClimateAgent:
    """Autonomous agent for climate analysis and tropical cyclone forecasting.

    The agent runs in a loop:
    1. Send user task + conversation history to Claude
    2. If Claude responds with tool calls, execute them and feed results back
    3. Repeat until Claude produces a final text response (no more tool calls)
    4. Return the final response to the user

    The agent maintains conversation history across interactions within a session,
    allowing follow-up questions and multi-step analyses.
    """

    def __init__(
        self,
        model: str | None = None,
        max_turns: int | None = None,
        verbose: bool = True,
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is required.\n"
                "Get your API key at https://console.anthropic.com/settings/keys"
            )
        self.client = Anthropic(api_key=api_key)
        self.model = model or AGENT_MODEL
        self.max_turns = max_turns or MAX_AGENT_TURNS
        self.verbose = verbose
        self.messages: list[dict[str, Any]] = []
        self.total_tool_calls = 0

    def _log(self, message: str) -> None:
        """Print a status message if verbose mode is enabled."""
        if self.verbose:
            print(f"  [agent] {message}", file=sys.stderr)

    def run(self, user_message: str) -> str:
        """Run the agent on a user message and return the final response.

        The agent will autonomously call tools as needed until it produces
        a complete text response.

        Args:
            user_message: The user's task or question.

        Returns:
            The agent's final text response.
        """
        self.messages.append({"role": "user", "content": user_message})

        for turn in range(self.max_turns):
            self._log(f"Turn {turn + 1}/{self.max_turns}")

            response = self.client.messages.create(
                model=self.model,
                max_tokens=8096,
                system=AGENT_SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=self.messages,
            )

            # Process the response
            assistant_content = response.content
            self.messages.append({"role": "assistant", "content": assistant_content})

            # Check if we're done (no tool use, just text)
            tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]

            if not tool_use_blocks:
                # Agent has finished - extract text response
                text_parts = [b.text for b in assistant_content if b.type == "text"]
                final_response = "\n".join(text_parts)
                self._log(f"Complete. Total tool calls this run: {self.total_tool_calls}")
                return final_response

            # Execute each tool call
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input
                self.total_tool_calls += 1

                self._log(f"Calling tool: {tool_name}({json.dumps(tool_input, default=str)[:200]})")
                result = dispatch_tool(tool_name, tool_input)
                self._log(f"  -> Result: {result[:200]}{'...' if len(result) > 200 else ''}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result,
                })

            self.messages.append({"role": "user", "content": tool_results})

        # If we exhaust turns, return whatever text we have
        self._log(f"Reached max turns ({self.max_turns})")
        final_texts = []
        for block in self.messages[-1].get("content", []):
            if hasattr(block, "text"):
                final_texts.append(block.text)
        return "\n".join(final_texts) if final_texts else (
            "I reached the maximum number of reasoning steps. "
            "Please refine your request or increase the turn limit."
        )

    def reset(self) -> None:
        """Clear conversation history to start a fresh session."""
        self.messages = []
        self.total_tool_calls = 0

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics about the current agent session."""
        return {
            "model": self.model,
            "total_messages": len(self.messages),
            "total_tool_calls": self.total_tool_calls,
            "max_turns": self.max_turns,
        }
