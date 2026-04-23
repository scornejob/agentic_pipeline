"""
ReAct (Reasoning + Acting) agentic pipeline.

Loop:
  1. Build system prompt with tool descriptions
  2. Call LLM → parse Thought / Action / Action Input
  3. Execute tool → append Observation
  4. Repeat until "Final Answer" or max_iterations is reached
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterator

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text

from src.agent.memory import Memory
from src.agent.tools import Tool, run_tool, tool_descriptions
from src.llm.provider import LLMProvider

console = Console()

# ── Parsed step ───────────────────────────────────────────────────────────────

@dataclass
class Step:
    thought: str = ""
    action: str = ""
    action_input: str = ""
    final_answer: str = ""
    raw: str = ""


# ── Parser ────────────────────────────────────────────────────────────────────

_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*(.+?)(?=\n|$)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.*?)(?=\nObservation:|\nThought:|$)", re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.DOTALL | re.IGNORECASE)


def _parse_step(text: str) -> Step:
    step = Step(raw=text)
    if m := _FINAL_RE.search(text):
        step.final_answer = m.group(1).strip()
        step.thought = (_THOUGHT_RE.search(text) or type('', (), {'group': lambda *a: ''})()).group(1)
        if callable(step.thought):
            step.thought = ""
        if m2 := _THOUGHT_RE.search(text):
            step.thought = m2.group(1).strip()
        return step
    if m := _THOUGHT_RE.search(text):
        step.thought = m.group(1).strip()
    if m := _ACTION_RE.search(text):
        step.action = m.group(1).strip()
    if m := _ACTION_INPUT_RE.search(text):
        step.action_input = m.group(1).strip()
    return step


# ── Pipeline ──────────────────────────────────────────────────────────────────

class AgentPipeline:
    def __init__(
        self,
        llm: LLMProvider,
        tools: dict[str, Tool],
        system_prompt_template: str,
        max_iterations: int = 20,
        verbose: bool = True,
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt_template = system_prompt_template
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.memory = Memory()

    def _build_system_prompt(self) -> str:
        desc = tool_descriptions(self.tools)
        return self.system_prompt_template.format(tool_descriptions=desc)

    def _print_step(self, step: Step, iteration: int) -> None:
        if not self.verbose:
            return
        parts = []
        if step.thought:
            parts.append(f"[bold cyan]Thought:[/bold cyan] {escape(step.thought)}")
        if step.action:
            parts.append(f"[bold yellow]Action:[/bold yellow] {escape(step.action)}")
        if step.action_input:
            parts.append(f"[bold yellow]Action Input:[/bold yellow] {escape(step.action_input)}")
        if step.final_answer:
            parts.append(f"[bold green]Final Answer:[/bold green] {escape(step.final_answer)}")
        console.print(Panel("\n".join(parts), title=f"[dim]Step {iteration}[/dim]", border_style="dim"))

    def _print_observation(self, obs: str) -> None:
        if not self.verbose:
            return
        console.print(Panel(escape(obs[:500] + ("…" if len(obs) > 500 else "")),
                            title="[dim]Observation[/dim]", border_style="dim magenta"))

    def run(self, task: str) -> str:
        """Run the ReAct loop for a given task and return the final answer."""
        self.memory.clear_except_system()
        self.memory.set_system(self._build_system_prompt())
        self.memory.add("user", task)

        if self.verbose:
            console.rule(f"[bold]Task: {escape(task[:80])}[/bold]")

        for iteration in range(1, self.max_iterations + 1):
            # ── Show spinner while the LLM is thinking ────────────────────────
            if self.verbose:
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[dim]Step {iteration}/{self.max_iterations}  thinking…[/dim]"),
                    BarColumn(bar_width=20),
                    TaskProgressColumn(),
                    transient=True,
                    console=console,
                ) as progress:
                    progress.add_task("", total=self.max_iterations, completed=iteration - 1)
                    resp = self.llm.chat(self.memory.get_messages())
            else:
                resp = self.llm.chat(self.memory.get_messages())
            raw_text = resp.content

            self.memory.add("assistant", raw_text)

            step = _parse_step(raw_text)
            self._print_step(step, iteration)

            # Finished
            if step.final_answer:
                return step.final_answer

            # No action parsed — ask the model to continue
            if not step.action:
                self.memory.add("user", "Please continue or provide a Final Answer.")
                continue

            # Execute tool
            if self.verbose:
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[dim]Running tool [bold yellow]{escape(step.action)}[/bold yellow]…[/dim]"),
                    transient=True,
                    console=console,
                ) as progress:
                    progress.add_task("", total=None)
                    observation = run_tool(step.action, step.action_input, self.tools)
            else:
                observation = run_tool(step.action, step.action_input, self.tools)
            self._print_observation(observation)

            # Feed observation back as a user message (standard ReAct trick)
            self.memory.add("user", f"Observation: {observation}\n\nContinue reasoning.")

        return "Agent reached maximum iterations without a final answer."
