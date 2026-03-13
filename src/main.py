"""
Entry point for the agentic pipeline.

Usage (inside container):
    python -m src.main                # interactive REPL
    python -m src.main "your task"    # single-shot task
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.prompt import Prompt

from src.agent.pipeline import AgentPipeline
from src.agent.tools import get_enabled_tools
from src.llm.provider import build_provider
from src.utils import load_config

console = Console()


def build_pipeline(cfg: dict) -> AgentPipeline:
    llm_cfg = cfg.get("llm", {})
    agent_cfg = cfg.get("agent", {})
    tools_cfg = cfg.get("tools", {})

    llm = build_provider(llm_cfg)
    tools = get_enabled_tools(tools_cfg)
    system_prompt = agent_cfg.get(
        "system_prompt",
        "You are a helpful AI assistant.\n\nAvailable tools:\n{tool_descriptions}",
    )

    return AgentPipeline(
        llm=llm,
        tools=tools,
        system_prompt_template=system_prompt,
        max_iterations=int(agent_cfg.get("max_iterations", 20)),
        verbose=bool(agent_cfg.get("verbose", True)),
    )


def main() -> None:
    cfg = load_config()
    pipeline = build_pipeline(cfg)

    llm_cfg = cfg.get("llm", {})
    model = llm_cfg.get("model", "?")
    provider = llm_cfg.get("provider", "ollama")

    console.print(f"\n[bold green]Agentic Pipeline[/bold green]  "
                  f"provider=[cyan]{provider}[/cyan]  model=[cyan]{model}[/cyan]")
    console.print("[dim]Type a task and press Enter. Type 'exit' or Ctrl-C to quit.[/dim]\n")

    # Single-shot mode: task passed as CLI argument
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        answer = pipeline.run(task)
        console.print(f"\n[bold green]Answer:[/bold green] {answer}\n")
        return

    # Interactive REPL
    while True:
        try:
            task = Prompt.ask("[bold blue]Task[/bold blue]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if task.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if not task.strip():
            continue

        try:
            answer = pipeline.run(task)
            console.print(f"\n[bold green]Answer:[/bold green] {answer}\n")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")


if __name__ == "__main__":
    main()
