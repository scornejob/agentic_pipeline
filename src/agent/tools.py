"""
Tool registry and built-in tool implementations.

Each tool is a callable that receives a single string input and returns a string.
All tools are registered via @tool() decorator.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import subprocess
import textwrap
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests


# ── Tool descriptor ───────────────────────────────────────────────────────────

@dataclass
class Tool:
    name: str
    description: str
    usage: str                          # shown in the system prompt
    fn: Callable[[str], str]
    enabled: bool = True


_REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, usage: str):
    """Decorator that registers a function as a named tool."""
    def decorator(fn: Callable[[str], str]) -> Callable[[str], str]:
        _REGISTRY[name] = Tool(name=name, description=description, usage=usage, fn=fn)
        return fn
    return decorator


def get_registry() -> dict[str, Tool]:
    return _REGISTRY


def get_enabled_tools(cfg: dict) -> dict[str, Tool]:
    """Return only the tools that are enabled in config."""
    enabled: dict[str, Tool] = {}
    for name, t in _REGISTRY.items():
        tool_cfg = cfg.get(name, {})
        if tool_cfg.get("enabled", True):
            t.enabled = True
            enabled[name] = t
        else:
            t.enabled = False
    return enabled


def tool_descriptions(tools: dict[str, Tool]) -> str:
    lines = []
    for t in tools.values():
        lines.append(f"- {t.name}: {t.description}")
        lines.append(f"  Usage: {t.usage}")
    return "\n".join(lines)


def run_tool(name: str, input_str: str, tools: dict[str, Tool]) -> str:
    if name not in tools:
        return f"Error: unknown tool '{name}'. Available: {', '.join(tools)}"
    try:
        return tools[name].fn(input_str)
    except Exception as exc:
        return f"Tool '{name}' raised an exception: {exc}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_input(input_str: str) -> dict:
    """Parse a JSON object from the model's Action Input block."""
    try:
        data = json.loads(input_str)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Fallback: treat raw string as {"input": input_str}
    return {"input": input_str.strip()}


def _safe_path(p: str, base_dir: str) -> Path:
    """Resolve path and ensure it stays inside base_dir."""
    base = Path(base_dir).resolve()
    resolved = (base / p).resolve()
    if not str(resolved).startswith(str(base)):
        raise PermissionError(f"Path '{p}' escapes the allowed directory '{base_dir}'")
    return resolved


# ── Built-in tools ────────────────────────────────────────────────────────────

@tool(
    name="calculator",
    description="Evaluate a mathematical expression. Safe: only math operations allowed.",
    usage='Action Input: {"expression": "2 ** 10 + sqrt(144)"}',
)
def _calculator(input_str: str) -> str:
    data = _parse_json_input(input_str)
    expr = data.get("expression", input_str).strip()
    # Whitelist: digits, operators, parens, spaces, dots, math function names
    allowed = re.compile(r'^[\d\s\+\-\*/\(\)\.\^%,]+$')
    safe_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_names["abs"] = abs
    safe_names["round"] = round
    if not allowed.match(re.sub(r'[a-zA-Z_,]+', '', expr)):
        pass  # allow math names; validated below
    try:
        result = eval(expr, {"__builtins__": {}}, safe_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool(
    name="python_eval",
    description="Execute a Python snippet and return stdout. No network access inside the snippet.",
    usage='Action Input: {"code": "for i in range(3): print(i)"}',
)
def _python_eval(input_str: str) -> str:
    cfg_timeout = int(os.environ.get("PYTHON_EVAL_TIMEOUT", "10"))
    data = _parse_json_input(input_str)
    code = data.get("code", input_str)
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    local_ns: dict = {}
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(textwrap.dedent(code), {"__builtins__": __builtins__}, local_ns)  # noqa: S102
    except Exception as exc:
        return f"Error: {exc}\nStderr: {stderr_buf.getvalue()}"
    output = stdout_buf.getvalue()
    if not output:
        # Return the last expression if nothing was printed
        last = list(local_ns.values())[-1] if local_ns else None
        output = str(last) if last is not None else "(no output)"
    return output[:4000]  # cap at 4 KB


@tool(
    name="shell",
    description="Run a shell command and capture its output. Use responsibly.",
    usage='Action Input: {"command": "ls -la /app/workspace"}',
)
def _shell(input_str: str) -> str:
    timeout = int(os.environ.get("SHELL_TIMEOUT", "30"))
    data = _parse_json_input(input_str)
    cmd = data.get("command", input_str)
    try:
        result = subprocess.run(
            cmd,
            shell=True,        # noqa: S602
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/app/workspace",
        )
        out = result.stdout + result.stderr
        return out[:4000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"


@tool(
    name="file_read",
    description="Read a text file from the agent workspace.",
    usage='Action Input: {"path": "notes.txt"}',
)
def _file_read(input_str: str) -> str:
    base_dir = os.environ.get("FILE_BASE_DIR", "/app/workspace")
    data = _parse_json_input(input_str)
    path = data.get("path", input_str).strip()
    try:
        full_path = _safe_path(path, base_dir)
        content = full_path.read_text(encoding="utf-8")
        return content[:8000]
    except PermissionError as exc:
        return f"Error: {exc}"
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool(
    name="file_write",
    description="Write text to a file in the agent workspace. Creates parent directories as needed.",
    usage='Action Input: {"path": "output.txt", "content": "Hello, world!"}',
)
def _file_write(input_str: str) -> str:
    base_dir = os.environ.get("FILE_BASE_DIR", "/app/workspace")
    data = _parse_json_input(input_str)
    path = data.get("path", "").strip()
    content = data.get("content", "")
    if not path:
        return "Error: 'path' is required"
    try:
        full_path = _safe_path(path, base_dir)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"Written {len(content)} characters to {path}"
    except PermissionError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error writing file: {exc}"


@tool(
    name="web_fetch",
    description="Fetch the text content of a URL. Returns a truncated plaintext version.",
    usage='Action Input: {"url": "https://example.com/some-page"}',
)
def _web_fetch(input_str: str) -> str:
    max_length = int(os.environ.get("WEB_FETCH_MAX_LENGTH", "50000"))
    timeout = int(os.environ.get("WEB_FETCH_TIMEOUT", "15"))
    data = _parse_json_input(input_str)
    url = data.get("url", input_str).strip()
    # Basic URL validation — only allow http/https
    if not re.match(r'^https?://', url, re.IGNORECASE):
        return "Error: only http/https URLs are allowed"
    headers = {"User-Agent": "AgenticPipeline/1.0 (research bot)"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type:
            # Very light HTML stripping (no external deps required)
            text = re.sub(r'<style[^>]*>.*?</style>', '', resp.text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s{2,}', ' ', text).strip()
        else:
            text = resp.text
        return text[:max_length]
    except requests.exceptions.RequestException as exc:
        return f"Error fetching URL: {exc}"
