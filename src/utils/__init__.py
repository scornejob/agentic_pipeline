"""
Shared config loader.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

_CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/app/config/config.yaml"))


def load_config() -> dict:
    load_dotenv()  # load .env if present
    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open() as f:
            return yaml.safe_load(f) or {}
    return {}
