"""Loads config.toml so every process reads the same settings."""

import tomllib
from pathlib import Path

TABLE_TENNIS_DIR = Path(__file__).resolve().parent.parent


def load_config():
    with open(TABLE_TENNIS_DIR / "config.toml", "rb") as f:
        return tomllib.load(f)
