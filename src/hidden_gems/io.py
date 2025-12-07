from __future__ import annotations

from pathlib import Path
import os


def repo_root() -> Path:
    # Return the repository root (two directories up from this file)
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def interim_dir() -> Path:
    return data_dir() / "interim"


def processed_dir() -> Path:
    return data_dir() / "processed"


def models_dir() -> Path:
    return repo_root() / "models"


def ensure_data_dirs():
    raw_dir().mkdir(parents=True, exist_ok=True)
    interim_dir().mkdir(parents=True, exist_ok=True)
    processed_dir().mkdir(parents=True, exist_ok=True)
    models_dir().mkdir(parents=True, exist_ok=True)


def raw_path(filename: str) -> Path:
    return raw_dir() / filename


def interim_path(filename: str) -> Path:
    return interim_dir() / filename


def processed_path(filename: str) -> Path:
    return processed_dir() / filename


def model_path(filename: str) -> Path:
    return models_dir() / filename
