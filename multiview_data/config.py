"""Persistent dataset path configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path


def config_path() -> Path:
    override = os.environ.get("VCI_DATASET_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "vci-dataset" / "datasets.json"


def output_config_path() -> Path:
    override = os.environ.get("VCI_DATASET_OUTPUT_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "vci-dataset" / "outputs.json"


def load_config() -> dict[str, str]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Dataset config is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Dataset config must contain an object: {path}")
    return {str(key): str(value) for key, value in data.items()}


def save_config(config: dict[str, str]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def resolve_output_path(dataset_kind: str, explicit_path: Path | None, *, persist: bool = True) -> Path:
    path = output_config_path()
    if path.is_file():
        try:
            config = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Dataset output config is not valid JSON: {path}") from exc
        if not isinstance(config, dict):
            raise SystemExit(f"Dataset output config must contain an object: {path}")
    else:
        config = {}

    if explicit_path is not None:
        output = explicit_path.expanduser().resolve()
        if persist:
            config[dataset_kind] = str(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        return output

    stored = config.get(dataset_kind)
    if stored:
        return Path(stored).expanduser().resolve()
    raise SystemExit(
        f"No stored output path for dataset kind {dataset_kind!r}. "
        "Pass --output-dir once to store it."
    )


def reset_dataset_path(dataset_kind: str) -> None:
    config = load_config()
    if dataset_kind in config:
        del config[dataset_kind]
        save_config(config)


def resolve_dataset_path(
    dataset_kind: str,
    explicit_path: Path | None,
    *,
    reset: bool,
    interactive: bool,
    default_path: Path | None = None,
    persist: bool = True,
) -> Path:
    if reset and persist:
        reset_dataset_path(dataset_kind)

    config = load_config()
    if reset and not persist:
        config.pop(dataset_kind, None)
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if persist:
            config[dataset_kind] = str(path)
            save_config(config)
        return path

    stored = config.get(dataset_kind)
    if stored:
        return Path(stored).expanduser().resolve()

    if default_path is not None:
        path = default_path.expanduser().resolve()
        if persist:
            config[dataset_kind] = str(path)
            save_config(config)
        return path

    if not interactive:
        raise SystemExit(
            f"No stored path for dataset kind {dataset_kind!r}. "
            "Pass --path or run interactively once."
        )

    answer = input(f"Path for dataset kind '{dataset_kind}': ").strip()
    if not answer:
        raise SystemExit("No dataset path provided.")
    path = Path(answer).expanduser().resolve()
    if persist:
        config[dataset_kind] = str(path)
        save_config(config)
    return path
