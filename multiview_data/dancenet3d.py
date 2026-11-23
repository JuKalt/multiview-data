"""Adapter for processed DanceNet3D sequences."""

from __future__ import annotations

from pathlib import Path

from .processed import DEFAULT_REQUIRED_KEYS, ProcessedVCIDataset


DEFAULT_DANCENET3D_PATHS = ()


class DanceNet3DDataset(ProcessedVCIDataset):
    kind = "dancenet3d"
    aliases = ("dancenet", "dance-net-3d")


def default_dancenet3d_path() -> Path | None:
    return next((path for path in DEFAULT_DANCENET3D_PATHS if path.is_dir()), None)


def load_dancenet3d_dataset(
    path: Path,
    *,
    required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
) -> DanceNet3DDataset:
    return DanceNet3DDataset(path, required_keys=required_keys)
