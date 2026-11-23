"""Dataset adapter registry."""

from __future__ import annotations

from pathlib import Path

from .actorshq import default_actorshq_path, load_actorshq_dataset
from .config import resolve_dataset_path
from .dancenet3d import default_dancenet3d_path, load_dancenet3d_dataset
from .dna import default_dna_path, load_dna_dataset
from .fourddress import default_fourddress_path, load_fourddress_dataset
from .processed import (
    DEFAULT_REQUIRED_KEYS,
    ProcessedVCIDataset,
    load_processed_dataset,
)
from .thuman import default_thuman_path, load_thuman_dataset
from .talkbody4d import default_talkbody4d_path, load_talkbody4d_dataset


AVAILABLE_DATASETS = {
    "processed": load_processed_dataset,
    "vci-processed": load_processed_dataset,
    "thuman": load_thuman_dataset,
    "thuman4": load_thuman_dataset,
    "thuman4d": load_thuman_dataset,
    "dna": load_dna_dataset,
    "dna-rendering": load_dna_dataset,
    "dna_rendering": load_dna_dataset,
    "dancenet3d": load_dancenet3d_dataset,
    "dancenet": load_dancenet3d_dataset,
    "dance-net-3d": load_dancenet3d_dataset,
    "talkbody4d": load_talkbody4d_dataset,
    "talkbody": load_talkbody4d_dataset,
    "talk-body-4d": load_talkbody4d_dataset,
    "4ddress": load_fourddress_dataset,
    "4d-dress": load_fourddress_dataset,
    "dress4d": load_fourddress_dataset,
    "actorshq": load_actorshq_dataset,
    "actors-hq": load_actorshq_dataset,
    "humanrf": load_actorshq_dataset,
}


def default_path_for_dataset(kind: str) -> Path | None:
    if kind in {"thuman", "thuman4", "thuman4d"}:
        return default_thuman_path()
    if kind in {"dna", "dna-rendering", "dna_rendering"}:
        return default_dna_path()
    if kind in {"dancenet3d", "dancenet", "dance-net-3d"}:
        return default_dancenet3d_path()
    if kind in {"talkbody4d", "talkbody", "talk-body-4d"}:
        return default_talkbody4d_path()
    if kind in {"4ddress", "4d-dress", "dress4d"}:
        return default_fourddress_path()
    if kind in {"actorshq", "actors-hq", "humanrf"}:
        return default_actorshq_path()
    return None


def normalize_dataset_kind(kind: str) -> str:
    normalized = kind.strip().lower()
    if normalized not in AVAILABLE_DATASETS:
        known = ", ".join(sorted(AVAILABLE_DATASETS))
        raise SystemExit(f"Unknown dataset kind {kind!r}. Available: {known}")
    return normalized


def parse_required_keys(value: str | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_REQUIRED_KEYS
    keys = tuple(key.strip() for key in value.split(",") if key.strip())
    if not keys:
        raise SystemExit("--required-keys must contain at least one key")
    return keys


def load_dataset(
    kind: str,
    *,
    path: Path | None = None,
    reset_path: bool = False,
    interactive: bool = True,
    persist_path: bool = True,
    required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
    warn_missing: bool = True,
) -> ProcessedVCIDataset:
    dataset_kind = normalize_dataset_kind(kind)
    dataset_default = (
        default_path_for_dataset(dataset_kind)
        if path is None and not reset_path
        else None
    )
    dataset_path = resolve_dataset_path(
        dataset_kind,
        path,
        reset=reset_path,
        interactive=interactive,
        default_path=dataset_default,
        persist=persist_path,
    )
    dataset = AVAILABLE_DATASETS[dataset_kind](
        dataset_path,
        required_keys=required_keys,
    )
    if warn_missing:
        dataset.warn_missing()
    return dataset
