"""Zero-copy adapter for the native ActorsHQ actor/sequence layout."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .processed import DEFAULT_REQUIRED_KEYS, MissingData


DEFAULT_ACTORSHQ_PATHS = ()
_SCALE_RE = re.compile(r"([124])x")


class ActorsHQDataset:
    """Expose every downloaded ActorsHQ sequence as one canonical subject."""

    kind = "actorshq"
    aliases = ("actors-hq", "humanrf")

    def __init__(self, root: Path, *, required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS) -> None:
        requested = root.expanduser().resolve()
        self.root = self._resolve_root(requested)
        self.required_keys = required_keys
        sequences = self._sequence_dirs(self.root)
        if not sequences:
            raise SystemExit(
                "No ActorsHQ sequences found. Expected "
                "Actor*/Sequence*/{1x,2x,4x}/{rgbs,masks,calibration.csv} under: "
                f"{self.root}"
            )
        self.subjects = [self._subject_dict(sequence) for sequence in sequences]

    @staticmethod
    def _is_scale(path: Path) -> bool:
        return bool(
            _SCALE_RE.fullmatch(path.name)
            and (path / "calibration.csv").is_file()
            and (path / "rgbs").is_dir()
            and (path / "masks").is_dir()
        )

    @classmethod
    def _is_sequence(cls, path: Path) -> bool:
        return path.is_dir() and any(
            cls._is_scale(child) for child in path.iterdir() if child.is_dir()
        )

    @classmethod
    def _resolve_root(cls, root: Path) -> Path:
        if cls._is_scale(root) or cls._is_sequence(root):
            return root
        if root.is_dir() and (
            any(cls._is_sequence(path) for path in root.iterdir() if path.is_dir())
            or any(
                cls._is_sequence(sequence)
                for actor in root.iterdir()
                if actor.is_dir()
                for sequence in actor.iterdir()
                if sequence.is_dir()
            )
        ):
            return root
        raise SystemExit(f"ActorsHQ dataset path is not a valid downloaded root: {root}")

    @classmethod
    def _sequence_dirs(cls, root: Path) -> list[Path]:
        if cls._is_scale(root):
            return [root.parent]
        if cls._is_sequence(root):
            return [root]
        direct = [path for path in root.iterdir() if cls._is_sequence(path)]
        nested = [
            sequence
            for actor in root.iterdir()
            if actor.is_dir()
            for sequence in actor.iterdir()
            if cls._is_sequence(sequence)
        ]
        return sorted({*direct, *nested})

    def _requested_scale(self, sequence: Path) -> Path:
        if self._is_scale(self.root) and self.root.parent == sequence:
            return self.root
        scales = sorted(
            (path for path in sequence.iterdir() if self._is_scale(path)),
            key=lambda path: int(path.name[:-1]),
        )
        return scales[0]

    @staticmethod
    def _smplx_path(sequence: Path, scale: Path) -> Path | None:
        for path in (sequence / "smpl_params.npz", scale / "smpl_params.npz"):
            if path.is_file():
                return path
        return None

    def _subject_dict(self, sequence: Path) -> dict[str, Any]:
        scale = self._requested_scale(sequence)
        item: dict[str, Any] = {
            "dataset": self.kind,
            "subject_id": f"{sequence.parent.name}_{sequence.name}",
            "subject_path": sequence,
            "images": scale / "rgbs",
            "masks": scale / "masks",
            "labels": None,
            "calibration": scale / "calibration.csv",
            "camera_id_mapping": None,
            "smplx": self._smplx_path(sequence, scale),
            "mhr": None,
        }
        item["missing_keys"] = tuple(
            key for key in self.required_keys if item.get(key) is None
        )
        return item

    def missing_data(self) -> list[MissingData]:
        return [
            MissingData(str(item["subject_id"]), tuple(item["missing_keys"]))
            for item in self.subjects
            if item["missing_keys"]
        ]

    def warn_missing(self) -> None:
        missing = self.missing_data()
        if not missing:
            return
        print("Warning: ActorsHQ is missing required keys for some sequences:")
        for entry in missing:
            print(f"  - {entry.subject_id}: {', '.join(entry.keys)}")
        if any("smplx" in entry.keys for entry in missing):
            print(
                "ActorsHQ does not include SMPL-X fits. Add a synchronized "
                "smpl_params.npz beside each sequence's scene.json before running "
                "the configured avatar methods."
            )


def validate_actorshq_selection(subject: dict[str, Any], selection: dict) -> None:
    """Validate explicit source camera IDs without decoding media."""
    import csv

    prefix = f"Run config subject {subject['subject_id']}"
    try:
        with Path(subject["calibration"]).open(newline="", encoding="utf-8") as stream:
            cameras = {
                int(row["name"].removeprefix("Cam")) - 1
                for row in csv.DictReader(stream)
                if row["name"].startswith("Cam")
            }
        missing = set(selection.get("camera_ids", ())) - cameras
        if missing:
            raise ValueError(f"unavailable camera IDs: {sorted(missing)}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{prefix}: {exc}") from exc


def default_actorshq_path() -> Path | None:
    for path in DEFAULT_ACTORSHQ_PATHS:
        try:
            return ActorsHQDataset._resolve_root(path)
        except SystemExit:
            continue
    return None


def load_actorshq_dataset(
    path: Path, *, required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS
) -> ActorsHQDataset:
    return ActorsHQDataset(path, required_keys=required_keys)
