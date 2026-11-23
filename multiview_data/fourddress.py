"""Zero-copy adapter for the native 4D-DRESS sequence layout."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .processed import DEFAULT_REQUIRED_KEYS, MissingData


DEFAULT_FOURDDRESS_PATHS = ()
_SUBJECT_RE = re.compile(r"\d{5}")
_OUTFITS = ("Inner", "Outer")


class FourDDressDataset:
    """Expose every 4D-DRESS motion sequence as one runnable avatar subject."""

    kind = "4ddress"
    aliases = ("4d-dress", "dress4d")

    def __init__(
        self,
        root: Path,
        *,
        required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
    ) -> None:
        self.root = self._resolve_root(root.expanduser().resolve())
        self.required_keys = required_keys
        sequences = self._sequence_dirs(self.root)
        if not sequences:
            raise SystemExit(
                "No 4D-DRESS sequences found. Expected "
                "<subject>/{Inner,Outer}/Take*/Capture under: "
                f"{self.root}"
            )
        self.subjects = [self._subject_dict(path) for path in sequences]

    @staticmethod
    def _resolve_root(root: Path) -> Path:
        if (
            (root / "Capture").is_dir()
            and root.name.startswith("Take")
            and root.parent.name in _OUTFITS
            and _SUBJECT_RE.fullmatch(root.parent.parent.name)
        ):
            return root
        candidates = (root, root / "4D-DRESS")
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            if _SUBJECT_RE.fullmatch(candidate.name):
                return candidate
            if any(
                path.is_dir() and _SUBJECT_RE.fullmatch(path.name)
                for path in candidate.iterdir()
            ):
                return candidate
        raise SystemExit(f"4D-DRESS dataset path is not a valid unpacked root: {root}")

    @staticmethod
    def _sequence_dirs(root: Path) -> list[Path]:
        if (root / "Capture").is_dir():
            return [root]
        subject_dirs = (
            [root]
            if _SUBJECT_RE.fullmatch(root.name)
            else [
                path
                for path in sorted(root.iterdir())
                if path.is_dir() and _SUBJECT_RE.fullmatch(path.name)
            ]
        )
        return [
            sequence
            for subject in subject_dirs
            for outfit in _OUTFITS
            for sequence in sorted((subject / outfit).glob("Take*"))
            if sequence.is_dir() and (sequence / "Capture").is_dir()
        ]

    @staticmethod
    def _has_camera_media(capture: Path, leaf: str) -> bool:
        return any(
            next((camera / leaf).glob("*.png"), None) is not None
            for camera in capture.iterdir()
            if camera.is_dir() and camera.name.isdigit() and (camera / leaf).is_dir()
        )

    def _subject_dict(self, sequence: Path) -> dict[str, Any]:
        subject_id, outfit, take = sequence.parts[-3:]
        capture = sequence / "Capture"
        calibration = capture / "cameras.pkl"
        smplx = sequence / "SMPLX"
        labels = sequence / "Semantic" / "labels"
        has_images = self._has_camera_media(capture, "images")
        has_masks = self._has_camera_media(capture, "masks")

        item: dict[str, Any] = {
            "dataset": self.kind,
            "subject_id": f"{subject_id}_{outfit}_{take}",
            "subject_path": sequence,
            "images": capture if has_images else None,
            "masks": capture if has_masks else None,
            "labels": labels if next(labels.glob("label-*.pkl"), None) else None,
            "calibration": calibration if calibration.is_file() else None,
            "camera_id_mapping": None,
            "smplx": smplx if next(smplx.glob("*_smplx.pkl"), None) else None,
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
        print("Warning: dataset is missing required keys for some subjects:")
        for entry in missing:
            print(f"  - {entry.subject_id}: {', '.join(entry.keys)}")


def default_fourddress_path() -> Path | None:
    for path in DEFAULT_FOURDDRESS_PATHS:
        try:
            return FourDDressDataset._resolve_root(path)
        except SystemExit:
            continue
    return None


def load_fourddress_dataset(
    path: Path,
    *,
    required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
) -> FourDDressDataset:
    return FourDDressDataset(path, required_keys=required_keys)
