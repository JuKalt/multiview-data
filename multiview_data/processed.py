"""Adapter for this repository's processed VCI/THUman-style dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

CANONICAL_KEYS = (
    "dataset",
    "subject_id",
    "subject_path",
    "images",
    "masks",
    "labels",
    "calibration",
    "camera_id_mapping",
    "smplx",
    "mhr",
    "missing_keys",
)

# Optional annotations extend only this adapter; shared canonical keys remain stable.
ANNOTATION_PATHS = {
    'smplx_confidence': 'smpl_params_confidence.npz',
    'mhrfromsmplx': 'mhr_from_smplx_params.npz',
    'semantic_general': 'semantic_parsing/general',
    'semantic_detailed': 'semantic_parsing/detailed',
    'semantic_confidence': 'semantic_parsing/confidence',
    'semantic_schema': 'semantic_parsing/schema.json',
    'semantic_metrics': 'semantic_parsing/metrics.json',
    'fit_metrics': 'fit_metrics.json',
    'subject_manifest': 'subject_manifest.json',
}

DEFAULT_REQUIRED_KEYS = ("images", "masks", "calibration", "smplx", "mhr")


@dataclass(frozen=True)
class MissingData:
    subject_id: str
    keys: tuple[str, ...]


class ProcessedVCIDataset:
    """Load processed subjects as dictionaries with stable keys."""

    kind = "processed"
    aliases = ("vci-processed", "thuman4")

    def __init__(
        self,
        root: Path,
        *,
        required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
    ) -> None:
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise SystemExit(f"Dataset path is not a directory: {self.root}")
        self.required_keys = required_keys
        self.subjects = [self._subject_dict(path) for path in self._subject_dirs()]

    def _subject_dirs(self) -> list[Path]:
        if (self.root / "images").is_dir():
            return [self.root]
        subjects: list[Path] = []
        for path in sorted(self.root.iterdir()):
            try:
                if path.is_dir() and (path / "images").is_dir():
                    subjects.append(path)
            except OSError:
                continue
        return subjects

    def _subject_dict(self, subject_path: Path) -> dict[str, Any]:
        images = subject_path / "images"
        masks = subject_path / "masks"
        labels = subject_path / "labels"
        calibration = subject_path / "calibration.json"
        camera_id_mapping = subject_path / "camera_id_mapping.json"
        smplx = subject_path / "smpl_params.npz"
        mhr = subject_path / "mhr_params.npz"

        item: dict[str, Any] = {
            "dataset": self.kind,
            "subject_id": subject_path.name,
            "subject_path": subject_path,
            "images": images if images.is_dir() else None,
            "masks": masks if masks.is_dir() else None,
            "labels": labels if labels.is_dir() else None,
            "calibration": calibration if calibration.is_file() else None,
            "camera_id_mapping": camera_id_mapping if camera_id_mapping.is_file() else None,
            "smplx": smplx if smplx.is_file() else None,
            "mhr": mhr if mhr.is_file() else None,
        }
        for key, relative in ANNOTATION_PATHS.items():
            path = subject_path / relative
            is_directory = key in ('semantic_general', 'semantic_detailed', 'semantic_confidence')
            item[key] = path if (path.is_dir() if is_directory else path.is_file()) else None
        if item['labels'] is None:
            item['labels'] = item['semantic_general']
        item["missing_keys"] = tuple(
            key for key in self.required_keys if item.get(key) is None
        )
        return {key: item[key] for key in (*CANONICAL_KEYS, *ANNOTATION_PATHS)}

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

def load_processed_dataset(
    path: Path,
    *,
    required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
) -> ProcessedVCIDataset:
    return ProcessedVCIDataset(path, required_keys=required_keys)
