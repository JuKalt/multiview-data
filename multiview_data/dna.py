"""Zero-copy adapter for the DNA-Rendering Part 1 SMC layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .processed import DEFAULT_REQUIRED_KEYS, MissingData

DEFAULT_DNA_PATHS = ()
DNA_DIRECTORIES = {
    "smplx": "dna_rendering_part1_annotations",
    "images": "dna_rendering_part1_main",
}


def _sequence_id(path: Path, key: str) -> str:
    suffix = {"smplx": "_annots", "images": ""}[key]
    if suffix and not path.stem.endswith(suffix):
        return ""
    return path.stem[: -len(suffix)] if suffix else path.stem


class DNADataset:
    """Expose original DNA sequences without extracting or copying SMC files."""

    kind = "dna"
    aliases = ("dna-rendering", "dna_rendering")

    def __init__(
        self,
        root: Path,
        *,
        required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
    ) -> None:
        self.root = self._resolve_root(root.expanduser().resolve())
        self.required_keys = required_keys
        self._indexes = {
            key: self._index_smc(self.root / directory, key)
            for key, directory in DNA_DIRECTORIES.items()
        }
        sequence_ids = sorted(
            set().union(*(set(index) for index in self._indexes.values()))
        )
        self.subjects = [self._subject_dict(value) for value in sequence_ids]

    @staticmethod
    def _resolve_root(root: Path) -> Path:
        for candidate in (root, root / "Part1"):
            if any(
                (candidate / directory).is_dir()
                for directory in DNA_DIRECTORIES.values()
            ):
                return candidate
        expected = ", ".join(DNA_DIRECTORIES.values())
        raise SystemExit(
            f"DNA dataset path must contain a Part1 RGB or annotation directory ({expected}): {root}"
        )

    @staticmethod
    def _index_smc(directory: Path, key: str) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for path in sorted(directory.rglob("*.smc")):
            sequence_id = _sequence_id(path, key)
            if not sequence_id:
                continue
            if sequence_id in index:
                raise SystemExit(
                    f"Multiple DNA {key} SMC files found for sequence {sequence_id}: "
                    f"{index[sequence_id]} and {path}"
                )
            index[sequence_id] = path.resolve()
        return index

    def _subject_dict(self, sequence_id: str) -> dict[str, Any]:
        annotations = self._indexes["smplx"].get(sequence_id)
        item: dict[str, Any] = {
            "dataset": self.kind,
            "subject_id": sequence_id,
            "subject_path": self.root,
            "images": self._indexes["images"].get(sequence_id),
            "masks": annotations,
            "labels": None,
            "calibration": annotations,
            "camera_id_mapping": None,
            "smplx": annotations,
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


def default_dna_path() -> Path | None:
    for path in DEFAULT_DNA_PATHS:
        try:
            return DNADataset._resolve_root(path)
        except SystemExit:
            continue
    return None


def load_dna_dataset(
    path: Path,
    *,
    required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
) -> DNADataset:
    return DNADataset(path, required_keys=required_keys)
