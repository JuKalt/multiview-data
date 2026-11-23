"""Adapter for the stored THuman4D dataset layout."""

from __future__ import annotations

import json
from pathlib import Path

from .processed import DEFAULT_REQUIRED_KEYS, ProcessedVCIDataset


DEFAULT_THUMAN_PATHS = ()


class THumanDataset(ProcessedVCIDataset):
    """Expose original THuman subjects through the canonical zero-copy schema."""

    kind = "thuman"
    aliases = ("thuman4", "thuman4d")

    def __init__(
        self,
        root: Path,
        *,
        required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
    ) -> None:
        requested_root = root.expanduser().resolve()
        super().__init__(requested_root, required_keys=required_keys)
        self.descriptor_root: Path | None = None
        subject_parents = {item["subject_path"].parent for item in self.subjects}
        if len(subject_parents) == 1 and requested_root not in subject_parents:
            self.descriptor_root = requested_root
            self.root = subject_parents.pop()

    def _descriptor_subject(self, path: Path) -> Path | None:
        descriptor = path / "data.json"
        if not descriptor.is_file():
            return None
        try:
            data = json.loads(descriptor.read_text())
            calibration = Path(data["calibration"]).expanduser().resolve()
            smplx = Path(data["smplx_params"]).expanduser().resolve()
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return None
        subject = calibration.parent
        if (
            calibration.is_file()
            and smplx.is_file()
            and smplx.parent == subject
            and (subject / "images").is_dir()
            and (subject / "masks").is_dir()
        ):
            return subject
        return None

    def _subject_dirs(self) -> list[Path]:
        native_subjects = super()._subject_dirs()
        if native_subjects:
            return native_subjects
        original_subjects = {
            subject
            for path in sorted(self.root.iterdir())
            if path.is_dir()
            for subject in [self._descriptor_subject(path)]
            if subject is not None
        }
        return sorted(original_subjects)


def default_thuman_path() -> Path | None:
    for path in DEFAULT_THUMAN_PATHS:
        if path.is_dir():
            return path
    return None


def load_thuman_dataset(
    path: Path,
    *,
    required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
) -> THumanDataset:
    return THumanDataset(path, required_keys=required_keys)
