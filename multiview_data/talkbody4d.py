"""Zero-copy adapter for the native TalkBody4D layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .processed import DEFAULT_REQUIRED_KEYS, MissingData


DEFAULT_TALKBODY4D_PATHS = ()


def _nested_directory(subject: Path, name: str) -> Path:
    direct = subject / name
    nested = direct / name
    return nested if nested.is_dir() else direct


class TalkBody4DDataset:
    """Expose each native TalkBody4D actor as one canonical subject."""

    kind = "talkbody4d"
    aliases = ("talkbody", "talk-body-4d")

    def __init__(
        self,
        root: Path,
        *,
        required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
    ) -> None:
        self.root = self._resolve_root(root.expanduser().resolve())
        self.required_keys = required_keys
        self.subjects = [self._subject_dict(path) for path in self._subject_dirs()]
        if not self.subjects:
            raise SystemExit(
                "No TalkBody4D actors found. Expected actor directories containing "
                f"calibration.json under: {self.root}"
            )

    @staticmethod
    def _resolve_root(root: Path) -> Path:
        if (root / "calibration.json").is_file():
            return root
        for candidate in (root, root / "data"):
            if candidate.is_dir() and any(
                path.is_dir() and (path / "calibration.json").is_file()
                for path in candidate.iterdir()
            ):
                return candidate
        raise SystemExit(f"TalkBody4D dataset path is not a valid root: {root}")

    def _subject_dirs(self) -> list[Path]:
        if (self.root / "calibration.json").is_file():
            return [self.root]
        return [
            path
            for path in sorted(self.root.iterdir())
            if path.is_dir() and (path / "calibration.json").is_file()
        ]

    def _subject_dict(self, subject: Path) -> dict[str, Any]:
        videos = _nested_directory(subject, "raw_videos")
        mattings = _nested_directory(subject, "mattings")
        smplx = _nested_directory(subject, "smplx_fitting")
        calibration = subject / "calibration.json"
        item: dict[str, Any] = {
            "dataset": self.kind,
            "subject_id": subject.name,
            "subject_path": subject,
            "images": videos if next(videos.glob("*.mp4"), None) else None,
            "masks": mattings if next(mattings.glob("*/*.png"), None) else None,
            "labels": None,
            "calibration": calibration if calibration.is_file() else None,
            "camera_id_mapping": None,
            "smplx": smplx if next(smplx.glob("*.json"), None) else None,
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
        print("Warning: TalkBody4D is missing required keys for some actors:")
        for entry in missing:
            print(f"  - {entry.subject_id}: {', '.join(entry.keys)}")


def default_talkbody4d_path() -> Path | None:
    for path in DEFAULT_TALKBODY4D_PATHS:
        try:
            return TalkBody4DDataset._resolve_root(path)
        except SystemExit:
            continue
    return None


def load_talkbody4d_dataset(
    path: Path,
    *,
    required_keys: tuple[str, ...] = DEFAULT_REQUIRED_KEYS,
) -> TalkBody4DDataset:
    return TalkBody4DDataset(path, required_keys=required_keys)
