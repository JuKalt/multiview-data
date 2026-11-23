"""Lightweight view and synchronized-frame capacity discovery."""

from __future__ import annotations

import ast
import csv
import json
import re
import struct
import subprocess
import zipfile
from pathlib import Path
from typing import Any

_FRAME_RE = re.compile(r"(\d+)$")
_FOURDDRESS_SMPLX_RE = re.compile(r"f(\d+)_smplx$")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _frame_id(path: Path) -> int | None:
    match = _FRAME_RE.search(path.stem)
    return int(match.group(1)) if match else None


def _media_frames(
    root: Path, *, capture_leaf: str | None = None
) -> dict[int, set[int]]:
    result = {}
    for camera in root.iterdir():
        if not camera.is_dir():
            continue
        if capture_leaf is None:
            if not camera.name.startswith("cam") or not camera.name[3:].isdigit():
                continue
            camera_id = int(camera.name[3:])
            media = camera
        else:
            if not camera.name.isdigit():
                continue
            camera_id = int(camera.name)
            media = camera / capture_leaf
        if not media.is_dir():
            continue
        frames = {
            value
            for path in media.iterdir()
            if path.suffix.lower() in _IMAGE_EXTENSIONS
            if (value := _frame_id(path)) is not None
        }
        if frames:
            result[camera_id] = frames
    return result


def _npz_frame_count(path: Path) -> int | None:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            name = next(
                (value for value in names if Path(value).stem == "transl"),
                next(
                    (
                        value
                        for value in names
                        if Path(value).stem == "global_orient"
                    ),
                    None,
                ),
            )
            if name is None:
                return None
            with archive.open(name) as stream:
                if stream.read(6) != b"\x93NUMPY":
                    return None
                major, _minor = stream.read(2)
                width = 2 if major == 1 else 4
                fmt = "<H" if major == 1 else "<I"
                header_size = struct.unpack(fmt, stream.read(width))[0]
                header = ast.literal_eval(
                    stream.read(header_size).decode("latin1")
                )
                shape = header.get("shape", ())
                return int(shape[0]) if shape else None
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None

def _actorshq_media_frames(root: Path, kind: str) -> dict[int, set[int]]:
    pattern = re.compile(rf"Cam(\d+)_{kind}(\d+)$")
    result = {}
    for camera in root.iterdir():
        if not camera.is_dir():
            continue
        for path in camera.iterdir():
            match = pattern.fullmatch(path.stem)
            if path.suffix.lower() in _IMAGE_EXTENSIONS and match:
                camera_id = int(match.group(1)) - 1
                result.setdefault(camera_id, set()).add(int(match.group(2)))
    return result


def _file_capacity(kind: str, subject: dict[str, Any]) -> dict[str, Any]:
    if kind == "actorshq":
        images = _actorshq_media_frames(Path(subject["images"]), "rgb")
        masks = _actorshq_media_frames(Path(subject["masks"]), "mask")
        with Path(subject["calibration"]).open(newline="", encoding="utf-8") as stream:
            calibrated = {int(row["name"].removeprefix("Cam")) - 1 for row in csv.DictReader(stream)}
        cameras = sorted(set(images) & set(masks) & calibrated)
        count = _npz_frame_count(Path(subject["smplx"]))
        smplx_frames = set(range(count)) if count is not None else None
    elif kind == "talkbody4d":
        videos = {
            int(path.stem): path
            for path in Path(subject["images"]).glob("*.mp4")
            if path.stem.isdigit()
        }
        images = {camera: set() for camera in videos}
        masks = _media_frames(Path(subject["masks"]), capture_leaf="")
        calibration = json.loads(Path(subject["calibration"]).read_text())
        calibrated = {
            int(name)
            for name in calibration.get("cameras", {})
            if str(name).isdigit()
        } & {
            int(str(name).split()[0])
            for name in calibration.get("camera_poses", {})
            if str(name).split()[0].isdigit()
        }
        cameras = sorted(set(videos) & set(masks) & calibrated)
        smplx_frames = {
            int(path.stem)
            for path in Path(subject["smplx"]).glob("*.json")
            if path.stem.isdigit()
        }
        images = {camera: masks[camera] for camera in cameras}
    elif kind == "4ddress":
        capture = Path(subject["images"])
        images = _media_frames(capture, capture_leaf="images")
        masks = _media_frames(capture, capture_leaf="masks")
        cameras = sorted(set(images) & set(masks))
        smplx_frames = {
            int(match.group(1))
            for path in Path(subject["smplx"]).glob("*_smplx.pkl")
            if (match := _FOURDDRESS_SMPLX_RE.search(path.stem)) is not None
        }
    else:
        images = _media_frames(Path(subject["images"]))
        masks = _media_frames(Path(subject["masks"]))
        calibration = json.loads(Path(subject["calibration"]).read_text())
        calibrated = {
            int(str(name)[3:])
            for name in calibration
            if str(name).startswith("cam") and str(name)[3:].isdigit()
        }
        cameras = sorted(set(images) & set(masks) & calibrated)
        count = _npz_frame_count(Path(subject["smplx"]))
        smplx_frames = set(range(count)) if count is not None else None

    common: set[int] | None = None
    for camera in cameras:
        paired = images[camera] & masks[camera]
        common = paired if common is None else common & paired
    common = common or set()
    if smplx_frames is not None:
        common &= smplx_frames
    return {
        "subject_id": str(subject["subject_id"]),
        "views": len(cameras),
        "timesteps": len(common),
    }


def _dna_capacities(
    subjects: list[dict[str, Any]],
    commands: list[list[str]],
    helper: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    payload = json.dumps(
        [
            {
                "subject_id": str(subject["subject_id"]),
                "images": str(subject["images"]),
                "smplx": str(subject["smplx"]),
            }
            for subject in subjects
        ]
    )
    errors = []
    for command in commands:
        result = subprocess.run(
            [*command, str(helper)],
            input=payload,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            try:
                return json.loads(result.stdout), None
            except json.JSONDecodeError:
                errors.append("metadata helper returned invalid JSON")
        else:
            errors.append(result.stderr.strip() or "metadata helper failed")
    return [], errors[-1] if errors else "no Python environment can read DNA metadata"


def dataset_capacity(
    kind: str,
    subjects: list[dict[str, Any]],
    *,
    dna_commands: list[list[str]],
    dna_helper: Path,
    verbose: bool,
) -> dict[str, Any]:
    entries = []
    error = None
    if kind == "dna":
        entries, error = _dna_capacities(subjects, dna_commands, dna_helper)
    else:
        for subject in subjects:
            try:
                entries.append(_file_capacity(kind, subject))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                error = f"{subject.get('subject_id')}: {exc}"

    measured = [
        entry
        for entry in entries
        if entry["views"] > 0 and entry["timesteps"] > 0
    ]
    result: dict[str, Any] = {
        "measured_subjects": len(measured),
        "views": {
            "min": min((entry["views"] for entry in measured), default=None),
            "max": max((entry["views"] for entry in measured), default=None),
        },
        "timesteps": {
            "min": min(
                (entry["timesteps"] for entry in measured), default=None
            ),
            "max": max(
                (entry["timesteps"] for entry in measured), default=None
            ),
        },
        "error": error,
    }
    if verbose:
        result["subjects"] = entries
    return result
