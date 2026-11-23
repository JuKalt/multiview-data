from __future__ import annotations

import base64
import json
import os
from pathlib import Path


_RGB_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAFElEQVR4nGNscFBgwAaYsIoOWgkAtKgA8A9Nk3kAAAAASUVORK5CYII="
)
_MASK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAAAAADhZOFXAAAAEElEQVR4nGP8zwABTAwUMQBJQQEPeWDz+AAAAABJRU5ErkJggg=="
)


def create_subject(root: Path, name: str = "subject00") -> Path:
    subject = root / name
    image_dir = subject / "images" / "cam00"
    mask_dir = subject / "masks" / "cam00"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (image_dir / "00000000.png").write_bytes(_RGB_PNG)
    (mask_dir / "00000000.png").write_bytes(_MASK_PNG)
    (subject / "calibration.json").write_text(
        json.dumps(
            {
                "cam00": {
                    "K": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "T": [0.0, 0.0, 0.0],
                }
            }
        )
    )
    # Dataset adapters only validate the presence of parameter files. Runtime
    # method smoke tests replace this sentinel with a valid NPZ file.
    (subject / "smpl_params.npz").write_bytes(b"smoke-test")
    return subject


def create_fourddress_root(
    root: Path,
    *,
    subject_id: str = "00122",
    outfit: str = "Inner",
    take: str = "Take2",
) -> dict[str, Path]:
    sequence = root / subject_id / outfit / take
    capture = sequence / "Capture"
    camera = capture / "0004"
    images = camera / "images"
    masks = camera / "masks"
    labels = sequence / "Semantic" / "labels"
    smplx = sequence / "SMPLX"
    for directory in (images, masks, labels, smplx):
        directory.mkdir(parents=True, exist_ok=True)
    (images / "capture-f00011.png").write_bytes(_RGB_PNG)
    (masks / "mask-f00011.png").write_bytes(_MASK_PNG)
    (labels / "label-f00011.pkl").write_bytes(b"4D-DRESS label fixture")
    (smplx / "mesh-f00011_smplx.pkl").write_bytes(b"4D-DRESS SMPL-X fixture")
    (capture / "cameras.pkl").write_bytes(b"4D-DRESS camera fixture")
    (sequence / "basic_info.pkl").write_bytes(b"4D-DRESS info fixture")
    return {
        "sequence": sequence,
        "capture": capture,
        "labels": labels,
        "calibration": capture / "cameras.pkl",
        "smplx": smplx,
    }


def create_dna_root(root: Path, sequence_id: str = "0001_01") -> dict[str, Path]:
    directories = {
        "smplx": root / "dna_rendering_part1_annotations",
        "images": root / "dna_rendering_part1_main",
        "masks": root / "dna_rendering_part1_kinect",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "smplx": directories["smplx"] / f"{sequence_id}_annots.smc",
        "images": directories["images"] / f"{sequence_id}.smc",
        "kinect": directories["masks"] / f"{sequence_id}_kinect.smc",
    }
    for path in paths.values():
        path.write_bytes(b"DNA SMC contract fixture")
    return paths | {"masks": paths["smplx"]}


def create_talkbody4d_root(root: Path, actor_id: str = "XG_01") -> dict[str, Path]:
    actor = root / actor_id
    videos = actor / "raw_videos" / "raw_videos"
    masks = actor / "mattings" / "mattings" / "001"
    smplx = actor / "smplx_fitting" / "smplx_fitting"
    for directory in (videos, masks, smplx):
        directory.mkdir(parents=True, exist_ok=True)
    (videos / "001.mp4").write_bytes(b"TalkBody4D video fixture")
    (masks / "000000.png").write_bytes(_MASK_PNG)
    params = {
        "betas": [[0.0] * 100],
        "expression": [[0.0] * 50],
        "left_hand_pose": [[0.0] * 45],
        "right_hand_pose": [[0.0] * 45],
        "jaw_pose": [[0.0] * 3],
        "leye_pose": [[0.0] * 3],
        "reye_pose": [[0.0] * 3],
        "body_pose": [[0.0] * 63],
        "Rh": [[0.0] * 3],
        "Th": [[0.0] * 3],
        "global_orient": [[0.0] * 3],
        "transl": [[0.0] * 3],
    }
    (smplx / "000000.json").write_text(json.dumps(params))
    calibration = actor / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": {
                    "001": {
                        "model": "standard",
                        "image_size": [8, 8],
                        "K": [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
                        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
                    }
                },
                "camera_poses": {
                    "001": {
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "T": [[0.0], [0.0], [0.0]],
                    }
                },
            }
        )
    )
    return {
        "actor": actor,
        "images": videos,
        "masks": masks.parent,
        "calibration": calibration,
        "smplx": smplx,
    }


def tree_snapshot(root: Path) -> tuple[str, ...]:
    return tuple(sorted(str(path.relative_to(root)) for path in root.rglob("*")))


def dataset_metadata_snapshot(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Snapshot dataset and immediate subject entries without walking media trees."""
    paths = [root]
    for subject in sorted(path for path in root.iterdir() if path.is_dir()):
        paths.append(subject)
        paths.extend(sorted(subject.iterdir()))
    return tuple(
        (
            str(path.relative_to(root)) if path != root else ".",
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in paths
    )


def configured_dataset_root(kind: str, environment_name: str) -> Path | None:
    if os.environ.get("VCI_TEST_INTEGRATION") == "0":
        return None
    """Return an explicit integration root, falling back to saved configuration."""
    raw = os.environ.get(environment_name)
    if raw:
        candidate = Path(raw).expanduser().resolve()
        return candidate if candidate.is_dir() else None

    config_raw = os.environ.get("VCI_DATASET_CONFIG")
    config_path = (
        Path(config_raw).expanduser()
        if config_raw
        else Path.home() / ".config" / "vci-dataset" / "datasets.json"
    )
    try:
        config = json.loads(config_path.read_text())
        candidate = Path(config[kind]).expanduser().resolve()
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return candidate if candidate.is_dir() else None
