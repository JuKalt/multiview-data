"""Read synchronized observations without knowing their native storage format."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import csv
import json
import pickle
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .dna_reader import close_reader, load_reader
from .loader import load_dataset

MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
_FRAME = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics in pixels and world-to-camera extrinsics in meters."""

    K: np.ndarray
    R: np.ndarray
    T: np.ndarray
    distortion: np.ndarray

    @classmethod
    def from_values(cls, K, R, T, distortion=()):
        camera = cls(*(np.asarray(values, dtype=float).reshape(shape) for values, shape in
                       ((K, (3, 3)), (R, (3, 3)), (T, (3,)), (distortion, (-1,)))))
        if (not all(np.isfinite(value).all() for value in (camera.K, camera.R, camera.T, camera.distortion))
                or not np.allclose(camera.K[2], [0, 0, 1])
                or min(camera.K[0, 0], camera.K[1, 1]) <= 0
                or not np.allclose(camera.R @ camera.R.T, np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(camera.R), 1, atol=1e-3)):
            raise ValueError("Invalid camera intrinsics or rigid extrinsics")
        return camera


@dataclass(frozen=True)
class Observation:
    image: np.ndarray | None
    mask: np.ndarray | None


def index_image_files(directory):
    index = {}
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            match = _FRAME.search(path.stem)
            if match is None:
                continue
            frame = int(match.group())
            if frame in index:
                # Some method exporters add an extension alias for one file.
                if path.samefile(index[frame]):
                    continue
                raise ValueError(f"Duplicate frame ID {frame} in {directory}")
            index[frame] = path
    return index


def _camera_folders(root, leaf=None):
    if root is None or not root.is_dir():
        return {}
    return {camera.name: camera / leaf if leaf else camera
            for camera in sorted(root.iterdir()) if camera.is_dir()}


def _read(path, mode=None):
    if path is None:
        return None
    with Image.open(path) as image:
        return np.array(image.convert(mode) if mode else image)


def _mask(values):
    if values is None:
        return None
    if values.ndim == 3:
        values = values.max(axis=-1)
    maximum = values.max(initial=0)
    # Binary masks may use 0/1, while mattes use the full integer precision.
    if maximum <= 1:
        return values > 0
    if values.dtype.kind == "f":
        return values >= 0.5
    if values.dtype == np.uint16:
        return values >= 32768
    return values >= 128


def _actor_cameras(path):
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not re.fullmatch(r"Cam\d+", row["name"]):
                continue
            width, height = int(row["w"]), int(row["h"])
            K = [[float(row["fx"]) * width, 0, float(row["px"]) * width],
                 [0, float(row["fy"]) * height, float(row["py"]) * height], [0, 0, 1]]
            vec = np.array([float(row[key]) for key in ("rx", "ry", "rz")])
            angle = np.linalg.norm(vec)
            rotation = np.eye(3)
            if angle:
                x, y, z = vec / angle
                cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                rotation += np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)
            world_to_camera = rotation.T
            center = np.array([float(row[key]) for key in ("tx", "ty", "tz")])
            yield row["name"], Camera.from_values(K, world_to_camera, -world_to_camera @ center)


class Sequence:
    """One recording. Use as a context manager to release any open native readers."""

    def __init__(self, descriptor, *, trusted_pickles=False):
        self._descriptor = dict(descriptor)
        self.kind = descriptor["dataset"]
        self.id = str(descriptor["subject_id"])
        self.root = Path(descriptor["subject_path"])
        self.trusted_pickles = trusted_pickles
        self._handles = {}
        self._frame_indexes = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        for key, handle in self._handles.items():
            if key[0] == "smc":
                close_reader(handle)
            else:
                handle.release()
        self._handles.clear()

    def _path(self, key):
        path = self._descriptor.get(key)
        return Path(path) if path is not None else None

    @property
    def asset_paths(self):
        """Compatibility escape hatch for adapters exporting native assets."""
        return dict(self._descriptor)

    @property
    def annotations(self):
        return tuple(name for name in ("smplx", "smplx_confidence", "mhr", "mhrfromsmplx",
                      "labels", "semantic_general", "semantic_detailed", "semantic_confidence")
                     if self._path(name) is not None)

    def _pickle(self, path):
        if not self.trusted_pickles:
            raise ValueError("Native 4D-DRESS pickle access requires trusted_pickles=True for trusted data")
        with path.open("rb") as stream:
            return pickle.load(stream)

    def _smc(self, key):
        token = ("smc", key)
        if token not in self._handles:
            path = self._path(key)
            if path is None:
                raise ValueError(f"DNA sequence {self.id} has no {key} container")
            self._handles[token] = load_reader(self._descriptor)(str(path))
        return self._handles[token]

    @cached_property
    def cameras(self):
        path = self._path("calibration")
        if path is None:
            return {}
        if self.kind == "actorshq":
            return dict(_actor_cameras(path))
        if self.kind == "dna":
            result = {}
            for name, camera in (self._smc("calibration").get_Calibration_all() or {}).items():
                # The public reader exposes camera-to-world RT.
                view = np.linalg.inv(np.asarray(camera["RT"], dtype=float).reshape(4, 4))
                result[f"{int(name):02d}"] = Camera.from_values(camera["K"], view[:3, :3], view[:3, 3], camera["D"])
            return result
        if self.kind == "4ddress":
            result = {}
            for name, camera in self._pickle(path).items():
                view = np.asarray(camera["extrinsics"], dtype=float).reshape(3, 4)
                result[f"{int(name):04d}"] = Camera.from_values(camera["intrinsics"], view[:, :3], view[:, 3])
            return result
        data = json.loads(path.read_text())
        if self.kind == "talkbody4d":
            intrinsics, poses = data["cameras"], data["camera_poses"]
            return {str(int(name)): Camera.from_values(intrinsics[name]["K"], poses[name]["R"],
                    poses[name]["T"], intrinsics[name].get("dist", [])) for name in intrinsics.keys() & poses.keys()}
        return {name: Camera.from_values(camera["K"], camera["R"], camera["T"], camera.get("distCoeff", []))
                for name, camera in data.items()}

    @cached_property
    def _sources(self):
        if self.kind == "4ddress":
            return {kind: _camera_folders(self._path(kind), kind) for kind in ("images", "masks")}
        if self.kind == "talkbody4d":
            root = self._path("images")
            videos = {str(int(path.stem)): path for path in root.glob("*.mp4") if path.stem.isdigit()} if root else {}
            masks = {str(int(name)): entries for name, entries in _camera_folders(self._path("masks")).items() if name.isdigit()}
            return {"videos": videos, "masks": masks}
        return {kind: _camera_folders(self._path(kind)) for kind in ("images", "masks")}

    def _files(self, kind, camera_id):
        key = kind, camera_id
        if key not in self._frame_indexes:
            if kind in self._sources:
                folder = self._sources[kind].get(camera_id)
            else:
                root = self._path(kind)
                folder = root / camera_id if root is not None else None
            self._frame_indexes[key] = index_image_files(folder) if folder is not None else {}
        return self._frame_indexes[key]

    def _video(self, camera_id):
        import cv2

        token = ("video", camera_id)
        if token not in self._handles:
            path = self._sources["videos"].get(camera_id)
            if path is None:
                return None
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f"Cannot open video {path}")
            self._handles[token] = capture
        return self._handles[token]

    @cached_property
    def _dna_cameras(self):
        container = self._smc("images").smc
        groups = ("Camera_5mp", "Camera_12mp")
        return {f"{int(name):02d}": (group, name)
                for group in groups if group in container
                for name in container[group] if "color" in container[group][name]}

    @property
    def camera_ids(self):
        if self.kind == "dna":
            return tuple(sorted(self._dna_cameras, key=int)) if self._path("images") else ()
        return tuple(sorted(set(self.cameras).union(*(index.keys() for index in self._sources.values()))))

    def frame_ids(self, camera_id):
        """Observed RGB IDs for the camera, independent of optional annotations."""
        if self.kind == "dna":
            location = self._dna_cameras.get(camera_id) if self._path("images") else None
            if location is None:
                return ()
            group, name = location
            return tuple(sorted(map(int, self._smc("images").smc[group][name]["color"])))
        if self.kind == "talkbody4d":
            import cv2

            capture = self._video(camera_id)
            return range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))) if capture else ()
        return tuple(sorted(self._files("images", camera_id)))

    def read_camera(self, camera_id):
        return self.cameras.get(camera_id)

    def observation_paths(self, camera_id):
        """Return file-backed RGB and mask indexes, for zero-copy exporters."""
        if self.kind in {"dna", "talkbody4d"}:
            raise ValueError(f"{self.kind} RGB is not stored as individual files")
        return tuple(dict(self._files(kind, camera_id)) for kind in ("images", "masks"))

    def read_observation(self, camera_id, frame_id):
        if frame_id < 0:
            raise ValueError("Frame IDs must be nonnegative")
        if self.kind == "dna":
            location = self._dna_cameras.get(camera_id) if self._path("images") else None
            if location is None:
                return Observation(None, None)
            group, name = location
            rgb = self._smc("images")
            frame = str(frame_id)
            image = (rgb.get_img(group, name, "color", Frame_id=frame, disable_tqdm=True)
                     if frame in rgb.smc[group][name]["color"] else None)
            mask = None
            if self._path("masks"):
                annotations = self._smc("masks")
                if f"Mask/{name}/mask/{frame}" in annotations.smc:
                    mask = annotations.get_mask(name, Frame_id=frame, disable_tqdm=True)
            image = image[..., ::-1].copy() if image is not None else None
        else:
            if self.kind == "talkbody4d":
                import cv2

                capture = self._video(camera_id)
                image = None
                if capture is not None:
                    if capture.get(cv2.CAP_PROP_POS_FRAMES) != frame_id:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ok, bgr = capture.read()
                    if ok:
                        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                image = _read(self._files("images", camera_id).get(frame_id), "RGB")
            mask = _read(self._files("masks", camera_id).get(frame_id))
        mask = _mask(mask)
        if image is not None and mask is not None and image.shape[:2] != mask.shape:
            raise ValueError(f"Unaligned RGB and mask for {self.id}/{camera_id}/{frame_id}")
        return Observation(image, mask)

    def read_annotation(self, name, *, frame_id=None, camera_id=None):
        """Read a named annotation. Native fit conventions are reported separately."""
        path = self._path(name)
        if path is None or name not in self.annotations:
            return None
        if self.kind == "dna":
            if name != "smplx":
                return None
            method = self._smc(name).get_SMPLx
            return method() if frame_id is None else method(Frame_id=frame_id)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        if name == "smplx" and self.kind in {"4ddress", "talkbody4d"}:
            if frame_id is None:
                raise ValueError("Specify a frame ID for a native per-frame SMPL-X annotation")
            if self.kind == "talkbody4d":
                matches = [file for file in path.glob("*.json") if file.stem.isdigit() and int(file.stem) == frame_id]
                return json.loads(matches[0].read_text()) if matches else None
            matches = [file for file in path.glob("*_smplx.pkl")
                       if (match := re.search(r"f(\d+)_smplx$", file.stem)) and int(match.group(1)) == frame_id]
            return self._pickle(matches[0]) if matches else None
        if name == "labels" and self.kind == "4ddress":
            matches = [file for file in path.glob("label-f*.pkl") if int(file.stem[len("label-f"):]) == frame_id]
            return self._pickle(matches[0]) if matches else None
        if camera_id is None or frame_id is None:
            raise ValueError("Specify camera and frame IDs for raster annotations")
        return _read(self._files(name, camera_id).get(frame_id))


def open_dataset(kind, path, *, trusted_pickles=False):
    """Discover all sequences at an explicit dataset path without writing config."""
    dataset = load_dataset(kind, path=Path(path), interactive=False, persist_path=False,
                           required_keys=(), warn_missing=False)
    return tuple(Sequence(item, trusted_pickles=trusted_pickles) for item in dataset.subjects)
