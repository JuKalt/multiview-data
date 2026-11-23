import csv
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from multiview_data import open_dataset

K = [[12., 0., 4.], [0., 12., 3.], [0., 0., 1.]]
R = np.eye(3).tolist()
T = [0., 0., 0.]


def image(path, mode="RGB", value=(90, 25, 5)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (8, 6), value).save(path)


class ReaderContractTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def assert_pair(self, sequence, camera_id, frame_id):
        with sequence:
            self.assertIn(camera_id, sequence.camera_ids)
            self.assertIn(frame_id, sequence.frame_ids(camera_id))
            calibration = sequence.read_camera(camera_id)
            np.testing.assert_allclose(calibration.K, K)
            np.testing.assert_allclose(calibration.R, R)
            view = sequence.read_observation(camera_id, frame_id)
            self.assertEqual(view.image.shape, (6, 8, 3))
            self.assertEqual(view.image.dtype, np.uint8)
            self.assertEqual(view.mask.dtype, bool)
            self.assertTrue(view.mask.all())
            self.assertIsNone(sequence.read_observation(camera_id, 300).image)

    def test_processed_thuman_dancenet_share_image_and_geometry_contract(self):
        for kind, stem in (("processed", "color_00000006"), ("thuman", "000006"), ("dancenet3d", "00000006")):
            with self.subTest(kind=kind):
                root = self.root / kind / "sequence"
                image(root / "images/cam00" / f"{stem}.png")
                image(root / "masks/cam00/mask_000006.png", "L", 1)
                image(root / "images/cam00/color_00000009.png")
                (root / "calibration.json").write_text(json.dumps({"cam00": {"K": K, "R": R, "T": T}}))
                sequence, = open_dataset(kind, root.parent)
                self.assert_pair(sequence, "cam00", 6)
                self.assertEqual(sequence.frame_ids("cam00"), (6, 9))
                self.assertIsNone(sequence.read_observation("cam00", 9).mask)
                self.assertIsNone(sequence.read_annotation("mhr"))

    def test_single_view_read_does_not_index_unselected_cameras(self):
        root = self.root / "sequence"
        image(root / "images/cam00/000001.png")
        image(root / "images/cam01/color_000001.png")
        image(root / "images/cam01/duplicate_000001.png")
        sequence, = open_dataset("processed", self.root)
        self.assertEqual(sequence.camera_ids, ("cam00", "cam01"))
        self.assertIsNotNone(sequence.read_observation("cam00", 1).image)
        with self.assertRaisesRegex(ValueError, "Duplicate frame"):
            sequence.frame_ids("cam01")

    def test_actorshq_csv_projection_and_native_ids(self):
        stage = self.root / "Actor01/Sequence1/4x"
        image(stage / "rgbs/Cam001/Cam001_rgb000016.jpg")
        image(stage / "masks/Cam001/Cam001_mask000016.png", "L", 255)
        columns = "name,w,h,rx,ry,rz,tx,ty,tz,fx,fy,px,py".split(",")
        with (stage / "calibration.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerow(["Cam001", 8, 6, 0, 0, 0, 1, 2, 3, 1.5, 2, .5, .5])
        sequence, = open_dataset("actorshq", self.root)
        self.assert_pair(sequence, "Cam001", 16)
        np.testing.assert_allclose(sequence.read_camera("Cam001").T, [-1, -2, -3])

    def test_fourddress_requires_consent_to_read_pickle_annotations(self):
        take = self.root / "00001/Inner/Take1"
        capture = take / "Capture"
        image(capture / "0004/images/capture-f00007.png")
        image(capture / "0004/masks/mask-f00007.png", "L", 255)
        (capture / "cameras.pkl").write_bytes(pickle.dumps({
            "0004": {"intrinsics": K, "extrinsics": np.column_stack([R, T])}
        }))
        labels = take / "Semantic/labels/label-f00007.pkl"
        labels.parent.mkdir(parents=True)
        labels.write_bytes(pickle.dumps({"scan_labels": np.array([1, 2])}))
        untrusted, = open_dataset("4ddress", self.root)
        with self.assertRaisesRegex(ValueError, "trusted_pickles"):
            untrusted.read_camera("0004")
        sequence, = open_dataset("4ddress", self.root, trusted_pickles=True)
        self.assert_pair(sequence, "0004", 7)
        np.testing.assert_array_equal(sequence.read_annotation("labels", frame_id=7)["scan_labels"], [1, 2])

    def test_talkbody_reads_video_and_matting_on_requested_frame(self):
        import cv2

        actor = self.root / "actor"
        image(actor / "mattings/0001/000001.png", "L", 255)
        videos = actor / "raw_videos"
        videos.mkdir()
        path = videos / "0001.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (8, 6))
        self.assertTrue(writer.isOpened(), "OpenCV must support MPEG-4 in the native reader test")
        writer.write(np.zeros((6, 8, 3), dtype=np.uint8))
        writer.write(np.full((6, 8, 3), [5, 25, 90], dtype=np.uint8))
        writer.release()
        (actor / "calibration.json").write_text(json.dumps(
            {"cameras": {"1": {"K": K}}, "camera_poses": {"1": {"R": R, "T": T}}}
        ))
        sequence, = open_dataset("talkbody4d", self.root)
        self.assert_pair(sequence, "1", 1)
        with sequence:
            rgb = sequence.read_observation("1", 1).image
            self.assertGreater(int(rgb[0, 0, 0]), int(rgb[0, 0, 2]))

    def test_dna_preserves_5mp_and_12mp_and_returns_rgb(self):
        import h5py

        main = self.root / "dna_rendering_part1_main"
        annotations = self.root / "dna_rendering_part1_annotations"
        main.mkdir()
        annotations.mkdir()
        with h5py.File(main / "person.smc", "w") as rgb:
            rgb.create_group("Camera_5mp/0/color/7")
            rgb.create_group("Camera_12mp/48/color/7")
        with h5py.File(annotations / "person_annots.smc", "w") as annotation:
            annotation.create_group("Mask/0/mask/7")
            annotation.create_group("Mask/48/mask/7")

        class OfficialReaderFixture:
            def __init__(self, path):
                self.smc = h5py.File(path, "r")

            def get_Calibration_all(self):
                return {key: {"K": K, "RT": np.eye(4), "D": np.zeros(5)} for key in ("00", "48")}

            def get_img(self, group, camera, *_args, **_kwargs):
                assert group == ({"0": "Camera_5mp", "48": "Camera_12mp"}[camera])
                return np.full((6, 8, 3), [5, 25, 90], dtype=np.uint8)

            def get_mask(self, *_args, **_kwargs):
                return np.ones((6, 8), dtype=np.uint8)

            def release(self):
                self.smc = None

        with patch("multiview_data.reader.load_reader", return_value=OfficialReaderFixture):
            sequence, = open_dataset("dna", self.root)
            for camera in ("00", "48"):
                self.assert_pair(sequence, camera, 7)
                with sequence:
                    self.assertEqual(sequence.read_observation(camera, 7).image[0, 0].tolist(), [90, 25, 5])
