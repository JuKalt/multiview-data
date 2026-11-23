"""List sequence IDs or inspect one synchronized multiview frame."""

import argparse
from pathlib import Path

from multiview_data import open_dataset


def shape(array):
    return array.shape if array is not None else "missing"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind")
    parser.add_argument("root", type=Path)
    parser.add_argument("--subject", help="Native sequence ID; omit to list IDs")
    parser.add_argument("--frame", type=int, help="Native frame ID; omit for first shared frame")
    parser.add_argument("--cameras", nargs="+", help="Native camera IDs; default: all")
    parser.add_argument("--trusted-pickles", action="store_true", help="Only for trusted 4D-DRESS inputs")
    args = parser.parse_args()
    if (args.frame is not None or args.cameras) and not args.subject:
        parser.error("--frame and --cameras require --subject")

    sequences = open_dataset(args.kind, args.root, trusted_pickles=args.trusted_pickles)
    if not args.subject:
        for sequence in sequences:
            print(sequence.id)
        return
    sequence = next((item for item in sequences if item.id == args.subject), None)
    if sequence is None:
        parser.error(f"unknown sequence: {args.subject!r}; omit --subject to list IDs")

    with sequence:
        cameras = args.cameras or sequence.camera_ids
        if not cameras or not set(cameras).issubset(sequence.camera_ids):
            parser.error(f"select from available cameras: {', '.join(sequence.camera_ids)}")
        shared = set.intersection(*(set(sequence.frame_ids(camera)) for camera in cameras))
        print(f"{sequence.id}: {len(shared)} shared RGB frames; first IDs: {sorted(shared)[:10]}")
        print("Annotations:", ", ".join(sequence.annotations) or "none")
        if args.frame is None and not shared:
            parser.error("no shared RGB frame; select fewer cameras or pass --frame")
        frame = args.frame if args.frame is not None else min(shared)
        for camera_id in cameras:
            camera = sequence.read_camera(camera_id)
            obs = sequence.read_observation(camera_id, frame)
            focal = f"{camera.K[0, 0]:g} px" if camera is not None else "missing"
            print(f"{camera_id} frame {frame}: RGB {shape(obs.image)}, mask {shape(obs.mask)}, fx {focal}")


if __name__ == "__main__":
    main()
