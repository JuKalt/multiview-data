# Multiview Data examples

Run from the Multiview Data root after a standalone or VCI Suite install.
`./run.sh` runs the installed Python without activation. Replace `/data/...` and
`example-subject` with your dataset root and a sequence ID. If another Conda
environment is active, leave it as it is; the launcher selects the recorded
environment in its own process.

## Try the reader without downloading a dataset

Create a two-camera processed fixture in a temporary directory:

```bash
DEMO_ROOT="$(mktemp -d)"
./run.sh - "$DEMO_ROOT" <<'PYTHON'
import json
from pathlib import Path
import sys
from PIL import Image

subject = Path(sys.argv[1]) / "demo"
calibration = {}
for index, camera in enumerate(("cam00", "cam01")):
    images = subject / "images" / camera
    masks = subject / "masks" / camera
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    Image.new("RGB", (8, 6), (90, 25, 5)).save(images / "color_000006.png")
    Image.new("L", (8, 6), 255).save(masks / "mask_000006.png")
    calibration[camera] = {"K": [[12, 0, 4], [0, 12, 3], [0, 0, 1]],
                           "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "T": [index, 0, 0]}
(subject / "calibration.json").write_text(json.dumps(calibration))
PYTHON
./run.sh examples/inspect_sequence.py processed "$DEMO_ROOT"
./run.sh examples/inspect_sequence.py processed "$DEMO_ROOT" --subject demo
```

Expect the ID `demo`, one shared RGB frame (native ID `6`), and RGB, mask,
and focal-length output for both cameras. The fixture contains no body fits
and is only for checking the reader.

## Find IDs and inspect synchronized views

```bash
./run.sh examples/inspect_sequence.py processed /data/processed
./run.sh examples/inspect_sequence.py processed /data/processed --subject example-subject
./run.sh examples/inspect_sequence.py processed /data/processed \
  --subject example-subject --cameras cam00 cam01 --frame 0
```

The first command lists native IDs. The second reports which RGB frame IDs
are shared across the cameras and opens the first shared frame. The third
requests a specific frame. A shared RGB frame does not imply that every camera
has a mask. Camera IDs and frame IDs retain their original numbering.

The same script reads the other supported datasets. For native DNA, point it
at an authorized copy of the official sample code before discovering IDs:

```bash
export DNA_RENDERING_SAMPLE_CODE=/data/dna/sample-code
./run.sh examples/inspect_sequence.py dna /data/dna/Part1
```

Use the same commands with `thuman`, `dancenet3d`, `talkbody4d`, `4ddress`, or
`actorshq`. To open native 4D-DRESS calibration, use `--trusted-pickles` only
if you trust the files. The [reader contract](../docs/contract.md) describes
expected directory layouts.

## Read a frame and its annotations in Python

```python
from multiview_data import open_dataset

sequences = open_dataset("processed", "/data/processed")
sequence = next(sequence for sequence in sequences if sequence.id == "example-subject")
with sequence:
    cameras = sequence.camera_ids[:2]
    if len(cameras) < 2:
        raise ValueError("Select a recording with at least two cameras")
    shared = set.intersection(*(set(sequence.frame_ids(camera)) for camera in cameras))
    if not shared:
        raise ValueError("The selected cameras have no shared RGB frame")
    frame_id = min(shared)
    for camera_id in cameras:
        camera = sequence.read_camera(camera_id)
        observation = sequence.read_observation(camera_id, frame_id)
        if camera is not None and observation.image is not None and observation.mask is not None:
            foreground_rgb = observation.image[observation.mask]
            print(camera_id, frame_id, foreground_rgb.shape)
    camera_id = cameras[0]
    if "smplx" in sequence.annotations:
        fit = sequence.read_annotation("smplx")
        if fit is not None:
            print({name: array.shape for name, array in fit.items()})
    if "semantic_general" in sequence.annotations:
        labels = sequence.read_annotation("semantic_general", camera_id=camera_id, frame_id=frame_id)
        if labels is not None:
            print(labels.shape)
```

This annotation snippet is for processed VCI. A processed SMPL-X file can
contain arrays for many frames. Native SMPL-X and labels keep their source
formats and may require an explicit frame ID. See the
[reader contract](../docs/contract.md#reader-contract) before mixing fits or
annotations from different datasets.
