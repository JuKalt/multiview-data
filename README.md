# Multiview Data

[Usage examples](examples/README.md): discover native IDs, inspect synchronized views, and load annotations.

Read synchronized human capture data through one Python API. Input media remains
in its original layout. This package does discovery and reading, with no fitting
or GPU dependency.

For standalone setup, put Conda on `PATH`. The installer uses `vci-multiview-data`
by default; set `VCI_CLI_ENV` to choose another environment. If you installed
through VCI Suite, skip the installer and use the component's `run.sh` directly.

```bash
./install.sh  # Standalone only; skip after installing through VCI Suite.
./run.sh -m unittest discover -v
```

Run these commands from the Multiview Data checkout (the `multiview-data/`
directory inside the suite). `./run.sh` uses the installed Python and native
decoder dependencies without shell activation. For DNA, obtain the official
`SMCReader.py` with your authorized dataset download and set
`DNA_RENDERING_SAMPLE_CODE` to its containing folder.

```python
from multiview_data import open_dataset

sequence, *_ = open_dataset("processed", "/path/to/processed")
with sequence:
    camera_id = sequence.camera_ids[0]
    frame_id = sequence.frame_ids(camera_id)[0]
    camera = sequence.read_camera(camera_id)
    observation = sequence.read_observation(camera_id, frame_id)
    if camera is not None and observation.image is not None:
        image = observation.image  # H × W × 3, uint8 RGB
        mask = observation.mask    # H × W bool, or None
```

| Reader | Media and calibration | Additional requirements |
| --- | --- | --- |
| `processed` | Exported VCI camera folders, JPEG/PNG and JSON | None |
| `thuman` | THuman camera folders and JSON | Dataset access |
| `dancenet3d` | Prepared DanceNet3D camera folders and JSON | Prepare raw COLMAP inputs in Capture Processing |
| `dna` | Native RGB and annotation SMC files, 5MP and 12MP | `native` extra and official SMCReader |
| `talkbody4d` | Native videos, mattings and camera JSON | `native` extra and an OpenCV build with MP4 support |
| `4ddress` | Native capture folders and camera pickles | `trusted_pickles=True` for trusted files |
| `actorshq` | Native resolution folders, images, masks and calibration CSV | Optional synchronized SMPL-X sidecar |

See [the contract](docs/contract.md) for conventions, layout details and native
annotation formats. Acquire external datasets under the dataset owners' terms.
