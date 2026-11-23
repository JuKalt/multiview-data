# Reader contract

- `open_dataset(kind, path)` discovers sequences without persisting paths or
  modifying inputs. Existing aliases are listed in `AVAILABLE_DATASETS`.
- `Sequence.id` identifies the native subject or take. Camera IDs are strings
  and retain the dataset's identity. Frame IDs are nonnegative integers and
  preserve gaps and the dataset's numbering. Call `frame_ids(camera_id)` for
  observed RGB frames. `read_observation` returns independent optional RGB and
  mask fields when an observation is missing. It raises on geometry mismatch,
  duplicate independent frame files or unavailable video decoders.
- RGB is H×W×3 uint8 in RGB channel order. Masks are H×W boolean foreground.
  Binary 0/1 masks are supported. Grayscale mattes are thresholded at half scale.
  Observations retain the native grid and lens distortion.
- A camera has a pixel-space `K`, a proper rotation `R`, translation `T`,
  and OpenCV distortion coefficients if available. Projection uses
  `X_camera = R @ X_world + T`. VCI and THuman-style inputs are expected
  to supply meters. Native readers apply the source formats' camera conventions
  and retain the world axes. There is no automatic world alignment across
  datasets. DNA camera-to-world transforms are inverted. ActorsHQ intrinsics
  are converted from normalized coordinates, and camera centers are converted
  to world-to-camera translations.
- Use `annotations` to discover supplied annotation kinds. This reports asset
  presence. `read_annotation` loads the supplied NPZ arrays, raster labels
  or native per-frame annotation. SMPL-X hand PCA, per-vertex labels and pose
  fields keep their source representation. Consumers that need converted
  body parameters or a particular label taxonomy must select and validate those
  explicitly. Native 4D-DRESS labels are per-vertex, and processed VCI labels
  are per-pixel. Pickled 4D-DRESS calibration and annotations are executable
  input and require explicit trust.
- Use `with sequence:` to release open SMC and video handles. File-backed
  indexes are built lazily once per selected camera and sequence. `observation_paths` supports
  exporters that can link media without decoding it.
- `load_dataset` retains the existing stable-key catalog for legacy method
  exporters. The catalog can optionally remember dataset locations; it has no
  repair mode. `open_dataset` is the API for new readers and integrations.

DNA reads `dna_rendering_part1_main/` and `dna_rendering_part1_annotations/`.
Place the licensed sample code next to the dataset or set
`DNA_RENDERING_SAMPLE_CODE` to the directory containing `SMCReader.py`.
TalkBody4D uses numeric camera video names, per-camera mattings and the native
`calibration.json`. 4D-DRESS discovers `<subject>/<outfit>/<take>/Capture`,
with per-camera `images` and `masks` and `cameras.pkl`. ActorsHQ discovers a
sequence's resolution folder containing `rgbs`, `masks` and `calibration.csv`.

The synthetic contract suite covers every reader, including sparse frame IDs,
BGR-to-RGB conversion, camera conventions and missing masks. Real data access
depends on the corresponding dataset license and availability.
