"""Load the separately distributed official DNA SMC reader."""

import importlib.util
import os
from pathlib import Path

def load_reader(subject: dict):
    candidates = []
    configured = os.environ.get("DNA_RENDERING_SAMPLE_CODE")
    if configured:
        candidates.append(Path(configured).expanduser())
    root = Path(subject.get("subject_path") or "").expanduser().resolve()
    for parent in (root, root.parent):
        candidates.extend(
            (
                parent / "dna_rendering_sample_code" / "SMCReader.py",
                parent
                / "dna_rendering_sample_code"
                / "dna_rendering_sample_code"
                / "SMCReader.py",
            )
        )
    for candidate in candidates:
        reader_path = candidate if candidate.is_file() else candidate / "SMCReader.py"
        if not reader_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("vci_dna_smc_reader", reader_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SMCReader
    raise SystemExit(
        "Could not find the official DNA-Rendering SMCReader.py. Extract "
        "dna_rendering_sample_code next to the DNA dataset, or set "
        "DNA_RENDERING_SAMPLE_CODE to its directory."
    )


def close_reader(reader) -> None:
    handle = getattr(reader, "smc", None)
    if handle is not None:
        handle.close()
    reader.release()
