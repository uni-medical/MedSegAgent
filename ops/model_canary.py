"""One real natural-language inference, then optional same-grid reference comparison.

Only use de-identified research inputs. References are opened after prediction and
are never used for routing, cropping or inference. All output remains private.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np
from dotenv import load_dotenv

from medsegagent import core
from medsegagent.agent import select_tool


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def run(args):
    started = time.monotonic()
    core.validate_input(str(args.input), task=args.expected_task)
    selected = await select_tool(args.text, "CT")
    assert selected.task == args.expected_task, asdict(selected)
    result = await core.segment(
        task=selected.task,
        input_path=str(args.input),
        output_dir=str(args.output / "runs"),
        targets=selected.targets,
    )
    image, mask = nib.load(args.input), nib.load(result["segmentation_path"])
    assert image.shape == mask.shape and np.allclose(image.affine, mask.affine)
    values = np.asarray(mask.dataobj)
    label = 2 if selected.task == "lung_nodules" else 1
    predicted = values == label
    assert set(np.unique(values)).issubset({0, label})
    assert predicted.any(), "Positive canary requires the lesion itself to be nonempty."
    report = {
        "platform": platform.platform(),
        "time": time.time(),
        "selection": asdict(selected),
        "result": result,
        "input_sha256": digest(args.input),
        "mask_sha256": digest(result["segmentation_path"]),
        "geometry_consistent": True,
        "lesion_label": label,
        "lesion_voxels": int(predicted.sum()),
        "pipeline_seconds": time.monotonic() - started,
        "limitation": "Positive engineering canary, possible training overlap; not clinical validation.",
    }
    if args.reference:
        reference = nib.load(args.reference)  # Sealed from all inference steps above.
        assert image.shape == reference.shape and np.allclose(image.affine, reference.affine)
        truth = np.asarray(reference.dataobj) == args.reference_label
        assert truth.any()
        intersection = int(np.count_nonzero(predicted & truth))
        report["reference"] = {
            "sha256": digest(args.reference),
            "label": args.reference_label,
            "voxels": int(truth.sum()),
            "intersection": intersection,
            "dice": 2 * intersection / (int(predicted.sum()) + int(truth.sum())),
            "note": "Single same-grid comparison; no clinical or independent performance estimate.",
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "result"}), flush=True)


def main():
    os.umask(0o077)
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--expected-task", choices=["lung_nodules", "liver_lesions"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-label", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
