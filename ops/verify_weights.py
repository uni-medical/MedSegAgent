"""Verify installed model bytes against the committed manifest; never download."""

import argparse
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from medsegagent.weights import weights_root


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("open-model-manifest.json")
    )
    args = parser.parse_args(argv)
    root = weights_root(args.root)
    manifest = json.loads(args.manifest.read_text())
    for model in manifest["models"]:
        model_id = model.get("model_id", model.get("task_id"))
        label = f"{model['task']} (Dataset{model_id})" if "task" in model else f"Dataset{model_id}"
        for item in model["files"]:
            path = root / model["folder"] / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise SystemExit(f"Missing or incomplete weights: {label} / {item['path']}")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != item["sha256"]:
                raise SystemExit(f"Weight checksum mismatch: {label} / {item['path']}")
        print(f"Verified {label}")


if __name__ == "__main__":
    main()
