"""Verify installed model bytes against the committed manifest; never download."""

import argparse
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv


def main():
    load_dotenv()
    from totalsegmentator.config import get_weights_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=get_weights_dir())
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("model-manifest.json").read_text())
    for model in manifest["models"]:
        for item in model["files"]:
            path = args.root / model["folder"] / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise SystemExit(f"Missing or incomplete weights: {model['task']} / {item['path']}")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != item["sha256"]:
                raise SystemExit(f"Weight checksum mismatch: {model['task']} / {item['path']}")
        print(f"Verified {model['task']} (Dataset{model['task_id']})")


if __name__ == "__main__":
    main()
