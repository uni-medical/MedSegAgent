"""Install verified example assets separately from Git; use the audited source catalog."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from medsegagent.core import validate_input


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("runtime/examples"))
    args = parser.parse_args()
    candidates = json.loads(args.catalog.read_text())
    assert {c["id"] for c in candidates} == {"liver-ct", "chest-ct", "abdomen-mr"}
    args.destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    cases = []
    labels = {
        "liver-ct": ["肝脏", "肝脏病灶"],
        "chest-ct": ["肺部", "肺结节"],
        "abdomen-mr": ["左右肾", "肝脏和脾脏"],
    }
    authors = {
        "liver-ct": "Wasserthal (2026)",
        "chest-ct": "LIDC-IDRI / TCIA",
        "abdomen-mr": "Wasserthal 等",
    }
    for candidate in candidates:
        example_id = candidate["id"]
        source = args.assets / (example_id + ".nii.gz")
        validate_input(str(source))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == candidate["sha256"]
        for name in (source.name, example_id + ".png"):
            dest = args.destination / name
            incoming = args.assets / name
            if dest.exists():
                assert (
                    hashlib.sha256(dest.read_bytes()).digest()
                    == hashlib.sha256(incoming.read_bytes()).digest()
                ), "Installed example differs"
            else:
                shutil.copyfile(incoming, dest)
                dest.chmod(0o600)
        cases.append(
            {
                "id": example_id,
                "title": "局部腹部 MRI" if example_id == "abdomen-mr" else candidate["title"],
                "modality": candidate["modality"],
                "description": candidate["description"],
                "filename": source.name,
                "preview": example_id + ".png",
                "license_file": "DATA-LICENSES.txt",
                "size_bytes": source.stat().st_size,
                "sha256": candidate["sha256"],
                "prompts": [
                    {"label": label, "text": text}
                    for label, text in zip(labels[example_id], candidate["prompts"], strict=True)
                ],
                "attribution": {
                    "label": authors[example_id],
                    "url": candidate["source_url"],
                    "license": candidate["license"],
                    "notice_url": f"/api/examples/{example_id}/license",
                },
            }
        )
    notices = []
    for filename in (
        "DATA-SOURCES.md",
        "LIDC-IDRI-DATA-LICENSE.txt",
        "TotalSegmentator-APACHE-2.0.txt",
    ):
        source = args.assets / filename
        shutil.copyfile(source, args.destination / filename)
        notices.append(filename + "\n\n" + source.read_text())
    (args.destination / "DATA-LICENSES.txt").write_text("\n\n".join(notices))
    shutil.copyfile(args.assets / "DATA-SOURCES.json", args.destination / "DATA-SOURCES.json")
    manifest = args.destination / "manifest.json.tmp"
    manifest.write_text(
        json.dumps({"version": 1, "cases": cases}, ensure_ascii=False, indent=2) + "\n"
    )
    manifest.replace(args.destination / "manifest.json")
    print(
        json.dumps(
            {"installed": len(cases), "total_image_bytes": sum(c["size_bytes"] for c in cases)}
        )
    )


if __name__ == "__main__":
    main()
