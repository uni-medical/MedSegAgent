#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def resolve_results_dir():
    for key in ("nnUNet_results", "RESULTS_FOLDER"):
        value = os.environ.get(key)
        if value:
            return key, Path(value)
    return None, None


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_label_value(value):
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, list):
        return [normalize_label_value(v) for v in value]
    return value


def first_dataset_json(dataset_dir):
    candidates = sorted(dataset_dir.glob("*/dataset.json"))
    if candidates:
        return candidates[0]
    direct = dataset_dir / "dataset.json"
    if direct.is_file():
        return direct
    return None


def checkpoint_count(trainer_dir):
    count = 0
    for pattern in ("**/checkpoint_final.pth", "**/checkpoint_best.pth"):
        count += len(list(trainer_dir.glob(pattern)))
    return count


def parse_model_folder_name(name):
    parts = name.split("__")
    return {
        "trainer": parts[0] if len(parts) > 0 else None,
        "plans": parts[1] if len(parts) > 1 else None,
        "configuration": parts[2] if len(parts) > 2 else None,
    }


def list_available_folds(trainer_dir):
    folds = []
    for fold_dir in sorted(p for p in trainer_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")):
        fold_name = fold_dir.name.removeprefix("fold_")
        fold_info = {
            "fold": fold_name,
            "has_checkpoint_final": (fold_dir / "checkpoint_final.pth").is_file(),
            "has_checkpoint_best": (fold_dir / "checkpoint_best.pth").is_file(),
        }
        folds.append(fold_info)
    return folds


def summarize_dataset(dataset_dir):
    trainer_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    dataset_json_path = first_dataset_json(dataset_dir)
    dataset_json = load_json(dataset_json_path) if dataset_json_path else {}

    labels = (dataset_json or {}).get("labels", {})
    channel_names = (dataset_json or {}).get("channel_names", {})
    input_modalities = (dataset_json or {}).get("input_modalities", {})
    modalities = list(channel_names.values()) or list(input_modalities.values())

    trainers = []
    for trainer_dir in trainer_dirs:
        parsed = parse_model_folder_name(trainer_dir.name)
        folds = list_available_folds(trainer_dir)
        trainer_info = {
            "name": trainer_dir.name,
            "trainer": parsed["trainer"],
            "plans": parsed["plans"],
            "configuration": parsed["configuration"],
            "has_dataset_json": (trainer_dir / "dataset.json").is_file(),
            "has_plans_json": (trainer_dir / "plans.json").is_file(),
            "checkpoint_count": checkpoint_count(trainer_dir),
            "folds": folds,
        }
        trainers.append(trainer_info)

    normalized_labels = {
        key: normalize_label_value(value)
        for key, value in labels.items()
        if key != "background"
    }

    return {
        "folder": dataset_dir.name,
        "dataset_name": (dataset_json or {}).get("name", dataset_dir.name),
        "modalities": modalities,
        "labels": normalized_labels,
        "trainer_count": len(trainers),
        "trainers": trainers,
    }


def collect_results(results_dir):
    dataset_dirs = sorted(
        p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("Dataset")
    )
    return [summarize_dataset(dataset_dir) for dataset_dir in dataset_dirs]


def main():
    parser = argparse.ArgumentParser(
        description="List datasets, trainers, modalities, and labels from an nnUNet results directory."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    env_key, results_dir = resolve_results_dir()
    if not results_dir:
        raise SystemExit("No nnUNet results directory found in nnUNet_results or RESULTS_FOLDER.")
    if not results_dir.exists():
        raise SystemExit(f"{env_key} points to a missing path: {results_dir}")

    datasets = collect_results(results_dir)

    if args.json:
        print(
            json.dumps(
                {
                    "env_var": env_key,
                    "results_dir": str(results_dir),
                    "dataset_count": len(datasets),
                    "datasets": datasets,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"{env_key}={results_dir}")
    print(f"datasets={len(datasets)}")
    for dataset in datasets:
        print(f"- {dataset['folder']} | {dataset['dataset_name']}")
        if dataset["modalities"]:
            print(f"  modalities: {', '.join(dataset['modalities'])}")
        print(f"  labels: {len(dataset['labels'])}")
        print(f"  trainers: {dataset['trainer_count']}")
        for trainer in dataset["trainers"]:
            print(
                f"    - {trainer['trainer']} | {trainer['configuration']} | folds={len(trainer['folds'])} | checkpoints={trainer['checkpoint_count']}"
            )


if __name__ == "__main__":
    main()
