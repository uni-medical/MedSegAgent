---
name: medsegagent-nnunet-runner
description: Inspect `nnUNet_results`, choose a deployed nnUNet task, and run or prepare segmentation inference. Use when a user wants to match a free-form target to deployed labels, select the right task, or run prediction and postprocessing.
---

# nnUNet Segmentation Runner

## Overview

Inspect what is deployed, choose the right nnUNet task, and run or prepare inference. Do not claim inference succeeded unless it was actually run and outputs were produced.

## Prerequisites

- An nnUNet v2 environment is available.
- `nnUNet_results` is configured. `nnUNet_raw` and `nnUNet_preprocessed` may also be needed.
- To actually run inference, the nnUNet CLI or Python API must be installed and usable.
- Custom trainers must be importable in the target environment.

## Workflow

1. Inspect the nnUNet environment first.
   - Prefer `nnUNet_results`.
   - Use `RESULTS_FOLDER` only as a legacy fallback.
   - If the environment is unclear, run `scripts/nnunetv2-doctor.sh` first.
   - Run `scripts/list_nnunet_results.py --json` to enumerate real datasets, model folders, modalities, labels, folds, and checkpoints.

2. Parse the request into:
   - modality
   - target
   - qualifiers such as left/right, tumor/cyst, whole organ vs substructure

3. Read metadata in this order:
   - `dataset.json` under nnUNet result folders for what is actually available
   - `dataset/*.json` in this repo for supplemental routing metadata
   - `dataset/README.md` only when adding or fixing dataset entries

4. Filter by modality, then match labels conservatively.
   - Preserve laterality and substructure specificity.
   - Keep tumor, cyst, lesion, vessel, and organ concepts separate.
   - Treat repo metadata as secondary when it disagrees with deployed nnUNet results.

5. Choose how to run inference.
   - Prefer `inference_instructions.txt` when it exists.
   - For default model folders, use `nnUNetv2_predict`, plus ensembling or postprocessing if needed.
   - For custom trainers, prefer exact model-folder inference through the Python predictor API.
   - Use all available trained folds by default unless the environment only supports a specific fold such as `all`.
   - Keep input naming and file endings consistent with the selected task `dataset.json`.

6. Return either:
   - a compact dataset-to-label mapping
   - or the selected task plus the exact inference command

```json
{
  "DatasetNameA": [1, 2],
  "DatasetNameB": [7]
}
```

Return `{}` if no suitable modality or label is available, and say why.

## Rules

- Prefer real nnUNet availability over paper claims or static repo metadata.
- If a dataset exists in repo metadata but not in `nnUNet_results`, describe it as known but not deployed.
- Group multiple trainer folders under one dataset unless the user explicitly asks about checkpoints or trainer variants.
- Prefer officially generated `inference_instructions.txt` when present.
- If input files do not match the trained task modality, naming scheme, or file ending, stop and say the input is not ready.
- Do not silently guess a configuration when multiple materially different deployed options exist.
- If repo metadata and nnUNet metadata disagree, report the mismatch explicitly.
- Do not guess custom trainer or plans settings from memory; read them from the selected model folder.

## Extending Coverage

When adding or updating routing metadata:

1. Follow `dataset/README.md`.
2. Keep names explicit, such as `kidney_left`.
3. Do not invent deployed availability if no trained output exists in `nnUNet_results`.

## References

- Read `references/nnunetv2.md` only when you need the official nnUNet v2 path conventions, result layout, prediction commands, or `dataset.json` details.
- Read `references/examples.md` only when you need compact routing or inference examples.
