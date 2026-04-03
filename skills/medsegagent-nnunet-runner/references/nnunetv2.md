# nnUNet v2 Notes

Use this reference when you need to verify official nnUNet v2 conventions.

## Environment Variables

- Official nnUNet v2 variables are `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results`.
- `nnUNet_results` is the primary source for trained or downloaded model availability.
- `RESULTS_FOLDER` is not the primary nnUNet v2 convention; use it only as a legacy fallback.

## Result Layout

The official training output layout is:

```text
nnUNet_results/DatasetXXX_NAME/TRAINER__PLANS__CONFIGURATION/fold_x
```

Typical trainer-level files:

- `dataset.json`
- `dataset_fingerprint.json`
- `plans.json`

Typical fold-level files:

- `checkpoint_final.pth`
- `checkpoint_best.pth`
- `debug.json`

`checkpoint_final.pth` is the standard inference checkpoint.

If `nnUNetv2_find_best_configuration` was run, nnUNet may also create:

- `inference_instructions.txt`
- `inference_information.json`

Prefer `inference_instructions.txt` when it exists.

## dataset.json

When reading nnUNet v2 metadata:

- modality or channel information usually comes from `channel_names`
- classes or regions come from `labels`
- `background` should be `0`

Label values may appear as integers, strings, or region definitions depending on dataset style. Normalize simple numeric strings when summarizing, but do not change semantic meaning.

## Practical Guidance

- Use nnUNet result-folder `dataset.json` as the source of truth for deployed labels and modalities.
- Use repo-local `dataset/*.json` as a supplemental routing library.
- If the two disagree, surface the mismatch instead of silently choosing one.

## Inference Commands

Core prediction pattern:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION
```

Notes:

- use `--save_probabilities` only if you intend to ensemble predictions later
- each configuration should write to its own output folder
- by default nnUNet uses all 5 folds as an ensemble
- if predicting with a single model trained on all cases, specify `-f all`

Ensembling:

```bash
nnUNetv2_ensemble -i FOLDER1 FOLDER2 -o OUTPUT_FOLDER -np NUM_PROCESSES
```

Postprocessing:

```bash
nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
```

## Python Predictor

For custom trainer folders, the safer path is to initialize from the trained model folder directly.

```python
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
```

Useful methods:

- `initialize_from_trained_model_folder(...)`
- `predict_from_files(...)`

This avoids reconstructing trainer or plans choices from memory when the deployed model folder is non-default.

## Official Sources

- `documentation/setting_up_paths.md`
- `documentation/dataset_format.md`
- `documentation/how_to_use_nnunet.md`
