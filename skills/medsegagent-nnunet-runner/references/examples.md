# Examples

## Example 1: Routing

Request:

```text
segment the spleen in CT
```

Answer style:

```json
{
  "Dataset009_Spleen": [1],
  "Dataset022_FLARE22": [3],
  "Dataset051_AMOS_CT": [1]
}
```

Note:

```text
Found CT tasks in nnUNet results. Labels come from task dataset.json files.
```

## Example 2: No Match

Request:

```text
segment the left kidney in MRI
```

Answer style:

```json
{}
```

Note:

```text
No deployed MRI task with a confirmed left-kidney label was found.
```

## Example 3: Prediction Command

Request:

```text
run liver segmentation on a CT case with a deployed nnUNet task
```

Answer style:

```text
Selected task: Dataset051_AMOS_CT
Selected labels: [6]
Selected model folder: nnUNetTrainer__nnUNetPlans__3d_fullres
Selected configuration: 3d_fullres
Command:
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 51 -c 3d_fullres
```

Note:

```text
Replace dataset ID, model folder, and paths with the deployed task you actually select.
```
