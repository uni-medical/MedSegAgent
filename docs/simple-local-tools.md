# CLI and MCP

Install and configure the app using the [quick start](../README.md#quick-start).
Prepare the required [model weights](deployment.md#prepare-models), then check the backend:

```bash
uv run medsegagent doctor
```

## Command line

```bash
uv run medsegagent run --modality CT --text "Segment the liver and kidneys" \
  --input /path/to/scan.nii.gz --output outputs
uv run medsegagent catalog --query lung --modality CT
uv run medsegagent catalog --task total_v3
uv run medsegagent weights
```

`run` accepts a NIfTI image or a local DICOM series directory. With `--modality` omitted,
the Agent can use a CT/MR declaration in the request or gather local modality evidence.
`route` previews a model selection using text alone:

```bash
uv run medsegagent route --modality CT --text "Segment the liver"
```

The catalog lists 33 available tasks: 27 CT and 6 MR. Search matches task names,
labels and composite regions. An explicit task query includes its exact labels,
quality modes, requirements, license and local weight readiness. `tasks` and
`weights-status` are aliases for `catalog` and `weights`.

`run` returns JSON with `completion` and `result`. Requested outputs are in
`result.outputs[]`; their regions identify `artifact_id` and voxel `values` in
`artifacts[].path`. Files retain their own label IDs, so use those returned selectors.
An incomplete request exits nonzero and retains any completed outputs. `route` returns
one selection for inspection; `run` executes the full request.

## MCP server

Start a stdio server from the repository root:

```bash
uv run --env-file .env medsegagent-mcp
```

Configure your MCP client with this command and the repository as its working directory.
The server exposes five tools:

| Tool | Arguments |
| --- | --- |
| `get_capabilities` | `query?`, `task?`, `modality?` |
| `detect_modality` | `input_path?`, `execution_id?`, `modality?`, `output_dir?` |
| `segment` | `input_path`, `targets`, `modality?`, `output_dir?`, `execution_id?`, `task?`, `quality?`, `supersedes?` |
| `inspect_artifact` | `execution_id`, `region_ids` |
| `compose_masks` | `execution_id`, `operation`, `region_ids`, `name` |

Start with `get_capabilities` to discover models and labels. For a known modality,
call `segment` with an input path, target names and `CT` or `MR`. For an unknown modality,
call `detect_modality` with the input path, then choose CT/MR in `segment` using the
returned evidence. The first valid segmentation choice establishes the execution's modality.

The response includes an `execution_id`. Use it for later operations on the same image;
`segment` also requires the original `input_path`. Each execution creates a separate
subdirectory under `output_dir`. A session retains up to eight inputs, and generated
files remain available after the stdio process ends.

Specify a `task` to choose a model, or an array such as `["total", "total_v3"]` to run
several models for the same targets. Use each task's exact label names. Omitting `task`
uses the default model for that modality and target. Omitting `quality` uses the model's
default; supported modes are listed by the catalog. Whole CT lungs expand to five lobes,
and whole MR lungs to the two native lung labels.

Inspection and composition use returned `region_ids`. Composition supports `union`,
`intersection` and `difference`; difference subtracts all later regions from the first.
Repeated requests within an execution reuse existing results for the same model and
quality. See the [tool reference](tool-management.md) for failed-attempt replacement.

`local_outputs[]` gives the files for requested objects. Each file's `labels` describes
its full contents, while `requested_regions[].label_ids` selects a particular target.
A shared file may include additional labels used to construct that target. Model and
quality metadata distinguish results from different runs.

The external MCP client chooses its own model and tool sequence. The server runs local
operations and returns their results directly to that client.

## Inputs and runtime

NIfTI validation checks dimensions, gzip integrity, finite intensities, spacing and
affine geometry. Local DICOM input uses a consistent, uncompressed single-frame CT/MR
series from one patient/study/series, converted with the bundled dcm2niix. Web and A2A
accept NIfTI files; local DICOM directories are available through CLI and MCP.

Backend directories contain `segmentation.nii.gz`, `state.json`, `run_report.json` and
`process.log`. An execution's `execution.json` records its regions and output references.
Native masks and normalization metadata are retained alongside these files.

CLI requests use a 7200-second deadline by default, with 24 model requests and 64 work-tool
calls. Configure `MEDSEGAGENT_TIMEOUT_SECONDS`, `MEDSEGAGENT_MAX_MODEL_REQUESTS` and
`MEDSEGAGENT_MAX_TOOL_CALLS` in `.env`. Device scheduling, caches and service storage are
covered in [Deployment](deployment.md).
