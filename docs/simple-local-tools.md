# Local tools

Install the pinned environment with `uv sync --frozen --group dev`. Run
`uv run medsegagent doctor` to check the actual backend. macOS defaults to MPS; Linux
uses CUDA (`MEDSEGAGENT_DEVICE=gpu`, with `CUDA_VISIBLE_DEVICES` selecting a card).

For natural language use `uv run medsegagent run --modality CT --text "Segment the liver"
--input /path/to/scan.nii.gz --output outputs` (on one line). The image is never serialized
into the provider payload. `uv run medsegagent route --modality CT --text "Segment the liver"`
is an image-free, real function-calling canary.

The stdio MCP entrypoint is `uv run medsegagent-mcp`. Its two tools are `segment_ct` and
`segment_mr`, accepting input_path, output_dir and targets. `output_dir` is always a **parent**:
each request creates its own atomic run subdirectory, even if many clients use the same
parent. Omit targets only for all structures. Empty lists, whitespace and unknown targets
fail with a readable MCP ToolError.

Each run writes segmentation.nii.gz, result.json, state.json, run_report.json and process.log.
No statistics.json is promised. State updates are atomic and synced to disk. Process logs
are private and are not returned over public HTTP. Native TotalSegmentator subprocesses
do not inherit the LLM API key or service tokens.

NIfTI is validated through the entire volume, including gzip integrity, bounded expansion,
raw spacing (before nibabel repairs headers), finite intensities, dimensionality and affine.
The output must use the requested labels and preserve the converted/input NIfTI geometry.

Local DICOM directories require a declared CT/MR tool, a single patient/study/series,
consistent geometry and supported uncompressed single-frame slices. A private snapshot is
converted with pinned dcm2niix before inference. Mismatched modality and mixed series are
rejected before the model is invoked. ZIP, arbitrary DICOM archives and Web DICOM upload
are not supported. dcm2niix is installed through uv; converter errors are recorded privately.

Runtime settings: MEDSEGAGENT_OUTPUT_ROOT, MEDSEGAGENT_LOCK_PATH,
MEDSEGAGENT_TIMEOUT_SECONDS (7200), MEDSEGAGENT_MAX_INPUT_BYTES (512 MiB),
MEDSEGAGENT_MAX_UNCOMPRESSED_BYTES (2 GiB), TOTALSEG_HOME_DIR and CUDA_VISIBLE_DEVICES.
Set TotalSegmentator config.json send_usage_stats to false for offline usage telemetry.
First inference may download model weights; preserve the cache for later runs.

Research use only; the fast models prioritize lower resolution and runtime. An empty mask
can reflect an absent structure or a model miss; completion is not a claim of correctness.
