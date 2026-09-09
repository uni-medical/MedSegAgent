# Development checks

## Automated checks

Install the development dependencies and run the existing suites:

```bash
uv sync --frozen --group dev
uv run --frozen pytest
uv run --frozen ruff check src tests ops
node tests/browser_state.cjs
git diff --check
```

The Python suite covers Agent execution, tool schemas, image handling, model readiness,
GPU scheduling, cancellation, records, uploads, authentication, MCP and A2A.
The Node suite exercises the browser client's state transitions with an in-memory DOM,
HTTP transport and viewer.

## Browser checks

Start the application using the [deployment guide](deployment.md), then open
[localhost:8767](http://127.0.0.1:8767) in a browser.

1. Confirm the workspace opens automatically as a guest, with the GitHub login option in the sidebar footer. Open an example and edit its suggested request.
2. Upload a NIfTI image and inspect the linked slices, 3D view and window controls.
3. Run a request, inspect the resulting labels, and download individual class masks.
4. Reopen the record, reuse its image, and check multiple results from one request.
5. Check a narrow viewport, cancellation, reload and network recovery.

## Inference and protocol checks

The utilities in `ops/` exercise real model and service integrations. Each provides
`--help` for its inputs and output directory:

| Utility | Coverage |
| --- | --- |
| `verify_weights.py` | Prepared model files and pinned checksums |
| `model_canary.py` | Model output geometry, labels and execution |
| `all_tasks_canary.py` | Public segmentation tasks and quality modes |
| `agent_canary.py` | Natural-language requests and composed results |
| `mcp_canary.py` | MCP tools through a real stdio client |
| `acceptance.py` | Web and A2A service integration |

Use public examples or synthetic fixtures and store generated artifacts under `outputs/`.
See [A2A integration](a2a.md) for client requests and task recovery.
