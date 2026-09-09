"""One input, reusable verified regions, and bounded local segmentation actions.

The provider sees opaque region identifiers and explicit measurements. File paths,
backend reports and exceptions stay in the local execution manifest. Native regions
share a backend label map; derived regions get independent binary masks so overlaps
between anatomical structures and lesions are never overwritten.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import math
import os
import re
import shutil
import stat
import uuid
from itertools import combinations
from pathlib import Path

from medsegagent import catalog, core
from medsegagent.result_metadata import model_provenance
from medsegagent.task_specs import TASK_SPECS
from medsegagent.tool_definitions import MAX_SEGMENT_PRODUCERS
from medsegagent.tool_definitions import WORK_TOOLS as _WORK_TOOLS

_MESSAGES = {
    "INVALID_ARGUMENTS": "The action arguments are invalid; use the declared tool schema.",
    "MODALITY_REQUIRED": "Choose CT or MR in the segment action's modality argument.",
    "MODALITY_CONFLICT": "The requested modality differs from the modality already bound to this execution.",
    "UNSUPPORTED_TARGET": "A target is not supported for this modality; check the public catalog.",
    "UNSUPPORTED_TASK": "The requested segmentation producer is unsupported.",
    "TASK_UNAVAILABLE": "The requested producer is unavailable under the public service policy.",
    "WEIGHTS_MISSING": "The producer's required model weights are not prepared.",
    "UNSUPPORTED_QUALITY": "This quality is unsupported for one or more requested targets.",
    "INPUT_INVALID": "The input could not be validated and frozen; no inference was started.",
    "INFERENCE_FAILED": "Local inference failed. Repeating the same failed job is disabled.",
    "PREVIOUS_FAILURE": "This job already failed; do not repeat it unchanged.",
    "OUTPUT_INVALID": "The output failed label or geometry validation and is unavailable.",
    "UNKNOWN_REGION": "A region identifier is not available in this execution.",
    "NAME_CONFLICT": "Choose a new derived-region name; existing objects cannot be redefined.",
    "ARTIFACT_CHANGED": "An artifact changed after validation and cannot be reused.",
    "GEOMETRY_MISMATCH": "The regions do not share the same input geometry.",
    "ACTION_FAILED": "The local action failed; private details were not sent to the model.",
    "TARGETS_PENDING": "Some requested targets still have no verified output.",
    "INVALID_SUPERSEDES": "Replacement declarations must identify failed, uncovered requests for the same target and choose one replacement producer.",
}


class _Fault(ValueError):
    def __init__(self, code: str):
        super().__init__(_MESSAGES[code])
        self.code = code


def _error(code: str, **fields) -> dict:
    return {"code": code, "message": _MESSAGES[code], "retryable": False, **fields}


def _same_geometry(left: dict, right: dict) -> bool:
    import numpy as np

    return (
        left["shape"] == right["shape"]
        and np.allclose(left["affine"], right["affine"], rtol=1e-5, atol=1e-4)
        and np.allclose(left["spacing"], right["spacing"], rtol=1e-5, atol=1e-6)
    )


class TaskExecution:
    """Local state for one immutable image and one Agent conversation.

    ``on_progress`` receives a safe dictionary; both sync and async callbacks work.
    Cancellation always propagates, after the core has stopped owned subprocesses
    and image readers. Successful coverage includes labels with zero voxels.
    """

    def __init__(
        self,
        input_path,
        modality=None,
        output_dir=None,
        on_progress=None,
        *,
        modality_source="user_declaration",
        on_inference_start=None,
        example_modality_hint=None,
    ):
        if not isinstance(modality_source, str) or modality_source not in {
            "user_declaration",
            "example_manifest",
        }:
            raise ValueError("Unsupported modality declaration source.")
        if example_modality_hint is not None and (
            not isinstance(example_modality_hint, str) or example_modality_hint not in {"CT", "MR"}
        ):
            raise ValueError("Unsupported example modality hint.")
        self.example_modality_hint = example_modality_hint
        if modality is not None:
            catalog.public_targets(
                modality
            )  # Validate explicit declarations and installed recipes.
        self.modality = modality
        self._declared_modality = modality
        self._declaration_source = modality_source
        self._modality_detection: dict | None = None
        self._intensity_features: list[float] | None = None
        self._nifti_sidecar: dict | None = None
        self._dicom_snapshot: Path | None = None
        self._source = Path(input_path).expanduser().resolve()
        parent = Path(output_dir or os.environ.get("MEDSEGAGENT_OUTPUT_ROOT", "outputs"))
        parent = parent.expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._prefix = uuid.uuid4().hex
        self._root = parent / ("execution-" + self._prefix)
        self._root.mkdir(mode=0o700)
        self._frozen: Path | None = None
        self._on_progress = on_progress
        self._on_inference_start = on_inference_start
        self._lock = asyncio.Lock()
        self._artifacts: dict[str, dict] = {}
        self._regions: dict[str, dict] = {}
        self._coverage: dict[tuple[str, str], dict[str, str]] = {}
        self._semantic: dict[tuple[str, str, str], str] = {}
        self._wanted: dict[tuple[str, str, str], tuple[str, ...]] = {}
        self._resolutions: dict[tuple[str, str, str], str] = {}
        self._outputs: dict[tuple[str, str, str | None], str] = {}
        self._backend_results: list[dict] = []
        self._inference_timings: list[dict] = []
        self._failed_jobs: dict[tuple, dict] = {}
        self._argument_errors: dict[str, dict] = {}
        self._compositions: dict[tuple, str] = {}

    @property
    def has_outputs(self) -> bool:
        return bool(self._outputs)

    @property
    def unresolved_failures(self) -> list[dict]:
        failures = list(self._argument_errors.values())
        replaced: dict[tuple[str, str], set[str]] = {}
        for key in self._resolutions:
            task, _target, quality = key
            replaced.setdefault((task, quality), set()).update(self._wanted[key])
        for signature, failure in self._failed_jobs.items():
            task, quality, native = signature
            covered = set(self._coverage.get((task, quality), {}))
            pending = set(native) - covered - replaced.get((task, quality), set())
            if pending:
                failures.append({**failure, "requested_targets": sorted(pending)})
        for task, target, quality in self._wanted:
            key = (task, target, quality)
            if key not in self._semantic and key not in self._resolutions:
                failures.append(
                    _error(
                        "TARGETS_PENDING", task=task, quality=quality, requested_targets=[target]
                    )
                )
        return copy.deepcopy(failures)

    @property
    def is_complete(self) -> bool:
        """Execution completeness only; this cannot prove natural-language coverage."""
        return self.has_outputs and not self.unresolved_failures

    def _safe_region(self, region: dict) -> dict:
        return {
            "region_id": region["id"],
            "artifact_id": region["artifact_id"],
            "target": region["target"],
            "task": region["task"],
            "quality": region["quality"],
            "name": region["name"],
            "voxels": region["voxels"],
            "volume_ml": region["volume_ml"],
            "empty": region["voxels"] == 0,
            **{
                key: region["provenance"][key]
                for key in ("operation", "source_region_ids")
                if key in region["provenance"]
            },
        }

    def snapshot(self) -> dict:
        """Only allowlisted engineering feedback and image-derived measurements."""
        return self._snapshot_for(self._regions)

    @property
    def modality_detection(self) -> dict | None:
        return copy.deepcopy(self._modality_detection)

    def _resolved_attempts(self, region_ids=None) -> list[dict]:
        return [
            {"task": task, "target": target, "quality": quality, "replacement_region_id": region_id}
            for (task, target, quality), region_id in self._resolutions.items()
            if region_ids is None or region_id in region_ids
        ]

    def _snapshot_for(self, region_ids) -> dict:
        """An action returns relevant regions; older identifiers remain in tool history."""
        selected = set(region_ids)
        artifacts = {self._regions[index]["artifact_id"] for index in selected}
        return {
            "modality": self.modality,
            "modality_detection": self.modality_detection,
            "regions": [
                self._safe_region(region)
                for index, region in self._regions.items()
                if index in selected
            ],
            "artifacts": [
                {
                    "artifact_id": row["id"],
                    "targets": row["targets"],
                    "geometry_validated": True,
                    **model_provenance(row, self._regions),
                }
                for row in self._artifacts.values()
                if row["id"] in artifacts
            ],
            "unresolved_failures": self.unresolved_failures,
            "resolved_attempts": self._resolved_attempts(selected),
            "summary": {
                "completed_targets": list(
                    dict.fromkeys(target for _task, target, _quality in self._outputs)
                ),
                "artifact_count": len(self._artifacts),
                "region_count": len(self._regions),
            },
        }

    def export_result(self) -> dict:
        """Host-only manifest. Adapters must never forward this dictionary to an LLM."""
        return copy.deepcopy(
            {
                "schema_version": 5,
                "modality": self.modality,
                "modality_detection": self.modality_detection,
                "artifacts": [
                    {**row, **model_provenance(row, self._regions)}
                    for row in self._artifacts.values()
                ],
                "regions": [
                    {**row, **model_provenance(row, self._regions)}
                    for row in self._regions.values()
                ],
                "outputs": [
                    {
                        "target": target,
                        "task": task,
                        "quality": quality,
                        "region_id": region_id,
                        "artifact_id": self._regions[region_id]["artifact_id"],
                        **model_provenance(self._regions[region_id], self._regions),
                    }
                    for (task, target, quality), region_id in self._outputs.items()
                ],
                "backend_results": self._backend_results,
                "inference_timings": self._inference_timings,
                "failed_attempts": list(self._failed_jobs.values()),
                "resolutions": self._resolved_attempts(),
                "unresolved_failures": self.unresolved_failures,
                "summary": self.snapshot()["summary"],
            }
        )

    result = export_result
    outcome = export_result

    def _save(self):
        core._write_json(self._root / "execution.json", self.export_result())

    async def _progress(self, **fields):
        if self._on_progress is not None:
            pending = self._on_progress(fields)
            if inspect.isawaitable(pending):
                await pending

    async def call(self, name: str, arguments: dict) -> dict:
        async with self._lock:
            previous_regions = set(self._regions)
            try:
                if name not in _WORK_TOOLS:
                    raise _Fault("INVALID_ARGUMENTS")
                if not isinstance(arguments, dict):
                    raise _Fault("INVALID_ARGUMENTS")
                self._argument_errors.pop("unknown_tool", None)
                if name == "get_capabilities":
                    if set(arguments) - {"modality", "task", "query"}:
                        raise _Fault("INVALID_ARGUMENTS")
                    try:
                        response = {"capabilities": catalog.get_agent_capabilities(**arguments)}
                    except (catalog.CatalogError, TypeError, ValueError):
                        raise _Fault("INVALID_ARGUMENTS") from None
                elif name == "detect_modality":
                    response = await self._detect_modality(arguments)
                elif name == "segment":
                    response = await self._segment(arguments)
                elif name == "inspect_artifact":
                    response = await self._inspect(arguments)
                else:
                    response = await self._compose(arguments)
                self._argument_errors.pop(name, None)
                self._save()
                return {
                    "ok": True,
                    "status": "completed",
                    **response,
                    "unresolved_failures": self.unresolved_failures,
                }
            except asyncio.CancelledError:
                self._save()
                raise
            except _Fault as exc:
                failure = _error(exc.code)
                if exc.code not in {"INFERENCE_FAILED", "PREVIOUS_FAILURE", "OUTPUT_INVALID"}:
                    key = name if name in _WORK_TOOLS else "unknown_tool"
                    self._argument_errors[key] = failure
                self._save()
                return {
                    "ok": False,
                    "status": "failed",
                    **failure,
                    **self._snapshot_for(set(self._regions) - previous_regions),
                }
            except Exception:  # noqa: BLE001 - adapter boundary must not expose paths or logs.
                self._argument_errors[name if isinstance(name, str) else "unknown_tool"] = _error(
                    "ACTION_FAILED"
                )
                self._save()
                return {
                    "ok": False,
                    "status": "failed",
                    **_error("ACTION_FAILED"),
                    **self._snapshot_for(set(self._regions) - previous_regions),
                }

    def _freeze(self, *, detected_modality=None, _stop_event=None) -> Path:
        directory = self._root / "input"
        directory.mkdir(mode=0o700, exist_ok=True)
        if self._source.is_dir():
            modality = detected_modality or self.modality
            if modality not in {"CT", "MR"}:
                raise _Fault("MODALITY_REQUIRED")
            destination = directory / "dicom"
            try:
                core._inspect_dicom_directory(
                    self._source,
                    "total" if modality == "CT" else "total_mr",
                    snapshot_dir=destination,
                    _stop_event=_stop_event,
                )
            except BaseException:
                shutil.rmtree(destination, ignore_errors=True)
                raise
            self._dicom_snapshot = destination
            return destination
        source = self._source
        if not source.name.lower().endswith((".nii", ".nii.gz")):
            raise _Fault("INPUT_INVALID")
        limit = core._positive_int("MEDSEGAGENT_MAX_INPUT_BYTES", core.DEFAULT_MAX_INPUT_BYTES)
        destination = directory / (
            "image.nii.gz" if source.name.lower().endswith(".gz") else "image.nii"
        )
        temporary = directory / "image.tmp"
        try:
            core._check_stop(_stop_event)
            descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as reader, temporary.open("xb") as writer:
                before = os.fstat(reader.fileno())
                if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
                    raise _Fault("INPUT_INVALID")
                os.chmod(temporary, 0o600)
                copied = 0
                while block := reader.read(1024 * 1024):
                    core._check_stop(_stop_event)
                    copied += len(block)
                    if copied > limit:
                        raise _Fault("INPUT_INVALID")
                    writer.write(block)
                after = os.fstat(reader.fileno())
                attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                if copied != before.st_size or any(
                    getattr(before, name) != getattr(after, name) for name in attributes
                ):
                    raise _Fault("INPUT_INVALID")
                core._check_stop(_stop_event)
                writer.flush()
                os.fsync(writer.fileno())
            core._check_stop(_stop_event)
            os.replace(temporary, destination)
            # Validate the exact private bytes and collect features in one image scan.
            inspected = core._inspect_nifti(
                destination, collect_intensity_stats=True, _stop_event=_stop_event
            )
            self._intensity_features = inspected["intensity_features"]
            if self._declared_modality is None:
                from medsegagent import modality

                self._nifti_sidecar = modality.probe_nifti_sidecar(source, _stop_event=_stop_event)
            return destination
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _public_detection(report) -> dict:
        """Only controlled detector metadata and finite summaries, never paths or headers."""
        if not isinstance(report, dict):
            raise _Fault("ACTION_FAILED")
        detected = report.get("modality")
        if detected is not None and (
            not isinstance(detected, str)
            or (detected != "other" and not re.fullmatch(r"[A-Z0-9_]{1,16}", detected))
        ):
            raise _Fault("ACTION_FAILED")
        source = report.get("source")
        if source not in {
            "dicom_metadata",
            "totalseg_intensity",
            "nifti_sidecar",
            "user_declaration",
            "example_manifest",
        }:
            raise _Fault("ACTION_FAILED")
        status = report.get("status")
        if status not in {"detected", "uncertain", "unsupported", "provided"}:
            raise _Fault("ACTION_FAILED")
        public = {
            "modality": detected,
            "source": source,
            "status": status,
            "supported": report.get("supported") is True and detected in {"CT", "MR"},
            "limitations": [
                code
                for code in report.get("limitations", [])
                if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_]{1,96}", code)
            ],
        }
        candidate = report.get("candidate_modality")
        if status == "detected" and candidate in {"CT", "MR", "US"}:
            public["candidate_modality"] = candidate
        agreement = report.get("vote_agreement")
        if (
            status == "detected"
            and type(agreement) in (float, int)
            and math.isfinite(agreement)
            and 0 <= agreement <= 1
        ):
            public["vote_agreement"] = float(agreement)
        statistics = report.get("intensity_statistics")
        if isinstance(statistics, dict):
            public["intensity_statistics"] = {
                key: float(value)
                for key in ("mean", "std", "min", "max")
                if type(value := statistics.get(key)) in (float, int) and math.isfinite(value)
            }
        return public

    def _classify_frozen(self, *, _stop_event=None):
        from medsegagent import modality

        core._check_stop(_stop_event)
        if self._nifti_sidecar is not None:
            return self._nifti_sidecar
        if self._intensity_features is None:
            inspected = core._inspect_nifti(
                self._frozen, collect_intensity_stats=True, _stop_event=_stop_event
            )
            self._intensity_features = inspected["intensity_features"]
        report = modality.classify_features(self._intensity_features)
        core._check_stop(_stop_event)
        return report

    async def _detect_modality(self, arguments):
        from medsegagent import modality

        if arguments:
            raise _Fault("INVALID_ARGUMENTS")
        cached = self._modality_detection is not None
        if not cached and self._declared_modality is not None:
            self._modality_detection = {
                "modality": self._declared_modality,
                "source": self._declaration_source,
                "status": "provided",
                "supported": True,
                "limitations": [],
                "declared_modality": self._declared_modality,
                "cached": False,
            }
        elif not cached and self.example_modality_hint is not None:
            self._modality_detection = {
                "modality": self.example_modality_hint,
                "source": "example_manifest",
                "status": "provided",
                "supported": True,
                "limitations": [],
                "cached": False,
            }
        elif not cached:
            metadata_source = self._dicom_snapshot is not None or (
                self._frozen is None
                and (
                    self._source.is_dir()
                    or not self._source.name.lower().endswith((".nii", ".nii.gz"))
                )
            )
            try:
                if metadata_source:
                    report = await core._validation(
                        modality.probe_dicom, self._dicom_snapshot or self._source
                    )
                    report = self._public_detection(report)
                    detected = report["modality"]
                    if (
                        report["status"] == "detected"
                        and report["supported"]
                        and self._frozen is None
                    ):
                        # Freeze supported series without committing the Agent's modality choice.
                        self._frozen = await core._validation(
                            self._freeze, detected_modality=detected
                        )
                else:
                    if self._frozen is None:
                        self._frozen = await core._validation(self._freeze)
                    report = self._public_detection(await core._validation(self._classify_frozen))
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - private input/model failures become an abstention.
                report = {
                    "modality": None,
                    "status": "uncertain",
                    "supported": False,
                    "source": "dicom_metadata" if metadata_source else "totalseg_intensity",
                    "limitations": ["INPUT_OR_DETECTOR_FAILED"],
                }
            report.update(declared_modality=self._declared_modality, cached=False)
            self._modality_detection = report
        else:
            self._modality_detection["cached"] = True
        return {
            "ok": True,
            "status": "completed",
            "modality": self.modality,
            "modality_detection": self.modality_detection,
            "cached": cached,
        }

    def _validate_supersedes(self, declarations, recipes) -> dict:
        """Bind explicit replacement intent to existing failed requests, before any work."""
        if not isinstance(declarations, list) or not 1 <= len(declarations) <= 512:
            raise _Fault("INVALID_ARGUMENTS")
        choices = {
            recipe["target"]: (recipe["task"], recipe["target"], recipe["quality"])
            for recipe in recipes
        }
        replacements = {}
        for row in declarations:
            if (
                not isinstance(row, dict)
                or set(row) != {"task", "target", "quality"}
                or any(not isinstance(value, str) for value in row.values())
            ):
                raise _Fault("INVALID_ARGUMENTS")
            key = (row["task"], row["target"], row["quality"])
            new = choices.get(row["target"])
            if key in replacements:
                raise _Fault("INVALID_ARGUMENTS")
            if key not in self._wanted or key in self._semantic or new is None or new == key:
                raise _Fault("INVALID_SUPERSEDES")
            covered = set(self._coverage.get((row["task"], row["quality"]), {}))
            pending = set(self._wanted[key]) - covered
            failed = set().union(
                *(
                    set(native)
                    for task, quality, native in self._failed_jobs
                    if (task, quality) == (row["task"], row["quality"])
                )
            )
            if not pending.intersection(failed):
                raise _Fault("INVALID_SUPERSEDES")
            existing = self._resolutions.get(key)
            if existing is not None and existing != self._semantic.get(new):
                raise _Fault("INVALID_SUPERSEDES")
            replacements[key] = new
        return replacements

    async def _segment(self, arguments: dict) -> dict:
        if (
            set(arguments) - {"targets", "quality", "modality", "task", "supersedes"}
            or "targets" not in arguments
        ):
            raise _Fault("INVALID_ARGUMENTS")
        chosen_modality = arguments.get("modality", self.modality)
        if chosen_modality is None:
            raise _Fault("MODALITY_REQUIRED")
        if not isinstance(chosen_modality, str) or chosen_modality not in {"CT", "MR"}:
            raise _Fault("INVALID_ARGUMENTS")
        if self.modality is not None and chosen_modality != self.modality:
            raise _Fault("MODALITY_CONFLICT")
        producer = arguments.get("task")
        if isinstance(producer, list):
            if (
                not 1 <= len(producer) <= MAX_SEGMENT_PRODUCERS
                or any(not isinstance(task, str) for task in producer)
                or len(set(producer)) != len(producer)
            ):
                raise _Fault("INVALID_ARGUMENTS")
            producers = producer
        else:
            producers = [producer]
        if any(
            task is not None and (not isinstance(task, str) or task not in TASK_SPECS)
            for task in producers
        ):
            raise _Fault("UNSUPPORTED_TASK")
        if len(producers) > 1 and "supersedes" in arguments:
            raise _Fault("INVALID_SUPERSEDES")
        try:
            recipes = [
                recipe
                for task in producers
                for recipe in catalog.resolve_targets(
                    chosen_modality, arguments["targets"], task=task
                )
            ]
        except (catalog.CatalogError, TypeError):
            raise _Fault("UNSUPPORTED_TARGET") from None
        requested_quality = arguments.get("quality")
        jobs: dict[tuple[str, str], list[str]] = {}
        for recipe in recipes:
            task = recipe["task"]
            try:
                _spec, quality, _targets = core.validate_task_options(
                    task, requested_quality, recipe["native_targets"]
                )
            except core.SegmentationError as exc:
                raise _Fault(exc.code if exc.code in _MESSAGES else "UNSUPPORTED_TARGET") from None
            recipe["quality"] = quality
            covered = self._coverage.get((task, quality), {})
            missing = jobs.setdefault((task, quality), [])
            for target in recipe["native_targets"]:
                if target not in covered and target not in missing:
                    missing.append(target)
        replacements = (
            self._validate_supersedes(arguments["supersedes"], recipes)
            if "supersedes" in arguments
            else {}
        )
        # No inference, input snapshot, or goal mutation occurs before the whole call validates.
        self._argument_errors.pop("segment", None)
        try:
            if self._frozen is None:
                self._frozen = await core._validation(
                    self._freeze, detected_modality=chosen_modality
                )
            elif self.modality is None and self._frozen.is_dir():
                # Detection may have frozen DICOM before the Agent selected a backend.
                # Apply the same deterministic preparation checks before committing its choice.
                await core._validation(
                    core._inspect_dicom_directory,
                    self._frozen,
                    "total" if chosen_modality == "CT" else "total_mr",
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - image parsing exceptions may include private paths.
            raise _Fault("INPUT_INVALID") from None
        self.modality = chosen_modality
        for recipe in recipes:
            self._wanted[(recipe["task"], recipe["target"], recipe["quality"])] = tuple(
                recipe["native_targets"]
            )
        reusable = set()
        for recipe in recipes:
            covered = self._coverage.get((recipe["task"], recipe["quality"]), {})
            reusable.update(covered[name] for name in recipe["native_targets"] if name in covered)
            semantic = self._semantic.get((recipe["task"], recipe["target"], recipe["quality"]))
            if semantic:
                reusable.add(semantic)
        if reusable:
            await core._validation(self._load_regions, sorted(reusable))
        cached = not any(jobs.values())
        await self._run_jobs(jobs, recipes)
        await self._materialize(recipes)
        relevant = set()
        for recipe in recipes:
            relevant.add(self._semantic[(recipe["task"], recipe["target"], recipe["quality"])])
            covered = self._coverage[(recipe["task"], recipe["quality"])]
            relevant.update(covered[name] for name in recipe["native_targets"])
        # Commit only after every requested replacement has a verified result. The old
        # producer's output/coverage and failure audit are never rewritten or invented.
        self._resolutions.update({old: self._semantic[new] for old, new in replacements.items()})
        return {
            **self._snapshot_for(relevant),
            "cached": cached,
            "requested_targets": list(dict.fromkeys(recipe["target"] for recipe in recipes)),
        }

    async def _run_jobs(self, jobs, recipes):
        pending = [(task, quality, missing) for (task, quality), missing in jobs.items() if missing]
        if not pending:
            return
        for task, quality, missing in pending:
            signature = (task, quality, tuple(sorted(missing)))
            if self._failed_jobs.get(signature, {}).get("code") in {
                "INFERENCE_FAILED",
                "OUTPUT_INVALID",
            }:
                raise _Fault("PREVIOUS_FAILURE")

        # Bound preprocessing as well as inference using the operator's existing device
        # capacity. Every core call still acquires its own host-wide device lease.
        capacity = core.GPUScheduler(device=core.device()).capacity
        semaphore = asyncio.Semaphore(capacity)
        failure = None

        async def infer(task, quality, missing, input_path, timing):
            def finished(status, outcome):
                fields = outcome if isinstance(outcome, dict) else getattr(outcome, "__dict__", {})
                timing.update(
                    task=task,
                    quality=quality,
                    status=status,
                    engine=fields.get("inference_engine"),
                    seconds=copy.deepcopy(fields.get("timings_seconds") or {}),
                )

            async with semaphore:
                await self._progress(
                    status="running", task=task, quality=quality, requested_targets=missing
                )
                try:
                    result = await core.segment(
                        task=task,
                        input_path=input_path,
                        output_dir=str(self._root / "runs"),
                        targets=missing,
                        speed=quality,
                        **(
                            {"on_inference_start": self._on_inference_start}
                            if self._on_inference_start is not None
                            else {}
                        ),
                    )
                except asyncio.CancelledError as exc:
                    finished("cancelled", exc)
                    raise
                except Exception as exc:  # noqa: BLE001 - backend paths and logs stay private.
                    finished("failed", exc)
                    code = (
                        exc.code
                        if isinstance(exc, core.SegmentationError)
                        and exc.code
                        in {
                            "WEIGHTS_MISSING",
                            "TASK_UNAVAILABLE",
                            "UNSUPPORTED_TASK",
                            "UNSUPPORTED_QUALITY",
                        }
                        else "INFERENCE_FAILED"
                    )
                    return _Fault(code)
                finished("completed", result)
                return result

        while pending:
            # Convert DICOM once before sharing its validated NIfTI with other jobs.
            count = 1 if self._frozen.is_dir() else len(pending)
            batch, pending = pending[:count], pending[count:]
            input_path = str(self._frozen)
            timings = [{} for _ in batch]
            workers = [
                asyncio.create_task(infer(task, quality, missing, input_path, timing))
                for (task, quality, missing), timing in zip(batch, timings, strict=True)
            ]
            try:
                # Only immutable-input inference runs concurrently. Region identifiers,
                # coverage and dependent compositions are registered in request order.
                for (task, quality, missing), worker in zip(batch, workers, strict=True):
                    result = await asyncio.shield(worker)
                    signature = (task, quality, tuple(sorted(missing)))
                    code = None
                    if isinstance(result, _Fault):
                        code = result.code
                    else:
                        try:
                            artifact = await core._validation(
                                self._read_backend, result, task, quality, missing
                            )
                        except Exception:  # noqa: BLE001 - output exceptions never become model text.
                            code = "OUTPUT_INVALID"
                    if code:
                        self._failed_jobs[signature] = _error(
                            code, task=task, quality=quality, requested_targets=missing
                        )
                        failure = failure or code
                        self._save()
                        continue
                    self._backend_results.append(copy.deepcopy(result))
                    self._artifacts[artifact["id"]] = artifact
                    covered = self._coverage.setdefault((task, quality), {})
                    for label in artifact["labels"]:
                        region = self._new_region(
                            label["name"],
                            artifact,
                            [label["id"]],
                            label["voxels"],
                            {
                                "operation": "native_label",
                                "task": task,
                                "quality": quality,
                                "native_targets": [label["name"]],
                            },
                        )
                        covered[label["name"]] = region["id"]
                    if self._frozen.is_dir():
                        self._frozen = Path(result["converted_input_path"]).resolve()
                    # Preserve verified partial output even if another producer fails.
                    await self._materialize(recipes)
                    self._save()
            except BaseException:
                # A caller cannot abandon running readers/children or release leases
                # while their cleanup is still in progress, even after repeated cancel.
                for worker in workers:
                    if not worker.done():
                        worker.cancel()
                drained = asyncio.gather(*workers, return_exceptions=True)
                while not drained.done():
                    try:
                        await asyncio.shield(drained)
                    except asyncio.CancelledError:
                        continue
                drained.result()
                raise
            finally:
                # Private timings remain ordered even when jobs finish out of order.
                # Jobs cancelled before admission have no core timing to report.
                self._inference_timings.extend(row for row in timings if row)
        if failure:
            raise _Fault(failure)

    async def _materialize(self, recipes: list[dict]):
        for recipe in recipes:
            task, target, quality = recipe["task"], recipe["target"], recipe["quality"]
            covered = self._coverage.get((task, quality), {})
            if not set(recipe["native_targets"]) <= set(covered):
                continue
            semantic_key = (task, target, quality)
            if semantic_key not in self._semantic:
                members = [covered[name] for name in recipe["native_targets"]]
                if len(members) == 1:
                    region_id = members[0]
                else:
                    region_id = await self._derive(
                        "union", members, target, task=task, quality=quality
                    )
                self._semantic[semantic_key] = region_id
            self._outputs[semantic_key] = self._semantic[semantic_key]

    def _read_backend(self, result, task, quality, native_targets, *, _stop_event=None):
        import nibabel as nib

        if not isinstance(result, dict) or not isinstance(result.get("labels"), list):
            raise _Fault("OUTPUT_INVALID")
        supplied_path = Path(result["segmentation_path"])
        path = supplied_path.resolve()
        if supplied_path.is_symlink() or not path.is_relative_to(self._root):
            raise _Fault("OUTPUT_INVALID")
        labels = result["labels"]
        ids = [row.get("id") for row in labels]
        names = [row.get("name") for row in labels]
        if (
            any(type(index) is not int or index <= 0 for index in ids)
            or len(set(ids)) != len(ids)
            or len(set(names)) != len(names)
            or set(names) != set(native_targets)
        ):
            raise _Fault("OUTPUT_INVALID")
        native = {name: index for index, name in catalog.native_labels(task).items()}
        if any(row.get("source_id", native[row["name"]]) != native[row["name"]] for row in labels):
            raise _Fault("OUTPUT_INVALID")
        geometry = core._inspect_nifti(
            path, label_map=dict(zip(ids, names, strict=True)), _stop_event=_stop_event
        )
        reference_path = self._frozen
        if reference_path.is_dir():
            reference_path = Path(result["converted_input_path"]).resolve()
            if not reference_path.is_relative_to(self._root):
                raise _Fault("OUTPUT_INVALID")
            core.validate_input(str(reference_path), _stop_event=_stop_event)
        reference = nib.load(reference_path)
        expected = {
            "shape": list(reference.shape),
            "spacing": list(reference.header.get_zooms()),
            "affine": reference.affine.tolist(),
        }
        actual = {
            "shape": geometry["shape"],
            "spacing": geometry["voxel_spacing"],
            "affine": geometry["affine"],
        }
        if not _same_geometry(expected, actual):
            raise _Fault("GEOMETRY_MISMATCH")
        unit = reference.header.get_xyzt_units()[0]
        factor = {"meter": 1000.0, "mm": 1.0, "micron": 0.001, "unknown": 1.0}[unit]
        spacing = [float(value) * factor for value in expected["spacing"]]
        measurement = {
            "method": "voxel_count_times_spacing_product",
            "spacing_mm": spacing,
            "voxel_volume_mm3": math.prod(spacing),
            "source_spatial_unit": unit,
            "unit_assumption": "assumed_mm" if unit == "unknown" else None,
        }
        counts = {row["id"]: row["voxels"] for row in geometry["labels"]}
        projected = []
        for row in labels:
            count = counts.get(row["id"], 0)
            projected.append(
                {
                    "id": row["id"],
                    "source_id": native[row["name"]],
                    "name": row["name"],
                    "color": core._label_color(row["id"]),
                    "voxels": count,
                    "volume_mm3": count * math.prod(spacing),
                    "volume_ml": count * math.prod(spacing) / 1000,
                }
            )
        return self._artifact(
            path,
            projected,
            actual,
            measurement,
            {
                "operation": "native_segmentation",
                "task": task,
                "native_targets": list(native_targets),
                "quality": quality,
            },
            task=task,
            quality=quality,
            _stop_event=_stop_event,
        )

    def _artifact(
        self, path, labels, geometry, measurement, provenance, *, _stop_event=None, **fields
    ):
        index = len(self._artifacts) + 1
        voxels = sum(label["voxels"] for label in labels)
        return {
            "id": f"artifact-{self._prefix}-{index}",
            "path": str(path),
            "name": f"mask-{index}.nii.gz",
            "targets": [row["name"] for row in labels],
            "labels": labels,
            "geometry": geometry,
            "schema_version": core.RESULT_SCHEMA_VERSION,
            "segmentation_shape": geometry["shape"],
            "segmentation_voxel_spacing": geometry["spacing"],
            "segmentation_affine": geometry["affine"],
            "volume_measurement": measurement,
            "nonzero_voxels": voxels,
            "detection_status": "target_detected" if voxels else "no_target_detected",
            "no_target_detected": not bool(voxels),
            "warning": core.WARNING,
            "sha256": core._file_digest(path, _stop_event),
            "size_bytes": path.stat().st_size,
            "provenance": provenance,
            **fields,
        }

    def _new_region(self, target, artifact, values, voxels, provenance):
        volume = voxels * artifact["volume_measurement"]["voxel_volume_mm3"]
        region = {
            "id": f"region-{self._prefix}-{len(self._regions) + 1}",
            "target": target,
            "task": provenance.get("task", "composition"),
            "quality": provenance.get("quality"),
            "name": target,
            "artifact_id": artifact["id"],
            "values": values,
            "voxels": voxels,
            "volume_mm3": volume,
            "volume_ml": volume / 1000,
            "provenance": provenance,
        }
        self._regions[region["id"]] = region
        return region

    def _region_ids(self, values):
        if (
            not isinstance(values, list)
            or not values
            or len(values) > 128
            or any(not isinstance(value, str) or value not in self._regions for value in values)
        ):
            raise _Fault("UNKNOWN_REGION")
        return list(dict.fromkeys(values))

    def _load_regions(self, ids, *, _stop_event=None):
        import nibabel as nib

        images = {}
        reference = None
        for region_id in ids:
            artifact = self._artifacts[self._regions[region_id]["artifact_id"]]
            if artifact["id"] in images:
                continue
            path = Path(artifact["path"])
            if path.is_symlink() or core._file_digest(path, _stop_event) != artifact["sha256"]:
                raise _Fault("ARTIFACT_CHANGED")
            image = nib.load(path)
            actual = {
                "shape": list(image.shape),
                "spacing": list(image.header.get_zooms()),
                "affine": image.affine.tolist(),
            }
            if not _same_geometry(actual, artifact["geometry"]):
                raise _Fault("ARTIFACT_CHANGED")
            if reference is not None and not _same_geometry(actual, reference):
                raise _Fault("GEOMETRY_MISMATCH")
            reference = actual
            images[artifact["id"]] = image
        return images

    async def _inspect(self, arguments):
        if set(arguments) != {"region_ids"}:
            raise _Fault("INVALID_ARGUMENTS")
        ids = self._region_ids(arguments["region_ids"])
        overlaps = await core._validation(self._overlaps, ids)
        return {
            "regions": [self._safe_region(self._regions[index]) for index in ids],
            "artifacts": [
                row
                for row in self.snapshot()["artifacts"]
                if row["artifact_id"] in {self._regions[index]["artifact_id"] for index in ids}
            ],
            "overlaps": overlaps,
            "overlaps_checked": len(ids) <= 16,
        }

    def _overlaps(self, ids, *, _stop_event=None):
        import numpy as np

        images = self._load_regions(ids, _stop_event=_stop_event)
        # Inspecting the whole 117-class anatomy need not allocate 6,786 pairwise masks.
        if len(ids) > 16:
            return []
        counts = {pair: 0 for pair in combinations(ids, 2)}
        if not counts:
            return []
        first = self._artifacts[self._regions[ids[0]]["artifact_id"]]
        for z in range(first["geometry"]["shape"][2]):
            core._check_stop(_stop_event)
            planes = {
                artifact_id: np.asanyarray(image.dataobj[:, :, z])
                for artifact_id, image in images.items()
            }
            masks = {
                region_id: np.isin(
                    planes[self._regions[region_id]["artifact_id"]],
                    self._regions[region_id]["values"],
                )
                for region_id in ids
            }
            for pair in counts:
                counts[pair] += int(np.count_nonzero(masks[pair[0]] & masks[pair[1]]))
        volume = first["volume_measurement"]["voxel_volume_mm3"]
        return [
            {"region_ids": list(pair), "voxels": count, "volume_ml": count * volume / 1000}
            for pair, count in counts.items()
        ]

    async def _compose(self, arguments):
        if set(arguments) != {"operation", "region_ids", "name"}:
            raise _Fault("INVALID_ARGUMENTS")
        operation, name = arguments["operation"], arguments["name"]
        ids = self._region_ids(arguments["region_ids"])
        if (
            not isinstance(operation, str)
            or operation not in {"union", "intersection", "difference"}
            or (operation == "difference" and len(ids) < 2)
            or not isinstance(name, str)
            or not name.strip()
            or len(name) > 80
            or re.search(r"[\\/:\x00-\x1f]", name)
            or ".." in name
        ):
            raise _Fault("INVALID_ARGUMENTS")
        name = name.strip()
        # Result names must also be publishable as binary class downloads.
        try:
            core.label_mask_filename({"id": 1, "name": name})
        except ValueError:
            raise _Fault("INVALID_ARGUMENTS") from None
        operands = tuple(sorted(ids)) if operation in {"union", "intersection"} else tuple(ids)
        cached = self._compositions.get((operation, operands, name))
        if name in catalog.public_targets(self.modality) or any(
            target == name and region_id != cached
            for (_task, target, _quality), region_id in self._outputs.items()
        ):
            raise _Fault("NAME_CONFLICT")
        region_id = await self._derive(operation, ids, name)
        self._outputs[("composition", name, None)] = region_id
        return {
            "regions": [self._safe_region(self._regions[region_id])],
            "artifacts": self._snapshot_for([region_id])["artifacts"],
        }

    async def _derive(self, operation, ids, name, *, task="composition", quality=None):
        operands = tuple(sorted(ids)) if operation in {"union", "intersection"} else tuple(ids)
        key = (operation, operands, name)
        if key in self._compositions:
            # Cache reuse still checks immutable source artifacts.
            await core._validation(self._load_regions, ids)
            return self._compositions[key]
        artifact = await core._validation(
            self._write_composition, operation, ids, name, task=task, quality=quality
        )
        self._artifacts[artifact["id"]] = artifact
        region = self._new_region(
            name, artifact, [1], artifact["nonzero_voxels"], artifact["provenance"]
        )
        self._compositions[key] = region["id"]
        return region["id"]

    def _write_composition(
        self, operation, ids, name, *, task="composition", quality=None, _stop_event=None
    ):
        import nibabel as nib
        import numpy as np

        images = self._load_regions(ids, _stop_event=_stop_event)
        reference = self._artifacts[self._regions[ids[0]]["artifact_id"]]
        image = images[reference["id"]]
        index = len(self._artifacts) + 1
        path = self._root / f"derived-{index}.nii.gz"
        temporary = self._root / f".derived-{index}.tmp.nii.gz"
        raw_path = self._root / f".derived-{index}.raw"
        data = None
        try:
            data = np.memmap(raw_path, dtype=np.uint8, mode="w+", shape=image.shape, order="F")
            os.chmod(raw_path, 0o600)
            count = 0
            for z in range(image.shape[2]):
                core._check_stop(_stop_event)
                planes = {
                    artifact_id: np.asanyarray(source.dataobj[:, :, z])
                    for artifact_id, source in images.items()
                }
                masks = [
                    np.isin(
                        planes[self._regions[region_id]["artifact_id"]],
                        self._regions[region_id]["values"],
                    )
                    for region_id in ids
                ]
                combined = masks[0].copy()
                for mask in masks[1:]:
                    if operation == "union":
                        combined |= mask
                    elif operation == "intersection":
                        combined &= mask
                    else:
                        combined &= ~mask
                data[:, :, z] = combined
                count += int(np.count_nonzero(combined))
            data.flush()
            header = image.header.copy()
            header.set_data_dtype(np.uint8)
            header.set_slope_inter(1, 0)
            nib.save(nib.Nifti1Image(data, image.affine, header), temporary)
            os.chmod(temporary, 0o600)
            geometry = core._inspect_nifti(temporary, label_map={1: name}, _stop_event=_stop_event)
            actual = {
                "shape": geometry["shape"],
                "spacing": geometry["voxel_spacing"],
                "affine": geometry["affine"],
            }
            if (
                not _same_geometry(actual, reference["geometry"])
                or geometry["nonzero_voxels"] != count
            ):
                raise _Fault("OUTPUT_INVALID")
            core._check_stop(_stop_event)
            os.replace(temporary, path)
            volume = count * reference["volume_measurement"]["voxel_volume_mm3"]
            labels = [
                {
                    "id": 1,
                    "name": name,
                    "color": core._label_color(1),
                    "voxels": count,
                    "volume_mm3": volume,
                    "volume_ml": volume / 1000,
                }
            ]
            return self._artifact(
                path,
                labels,
                actual,
                reference["volume_measurement"],
                {
                    "operation": operation,
                    "source_region_ids": list(ids),
                    "task": task,
                    "quality": quality,
                },
                _stop_event=_stop_event,
                task=task,
                quality=quality,
            )
        finally:
            if data is not None:
                data._mmap.close()
            temporary.unlink(missing_ok=True)
            raw_path.unlink(missing_ok=True)
