import pytest

from medsegagent import core
from medsegagent.totalseg_worker import sequential_predictor


class Predictor:
    def predict_from_files_sequential(self, source, destination, **options):
        return "sequential", source, destination, options

    def predict_from_files(self, source, destination, **options):
        return "partitioned", source, destination, options


@pytest.mark.parametrize("cropped", [True, False])
@pytest.mark.parametrize("probabilities", [True, False])
def test_adapter_preserves_numerical_and_cascade_options(cropped, probabilities):
    adapted = sequential_predictor(Predictor)()
    result = adapted.predict_from_files(
        "source",
        "destination",
        save_probabilities=probabilities,
        overwrite=False,
        folder_with_segs_from_prev_stage="previous",
        num_processes_preprocessing=4,
        num_processes_segmentation_export=5,
        use_cropped_logits_resampling=cropped,
    )
    assert result == (
        "sequential",
        "source",
        "destination",
        {
            "save_probabilities": probabilities,
            "overwrite": False,
            "folder_with_segs_from_prev_stage": "previous",
            "use_cropped_logits_resampling": cropped,
        },
    )
    assert type(Predictor()) is Predictor


def test_partitioned_prediction_retains_upstream_contract():
    result = sequential_predictor(Predictor)().predict_from_files(
        "input",
        "output",
        num_parts=2,
        part_id=1,
        num_processes_preprocessing=7,
        use_cropped_logits_resampling=True,
    )
    assert result[0] == "partitioned"
    assert result[3]["num_parts"] == 2
    assert result[3]["part_id"] == 1
    assert result[3]["num_processes_preprocessing"] == 7
    assert result[3]["use_cropped_logits_resampling"] is True


def test_worker_and_cli_use_identical_segmentation_options(monkeypatch, tmp_path):
    kwargs = {
        "task": "total",
        "input_path": tmp_path / "ct.nii",
        "output_dir": tmp_path,
        "targets": ["liver"],
        "speed": "fast",
    }
    monkeypatch.setenv("MEDSEGAGENT_TOTALSEG_ENGINE", "sequential")
    optimized = core._build_command(**kwargs)
    monkeypatch.setenv("MEDSEGAGENT_TOTALSEG_ENGINE", "cli")
    monkeypatch.setattr(core, "which", lambda _: "/venv/bin/TotalSegmentator")
    baseline = core._build_command(**kwargs)
    assert optimized[:3] == [core.os.sys.executable, "-m", "medsegagent.totalseg_worker"]
    assert optimized[3:] == baseline[1:]
    monkeypatch.setenv("MEDSEGAGENT_TOTALSEG_ENGINE", "invalid")
    with pytest.raises(core.SegmentationError, match="ENGINE"):
        core._build_command(**kwargs)
