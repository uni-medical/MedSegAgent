"""Single-request TotalSeg process using upstream sequential nnUNet I/O.

Keep TotalSeg's CLI, model options and runtime patches. Only its predictor factory
uses a subclass: preprocessing/export run in this disposable inference process,
instead of importing Torch again in extra Python workers. The service process and
installed dependency files are never patched. GPU leases and process cancellation
remain owned by core.
"""

from __future__ import annotations

import importlib.metadata


def sequential_predictor(base):
    class SequentialPredictor(base):
        def predict_from_files(
            self,
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=3,
            num_processes_segmentation_export=3,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
            use_cropped_logits_resampling=False,
        ):
            # TotalSeg submits one case (possibly split internally). Keep the
            # upstream partitioned API intact if a future caller requests it.
            if num_parts != 1 or part_id != 0:
                return super().predict_from_files(
                    list_of_lists_or_source_folder,
                    output_folder_or_list_of_truncated_output_files,
                    save_probabilities=save_probabilities,
                    overwrite=overwrite,
                    num_processes_preprocessing=num_processes_preprocessing,
                    num_processes_segmentation_export=num_processes_segmentation_export,
                    folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
                    num_parts=num_parts,
                    part_id=part_id,
                    use_cropped_logits_resampling=use_cropped_logits_resampling,
                )
            return self.predict_from_files_sequential(
                list_of_lists_or_source_folder,
                output_folder_or_list_of_truncated_output_files,
                save_probabilities=save_probabilities,
                overwrite=overwrite,
                folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
                use_cropped_logits_resampling=use_cropped_logits_resampling,
            )

    return SequentialPredictor


def main():
    expected = {"TotalSegmentator": "2.18.0", "nnunetv2": "2.8.1"}
    for package, version in expected.items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(
                f"Sequential adapter requires {package}=={version}. "
                "Use MEDSEGAGENT_TOTALSEG_ENGINE=cli until the new version is validated."
            )
    # Complete TotalSeg's lazy imports and cropped-logit patches before deriving
    # the predictor; otherwise its lazy import can replace our factory.
    from totalsegmentator import nnunet
    from totalsegmentator.bin.TotalSegmentator import main as upstream_main

    original = nnunet.nnUNetPredictor
    nnunet.nnUNetPredictor = sequential_predictor(original)
    try:
        upstream_main()
    finally:
        nnunet.nnUNetPredictor = original


if __name__ == "__main__":
    main()
