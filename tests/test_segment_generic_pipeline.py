import configparser
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import nibabel as nib

from astril import segment as seg_mod


def _write_nifti(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), str(path))


def _fake_process_subject_with_models(seg_config_file, subject_index, loaded_models,
                                      slice_batch_size, overwrite, segment_suffix, tiebreaker_model,
                                      debug_models, extra_channel_paths=None, brainiac_paths_list=None,
                                      return_merged=False):
    """
    Stands in for the real (torch/GPU) astril.segment_GBM.process_subject_with_models: mirrors
    its real out_path-naming logic (reading maskPattern/output_directory back out of the
    segmentation config file it's given) and writes a plausible label file, so the orchestration
    around it -- config building, per-subject scoping, derived-output wiring, skip/overwrite --
    is exercised the same way it would be for real, without needing real trained weights or a GPU.
    """
    cp = configparser.ConfigParser()
    cp.read(seg_config_file)
    cfg = cp["DEFAULT"]
    with open(cfg["mask_paths_file"]) as f:
        mask_paths = [line.strip() for line in f if line.strip()]
    mask_path = Path(mask_paths[subject_index])

    mask_pattern = cfg["maskPattern"]
    base_name = mask_path.name
    if ".nii.gz" in mask_pattern:
        seg_name = base_name.replace(mask_pattern, segment_suffix)
    else:
        seg_name = base_name.replace(".nii.gz", segment_suffix)
    output_directory = cfg["output_directory"]
    out_dir = mask_path.parent if output_directory == "in_place" else Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / seg_name
    if out_path.exists() and not overwrite:
        return None

    mask_img = nib.load(str(mask_path))
    foreground = mask_img.get_fdata() > 0.5
    label = foreground.astype(np.uint8)
    nib.save(nib.Nifti1Image(label, mask_img.affine), str(out_path))

    if return_merged:
        probs = np.zeros(mask_img.shape + (2,), dtype=np.float32)
        probs[..., 1][foreground] = 1.0
        probs[..., 0][~foreground] = 1.0
        return label, probs, mask_img.affine
    return None


class TestGenericSegmentationEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        self.models_root = self.tmp / "models_root"
        self.family_root = self.models_root / "TestFamily"
        self.family_root.mkdir(parents=True)
        for name in ["StageA_Axial", "StageB_Axial"]:
            (self.family_root / f"{name}_train_parameters.cfg").write_text(
                "[DEFAULT]\n"
                "slicing_plane = axial\n"
                "num_input_slices = 3\n"
                "num_output_slices = 1\n"
                "minimum_height_width = 8\n"
                "num_classes = 2\n"
                "optional_channels = t1n\n"
            )
        pipeline = {
            "model_family": "TestFamily",
            "labels": {"0": "background", "1": "tumor"},
            "initial_inputs": {
                "t1c": {"kind": "channel", "pattern": "_t1c.nii.gz", "required": True},
                "t1n": {"kind": "channel", "pattern": "_t1n.nii.gz", "required": False},
                "brainmask": {"kind": "mask", "pattern": "_brainmask.nii.gz", "required": True},
            },
            "stages": [
                {
                    "name": "stage_a",
                    "models": [{"plane": "Axial", "dir": "StageA_Axial", "cfg": "StageA_Axial_train_parameters.cfg"}],
                    "inputs": ["t1c", "t1n"],
                    "mask": "brainmask",
                    "merge": {"method": "average_prob", "tiebreaker": "Axial"},
                    "label_output_suffix": "_stageA_seg.nii.gz",
                    "is_final": False,
                    "derived_outputs": [
                        {
                            "name": "gate", "op": "threshold_gate", "source": "probabilities",
                            "background_index": 0, "threshold": 0.5, "dilation_voxels": 0,
                            "intersect_with": "brainmask", "pattern": "_gate.nii.gz",
                        }
                    ],
                },
                {
                    "name": "stage_b",
                    "models": [{"plane": "Axial", "dir": "StageB_Axial", "cfg": "StageB_Axial_train_parameters.cfg"}],
                    "inputs": ["t1c", "t1n"],
                    "mask": "stage_a.gate",
                    "merge": {"method": "average_logit", "tiebreaker": "Axial"},
                    "label_output_suffix": None,
                    "is_final": True,
                    "derived_outputs": [],
                },
            ],
            "final_postprocessing": [],
        }
        (self.family_root / "pipeline.json").write_text(json.dumps(pipeline))

        self.input_dir = self.tmp / "input"
        shape = (6, 6, 6)
        mask = np.zeros(shape)
        mask[1:5, 1:5, 1:5] = 1
        _write_nifti(self.input_dir / "P1_ExamA" / "P1_ExamA_t1c.nii.gz", mask)
        _write_nifti(self.input_dir / "P1_ExamA" / "P1_ExamA_t1n.nii.gz", mask)
        _write_nifti(self.input_dir / "P1_ExamA" / "P1_ExamA_brainmask.nii.gz", mask)
        # P2 deliberately omits the optional t1n channel.
        _write_nifti(self.input_dir / "P2_ExamB" / "P2_ExamB_t1c.nii.gz", mask)
        _write_nifti(self.input_dir / "P2_ExamB" / "P2_ExamB_brainmask.nii.gz", mask)

        # Always-needed patches (independent of how the model is selected).
        patch("astril.run_segmentation.load_models_for_config", return_value=["fake_model"]).start()
        self.addCleanup(patch.stopall)
        self.mock_run = patch.object(
            seg_mod, "process_subject_with_models", side_effect=_fake_process_subject_with_models
        ).start()

    def _assert_exam_outputs_ok(self, exam_dir, exam_name, expected_family="TestFamily"):
        # The final output existing at all proves stage_b's config successfully found the gate
        # stage_a produced (mask="stage_a.gate") -- if that reference had resolved wrong,
        # stage_b's create_segmentation_config would have matched zero subjects and the fake
        # process_subject_with_models would have raised an IndexError instead.
        self.assertTrue((exam_dir / f"{exam_name}_seg.nii.gz").exists(), f"final seg missing for {exam_name}")
        # Both the gate (a derived, intermediate artifact) and stage_a's own intermediate label
        # output are cleaned up after the subject completes, matching segment_GBM.py's existing
        # cleanup_intermediate_files behavior for _Model_A_gate.nii.gz.
        self.assertFalse((exam_dir / f"{exam_name}_gate.nii.gz").exists(), "derived gate should be cleaned up")
        self.assertFalse((exam_dir / f"{exam_name}_stageA_seg.nii.gz").exists(), "intermediate output should be cleaned up")

        provenance = json.loads((exam_dir / "segmentation_provenance.json").read_text())
        self.assertEqual(provenance["model_family"], expected_family)
        self.assertEqual(provenance["labels"], {"1": "tumor"})

    def test_model_family_resolves_via_locate_models_dir(self):
        with patch("astril.models_download.locate_models_dir", return_value=self.models_root):
            seg_mod.run_segmentation(
                input_dir=str(self.input_dir), model_family="TestFamily", segment_suffix="_seg.nii.gz",
            )
        for exam in ["P1_ExamA", "P2_ExamB"]:
            self._assert_exam_outputs_ok(self.input_dir / exam, exam)

    def test_model_dir_bypasses_family_registry_entirely(self):
        # Deliberately do NOT patch locate_models_dir here: --model_dir must never call it, so a
        # self-trained model a user hasn't published anywhere works without astril's download
        # registry knowing about it at all. If run_segmentation ever called locate_models_dir in
        # this path, it would hit the real, unpatched function and likely fail or resolve wrong.
        seg_mod.run_segmentation(
            input_dir=str(self.input_dir), model_dir=str(self.family_root), segment_suffix="_seg.nii.gz",
        )
        for exam in ["P1_ExamA", "P2_ExamB"]:
            # Provenance's model_family comes from the contract's own declared identity, not from
            # a --model_family argument (none was given here).
            self._assert_exam_outputs_ok(self.input_dir / exam, exam, expected_family="TestFamily")

    def test_model_selection_requires_exactly_one_of_family_or_dir(self):
        with self.assertRaises(ValueError):
            seg_mod.run_segmentation(str(self.input_dir))
        with self.assertRaises(ValueError):
            seg_mod.run_segmentation(str(self.input_dir), model_family="TestFamily", model_dir=str(self.family_root))

    def test_no_matching_files_raises_loud_error_naming_expected_patterns(self):
        # Rename every input file to something that doesn't match the contract's patterns at
        # all -- the naive-user scenario. Matching is substring-based, so the replacement must
        # not merely add a prefix (which would still contain the original pattern) -- it has to
        # replace the pattern-bearing suffix itself.
        renames = {
            "_t1c.nii.gz": "_T1_POST.nii.gz",
            "_t1n.nii.gz": "_T1_PRE.nii.gz",
            "_brainmask.nii.gz": "_mask.nii.gz",
        }
        for exam in ["P1_ExamA", "P2_ExamB"]:
            exam_dir = self.input_dir / exam
            for f in list(exam_dir.iterdir()):
                new_name = f.name
                for old_pattern, new_pattern in renames.items():
                    new_name = new_name.replace(old_pattern, new_pattern)
                f.rename(exam_dir / new_name)

        with self.assertRaises(RuntimeError) as ctx:
            seg_mod.run_segmentation(str(self.input_dir), model_dir=str(self.family_root), segment_suffix="_seg.nii.gz")
        message = str(ctx.exception)
        self.assertIn("_t1c.nii.gz", message)
        self.assertIn("_brainmask.nii.gz", message)
        self.assertIn("input_override", message)

    def test_skip_existing_final_output_unless_overwrite(self):
        seg_mod.run_segmentation(str(self.input_dir), model_dir=str(self.family_root), segment_suffix="_seg.nii.gz")
        first_run_calls = self.mock_run.call_count
        self.assertGreater(first_run_calls, 0)

        seg_mod.run_segmentation(str(self.input_dir), model_dir=str(self.family_root), segment_suffix="_seg.nii.gz")
        self.assertEqual(self.mock_run.call_count, first_run_calls, "second run should skip both subjects entirely")

        seg_mod.run_segmentation(
            str(self.input_dir), model_dir=str(self.family_root), segment_suffix="_seg.nii.gz",
            overwrite_existing_outputs=True,
        )
        self.assertGreater(self.mock_run.call_count, first_run_calls, "overwrite=True should re-run both subjects")

    def test_input_overrides_replace_default_pattern(self):
        # Rename P1's brainmask to a nonstandard pattern and confirm an override finds it.
        exam_dir = self.input_dir / "P1_ExamA"
        (exam_dir / "P1_ExamA_brainmask.nii.gz").rename(exam_dir / "P1_ExamA_custom-mask.nii.gz")
        (self.input_dir / "P2_ExamB").rename(self.tmp / "P2_ExamB_excluded")  # keep this test single-subject

        seg_mod.run_segmentation(
            str(self.input_dir), model_dir=str(self.family_root), segment_suffix="_seg.nii.gz",
            input_overrides={"brainmask": "_custom-mask.nii.gz"},
        )
        self.assertTrue((exam_dir / "P1_ExamA_seg.nii.gz").exists())


if __name__ == "__main__":
    unittest.main()
