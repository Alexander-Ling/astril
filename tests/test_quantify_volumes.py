import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import nibabel as nib

from astril.quantify_volumes import quantify_segmentation_volumes


def _write_seg(path, label_values, shape=(4, 4, 4)):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(shape, dtype=np.uint8)
    for i, level in enumerate(label_values):
        data.flat[i] = level
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(path))


def _read_header_and_rows(output_file):
    lines = Path(output_file).read_text().splitlines()
    return lines[0].split("\t"), [line.split("\t") for line in lines[1:]]


class TestQuantifyVolumes(unittest.TestCase):
    def test_no_contract_or_total_levels_raises(self):
        # The level set must be pinned down by a contract or an explicit count -- never inferred
        # by scanning voxel data for whichever values happen to appear (fragile/data-dependent).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_seg(root / "examA" / "examA_GBM-seg.nii.gz", [1, 3])
            out = root / "volumes.tsv"

            with self.assertRaises(ValueError):
                quantify_segmentation_volumes(str(root), "_GBM-seg.nii.gz", str(out))

    def test_explicit_total_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_seg(root / "examA" / "examA_GBM-seg.nii.gz", [1])
            out = root / "volumes.tsv"

            quantify_segmentation_volumes(str(root), "_GBM-seg.nii.gz", str(out), total_levels=4)

            header, _rows = _read_header_and_rows(out)
            self.assertEqual(header, [
                "File_path", "File_name",
                "Segment1Volume_mm3", "Segment2Volume_mm3",
                "Segment3Volume_mm3", "Segment4Volume_mm3",
            ])

    def test_explicit_labels_json_contract_names_columns_and_forces_full_level_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_seg(root / "examA" / "examA_GBM-seg.nii.gz", [1])
            contract = root / "labels.json"
            contract.write_text(json.dumps({
                "model_family": "GBM_seg_v2",
                "labels": {
                    "0": "background",
                    "1": "necrosis_non_enhancing_tumor",
                    "2": "edema_non_flair_hyperintensity",
                },
            }))
            out = root / "volumes.tsv"

            quantify_segmentation_volumes(
                str(root), "_GBM-seg.nii.gz", str(out), labels_json=str(contract)
            )

            header, rows = _read_header_and_rows(out)
            self.assertEqual(header, [
                "File_path", "File_name",
                "necrosis_non_enhancing_tumor_Volume_mm3",
                "edema_non_flair_hyperintensity_Volume_mm3",
            ])
            # Level 2 is declared by the contract but absent from this exam's data -- still
            # reported (as 0), not omitted, so every row shares the same column set.
            self.assertEqual(rows[0][3], "0")

    def test_provenance_sidecar_is_used_as_an_automatic_contract(self):
        # segment_GBM.py writes segmentation_provenance.json next to its output; quantify should
        # pick it up automatically (still a contract, just supplied by the segmentation step
        # rather than by hand) when no explicit total_levels/labels_json is given.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exam_dir = root / "examA"
            _write_seg(exam_dir / "examA_GBM-seg.nii.gz", [1])
            provenance = {
                "model_family": "GBM_seg_v2",
                "labels": {"1": "necrosis_non_enhancing_tumor", "2": "edema_non_flair_hyperintensity"},
            }
            (exam_dir / "segmentation_provenance.json").write_text(json.dumps(provenance))
            out = root / "volumes.tsv"

            quantify_segmentation_volumes(str(root), "_GBM-seg.nii.gz", str(out))

            header, rows = _read_header_and_rows(out)
            self.assertEqual(header, [
                "File_path", "File_name",
                "necrosis_non_enhancing_tumor_Volume_mm3",
                "edema_non_flair_hyperintensity_Volume_mm3",
            ])
            self.assertEqual(rows[0][3], "0")

    def test_explicit_contract_overrides_provenance_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exam_dir = root / "examA"
            _write_seg(exam_dir / "examA_GBM-seg.nii.gz", [1])
            (exam_dir / "segmentation_provenance.json").write_text(json.dumps({
                "model_family": "GBM_seg_v2",
                "labels": {"1": "from_sidecar"},
            }))
            out = root / "volumes.tsv"

            quantify_segmentation_volumes(
                str(root), "_GBM-seg.nii.gz", str(out),
                level_labels={1: "from_explicit_override"},
            )

            header, _rows = _read_header_and_rows(out)
            self.assertEqual(header, ["File_path", "File_name", "from_explicit_override_Volume_mm3"])


if __name__ == "__main__":
    unittest.main()
