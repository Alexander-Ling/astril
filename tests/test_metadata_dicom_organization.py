import os
import signal
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pydicom
import pandas as pd
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from astril.preprocess import create_patient_metadata, organize_dicoms, plan_dicom_to_nifti_conversion
from astril import preprocessing_utils


def _write_dicom(path, *, patient, study, series, sop, description, modality="MR", sop_class_uid=None, patient_name="Example^Patient", study_date="20260115"):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class_uid or MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PatientID = patient
    ds.PatientName = patient_name
    ds.StudyInstanceUID = study
    ds.StudyID = "exam-label"
    ds.StudyDate = study_date
    ds.SeriesInstanceUID = series
    ds.SeriesNumber = "7"
    ds.SeriesDescription = description
    if modality:
        ds.Modality = modality
    if sop_class_uid:
        ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path), write_like_original=False)


class MetadataDICOMOrganizationTests(unittest.TestCase):
    def test_organization_classifies_modalities_in_both_input_structures(self):
        layouts = {
            "with_mr": lambda name: Path("Patient1") / "Exam1" / "MR" / name / ("001.dcm"),
            "direct_extensionless": lambda name: Path("Patient1") / "Exam1" / name / "001",
        }
        expected = {"MR": "MRI", "CT": "CT", "US": "Ultrasound", "OT": "Unknown"}
        for layout_name, make_path in layouts.items():
            with self.subTest(layout=layout_name), tempfile.TemporaryDirectory() as temp:
                root, out, index = Path(temp) / "source", Path(temp) / "normalized", Path(temp) / "index.csv"
                for number, (modality, sequence_type) in enumerate(expected.items(), start=1):
                    _write_dicom(
                        root / make_path(f"sequence_{number}"), patient="P000", study=generate_uid(),
                        series=generate_uid(), sop=generate_uid(), description=f"sequence {number}",
                        modality=modality, sop_class_uid="1.2.3.4" if modality == "OT" else None,
                    )
                frame = organize_dicoms(str(root), str(out), index_out=str(index), show_progress=False, n_workers=1)
                copied = frame[frame["status"] == "copied"]
                self.assertEqual(len(copied), len(expected))
                self.assertEqual(dict(zip(copied["modality"], copied["sequence_type"])), expected)
                self.assertEqual(set(copied["sequence_type_source"]), {"modality", "unknown"})

    def test_interrupt_returns_partial_index_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out, index = Path(temp) / "source", Path(temp) / "normalized", Path(temp) / "index.csv"
            for number in (1, 2):
                _write_dicom(
                    root / "Patient1" / "Exam1" / "MR" / "T1n" / f"{number:03d}.dcm",
                    patient="PINT", study=generate_uid(), series=generate_uid(), sop=generate_uid(), description="T1",
                )
            original_reader = preprocessing_utils._safe_dcmread
            state = {"sent": False}

            def _read_and_interrupt(path):
                dataset = original_reader(path)
                if not state["sent"]:
                    state["sent"] = True
                    signal.raise_signal(signal.SIGINT)
                return dataset

            with patch("astril.preprocessing_utils._safe_dcmread", side_effect=_read_and_interrupt):
                result = organize_dicoms(str(root), str(out), index_out=str(index), show_progress=False, n_workers=1)

            self.assertTrue(index.exists())
            self.assertTrue(len(result) >= 1)
            self.assertTrue((result["status"] == "interrupted_metadata").all())

    def test_organization_ignores_source_folder_names_and_indexes_nested_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            out = Path(temp) / "normalized"
            index = Path(temp) / "index.csv"
            study = generate_uid()
            series_a = generate_uid()
            series_b = generate_uid()
            _write_dicom(root / "random" / "one" / "DICOM" / "a.dcm", patient="P001", study=study, series=series_a, sop=generate_uid(), description="T1 axial")
            _write_dicom(root / "unrelated" / "two" / "deep" / "b.dcm", patient="P001", study=study, series=series_b, sop=generate_uid(), description="T2 axial")
            (root / "notes.txt").parent.mkdir(parents=True, exist_ok=True)
            (root / "notes.txt").write_text("not dicom", encoding="utf-8")

            frame = organize_dicoms(str(root), str(out), index_out=str(index), show_progress=False, n_workers=1)

            self.assertEqual(set(frame["status"]), {"copied", "quarantined"})
            copied = frame[frame["status"] == "copied"]
            self.assertEqual(len(copied), 2)
            self.assertTrue(all(Path(p).exists() for p in copied["normalized_path"]))
            self.assertEqual({Path(p).parts[-4] for p in copied["normalized_path"]}, {"Patient_P001_" + __import__("hashlib").sha1(b"P001").hexdigest()[:10]})
            self.assertEqual(len({Path(p).parts[-3] for p in copied["normalized_path"]}), 1)
            self.assertEqual(len({Path(p).parts[-2] for p in copied["normalized_path"]}), 2)
            quarantined = frame[frame["status"] == "quarantined"].iloc[0]
            self.assertTrue(Path(quarantined["quarantine_path"]).exists())
            unresolved = index.with_name(index.stem + "_unresolved.csv")
            self.assertTrue(unresolved.exists())
            self.assertEqual(len(pd.read_csv(unresolved)), 1)

            rerun = organize_dicoms(str(root), str(out), show_progress=False, n_workers=1)
            self.assertEqual((rerun["status"] == "skipped_existing").sum(), 2)

    def test_one_study_uid_uses_one_exam_directory_when_dates_differ(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out, index = Path(temp) / "source", Path(temp) / "normalized", Path(temp) / "index.csv"
            study = generate_uid()
            series = generate_uid()
            _write_dicom(root / "source_a" / "001.dcm", patient="P005", study=study, series=series, sop=generate_uid(), description="T1", study_date="20260115")
            _write_dicom(root / "source_b" / "002.dcm", patient="P005", study=study, series=series, sop=generate_uid(), description="T1", study_date="20260116")

            frame = organize_dicoms(str(root), str(out), index_out=str(index), show_progress=False, n_workers=1)

            copied = frame[frame["status"] == "copied"]
            exam_dirs = {str(Path(path).parent.parent) for path in copied["normalized_path"]}
            self.assertEqual(len(exam_dirs), 1)

    def test_series_uid_collision_keeps_largest_parent_group(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out, index = Path(temp) / "source", Path(temp) / "normalized", Path(temp) / "index.csv"
            series = generate_uid()
            retained_study = generate_uid()
            duplicate_study = generate_uid()

            for number in range(3):
                _write_dicom(
                    root / "export_a" / f"retained_{number:03d}.dcm",
                    patient="P006", study=retained_study, series=series,
                    sop=generate_uid(), description="T1",
                )
            for number in range(2):
                _write_dicom(
                    root / "export_b" / f"duplicate_{number:03d}.dcm",
                    patient="P006", study=duplicate_study, series=series,
                    sop=generate_uid(), description="T1",
                )

            frame = organize_dicoms(str(root), str(out), index_out=str(index), show_progress=False, n_workers=1)

            retained = frame[frame["study_instance_uid"] == retained_study]
            quarantined = frame[frame["study_instance_uid"] == duplicate_study]
            self.assertEqual(set(retained["status"]), {"copied"})
            self.assertEqual(set(quarantined["status"]), {"quarantined"})
            self.assertTrue(all(Path(path).exists() for path in retained["normalized_path"]))
            self.assertTrue(all(Path(path).exists() for path in quarantined["quarantine_path"]))
            self.assertTrue(all("conflicting_parent_for_series_uid" in str(error) for error in quarantined["error"]))
            unresolved = pd.read_csv(index.with_name(index.stem + "_unresolved.csv"))
            self.assertEqual(len(unresolved), len(quarantined))

    def test_patient_metadata_can_be_created_from_index_without_mr_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            out = Path(temp) / "normalized"
            index = Path(temp) / "index.csv"
            metadata = Path(temp) / "patients.csv"
            _write_dicom(root / "vendor" / "anything" / "scan", patient="P002", study=generate_uid(), series=generate_uid(), sop=generate_uid(), description="T1")
            organize_dicoms(str(root), str(out), index_out=str(index), show_progress=False, n_workers=1)

            table = create_patient_metadata(str(out), str(metadata), index_path=str(index), show_progress=False, n_workers=1)

            self.assertEqual(len(table), 1)
            self.assertEqual(table.iloc[0]["dicomPatientID"], "p002")
            self.assertTrue(str(table.iloc[0]["Directory"]).startswith("Patient_P002_"))

    def test_index_metadata_aggregates_names_to_one_row_per_patient_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            out = Path(temp) / "normalized"
            index = Path(temp) / "index.csv"
            metadata = Path(temp) / "patients.csv"
            study = generate_uid()
            _write_dicom(root / "vendor" / "first", patient="P004", study=study, series=generate_uid(), sop=generate_uid(), description="T1", patient_name="Alpha^Patient")
            _write_dicom(root / "vendor" / "second", patient="P004", study=study, series=generate_uid(), sop=generate_uid(), description="T2", patient_name="Beta^Patient")
            organize_dicoms(str(root), str(out), index_out=str(index), show_progress=False, n_workers=1)

            table = create_patient_metadata(str(out), str(metadata), index_path=str(index), show_progress=False, n_workers=1)

            self.assertEqual(len(table), 1)
            self.assertEqual(table.iloc[0]["patientName"], "alpha patient; beta patient")
            self.assertEqual(table.iloc[0]["dicomPatientID"], "p004")

    def test_planner_uses_indexed_exam_without_mr_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            out = Path(temp) / "normalized"
            index = Path(temp) / "index.csv"
            metadata = Path(temp) / "patients.csv"
            plan = Path(temp) / "plan.csv"
            _write_dicom(root / "vendor" / "exam-x" / "scan", patient="P003", study=generate_uid(), series=generate_uid(), sop=generate_uid(), description="T1")
            organize_dicoms(str(root), str(out), index_out=str(index), show_progress=False, n_workers=1)
            create_patient_metadata(str(out), str(metadata), index_path=str(index), show_progress=False, n_workers=1)
            patient_table = pd.read_csv(metadata, dtype=str).fillna("")
            patient_table.loc[0, "patientID"] = "P003"
            patient_table.loc[0, "day0Date"] = "2026-01-01"
            patient_table.to_csv(metadata, index=False)

            series_folder = str(Path(pd.read_csv(index).iloc[0]["normalized_path"]).parent)
            fake_classification = pd.DataFrame([{
                "folder": series_folder, "series_number": 7, "acq_dt": None,
                "acq_dt_iso": "", "manufacturer": "", "modality": "MR",
                "series_description": "T1", "protocol_name": "", "sequence_name": "",
                "image_type": "", "te": None, "tr": None, "ti": None,
                "flip_angle": None, "b_value": None, "primary_secondary": "PRIMARY",
                "is_fspgr": False, "base_type": "T1", "final_label": "T1",
                "is_postcontrast": False, "is_flair": False, "reason": "",
                "confidence": 1.0, "plane": "Axial", "matrix": "", "voxel_mm": "",
                "n_slices": 20, "mr_acq_type": "", "pulse_sequence_name": "",
                "is_derived": False,
            }])
            with patch("astril.preprocessing_utils.classify_exam_series", return_value=fake_classification) as legacy_classifier, \
                 patch("astril.preprocessing_utils._safe_dcmread", side_effect=AssertionError("indexed planner reread a DICOM")):
                result = plan_dicom_to_nifti_conversion(
                    str(metadata), str(out), str(Path(temp) / "nifti"),
                    plan_out=str(plan), show_progress=False, n_workers=2,
                    dicom_index=str(index), min_slices=0,
                )

            self.assertTrue(plan.exists())
            self.assertEqual(result["n_exams"], 1)
            self.assertEqual(result["n_series"], 1)
            self.assertEqual(result["n_selected"], 1)
            self.assertEqual(result["n_errors"], 0)
            self.assertFalse(legacy_classifier.called)
            planned = pd.read_csv(plan, dtype=str, keep_default_na=False)
            planned = planned[~planned["Directory"].isin(["", "-"])]
            self.assertTrue(all(str(value).startswith("Patient_") for value in planned["Directory"].unique()))
            self.assertTrue(all("Exam_" in str(value) for value in planned["ExamDirectory"].unique()))
            self.assertEqual(set(planned["sequence_type"]), {"MRI"})


if __name__ == "__main__":
    unittest.main()
