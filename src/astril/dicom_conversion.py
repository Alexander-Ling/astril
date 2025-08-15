import argparse
from pathlib import Path
from .preprocessing_utils import ensure_dicom2nifti_installed


def convert_single_dicom_series_to_nifti(dicom_series_dir, output_path, reorient=True, compress=True):
    import dicom2nifti
    import tempfile
    import os

    dicom_series_dir = Path(dicom_series_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dicom_series_dir.exists() or not any(dicom_series_dir.glob("*.dcm")):
        raise FileNotFoundError(f"No DICOM files found in {dicom_series_dir}")

    with tempfile.TemporaryDirectory() as tmpdir:
        dicom2nifti.convert_directory(
            dicom_series_dir, tmpdir,
            reorient=reorient,
            compression=compress
        )

        final_file = next(Path(tmpdir).glob("*.nii*"))
        final_file.rename(output_path)

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a single DICOM series to NIfTI (.nii.gz)."
    )
    parser.add_argument("--dicom_dir", type=str, required=True,
                        help="Path to DICOM series directory containing only one scan.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output NIfTI file path (.nii or .nii.gz).")
    parser.add_argument("--no_reorient", action="store_true",
                        help="Disable reorientation to standard (RAS) space.")
    parser.add_argument("--no_compress", action="store_true",
                        help="Disable gzip compression (.nii instead of .nii.gz).")

    args = parser.parse_args()
    ensure_dicom2nifti_installed()

    output_path = convert_single_dicom_series_to_nifti(
        dicom_series_dir=args.dicom_dir,
        output_path=args.output_path,
        reorient=not args.no_reorient,
        compress=not args.no_compress
    )

    print(f"Saved: {output_path}")