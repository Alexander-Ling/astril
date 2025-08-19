
# astril: Automated Segmentation Toolkit for Radiology Image Libraries
### Authors:
- Alexander Ling (alexander.l.ling@gmail.com; alling@bwh.harvard.edu)
-  E. Antonio Chiocca
### License: Creative Commons CC BY-NC-SA 4.0
### License Holders:
- Alexander Ling
- E. Antonio Chiocca
- Mass General Brigham
### Installation Requirements:
- Python 3.11 or higher
### Description:
astril is a python package designed to streamline radiology image pre-processing (i.e. scan type identification, DICOM -> NIFTI conversion, co-registration, skull stripping, normalization, etc.), segmentation, and model training. The package was built to make it easy for other researchers to trivially apply our glioblastoma segmentation algorithms to their own brain MRI scans while also providing the tools to refine our models or train their own from scratch. Most of the functions in this package should also work with radiology images other than MRIs, though the functionality and models are currently tailred specifically to MRI images.

A detailed description of the package and instructions for its use will be provided in the coming months as we finalize the package and its associated manuscript.

### General Workflow (rough draft)
---
#### Format for radiology image inputs:
astril is designed to perform image pre-processing and segmentation on NIFTI (.nii.gz) images. However, astril includes functionality to convert DICOM imaging datasets to NIFTI images and organize them into the folder structure expected by astril's processing functions.
If starting from a set of DICOM images, they should be organized by patient, radiology type, study, and series as in the following example:
```
DICOM_Library/
├── Patient1/ 					{Patient}
│   ├── MR/ 					{Radiology Type}
|	|	├── Study1/ 			{Study}
|	|	|	├── T1n/ 			{Series}
|	|	|	|	├── 001.dcm
|	|	|	|	├── 002.dcm
|	|	|	|	├── ...
|	|	|	├── T1c/
|	|	|	|	├── 145.dcm
|	|	|	|	├── 146.dcm
|	|	|	|	├── ...
|	|	|	├── ...
|	|	├── Study2/
|	|	|	├── T1n/
|	|	|	|	├── 001.dcm
|	|	|	|	├── 002.dcm
|	|	|	|	├── ...
|	|	|	├── ...
|	|	├── ...
│   ├── ...
├── ...
```
If starting from a set of NIFTI images, they should be organized by patient, study, and series as in the following example:
```
NIFTI_Library/
├── Patient1/ 					{Patient}
|   ├── Study1/ 				{Study}
|	|	├── T1c.nii.gz 			{Series}
|	|	├── T1c.nii.gz 
|	|	├── ...			
|   ├── ...
├── ...
```
---
#### Pre-processing files from DICOM library
1. (Optional) Ensure that all of the DICOM files for each series are properly separated into their respective directories. I have occasionally found that DICOM libraries can have misplaced dicom files such that a single series has been divided into multiple folder or dicom files from multiple series have been placed in the same folder. The following command will check that all dicom files are correctly separated. **Note: The .dcm files in your library must have meaningful SeriesInstanceUID or SeriesNumber, SeriesDescription, and ProtocolName metadata fields for this to work. If these fields have been stripped, this function will not work as intended and <span style="color: red;">could scramble your library.</span>**
```
usage: preprocess.py [-h] --dir DIR [--no-progress] [--logOut LOGOUT] [--outDir OUTDIR]
                     [--n_workers N_WORKERS] [--dryRun] [--in_place]

Ensure each series folder contains only one scan; demix if needed.

options:
  -h, --help            show this help message and exit
  --dir DIR             Root directory containing patient/exam/MR folders
  --no-progress         Disable progress bar
  --logOut LOGOUT       Optional path for the move log (.csv | .tsv). If omitted, a default
                        demix_log_{date}_{time}.csv is written under --dir
  --outDir OUTDIR       If provided, write a fully de-mixed COPY of --dir under this path (original
                        tree left unchanged).If omitted, you MUST also pass --in_place to allow
                        moving files within --dir.
  --n_workers N_WORKERS
                        Number of CPU threads to use while reading DICOM metadata and moving files.
  --dryRun              Compute assignments and write the demix log, but do NOT move/copy any files.
  --in_place            Allow demixing IN PLACE within --dir (moves files). If not set, you must
                        specify --outDir.
```
2. Create a spreadsheet of patient-level metadata to help with organizing/deidentifying your dataset during NIFTI file conversion. A template patient metadata spreadsheet can be generated using the following function.
```
astril.preprocess create_patient_metadata
options:
  -h, --help            show this help message and exit
  --dir DIR             Root directory with {Patient_folder}/.../MR/{series}
  --metadataOut METADATAOUT
                        Output path (.csv | .tsv | .xlsx)
  --previousMetadata [PREVIOUSMETADATA ...]
                        Zero or more previous metadata tables (.csv|.tsv|.xlsx) to be used to help populate metadata or to omit patient folders that have been previously analysed.
  --omitPrevious        Omit rows whose Directory appears in previous metadata
  --subdirs SUBDIRS [SUBDIRS ...]
                        One or more subfolder names to search under each patient folder (default: MR). Example: --subdirs MR MR2
  --excludeEmpty        If set, exclude patient folders where no DICOM files were found under the chosen subfolders
  
Example Usage:
py -m astril.preprocess create_patient_metadata --dir ./My_DICOM_Library/ --metadataOut ./Patient_Metadata.csv
```
This will generate a metadata table with the following columns:
- **Directory:** The immediate subdirectory of --dir to which the metadata in a given row pertains. Remember, each of these directories should contain scans for only a single patient.
- **patientID:** An arbitrary identifier that you will assign to this patient. This is the identifier that will be used when DICOM files are converted to NIFTI files.
- **patientName:** The name of the patient as extracted from the DICOM metadata.
- **dicomPatientID:** The patientID for this patient as extracted from the DICOM metadata.
- **day0Date:** A date (mm/dd/yyyy) that should be used as day 0 when organizing scans by timepoint.

<u>Example of an Incomplete Table</u>
|Directory|patientID|patientName|dicomPatientID|day0Date|
 |-|-|-|-|-|
 |Patient1| |Jane Doe|1234567||
 |Patient2| |John Doe|7654321||

<u>Example of a Completed Table</u>
|Directory|patientID|patientName|dicomPatientID|day0Date|
 |-|-|-|-|-|
 |Patient1|Subject1|Jane Doe|1234567|01/21/1900|
 |Patient2|Subject2|John Doe|7654321|01/22/1900|