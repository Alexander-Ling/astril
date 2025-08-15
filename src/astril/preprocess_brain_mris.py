"""
preprocess_brain_mris.py
Author: Alex Ling
E. Antonio Chiocca Group, BWH
Description: Full MRI preprocessing pipeline for brain scans (T1c, T1n, T2f, T2w), replicating legacy shell pipeline.
"""

import os
import sys
import argparse
import tempfile
import shutil
import json

from astril.preprocess import (
    register_images,
    perform_mri_math,
    run_hd_bet,
    normalize_masked_image,
    resize_mri,
)

def run_preprocessing_pipeline(
    t1c_path,
    t1n_path,
    t2f_path,
    t2w_path,
    output_dir,
    temp_dir,
    co_register_path=None,
    save_scans_with_skulls=False,
    final_dims=(240, 240, 155),
    final_voxels=(1.0, 1.0, 1.0),
    patientID=None,
    timepoint=None,
    scanID=None,
    debug=False,
):
    print(f"[Info] Using temporary directory: {temp_dir}")

    transform_records = {
        "T1c": {},
        "T1n": {},
        "T2f": {},
        "T2w": {},
    }

    transform_basedir = "transforms"
    transform_dir = os.path.join(output_dir, transform_basedir)
    os.makedirs(transform_dir, exist_ok=True)
    temp_transform_dir = os.path.join(temp_dir, transform_basedir)
    os.makedirs(temp_transform_dir, exist_ok=True)

    name_fields = os.path.basename(t1c_path).split('_')

    if patientID is None:
        if len(name_fields) >= 1:
            patientID = name_fields[0]
        else:
            raise ValueError(
                "[Error] Could not extract patientID from T1c filename. "
                "Please provide it explicitly with --patientID."
            )
    print(f"[Info] PatientID = {patientID}")

    if timepoint is None:
        if len(name_fields) >= 2:
            timepoint = name_fields[1]
        else:
            raise ValueError(
                "[Error] Could not extract timepoint from T1c filename. "
                "Please provide it explicitly with --timepoint."
            )
    print(f"[Info] Timepoint = {timepoint}")

    if scanID is None:
        if len(name_fields) >= 3:
            scanID = name_fields[2]
        else:
            raise ValueError(
                "[Error] Could not extract scanID from T1c filename. "
                "Please provide it explicitly with --scanID."
            )
    print(f"[Info] ScanID = {scanID}")

    basename_prefix = f"{patientID}_{timepoint}_{scanID}"
        
    print("[Step 1] Register T1n, T2f, T2w to T1c")
    t1n_reg = os.path.join(temp_dir, f"{basename_prefix}_T1n_reg.nii.gz")
    t2f_reg = os.path.join(temp_dir, f"{basename_prefix}_T2f_reg.nii.gz")
    t2w_reg = os.path.join(temp_dir, f"{basename_prefix}_T2w_reg.nii.gz")

    tfm_t1n = os.path.join(temp_dir, "T1n_to_T1c.tfm")
    tfm_t2f = os.path.join(temp_dir, "T2f_to_T1c.tfm")
    tfm_t2w = os.path.join(temp_dir, "T2w_to_T2f.tfm")

    register_images(t1c_path, t1n_path, t1n_reg, transform_path=tfm_t1n, save_dummy_ref=True, verbose=False)
    register_images(t1c_path, t2f_path, t2f_reg, transform_path=tfm_t2f, save_dummy_ref=True, verbose=False)
    register_images(t2f_reg, t2w_path, t2w_reg, transform_path=tfm_t2w, save_dummy_ref=True, verbose=False)

    for tfm, label in zip([tfm_t1n, tfm_t2f, tfm_t2w], ["T1n", "T2f", "T2w"]):
        
        tfm_dest = os.path.join(transform_dir, os.path.basename(tfm))
        shutil.move(tfm, tfm_dest)
        
        moving_ref_dummy = tfm.replace(".tfm", "_moving_ref.nii.gz")
        tfm_moving_ref_dest = os.path.join(transform_dir, os.path.basename(moving_ref_dummy))
        shutil.move(moving_ref_dummy, tfm_moving_ref_dest)

        fixed_ref_dummy = tfm.replace(".tfm", "_fixed_ref.nii.gz")
        tfm_fixed_ref_dest = os.path.join(transform_dir, os.path.basename(fixed_ref_dummy))
        shutil.move(fixed_ref_dummy, tfm_fixed_ref_dest)
        
        transform_records[label]["initial_registration"] = {
            "transform": f"./{transform_basedir}/{os.path.basename(tfm)}",
            "fixed_reference": f"./{transform_basedir}/{os.path.basename(fixed_ref_dummy)}",
            "moving_reference": f"./{transform_basedir}/{os.path.basename(moving_ref_dummy)}"
        }

    if co_register_path:
        print("[Step 2] Co-register all scans to provided reference")
        t1c_coreg = os.path.join(temp_dir, f"{basename_prefix}_T1c_coreg.nii.gz")
        coreg_tfm = os.path.join(temp_dir, "T1c_to_coreg.tfm")
        
        register_images(co_register_path, t1c_path, t1c_coreg, transform_path=coreg_tfm, save_dummy_ref=True, verbose=False)

        coreg_tfm_dest = os.path.join(transform_dir, os.path.basename(coreg_tfm))
        shutil.move(coreg_tfm, coreg_tfm_dest)

        coreg_moving_ref_dummy = coreg_tfm.replace(".tfm", "_moving_ref.nii.gz")
        coreg_tfm_moving_ref_dest = os.path.join(transform_dir, os.path.basename(coreg_moving_ref_dummy))
        shutil.move(coreg_moving_ref_dummy, coreg_tfm_moving_ref_dest)

        coreg_fixed_ref_dummy = coreg_tfm.replace(".tfm", "_fixed_ref.nii.gz")
        coreg_tfm_fixed_ref_dest = os.path.join(transform_dir, os.path.basename(coreg_fixed_ref_dummy))
        shutil.move(coreg_fixed_ref_dummy, coreg_tfm_fixed_ref_dest)

        t1n_reg_coreg = os.path.join(temp_dir, f"{basename_prefix}_T1n_coreg.nii.gz")
        t2f_reg_coreg = os.path.join(temp_dir, f"{basename_prefix}_T2f_coreg.nii.gz")
        t2w_reg_coreg = os.path.join(temp_dir, f"{basename_prefix}_T2w_coreg.nii.gz")
        
        for label, src, out in zip(["T1n", "T2f", "T2w"], [t1n_reg, t2f_reg, t2w_reg], [t1n_reg_coreg, t2f_reg_coreg, t2w_reg_coreg]):
            register_images(co_register_path, src, out, transform_path=coreg_tfm_dest, apply_only=True, save_dummy_ref=False, verbose=False)
            
        for label in ["T1c", "T1n", "T2f", "T2w"]:
            transform_records[label]["coregistration"] = {
                "transform": f"./{transform_basedir}/{os.path.basename(coreg_tfm)}",
                "fixed_reference": f"./{transform_basedir}/{os.path.basename(coreg_fixed_ref_dummy)}",
                "moving_reference": f"./{transform_basedir}/{os.path.basename(coreg_moving_ref_dummy)}"
            }

        t1c_reg_final = t1c_coreg
        t1n_reg_final = t1n_reg_coreg
        t2f_reg_final = t2f_reg_coreg
        t2w_reg_final = t2w_reg_coreg

    else:
        t1c_reg_final = t1c_path
        t1n_reg_final = t1n_reg
        t2f_reg_final = t2f_reg
        t2w_reg_final = t2w_reg

    outputs_to_save = []
    if save_scans_with_skulls:
        for label, path in zip(["T1c", "T1n", "T2f", "T2w"], [t1c_reg_final, t1n_reg_final, t2f_reg_final, t2w_reg_final]):
            out_path = os.path.join(temp_dir, f"{basename_prefix}_{label}.nii.gz")
            resize_mri(
                input_filepath=path,
                output_filepath=out_path,
                target_shape=final_dims,
                target_voxel_dims=final_voxels,
                interp=1,
                save_padding_record=False
            )
            outputs_to_save.append(out_path)

    print("[Step 3] Skull-strip and mask all scans")
    brain_mask = os.path.join(temp_dir, f"{basename_prefix}_brainmask_temp.nii.gz")
    t1c_brain = os.path.join(temp_dir, f"{basename_prefix}_T1c_brain_temp.nii.gz")
    run_hd_bet(t1c_reg_final, output_path=t1c_brain, mask_path=brain_mask, overwrite_existing=1, device="cpu")

    masked_paths = {}
    for label, path in zip(["T1n", "T2f", "T2w"], [t1n_reg_final, t2f_reg_final, t2w_reg_final]):
        out_path = os.path.join(temp_dir, f"{basename_prefix}_{label}_brain_temp.nii.gz")
        class Args: pass
        args = Args()
        args.applymask = True
        args.input = path
        args.mask = brain_mask
        args.output = out_path
        args.average = args.operation = args.inputs = None
        perform_mri_math(args)
        masked_paths[label] = out_path

    print("[Step 4] Normalize brain-extracted scans")
    norm_paths = {}
    for label in ["T1c", "T1n", "T2f", "T2w"]:
        input_path = t1c_brain if label == "T1c" else masked_paths[label]
        norm_out = os.path.join(temp_dir, f"{basename_prefix}_{label}_brain_norm_temp.nii.gz")
        normalize_masked_image(input_path, brain_mask, norm_out)
        norm_paths[label] = norm_out

    print("[Step 5] Resize and collect all output images")
    for label in ["T1c", "T1n", "T2f", "T2w"]:
        for suffix in ["brain_temp", "brain_norm_temp"]:
            path = os.path.join(temp_dir, f"{basename_prefix}_{label}_{suffix}.nii.gz")
            suffix_clean = suffix.replace("_temp", "")
            resized = os.path.join(temp_dir, f"{basename_prefix}_{label}_{suffix_clean}.nii.gz")

            if os.path.exists(path):

                if debug:
                    print(f"[Debug] Appending resized file to outputs: {os.path.basename(resized)}")
                    
                if suffix == "brain_temp":
                    pad_basename = f"{label}_padding.txt"
                    pad_path = os.path.join(transform_dir, pad_basename)
                    record_path = f"./{transform_basedir}/{pad_basename}"
                    resize_mri(
                        input_filepath=path,
                        output_filepath=resized,
                        target_shape=final_dims,
                        target_voxel_dims=final_voxels,
                        interp=1,
                        save_padding_record=True,
                        padding_record_path=pad_path
                    )
                    outputs_to_save.append(resized)
                    transform_records[label]["final_resize"] = record_path
                else:
                    resize_mri(
                        input_filepath=path,
                        output_filepath=resized,
                        target_shape=final_dims,
                        target_voxel_dims=final_voxels,
                        interp=1,
                        save_padding_record=False
                    )
                    outputs_to_save.append(resized)
            else:
                raise ValueError(
                    f"[Error] Attempted to resize {path}, but file does not exist."
                )

    brainmask_resized = os.path.join(temp_dir, f"{basename_prefix}_brainmask.nii.gz")
    resize_mri(
        input_filepath=brain_mask,
        output_filepath=brainmask_resized,
        target_shape=final_dims,
        target_voxel_dims=final_voxels,
        interp=0,
        save_padding_record=False
    )
    outputs_to_save.append(brainmask_resized)

    print("[Final Step] Moving outputs to:", output_dir)
    
    if debug:
        print("[Debug] Outputs to save:")

    for src in outputs_to_save:
        if debug:
            print(" -", os.path.basename(src))
        shutil.move(src, os.path.join(output_dir, os.path.basename(src)))

    for label, record in transform_records.items():
        with open(os.path.join(output_dir, f"{basename_prefix}_{label}_transform_record.json"), 'w') as f:
            json.dump(record, f, indent=2)

    print("[Done] All selected outputs saved to:", output_dir)


def preprocess_brain_mris(
    t1c_path,
    t1n_path,
    t2f_path,
    t2w_path,
    output_dir,
    co_register_path=None,
    save_scans_with_skulls=False,
    final_dims=(240, 240, 155),
    final_voxels=(1.0, 1.0, 1.0),
    debug=False,
    patientID=None,
    timepoint=None,
    scanID=None,
):
    os.makedirs(output_dir, exist_ok=True)

    if debug:
        temp_dir = tempfile.mkdtemp()
        print(f"[Debug] Temporary directory retained at: {temp_dir}")
        try:
            run_preprocessing_pipeline(
                t1c_path=t1c_path,
                t1n_path=t1n_path,
                t2f_path=t2f_path,
                t2w_path=t2w_path,
                output_dir=output_dir,
                co_register_path=co_register_path,
                save_scans_with_skulls=save_scans_with_skulls,
                final_dims=final_dims,
                final_voxels=final_voxels,
                temp_dir=temp_dir,
                patientID=patientID,
                timepoint=timepoint,
                scanID=scanID,
                debug=True
            )
        except Exception as e:
            print(f"[Error] {e}")
            raise
        # Do not remove temp_dir
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
             run_preprocessing_pipeline(
                t1c_path=t1c_path,
                t1n_path=t1n_path,
                t2f_path=t2f_path,
                t2w_path=t2w_path,
                output_dir=output_dir,
                co_register_path=co_register_path,
                save_scans_with_skulls=save_scans_with_skulls,
                final_dims=final_dims,
                final_voxels=final_voxels,
                temp_dir=temp_dir,
                patientID=patientID,
                timepoint=timepoint,
                scanID=scanID,
                debug=False
            )
            


def main():
    parser = argparse.ArgumentParser(description="Full MRI preprocessing pipeline for T1c, T1n, T2f, T2w")
    parser.add_argument("--t1c", required=True, help="T1c image path")
    parser.add_argument("--t1n", required=True, help="T1n image path")
    parser.add_argument("--t2f", required=True, help="T2f image path")
    parser.add_argument("--t2w", required=True, help="T2w image path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--patientID", help="Flag to indicate which patient this scan is from. If not provided, defaults to the first _ separated field of the input T1c filename.")
    parser.add_argument("--timepoint", help="Flag to indicate timepoint of this scan. If not provided, defaults to the second _ separated field of the input T1c filename.")
    parser.add_argument("--scanID", help="Unique identifier for this sequence of scans. If not provided, defaults to the third _ separated field of the input T1c filename.")
    parser.add_argument("--co_register", help="Optional reference image to co-register all scans to")
    parser.add_argument("--save_scans_with_skulls", action="store_true", help="Save scans after registration but before skull-stripping. WARNING: May contain PHI in the form of a patient's face scan unless input scans were de-faced prior to processing.")
    parser.add_argument("--final_dims", default="240,240,155", help="Final data dimensions (default: 240,240,155)")
    parser.add_argument("--final_voxels", default="1.0,1.0,1.0", help="Final voxel sizes (default: 1.0,1.0,1.0)")
    parser.add_argument("--debug", action="store_true", help="Keep intermediate files and temp directory after execution")


    args = parser.parse_args()
    dims = tuple(map(int, args.final_dims.split(",")))
    voxels = tuple(map(float, args.final_voxels.split(",")))

    preprocess_brain_mris(
        t1c_path=args.t1c,
        t1n_path=args.t1n,
        t2f_path=args.t2f,
        t2w_path=args.t2w,
        output_dir=args.output,
        co_register_path=args.co_register,
        save_scans_with_skulls=args.save_scans_with_skulls,
        final_dims=dims,
        final_voxels=voxels,
        debug=args.debug,
        patientID=args.patientID,
        timepoint=args.timepoint,
        scanID=args.scanID,
    )


if __name__ == "__main__":
    main()
