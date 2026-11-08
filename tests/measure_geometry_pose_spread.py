"""Repeat the synthetic geometry pipeline and record every scan's max pose errors (not a unit test).

Uses tests/test_preprocess_geometry.py's phantom builder and error definitions:
  pose   = max error of the recorded transform chain vs the known pose over interior brain points (mm)
  native = max error of native_space header placement vs the known pose (mm)

  python tests/measure_geometry_pose_spread.py --runs 30 [--seed none|N]   (backend from ASTRIL_TEST_REGISTRATION_BACKEND)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_preprocess_geometry as T  # noqa: E402


def one_run(seed):
    if seed != "default":
        import astril.preprocess_single_brain_mri as psbm
        orig = psbm.preprocess_single_brain_mri

        def patched(*a, **k):
            k["registration_seed"] = None if seed == "none" else int(seed)
            return orig(*a, **k)
        T.preprocess_single_brain_mri = patched
    C = T.PreprocessGeometryTests
    C.setUpClass()
    try:
        out = {}
        for lbl in T.SCANS:
            _rec, chain = T._record_chain(C.out_dir, lbl)
            got = T._apply_chain(chain, C.brain_points)
            inv = C.poses[lbl].GetInverse()
            want = np.asarray([inv.TransformPoint(tuple(p)) for p in C.brain_points])
            pose = float(np.linalg.norm(got - want, axis=1).max())
            native = sitk.ReadImage(str(Path(C.out_dir) / "native_space" / f"{T.PATIENT}_{T.TIMEPOINT}_{lbl}_native_brain.nii.gz"))
            grid = C.grids[lbl]
            size = np.asarray(grid.GetSize())
            corners = np.array(np.meshgrid(*[[0.25, 0.75]] * 3, indexing="ij")).reshape(3, -1).T
            idxs = [[int(v) for v in np.rint(c * (size - 1))] for c in np.vstack([corners, [0.5, 0.5, 0.5]])]
            g = np.asarray([native.TransformIndexToPhysicalPoint(i) for i in idxs])
            w = np.asarray([C.poses[lbl].TransformPoint(grid.TransformIndexToPhysicalPoint(i)) for i in idxs])
            out[lbl] = dict(pose=pose, native=float(np.linalg.norm(g - w, axis=1).max()))
        return out
    finally:
        C.tearDownClass()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed", default="default", help="'default' (pipeline default), 'none' (wall clock) or an int")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for i in range(a.runs):
        t0 = time.perf_counter()
        r = one_run(a.seed)
        rows.append(r)
        worst = max(r.items(), key=lambda kv: kv[1]["pose"])
        print(f"run {i + 1:2d} ({time.perf_counter() - t0:4.1f}s): " +
              " ".join(f"{k} {v['pose']:.2f}/{v['native']:.2f}" for k, v in r.items()) +
              f" | worst {worst[0]} {worst[1]['pose']:.2f} mm", flush=True)
    print("\nper-scan max pose error (mm) over runs: min / median / max | runs >= 1 mm")
    for lbl in T.SCANS:
        v = np.array([r[lbl]["pose"] for r in rows])
        n = np.array([r[lbl]["native"] for r in rows])
        print(f"  {lbl:10s} pose {v.min():.2f} / {np.median(v):.2f} / {v.max():.2f} | {int((v >= 1).sum())}   "
              f"native {n.min():.2f} / {np.median(n):.2f} / {n.max():.2f} | {int((n >= 1).sum())}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
