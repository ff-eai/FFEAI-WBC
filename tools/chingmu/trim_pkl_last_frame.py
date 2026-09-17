#!/usr/bin/env python3
"""Drop the final frame of every motion_lib PKL under --pkl-dir (in place, idempotent via log).

Why: MotionDecode CSVs carry a corrupt final row (root teleport / joint step on the
last 120 fps frame; ~40 % of clips, see validation per_clip.tsv end_glitch). After
the 120->30 stride-4 downsample that row survives whenever (T-1) % 4 == 0, i.e. in
~10 % of clips. Dropping the last 30 fps frame of *every* clip (33 ms) removes it
uniformly without depending on thresholds. A trim_log.tsv records before/after
lengths; PKLs already listed there are skipped on re-runs.
"""
import argparse
import csv
import glob
import os
import joblib
from multiprocessing import Pool

ARRAY_KEYS = ["root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints"]


def trim_one(path):
    d = joblib.load(path)
    out = {}
    before = after = None
    for k, e in d.items():
        T = e["dof"].shape[0]
        before = T
        if T < 3:
            return (path, T, T, "skipped_too_short")
        e2 = dict(e)
        for ak in ARRAY_KEYS:
            if ak in e2 and hasattr(e2[ak], "shape") and e2[ak].shape[0] == T:
                e2[ak] = e2[ak][:-1]
        after = T - 1
        out[k] = e2
    tmp = path + ".tmp"
    joblib.dump(out, tmp)
    os.replace(tmp, path)
    return (path, before, after, "trimmed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-dir", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    log_path = os.path.join(args.pkl_dir, "trim_log.tsv")
    done = set()
    if os.path.exists(log_path):
        with open(log_path) as f:
            for row in csv.reader(f, delimiter="\t"):
                if row and row[0] != "path":
                    done.add(row[0])
    files = sorted(p for p in glob.glob(os.path.join(args.pkl_dir, "**", "*.pkl"), recursive=True)
                   if os.path.basename(p) != "metadata.pkl" and p not in done)
    print(f"{len(files)} pkls to trim ({len(done)} already logged)")
    new = 0
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if not done:
            w.writerow(["path", "frames_before", "frames_after", "action"])
        with Pool(args.workers) as pool:
            for res in pool.imap_unordered(trim_one, files, chunksize=16):
                w.writerow(res); new += 1
                if new % 5000 == 0:
                    print(f"  {new}/{len(files)}", flush=True)
    print(f"done: {new} processed; log {log_path}")


if __name__ == "__main__":
    main()
