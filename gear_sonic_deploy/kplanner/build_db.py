#!/usr/bin/env python3
"""Build the motion-matching database for the FF Master kinematic planner.

Input: motion-library PKLs (the training format: dof in MuJoCo order at 30 fps,
root_trans_offset, root_rot x,y,z,w). Output: one .npz with every frame of the
selected clips resampled to the 50 Hz control rate, plus the per-frame feature
vector the planner searches on.

Feature vector per frame (all in that frame's heading frame, x forward):
  [0:3]   root linear velocity
  [3:9]   root xy position 0.2 / 0.4 / 0.6 s ahead, relative to the current root
  [9:15]  heading direction (cos, sin) 0.2 / 0.4 / 0.6 s ahead
  [15:21] left / right foot position relative to the root
  [21:27] left / right foot velocity

Example (sim venv, repo root):
  .venv_sim/bin/python gear_sonic_deploy/kplanner/build_db.py \
      --pkl-dir data/ffmaster_motions/lab_train/chingmu_stage_loco \
      --out data/kplanner/ffmaster_loco_db.npz
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffmaster_kinematics import (  # noqa: E402
    CONTROL_FPS, MJ_TO_ISAAC, FFMasterModel, finite_diff, resample_clip, rot2d,
    wrap_angle, yaw_from_quat_wxyz,
)

DEFAULT_INCLUDE = (r"Walking|walking|Jogging|running|Sprint|Gait|Speed_Transition|Direction_Transition|Turning_in_Place|Lateral|Arms_Folded")
DEFAULT_EXCLUDE = r"Jump|jump|Leap|Climb|Stair|Crawl|Balance|Sit|Lie|Prone|Action_Transition"
LOOKAHEAD = (10, 20, 30)  # frames at 50 Hz: 0.2, 0.4, 0.6 s
FEAT_DIM = 27
FEAT_GROUPS = {  # slice -> default weight. Pose (foot) terms must stay strong: weaker pose weights make the
    "root_vel": (slice(0, 3), 1.0),  # matcher re-select a frame at every search and the blended output degenerates
    "traj_pos": (slice(3, 9), 1.0),  # into one static pose dragged along the path (no stepping).
    "traj_dir": (slice(9, 15), 0.75),
    "foot_pos": (slice(15, 21), 0.75),
    "foot_vel": (slice(21, 27), 1.0),
}


def load_pkl(path: str):
    d = joblib.load(path)
    key = list(d)[0]
    m = d[key]
    return key, m["dof"].astype(np.float64), m["root_trans_offset"].astype(np.float64), \
        m["root_rot"].astype(np.float64), float(m["fps"])


def clip_features(model: FFMasterModel, dof_mj, root_pos, root_quat, fps=CONTROL_FPS):
    T = len(dof_mj)
    yaw = yaw_from_quat_wxyz(root_quat)
    root_vel = np.gradient(root_pos, 1.0 / fps, axis=0)
    feet = np.zeros((T, 2, 3))
    for i in range(T):
        _, lf, rf = model.fk(root_pos[i], root_quat[i], dof_mj[i])
        feet[i, 0], feet[i, 1] = lf, rf
    feet_vel = np.gradient(feet, 1.0 / fps, axis=0)
    feat = np.full((T, FEAT_DIM), np.nan, dtype=np.float32)
    kmax = max(LOOKAHEAD)
    for i in range(T - kmax):
        R = rot2d(-yaw[i])
        R3 = np.eye(3)
        R3[:2, :2] = R
        f = np.empty(FEAT_DIM)
        f[0:3] = R3 @ root_vel[i]
        for n, k in enumerate(LOOKAHEAD):
            f[3 + 2 * n: 5 + 2 * n] = R @ (root_pos[i + k, :2] - root_pos[i, :2])
            dy = wrap_angle(yaw[i + k] - yaw[i])
            f[9 + 2 * n: 11 + 2 * n] = (np.cos(dy), np.sin(dy))
        f[15:18] = R3 @ (feet[i, 0] - root_pos[i])
        f[18:21] = R3 @ (feet[i, 1] - root_pos[i])
        f[21:24] = R3 @ feet_vel[i, 0]
        f[24:27] = R3 @ feet_vel[i, 1]
        feat[i] = f
    return feat, yaw.astype(np.float32), feet.astype(np.float32), root_vel.astype(np.float32)


def mirror_clip(dof_mj, root_pos, root_quat, perm, sign):
    n = dof_mj.shape[1]
    out = np.zeros_like(dof_mj)
    for j in range(n):
        out[:, perm[j]] = sign[j] * dof_mj[:, j]
    pos = root_pos * np.array([1.0, -1.0, 1.0])
    quat = root_quat * np.array([1.0, -1.0, 1.0, -1.0])
    return out, pos, quat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl-dir", action="append", required=True, help="directory of motion-library PKLs (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--include", default=DEFAULT_INCLUDE, help="regex on clip stem to include")
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="regex on clip stem to exclude")
    ap.add_argument("--min-future", type=int, default=60, help="frames a match must have left in its clip (1.2 s)")
    ap.add_argument("--no-mirror", action="store_true", help="do not add left/right mirrored copies")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--max-step-deg", type=float, default=18.0,
                    help="drop clips with a larger per-tick joint step (solver jumps / implausible velocity)")
    args = ap.parse_args()

    files = []
    seen = set()
    for d in args.pkl_dir:
        for p in sorted(glob.glob(os.path.join(d, "**", "*.pkl"), recursive=True)):
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in seen:
                continue
            if not re.search(args.include, stem) or re.search(args.exclude, stem):
                continue
            seen.add(stem)
            files.append(p)
    if args.max_clips:
        files = files[: args.max_clips]
    if not files:
        sys.exit("no clips selected")
    print(f"[build_db] {len(files)} clips selected")

    model = FFMasterModel()
    mirror = None
    if not args.no_mirror:
        mirror = model.derive_mirror_map()
        if mirror is None:
            print("[build_db] WARNING: could not derive a consistent mirror map; mirroring disabled")
        else:
            perm, sign, err = mirror
            print(f"[build_db] mirror map derived by FK, max error {err * 1000:.3f} mm")

    cols = {k: [] for k in ("joint_pos", "root_pos", "root_quat", "yaw", "foot_pos", "root_vel", "feat",
                            "clip_id", "frame", "valid")}
    clip_names, clip_start, clip_len, clip_mirrored, clip_speed = [], [], [], [], []

    def add_clip(name, dof_mj, root_pos, root_quat, mirrored):
        T = len(dof_mj)
        feat, yaw, feet, root_vel = clip_features(model, dof_mj, root_pos, root_quat)
        valid = np.zeros(T, bool)
        valid[: max(0, T - args.min_future)] = True
        valid &= ~np.isnan(feat[:, 0])
        cid = len(clip_names)
        clip_names.append(name + ("_mirror" if mirrored else ""))
        clip_start.append(sum(clip_len))
        clip_len.append(T)
        clip_mirrored.append(mirrored)
        speed = float(np.linalg.norm(root_vel[:, :2], axis=1).mean())
        clip_speed.append(speed)
        cols["joint_pos"].append(dof_mj[:, MJ_TO_ISAAC].astype(np.float32))
        cols["root_pos"].append(root_pos.astype(np.float32))
        cols["root_quat"].append(root_quat.astype(np.float32))
        cols["yaw"].append(yaw)
        cols["foot_pos"].append(feet.reshape(T, 6))
        cols["root_vel"].append(root_vel)
        cols["feat"].append(np.nan_to_num(feat))
        cols["clip_id"].append(np.full(T, cid, np.int32))
        cols["frame"].append(np.arange(T, dtype=np.int32))
        cols["valid"].append(valid)
        return speed

    for p in files:
        key, dof_mj, root_pos, root_rot_xyzw, fps = load_pkl(p)
        dof50, pos50, quat50 = resample_clip(dof_mj, root_pos, root_rot_xyzw, fps)
        # MotionDecode clips carry a corrupt last frame; drop the last 50 Hz frame.
        dof50, pos50, quat50 = dof50[:-1], pos50[:-1], quat50[:-1]
        if len(dof50) < args.min_future + max(LOOKAHEAD) + 10:
            print(f"  skip {key}: too short ({len(dof50)} frames)")
            continue
        step_deg = float(np.degrees(np.abs(np.diff(dof50, axis=0)).max()))
        if step_deg > args.max_step_deg:
            print(f"  skip {key}: per-tick joint step {step_deg:.1f} deg > {args.max_step_deg}")
            continue
        speed = add_clip(key, dof50.astype(np.float64), pos50.astype(np.float64), quat50.astype(np.float64), False)
        if mirror is not None:
            perm, sign, _ = mirror
            mdof, mpos, mquat = mirror_clip(dof50.astype(np.float64), pos50.astype(np.float64),
                                            quat50.astype(np.float64), perm, sign)
            # sanity: full-body FK of the mirrored frame must equal the reflected original
            f0 = 60
            rp, rr = model.fk_all(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(29))
            op, orr = model.fk_all(np.zeros(3), np.array([1.0, 0, 0, 0]), dof50[f0].astype(np.float64))
            mp, mr = model.fk_all(np.zeros(3), np.array([1.0, 0, 0, 0]), mdof[f0])
            M = np.diag([1.0, -1.0, 1.0])
            dp_ref = (op - rp)[model.body_perm] * np.array([1.0, -1.0, 1.0])
            dr_ref = M @ (orr @ np.transpose(rr, (0, 2, 1)))[model.body_perm] @ M
            err = max(np.abs((mp - rp) - dp_ref).max(), np.abs((mr @ np.transpose(rr, (0, 2, 1))) - dr_ref).max())
            if err > 2e-2:
                print(f"  WARNING mirror FK error {err * 1000:.1f} mm on {key}; skipping mirrored copy")
            else:
                add_clip(key, mdof, mpos, mquat, True)
        print(f"  {key}: {len(dof50)} frames @50 Hz, mean speed {speed:.2f} m/s")

    data = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    valid = data["valid"]
    feat_mean = data["feat"][valid].mean(0)
    feat_std = data["feat"][valid].std(0) + 1e-6
    weights = np.ones(FEAT_DIM, np.float32)
    for name, (sl, w) in FEAT_GROUPS.items():
        weights[sl] = w
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        fps=np.float32(CONTROL_FPS),
        feat_mean=feat_mean.astype(np.float32),
        feat_std=feat_std.astype(np.float32),
        feat_weights=weights,
        lookahead=np.array(LOOKAHEAD, np.int32),
        clip_names=np.array(clip_names),
        clip_start=np.array(clip_start, np.int64),
        clip_len=np.array(clip_len, np.int64),
        clip_mirrored=np.array(clip_mirrored, bool),
        clip_speed=np.array(clip_speed, np.float32),
        **data,
    )
    n = len(valid)
    print(f"[build_db] wrote {args.out}: {n} frames ({n / CONTROL_FPS / 60:.1f} min), {valid.sum()} searchable, "
          f"{len(clip_names)} clips ({sum(clip_mirrored)} mirrored)")
    sp = np.linalg.norm(data["root_vel"][valid][:, :2], axis=1)
    hist, edges = np.histogram(sp, bins=[0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5])
    print("[build_db] planar speed histogram (m/s):", ", ".join(f"{edges[i]:.1f}-{edges[i + 1]:.1f}: {h}" for i, h in enumerate(hist)))


if __name__ == "__main__":
    main()
