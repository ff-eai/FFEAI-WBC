#!/usr/bin/env python3
"""CSV-only feasibility validation for FF Master SOMA-schema motion CSVs (chingmu/MotionDecode).

Why CSV-only: the soma-retargeter validator needs the source BVH per clip
(fidelity/semantic/grades). The retargeted MotionDecode release is CSV-only, so
this tool implements the *robot-side* gates only, mirroring the definitions in
soma_retargeter/validation/{metrics,config}.py (feasibility block) and the
parsing of gear_sonic/data_process/convert_soma_csv_to_motion_lib.py.

Per clip it reports structural checks, root sanity, joint feasibility (MJCF
limits, velocity limits, jumps), MuJoCo-FK foot checks (sole penetration /
floating / contact / skate), vendor QC cross-references and a content hash for
duplicate detection. Gates are provisional and listed per clip so the filter
policy can be applied separately (see build_sets.py).

Usage (training environment):
  python tools/chingmu/validate_ffmaster_csv.py --root <DATA>/csv/ff_master \
      --out data/chingmu/validation --workers 16
Pilot: add --sample 3 (per leaf category) or --limit 200.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MJCF = os.path.join(
    REPO, "gear_sonic/data/assets/robot_description/mjcf/ffmaster_sonic_29dof.xml"
)

# FF Master CSV joint columns in MJCF order (== convert_soma_csv_to_motion_lib.FFMASTER_CSV_JOINT_NAMES)
JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_pitch", "waist_roll",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_yaw", "left_wrist_pitch", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_yaw", "right_wrist_pitch", "right_wrist_roll",
]
HEAD = ["head_yaw", "head_pitch"]
WRIST_IDX = [i for i, n in enumerate(JOINTS) if "wrist" in n]
ROOT_COLS = ["root_translateX", "root_translateY", "root_translateZ",
             "root_rotateX", "root_rotateY", "root_rotateZ"]

# Velocity limits (rad/s) from ffmaster.urdf, copied from soma_retargeter/validation/config.py
VEL_LIMITS = {
    "left_hip_pitch": 11.936, "left_hip_roll": 11.936, "left_hip_yaw": 11.936,
    "left_knee": 11.936, "left_ankle_pitch": 13.087, "left_ankle_roll": 15.077,
    "right_hip_pitch": 11.936, "right_hip_roll": 11.936, "right_hip_yaw": 11.936,
    "right_knee": 11.936, "right_ankle_pitch": 13.088, "right_ankle_roll": 15.077,
    "waist_yaw": 11.936, "waist_pitch": 13.088, "waist_roll": 13.088,
    "left_shoulder_pitch": 13.088, "left_shoulder_roll": 13.088, "left_shoulder_yaw": 15.077,
    "left_elbow": 15.077, "left_wrist_yaw": 15.077, "left_wrist_pitch": 4.188, "left_wrist_roll": 4.188,
    "right_shoulder_pitch": 13.088, "right_shoulder_roll": 13.088, "right_shoulder_yaw": 15.077,
    "right_elbow": 15.077, "right_wrist_yaw": 15.077, "right_wrist_pitch": 4.188, "right_wrist_roll": 4.188,
}
VLIM = np.array([VEL_LIMITS[j] for j in JOINTS])

# Provisional thresholds (gates marked G; others diagnostic D); gate values follow the soma-retargeter defaults where they exist.
THR = {
    "min_frames": 120,            # G  1 s @120 fps
    "dq_deg": 25.0,               # G  single-frame joint jump (soma-retargeter gate_dq_deg)
    "root_jump_m": 0.15,          # G  single-frame root jump (gate_root_jump_m)
    "root_z_min": 0.0,            # G  pelvis below floor
    "root_z_max": 1.20,           # G  (gate_root_z_max)
    "poslimit_margin_deg": -10.0, # G  gross out-of-range (converter does not clamp) -- provisional
    "sole_penetration_m": -0.05,  # G  "deep" penetration level (sole 5 cm under the floor) ...
    "deep_penetration_frac_max": 0.05,  # G  ... sustained on > 5 % of UPRIGHT frames (root_z > 0.45 m);
                                        #    ground work (sit/lie/crawl) sinks feet in the retarget -> diagnostic only
    "sole_gross_m": -0.15,        # G  or any frame with the sole 15 cm under the floor
    "floating_sole_m": 0.10,      # G  lowest sole never within 10 cm of floor (persistently airborne)
    "sat_margin_rad": 0.035,      # D  ~2 deg from a limit = saturated
    "contact_sole_m": 0.03,       # D  sole within 3 cm of floor ...
    "contact_vz_max": 0.05,       # D  ... and |vz| < 5 cm/s  => planted
    "airborne_sole_m": 0.10,      # D  both soles above 10 cm
    "frozen_run_s": 3.0,          # D  no motion for > 3 s (mocap dropout)
    "root_speed_max": 8.0,        # D  implausible planar speed (fps assumption check)
    "penetration_frac_soft": -0.02,  # D  sole under -2 cm
}
GATES = ["bad_header", "nonfinite", "nonmonotonic_frame", "too_short", "dq_jump", "root_jump",
         "root_z_range", "poslimit_gross", "sole_penetration", "floating", "vendor_flagged"]
# End-of-clip glitch: MotionDecode CSVs carry a corrupt FINAL row (root teleport + joint step).
# It is reported separately (end_glitch) and excluded from the jump gates, because the
# pipeline trims the last frame instead of rejecting the clip.
END_GLITCH_POS_M = 0.05
END_GLITCH_DQ_DEG = 10.0

_M = None
_D = None
_QADR = None
_LIMS = None
_SOLE = None
_ANKLE = None


def _init_worker(mjcf_path):
    global _M, _D, _QADR, _LIMS, _SOLE, _ANKLE
    import mujoco
    _M = mujoco.MjModel.from_xml_path(mjcf_path)
    _D = mujoco.MjData(_M)
    names = {mujoco.mj_id2name(_M, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(_M.njnt)}
    qadr, lo, up = [], [], []
    for jn in JOINTS:
        j = names[jn + "_joint"]
        qadr.append(int(_M.jnt_qposadr[j]))
        lo.append(float(_M.jnt_range[j][0])); up.append(float(_M.jnt_range[j][1]))
    _QADR = np.array(qadr)
    _LIMS = (np.array(lo), np.array(up))
    _SOLE, _ANKLE = {}, {}
    for side in ("left", "right"):
        b = mujoco.mj_name2id(_M, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        sph = [g for g in range(_M.ngeom)
               if _M.geom_bodyid[g] == b and _M.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        _SOLE[side] = (np.array(sph), _M.geom_size[sph, 0].copy())
        _ANKLE[side] = b


def _euler_xyz_to_quat_wxyz(euler_deg):
    """Match convert_soma_csv_to_motion_lib: scipy from_euler('xyz', degrees=True) (extrinsic)."""
    from scipy.spatial.transform import Rotation as R
    q = R.from_euler("xyz", euler_deg, degrees=True).as_quat()  # xyzw
    return q[:, [3, 0, 1, 2]]


def _tilt_deg(q_wxyz):
    """Angle between body z-axis and world z (0 = upright)."""
    w, x, y, z = q_wxyz.T
    # third column of rotation matrix, z component: 1 - 2(x^2 + y^2)
    cz = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))


def _read_csv(path):
    with open(path, "r", newline="") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    return header, data


UPRIGHT_ROOT_Z = 0.45  # pelvis height above which the robot is considered upright


def _fk_feet(root_pos, quat_wxyz, q, fps_native, fk_step):
    import mujoco
    idx = np.arange(0, q.shape[0], fk_step)
    n = len(idx)
    sole = {s: np.empty(n) for s in _SOLE}
    ankle_xy = {s: np.empty((n, 2)) for s in _SOLE}
    qpos = np.zeros(_M.nq)
    for k, i in enumerate(idx):
        qpos[0:3] = root_pos[i]; qpos[3:7] = quat_wxyz[i]; qpos[_QADR] = q[i]
        _D.qpos[:] = qpos
        mujoco.mj_kinematics(_M, _D)
        for s, (gids, r) in _SOLE.items():
            sole[s][k] = float((_D.geom_xpos[gids, 2] - r).min())
            ankle_xy[s][k] = _D.xpos[_ANKLE[s], 0:2]
    fps = fps_native / fk_step
    out = {}
    per_contact, per_skate_mean, per_skate_max, mins = [], [], [], []
    for s in _SOLE:
        z = sole[s]
        vz = np.gradient(z) * fps if n > 1 else np.zeros(n)
        contact = (z < THR["contact_sole_m"]) & (np.abs(vz) < THR["contact_vz_max"])
        per_contact.append(float(contact.mean()))
        mins.append(float(z.min()))
        if contact.sum() >= 2 and n > 1:
            vh = np.linalg.norm(np.gradient(ankle_xy[s], axis=0) * fps, axis=1)
            per_skate_mean.append(float(vh[contact].mean())); per_skate_max.append(float(vh[contact].max()))
        else:
            per_skate_mean.append(0.0); per_skate_max.append(0.0)
    lowest = np.minimum(sole["left"], sole["right"])
    upright = root_pos[idx, 2] > UPRIGHT_ROOT_Z
    deep = lowest < THR["sole_penetration_m"]
    out["deep_pen_upright_frac"] = round(float((deep & upright).sum() / max(int(upright.sum()), 1)), 4)
    out["upright_frac"] = round(float(upright.mean()), 4)
    out["sole_min_m"] = round(min(mins), 4)
    out["sole_median_lowest_m"] = round(float(np.median(lowest)), 4)
    out["penetration_frac"] = round(float((lowest < THR["penetration_frac_soft"]).mean()), 4)
    out["deep_penetration_frac"] = round(float((lowest < THR["sole_penetration_m"]).mean()), 4)
    out["airborne_frac"] = round(float((lowest > THR["airborne_sole_m"]).mean()), 4)
    out["persistently_airborne"] = bool(lowest.min() > THR["floating_sole_m"])
    out["contact_frac_min"] = round(min(per_contact), 4)
    out["contact_frac_max"] = round(max(per_contact), 4)
    out["footskate_mean"] = round(float(np.mean(per_skate_mean)), 4)
    out["footskate_max"] = round(float(max(per_skate_max)), 4)
    return out


def validate_one(job):
    path, rel, fps_native, fk_step, flagged_keys, vendor = job
    rec = {"stem": os.path.splitext(os.path.basename(path))[0], "rel": rel,
           "category": os.path.dirname(rel), "leaf": os.path.basename(os.path.dirname(rel)),
           "status": "ok", "error": ""}
    fails, diag = [], []
    try:
        with open(path, "rb") as f:
            rec["md5"] = hashlib.md5(f.read()).hexdigest()
        header, data = _read_csv(path)
        col = {h: i for i, h in enumerate(header)}
        need = ["Frame"] + ROOT_COLS + [j + "_joint_dof" for j in JOINTS]
        missing = [c for c in need if c not in col]
        rec["n_cols"] = len(header)
        rec["has_head_cols"] = all(h + "_joint_dof" in col for h in HEAD)
        if missing:
            fails.append("bad_header"); rec["error"] = f"missing {missing[:3]}"
            rec["gate_failures"] = ";".join(fails); rec["verdict"] = "FAIL"; return rec
        T = data.shape[0]
        rec["n_frames"] = int(T); rec["dur_s"] = round(T / fps_native, 2)
        if not np.isfinite(data).all():
            fails.append("nonfinite")
        frame = data[:, col["Frame"]]
        if T > 1 and not np.all(np.diff(frame) > 0):
            fails.append("nonmonotonic_frame")
        if T < THR["min_frames"]:
            fails.append("too_short")
        root_pos = data[:, [col[c] for c in ROOT_COLS[:3]]] / 100.0
        euler = data[:, [col[c] for c in ROOT_COLS[3:]]]
        q = np.deg2rad(data[:, [col[j + "_joint_dof"] for j in JOINTS]])
        wr = q[:, WRIST_IDX]
        rec["wrists_zero"] = bool(np.abs(wr).max() < 1e-6) if T else True
        rec["wrist_abs_max_deg"] = round(float(np.degrees(np.abs(wr).max())), 2) if T else 0.0
        if rec["has_head_cols"]:
            hd = data[:, [col[h + "_joint_dof"] for h in HEAD]]
            rec["head_zero"] = bool(np.abs(hd).max() < 1e-6)
        else:
            rec["head_zero"] = None
        # --- root
        rec["root_z_min"] = round(float(root_pos[:, 2].min()), 4)
        rec["root_z_max"] = round(float(root_pos[:, 2].max()), 4)
        rec["root_z_mean"] = round(float(root_pos[:, 2].mean()), 4)
        droot = np.linalg.norm(np.diff(root_pos, axis=0), axis=1) if T > 1 else np.zeros(1)
        rec["root_jump_max_m"] = round(float(droot.max()), 4)
        rec["last_root_jump_m"] = round(float(droot[-1]), 4)
        rec["first_root_jump_m"] = round(float(droot[0]), 4)
        droot_inner = droot[:-1] if T > 2 else droot
        rec["root_jump_max_inner_m"] = round(float(droot_inner.max()), 4)
        vroot = np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1) * fps_native if T > 1 else np.zeros(1)
        rec["root_speed_max"] = round(float((vroot[:-1] if T > 2 else vroot).max()), 3)
        rec["path_m"] = round(float(droot.sum()), 3)
        quat = _euler_xyz_to_quat_wxyz(euler)
        tilt = _tilt_deg(quat)
        rec["tilt_max_deg"] = round(float(tilt.max()), 1)
        rec["tilt_p95_deg"] = round(float(np.percentile(tilt, 95)), 1)
        if rec["root_z_min"] < THR["root_z_min"] or rec["root_z_max"] > THR["root_z_max"]:
            fails.append("root_z_range")
        if rec["root_jump_max_inner_m"] > THR["root_jump_m"]:
            fails.append("root_jump")
        if rec["root_speed_max"] > THR["root_speed_max"]:
            diag.append("speed_implausible")
        # --- joints
        lo, up = _LIMS
        margin = np.minimum(q - lo[None], up[None] - q)
        rec["poslimit_violation_frac"] = round(float((margin < 0).mean()), 5)
        rec["poslimit_min_margin_deg"] = round(float(np.degrees(margin.min())), 2)
        worst = int(np.unravel_index(np.argmin(margin), margin.shape)[1])
        rec["poslimit_worst_joint"] = JOINTS[worst]
        rec["saturation_frac"] = round(float((margin < THR["sat_margin_rad"]).mean()), 4)
        if rec["poslimit_min_margin_deg"] < THR["poslimit_margin_deg"]:
            fails.append("poslimit_gross")
        if T > 1:
            qd = np.diff(q, axis=0) * fps_native
            if T > 2:
                qd = qd[:-1]  # exclude the (possibly corrupt) final transition
            absqd = np.abs(qd)
            rec["vel_violation_frac"] = round(float((absqd > VLIM[None]).mean()), 5)
            rec["vel_max_ratio"] = round(float((absqd / VLIM[None]).max()), 3)
            qdd = np.diff(qd, axis=0) * fps_native if T > 2 else np.zeros((1, q.shape[1]))
            rec["accel_p95"] = round(float(np.percentile(np.abs(qdd), 95)), 1)
            dq = np.abs(np.diff(q, axis=0)).max(axis=1)
            rec["last_dq_deg"] = round(float(np.degrees(dq[-1])), 2)
            dq_inner = dq[:-1] if T > 2 else dq
            rec["dq_max_deg"] = round(float(np.degrees(dq_inner.max())), 2)
            rec["end_glitch"] = bool(rec["last_root_jump_m"] > END_GLITCH_POS_M
                                     or rec["last_dq_deg"] > END_GLITCH_DQ_DEG)
            if rec["end_glitch"]:
                diag.append("end_glitch")
            med = np.median(dq_inner); mad = np.median(np.abs(dq_inner - med)) + 1e-9
            rec["discontinuity_spikes"] = int((dq_inner > med + 8.0 * mad).sum())
            if rec["dq_max_deg"] > THR["dq_deg"]:
                fails.append("dq_jump")
            still = (dq < 1e-3) & (droot < 1e-4)
            # longest run of still frames
            best = cur = 0
            for s_ in still:
                cur = cur + 1 if s_ else 0
                best = max(best, cur)
            rec["frozen_run_s"] = round(best / fps_native, 2)
            if rec["frozen_run_s"] > THR["frozen_run_s"]:
                diag.append("frozen")
        # --- FK feet
        fk = _fk_feet(root_pos[:-1], quat[:-1], q[:-1], fps_native, fk_step) if T > 2 else _fk_feet(root_pos, quat, q, fps_native, fk_step)
        rec.update(fk)
        if fk["deep_pen_upright_frac"] > THR["deep_penetration_frac_max"] or fk["sole_min_m"] < THR["sole_gross_m"]:
            fails.append("sole_penetration")
        if fk["deep_penetration_frac"] > THR["deep_penetration_frac_max"]:
            diag.append("deep_penetration")
        if fk["persistently_airborne"]:
            fails.append("floating")
        # --- vendor
        parts = rel.split("/")
        keys = {"/".join(parts[-2:]), "/".join(parts[-3:])}
        rec["vendor_flagged"] = bool(keys & flagged_keys)
        if rec["vendor_flagged"]:
            fails.append("vendor_flagged")
        vk = keys & set(vendor.keys())
        if vk:
            v = vendor[next(iter(vk))]
            rec["vendor_max_solver_jump_deg"] = v.get("ffmaster_max_solver_jump_deg")
            rec["vendor_max_motion_jump_deg"] = v.get("ffmaster_max_motion_jump_deg")
            rec["vendor_sat_hw_cnt"] = v.get("ffmaster_sat_hw_cnt")
    except Exception as e:  # noqa: BLE001
        rec["status"] = "fail"; rec["error"] = f"{type(e).__name__}: {e}"; fails.append("exception")
    rec["gate_failures"] = ";".join(fails)
    rec["diagnostics"] = ";".join(diag)
    rec["verdict"] = "PASS" if not fails else "FAIL"
    return rec


def load_vendor(root):
    flagged = set()
    for fp in glob.glob(os.path.join(root, "*_joint_report_flagged.txt")):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    flagged.add(line)
                    flagged.add(os.path.splitext(line)[0])
    vendor = {}
    for fp in glob.glob(os.path.join(root, "*_joint_report.csv")):
        with open(fp, newline="") as f:
            for row in csv.DictReader(f):
                m = row.get("motion")
                if m:
                    vendor[m] = row
    # normalise flagged keys to "<...>/<stem>" without extension, and with .csv
    keys = set()
    for k in flagged:
        keys.add(k); keys.add(k + ".csv" if not k.endswith(".csv") else k[:-4])
    return keys, vendor


def discover(root, exclude_dirs):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        top = rel_dir.split(os.sep)[0]
        if top in exclude_dirs:
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.endswith(".csv") and "joint_report" not in fn:
                p = os.path.join(dirpath, fn)
                files.append((p, os.path.relpath(p, root)))
    files.sort(key=lambda t: t[1])
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mjcf", default=DEFAULT_MJCF)
    ap.add_argument("--fps", type=float, default=120.0, help="native CSV frame rate")
    ap.add_argument("--fk-step", type=int, default=4, help="FK every k-th frame (4 => 30 fps)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--exclude-dirs", nargs="*", default=["Standing_High_Jump"],
                    help="top-level dirs to skip (default: the byte-identical duplicate folder)")
    ap.add_argument("--sample", type=int, default=0, help="pilot: N clips per leaf category")
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N clips")
    ap.add_argument("--files", nargs="*", help="explicit CSV paths (root still used for vendor lists)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    flagged_keys, vendor = load_vendor(args.root)
    if args.files:
        files = [(os.path.abspath(p), os.path.relpath(os.path.abspath(p), args.root)) for p in args.files]
    else:
        files = discover(args.root, set(args.exclude_dirs))
    if args.sample:
        by_leaf = defaultdict(list)
        for p, rel in files:
            by_leaf[os.path.dirname(rel)].append((p, rel))
        rng = np.random.default_rng(0)
        files = []
        for leaf in sorted(by_leaf):
            lst = by_leaf[leaf]
            pick = rng.choice(len(lst), size=min(args.sample, len(lst)), replace=False)
            files += [lst[i] for i in sorted(pick)]
    if args.limit:
        files = files[: args.limit]
    print(f"[validate] {len(files)} clips, {len(flagged_keys)//2} vendor-flagged entries, "
          f"{len(vendor)} vendor report rows, workers={args.workers}", flush=True)

    jobs = [(p, rel, args.fps, args.fk_step, flagged_keys, vendor) for p, rel in files]
    t0 = time.time()
    recs = []
    with Pool(args.workers, initializer=_init_worker, initargs=(args.mjcf,)) as pool:
        for i, rec in enumerate(pool.imap_unordered(validate_one, jobs, chunksize=4), 1):
            recs.append(rec)
            if i % 2000 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    recs.sort(key=lambda r: r["rel"])

    # duplicate content detection (across the whole set)
    by_md5 = defaultdict(list)
    by_stem = defaultdict(list)
    for r in recs:
        by_md5[r.get("md5")].append(r["rel"]); by_stem[r["stem"]].append(r["rel"])
    for r in recs:
        r["dup_content_n"] = len(by_md5[r.get("md5")])
        r["dup_stem_n"] = len(by_stem[r["stem"]])

    cols = ["stem", "leaf", "category", "rel", "verdict", "gate_failures", "diagnostics", "status", "error",
            "n_frames", "dur_s", "n_cols", "has_head_cols", "wrists_zero", "wrist_abs_max_deg", "head_zero",
            "root_z_min", "root_z_max", "root_z_mean", "root_jump_max_m", "root_jump_max_inner_m", "last_root_jump_m",
            "first_root_jump_m", "last_dq_deg", "end_glitch", "root_speed_max", "path_m",
            "tilt_max_deg", "tilt_p95_deg",
            "poslimit_violation_frac", "poslimit_min_margin_deg", "poslimit_worst_joint", "saturation_frac",
            "vel_violation_frac", "vel_max_ratio", "accel_p95", "dq_max_deg", "discontinuity_spikes", "frozen_run_s",
            "sole_min_m", "sole_median_lowest_m", "penetration_frac", "deep_penetration_frac", "deep_pen_upright_frac", "upright_frac",
            "airborne_frac", "persistently_airborne",
            "contact_frac_min", "contact_frac_max", "footskate_mean", "footskate_max",
            "vendor_flagged", "vendor_max_solver_jump_deg", "vendor_max_motion_jump_deg", "vendor_sat_hw_cnt",
            "md5", "dup_content_n", "dup_stem_n"]
    with open(os.path.join(args.out, "per_clip.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(recs)

    # summaries
    n = len(recs)
    gate_counts = {g: sum(1 for r in recs if g in r["gate_failures"].split(";")) for g in GATES + ["exception"]}
    diag_counts = defaultdict(int)
    for r in recs:
        for d in r["diagnostics"].split(";"):
            if d:
                diag_counts[d] += 1
    per_cat = defaultdict(lambda: {"n": 0, "pass": 0, "dur_s": 0.0, "end_glitch": 0, "sole_med": [], "deep": [], "rootz": []})
    for r in recs:
        c = per_cat[r["category"]]
        c["n"] += 1; c["pass"] += r["verdict"] == "PASS"; c["dur_s"] += r.get("dur_s", 0) or 0
        c["end_glitch"] += bool(r.get("end_glitch"))
        if r.get("sole_median_lowest_m") is not None:
            c["sole_med"].append(r["sole_median_lowest_m"]); c["deep"].append(r.get("deep_penetration_frac", 0)); c["rootz"].append(r.get("root_z_min", 0))
    gate_names = GATES
    with open(os.path.join(args.out, "per_category.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["category", "n", "pass", "fail", "pass_rate", "hours", "end_glitch", "med_sole_lowest_m",
                    "med_deep_pen_frac", "med_root_z_min"] + gate_names)
        for c in sorted(per_cat):
            v = per_cat[c]
            gc = [sum(1 for r in recs if r["category"] == c and g in r["gate_failures"].split(";")) for g in gate_names]
            w.writerow([c, v["n"], v["pass"], v["n"] - v["pass"], round(v["pass"] / max(v["n"], 1), 4), round(v["dur_s"] / 3600, 2),
                        v["end_glitch"], round(float(np.median(v["sole_med"])), 4) if v["sole_med"] else "",
                        round(float(np.median(v["deep"])), 4) if v["deep"] else "",
                        round(float(np.median(v["rootz"])), 3) if v["rootz"] else ""] + gc)
    summ = {
        "clips": n, "pass": sum(r["verdict"] == "PASS" for r in recs),
        "hours_total": round(sum((r.get("dur_s") or 0) for r in recs) / 3600, 2),
        "gate_counts": gate_counts, "diagnostic_counts": dict(diag_counts),
        "end_glitch_clips": sum(1 for r in recs if r.get("end_glitch")),
        "end_glitch_row_survives_div4": sum(1 for r in recs if r.get("end_glitch") and ((r.get("n_frames", 1) - 1) % 4 == 0)),
        "dup_content_groups": sum(1 for v in by_md5.values() if len(v) > 1),
        "dup_stem_groups": sum(1 for v in by_stem.values() if len(v) > 1),
        "thresholds": THR, "root": args.root, "fps": args.fps, "fk_fps": args.fps / args.fk_step,
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
