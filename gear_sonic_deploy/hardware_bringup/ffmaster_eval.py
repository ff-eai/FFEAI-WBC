#!/usr/bin/env python3
"""FF Master session evaluator — offline, run on the laptop after a session.

Input : a session directory produced by ffmaster_record.py (bus.csv + meta.json)
Output: <session>/report.md + <session>/metrics.json  (+ PNG plots with --plots)

What it does:
 1. Segments the timeline by command signature:
      ENGAGED (ckp>0) / DAMP (ckp==0 & ckd>0) / IDLE (no fresh commands)
 2. Inside ENGAGED, splits HOLD (commanded pose static — the frame-0 stand)
    from PLAY (commanded pose moving — a motion being tracked).
 3. Identifies WHICH reference clip each PLAY window is, by normalised
    cross-correlation of the commanded trajectory against every clip in the
    reference library (clips stored in IsaacLab order; converted to MuJoCo).
    Commands = default + action*scale (policy output), not the raw reference,
    so matching uses a generous threshold and reports its score.
 4. Metrics per window + session summary:
      PD tracking   RMSE / max |q - cq| per joint (how well motors follow)
      reference     RMSE |q - clip_q| on the identified, time-aligned clip
      posture       pelvis tilt mean/max, |gyro| max
      dynamics      |dq| max, |effort| max & RMS
      smoothness    mean & max |Δcq| per 20 ms tick (command step size)
      integrity     row-rate, max data gap, damp events
 5. Verdicts per PLAY window: PASS / CHECK / FAIL from thresholds below
    (tune as experience accumulates — they are in one place, THRESHOLDS).

Usage:
    python3 ffmaster_eval.py <session_dir> \
        [--reference gear_sonic_deploy/reference/ffmaster] [--plots]

numpy required; matplotlib only with --plots.
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

# FF Master IsaacLab(BFS) -> MuJoCo joint order (policy_parameters_ffmaster.hpp:113)
ISAACLAB_TO_MUJOCO = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                      11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]

JOINT_NAMES = [
    "L_hip_p", "L_hip_r", "L_hip_y", "L_knee", "L_ank_p", "L_ank_r",
    "R_hip_p", "R_hip_r", "R_hip_y", "R_knee", "R_ank_p", "R_ank_r",
    "waist_y", "waist_p", "waist_r",
    "L_sho_p", "L_sho_r", "L_sho_y", "L_elb", "L_wr_y", "L_wr_p", "L_wr_r",
    "R_sho_p", "R_sho_r", "R_sho_y", "R_elb", "R_wr_y", "R_wr_p", "R_wr_r"]

CLIP_FPS = 50.0

# Calibrated 2026-08-18 against a gantry session the operator visually rated
# GOOD (salute x6, greeting x7, reaching_up x6): observed-good landed at
# PD 0.085-0.10, ref 0.13-0.25, tilt <=7.7 deg, gyro <=1.13, cmd step 0.24-0.34.
# PASS = comfortably inside observed-good; CHECK = beyond it; FAIL = far out.
THRESHOLDS = {
    "pd_rmse_pass": 0.12,      # rad, mean over joints — motors follow commands
    "pd_rmse_check": 0.20,
    "ref_rmse_pass": 0.30,     # rad, mean over joints — robot follows the clip
    "ref_rmse_check": 0.50,
    "tilt_max_pass": 12.0,     # deg — HOLD windows only (no reference there)
    "tilt_max_check": 20.0,    # beyond this = fall/near-fall
    # PLAY windows are graded on tilt ERROR vs the reference clip attitude.
    # CHECK bound = 11.5 deg = 0.2 rad = the training run's own
    # exceeded_anchor_ori termination threshold (config.yaml:723 of
    # sonic_ffmaster_spherefeet_validated_v2-20260728_111209): episodes whose root
    # attitude drifted more than 0.2 rad from the reference were terminated
    # as failures during training — so the deployed policy never operated
    # beyond it.
    "tilt_err_pass": 6.0,      # deg
    "tilt_err_check": 11.5,
    "gyro_max_pass": 2.0,      # rad/s
    "gyro_max_check": 4.0,
    "cmd_step_max_pass": 0.40,  # rad per 20 ms tick (policy action rate)
    "cmd_step_max_check": 0.60,
    "match_score_min": 0.55,   # below this, clip identity = unidentified
    "min_run_frac": 0.4,       # PLAY shorter than this x clip length = TRANS
}

N = 29



# Full model joint names, FF Master MuJoCo order (matches bridge.yaml joint_names)
FULL_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint"]

_MJCF_CANDIDATES = [
    # laptop: repo layout (hardware_bringup -> repo root -> gear_sonic)
    os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "gear_sonic",
        "data", "assets", "robot_description", "mjcf",
        "ffmaster_sonic_29dof.xml")),
    # robot: staged next to the deployment
    os.path.expanduser("~/sonic_deployment/gear_sonic/data/assets/"
                       "robot_description/mjcf/ffmaster_sonic_29dof.xml"),
]
DEFAULT_MJCF = next((c for c in _MJCF_CANDIDATES if os.path.isfile(c)),
                    _MJCF_CANDIDATES[0])


def load_fk_model(mjcf_path):
    """Minimal MJCF kinematic-tree parser for FK — no MuJoCo dependency.

    Valid for this model family: every kinematic joint is a hinge located at
    its body origin (pos="0 0 0") with a unit axis, and chain bodies carry only
    a `pos` offset (the single euler on the pelvis is 0 0 0). Verified against
    ffmaster_sonic_29dof.xml (the training model) on 2026-08-19.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(mjcf_path).getroot()
    world = root.find("worldbody")
    bodies = []          # dicts: name, parent (idx), pos, joint, axis

    def rec(el, parent_idx):
        for b in el.findall("body"):
            pos = np.array([float(v) for v in b.get("pos", "0 0 0").split()])
            j = b.find("joint")
            jname, axis = None, None
            if j is not None and j.get("name"):
                jname = j.get("name")
                axis = np.array([float(v)
                                 for v in j.get("axis", "0 0 1").split()])
            idx = len(bodies)
            bodies.append(dict(name=b.get("name"), parent=parent_idx,
                               pos=pos, joint=jname, axis=axis))
            rec(b, idx)

    rec(world, -1)
    return bodies


def _axis_angle_R(axis, angle):
    x, y, z = axis
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def fk_body_positions(model, qmap):
    """Root pinned at origin/identity -> world positions of all bodies (m)."""
    n = len(model)
    R = [None] * n
    P = [None] * n
    out = np.zeros((n, 3))
    for i, b in enumerate(model):
        if b["parent"] < 0:
            Rp, Pp = np.eye(3), np.zeros(3)        # root pinned (mpjpe_local)
        else:
            Rp, Pp = R[b["parent"]], P[b["parent"]]
        Pw = Pp + Rp @ b["pos"] if b["parent"] >= 0 else np.zeros(3)
        Rw = Rp
        if b["joint"] is not None and b["joint"] in qmap:
            Rw = Rp @ _axis_angle_R(b["axis"], qmap[b["joint"]])
        R[i], P[i] = Rw, Pw
        out[i] = Pw
    return out


def mpjpe_local_series(model, q_meas_50, q_ref_50):
    """Per-frame local MPJPE in mm between two (T,29) MuJoCo-order angle
    arrays, FK'd through the same model with the pelvis pinned. Skips frames
    containing NaNs. This is the hardware analog of the trainer's mpjpe_l."""
    T = min(len(q_meas_50), len(q_ref_50))
    vals = np.full(T, np.nan)
    for k in range(T):
        if np.isnan(q_meas_50[k]).any() or np.isnan(q_ref_50[k]).any():
            continue
        pm = fk_body_positions(model, dict(zip(FULL_JOINT_NAMES, q_meas_50[k])))
        pr = fk_body_positions(model, dict(zip(FULL_JOINT_NAMES, q_ref_50[k])))
        vals[k] = np.linalg.norm(pm[1:] - pr[1:], axis=1).mean() * 1000.0
    return vals


# ---------------------------------------------------------------- loading

def find_bridge_cmdlog(session_dir, t0, t1):
    """Newest bridge_cmd_*.csv whose time range overlaps [t0, t1]. Searches the
    session dir, then its parent (the recordings/ convention)."""
    import glob
    cands = (glob.glob(os.path.join(session_dir, "bridge_cmd_*.csv"))
             + glob.glob(os.path.join(os.path.dirname(
                 os.path.normpath(session_dir)), "bridge_cmd_*.csv")))
    best = None
    for p in sorted(cands, reverse=True):
        try:
            with open(p) as f:
                f.readline()
                first = f.readline().split(",")[0]
            if not first:
                continue
            ct0 = float(first)
            if ct0 <= t1:            # started before the recording ended
                best = p
                break
        except (OSError, ValueError):
            continue
    return best


def merge_commands(S, cmd_path):
    """Attach cq/ckp/ckd/cmd_age to each bus row: latest command at-or-before
    the row's timestamp (same semantics as a live subscription)."""
    with open(cmd_path) as f:
        f.readline()
    C = np.genfromtxt(cmd_path, delimiter=",", skip_header=1)
    if C.ndim == 1:
        C = C[None, :]
    ct = C[:, 0]
    idx = np.searchsorted(ct, S["t"], side="right") - 1
    valid = idx >= 0
    idx = np.clip(idx, 0, len(ct) - 1)
    nanrow = np.full(N, np.nan)
    S["cq"] = np.where(valid[:, None], C[idx, 1:1 + N], nanrow)
    S["ckp"] = np.where(valid[:, None], C[idx, 1 + N:1 + 2 * N], nanrow)
    S["ckd"] = np.where(valid[:, None], C[idx, 1 + 2 * N:1 + 3 * N], nanrow)
    S["cmd_age"] = np.where(valid, S["t"] - ct[idx], np.nan)
    print("merged commands from %s (%d rows, overlap %.1f s)"
          % (os.path.basename(cmd_path), len(C),
             max(0.0, min(S["t"][-1], ct[-1]) - max(S["t"][0], ct[0]))))


def load_session(session_dir, commands=None):
    path = os.path.join(session_dir, "bus.csv")
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[None, :]
    col = {name: i for i, name in enumerate(header)}

    def block(prefix):
        return data[:, [col["%s%02d" % (prefix, i)] for i in range(N)]]

    S = {
        "t": data[:, col["t"]],
        "q": block("q"), "dq": block("dq"), "eff": block("eff"),
        "tilt": data[:, col["tilt_deg"]],
        "gyr": data[:, [col["gyr_x"], col["gyr_y"], col["gyr_z"]]],
    }
    meta_path = os.path.join(session_dir, "meta.json")
    S["meta"] = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    if "cq00" in col:
        # legacy recording with embedded command columns
        S["cq"], S["ckp"], S["ckd"] = block("cq"), block("ckp"), block("ckd")
        S["cmd_age"] = data[:, col["cmd_age_s"]]
    else:
        cmd_path = commands or find_bridge_cmdlog(session_dir,
                                                  S["t"][0], S["t"][-1])
        if cmd_path is None:
            sys.exit("states-only recording and no bridge_cmd_*.csv found — "
                     "pass --commands <path> (the bridge prints it at start)")
        merge_commands(S, cmd_path)
    return S


def _read_csv_matrix(path):
    rows = list(csv.reader(open(path)))
    try:
        [float(x) for x in rows[0]]
        return np.array(rows, dtype=float)
    except ValueError:
        return np.array(rows[1:], dtype=float)


def load_reference_library(ref_dir):
    """Per clip: joint targets (MuJoCo order) + reference pelvis tilt (deg).
    Tilt comes from body_quat.csv (wxyz — verified empirically 2026-08-18:
    salute stays ~2 deg upright, reaching_down bends to 45 deg; the xyzw
    reading gives a nonsensical constant ~87 deg)."""
    lib = {}
    for name in sorted(os.listdir(ref_dir)):
        jp = os.path.join(ref_dir, name, "joint_pos.csv")
        if not os.path.isfile(jp):
            continue
        arr = _read_csv_matrix(jp)
        if arr.shape[1] < N:
            continue
        # clip columns are IsaacLab-ordered; mujoco[i] = clip[il2mj[i]] is
        # exactly arr[:, ISAACLAB_TO_MUJOCO] (verified against the binary's
        # target-motion writer, g1_deploy_onnx_ref.cpp:4067).
        entry = {"q": arr[:, :N][:, ISAACLAB_TO_MUJOCO], "tilt": None}
        bq = os.path.join(ref_dir, name, "body_quat.csv")
        if os.path.isfile(bq):
            quat = _read_csv_matrix(bq)[:, :4]        # wxyz
            gz = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
            entry["tilt"] = np.degrees(np.arccos(np.clip(np.abs(gz), -1, 1)))
        lib[name] = entry
    return lib


# ---------------------------------------------------------------- segmenting

def segment(S, fresh_cmd_s=0.5):
    """Return list of (kind, i0, i1) with kind in ENGAGED/DAMP/IDLE."""
    ckp0 = S["ckp"][:, 0]
    ckd0 = S["ckd"][:, 0]
    fresh = S["cmd_age"] < fresh_cmd_s
    kind = np.where(~np.isfinite(ckp0) | ~fresh, 0,           # IDLE
                    np.where(ckp0 > 1e-6, 2, np.where(ckd0 > 1e-6, 1, 0)))
    names = {0: "IDLE", 1: "DAMP", 2: "ENGAGED"}
    segs = []
    i0 = 0
    for i in range(1, len(kind) + 1):
        if i == len(kind) or kind[i] != kind[i0]:
            segs.append((names[int(kind[i0])], i0, i))
            i0 = i
    return segs


def split_hold_play(S, i0, i1, win_s=0.4, move_thresh=0.01):
    """Inside an ENGAGED segment, split by commanded-pose movement."""
    t, cq = S["t"][i0:i1], S["cq"][i0:i1]
    n = len(t)
    if n < 10:
        return [("HOLD", i0, i1)]
    dt = np.median(np.diff(t))
    w = max(3, int(win_s / max(dt, 1e-4)))
    moving = np.zeros(n, bool)
    for i in range(n):
        j = min(n - 1, i + w)
        moving[i] = np.max(np.abs(cq[j] - cq[i])) > move_thresh
    out = []
    s = 0
    for i in range(1, n + 1):
        if i == n or moving[i] != moving[s]:
            out.append(("PLAY" if moving[s] else "HOLD", i0 + s, i0 + i))
            s = i
    return [(k, a, b) for k, a, b in out if (S["t"][b - 1] - S["t"][a]) > 0.3]


# ---------------------------------------------------------------- clip ID

def resample_to_clip_rate(t, x):
    t0, t1 = t[0], t[-1]
    n = max(2, int(round((t1 - t0) * CLIP_FPS)))
    tq = np.linspace(t0, t1, n)
    out = np.empty((n, x.shape[1]))
    for j in range(x.shape[1]):
        out[:, j] = np.interp(tq, t, x[:, j])
    return out


def identify_clip(cmd50, lib):
    """Best (name, lag_frames, score) by normalised cross-correlation on the
    highest-variance joints of the window."""
    var = cmd50.var(axis=0)
    joints = list(np.argsort(var)[-5:])
    best = (None, 0, -1.0)
    a = cmd50[:, joints]
    a = a - a.mean(axis=0)
    na = np.linalg.norm(a)
    if na < 1e-9:
        return best
    for name, clip in lib.items():
        if name == "__fk_model__":
            continue
        b_full = clip["q"][:, joints]
        max_lag = max(0, len(b_full) - len(a))
        step = max(1, int(CLIP_FPS / 10))          # 100 ms lag resolution
        for lag in range(0, max_lag + 1, step):
            b = b_full[lag:lag + len(a)]
            if len(b) < len(a) * 0.8:
                break
            bb = b - b.mean(axis=0)
            denom = na * np.linalg.norm(bb)
            if denom < 1e-9:
                continue
            score = float((a[:len(bb)] * bb).sum() / denom)
            if score > best[2]:
                best = (name, lag, score)
    return best


# ---------------------------------------------------------------- metrics

def window_metrics(S, i0, i1, lib=None):
    q, cq = S["q"][i0:i1], S["cq"][i0:i1]
    m = {}
    err = q - cq
    m["pd_rmse_mean"] = float(np.sqrt(np.nanmean(err ** 2)))
    per = np.sqrt(np.nanmean(err ** 2, axis=0))
    worst = np.argsort(per)[-5:][::-1]
    m["pd_rmse_worst"] = [(JOINT_NAMES[j], round(float(per[j]), 4))
                          for j in worst]
    m["pd_err_max"] = float(np.nanmax(np.abs(err)))
    m["tilt_mean"] = float(np.nanmean(S["tilt"][i0:i1]))
    m["tilt_max"] = float(np.nanmax(S["tilt"][i0:i1]))
    m["gyro_max"] = float(np.nanmax(np.abs(S["gyr"][i0:i1])))
    m["dq_max"] = float(np.nanmax(np.abs(S["dq"][i0:i1])))
    m["eff_max"] = float(np.nanmax(np.abs(S["eff"][i0:i1])))
    m["eff_rms"] = float(np.sqrt(np.nanmean(S["eff"][i0:i1] ** 2)))
    t = S["t"][i0:i1]
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.002
    step = int(round(0.02 / max(dt, 1e-4)))       # command steps per 20 ms
    if len(cq) > step:
        d = np.abs(cq[step:] - cq[:-step])
        m["cmd_step_mean"] = float(np.nanmean(d))
        m["cmd_step_max"] = float(np.nanmax(d))
    m["duration_s"] = float(t[-1] - t[0]) if len(t) else 0.0
    m["clip"] = None

    if lib:
        cmd50 = resample_to_clip_rate(t, cq)
        name, lag, score = identify_clip(cmd50, lib)
        if name and score >= THRESHOLDS["match_score_min"]:
            m["clip"] = name
            m["clip_score"] = round(score, 3)
            m["clip_lag_frames"] = lag
            clip = lib[name]["q"]
            q50 = resample_to_clip_rate(t, q)
            b = clip[lag:lag + len(q50)]
            k = min(len(b), len(q50))
            # pelvis-attitude error vs the reference's own attitude — a crouch
            # clip EXPECTS a big lean, so absolute tilt is not graded on runs.
            fkm = lib.get("__fk_model__")
            if fkm is not None:
                mp = mpjpe_local_series(fkm, q50[:k], b[:k])
                if np.isfinite(mp).any():
                    m["mpjpe_mean"] = float(np.nanmean(mp))
                    m["mpjpe_max"] = float(np.nanmax(mp))
            rerr = q50[:k] - b[:k]
            m["ref_rmse_mean"] = float(np.sqrt((rerr ** 2).mean()))
            rper = np.sqrt((rerr ** 2).mean(axis=0))
            rworst = np.argsort(rper)[-5:][::-1]
            m["ref_rmse_worst"] = [(JOINT_NAMES[j], round(float(rper[j]), 4))
                                   for j in rworst]
        else:
            m["clip"] = "UNIDENTIFIED"
            m["clip_score"] = round(score, 3) if name else None
    return m


def verdict(m, is_play):
    T = THRESHOLDS
    checks = []

    def grade(val, p, c, label, invert=False):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return
        ok = val <= p
        warn = val <= c
        checks.append((label, val, "PASS" if ok else
                       ("CHECK" if warn else "FAIL")))

    grade(m.get("pd_rmse_mean"), T["pd_rmse_pass"], T["pd_rmse_check"],
          "PD tracking")
    if is_play and isinstance(m.get("ref_rmse_mean"), float):
        grade(m["ref_rmse_mean"], T["ref_rmse_pass"], T["ref_rmse_check"],
              "reference tracking")
    if is_play and isinstance(m.get("tilt_err_max"), float):
        grade(m["tilt_err_max"], T["tilt_err_pass"], T["tilt_err_check"],
              "tilt error vs reference")
    elif not is_play:
        grade(m.get("tilt_max"), T["tilt_max_pass"], T["tilt_max_check"],
              "max tilt")
    grade(m.get("gyro_max"), T["gyro_max_pass"], T["gyro_max_check"],
          "max gyro")
    grade(m.get("cmd_step_max"), T["cmd_step_max_pass"],
          T["cmd_step_max_check"], "max cmd step")
    order = {"PASS": 0, "CHECK": 1, "FAIL": 2}
    overall = max((c[2] for c in checks), key=lambda v: order[v],
                  default="PASS")
    return overall, checks


# ---------------------------------------------------------------- report

def write_report(session_dir, S, results, integrity, motion=None, agg=None):
    lines = []
    meta = S["meta"]
    lines.append("# FF Master session report — %s" % os.path.basename(
        os.path.normpath(session_dir)))
    lines.append("")
    if meta:
        lines.append("- note: %s" % (meta.get("note") or "—"))
        lines.append("- started: %s   duration: %ss   rate: %s Hz   "
                     "max gap: %s ms"
                     % (meta.get("started_utc"), meta.get("duration_s"),
                        meta.get("mean_rate_hz"), meta.get("max_gap_ms")))
        lines.append("- system state at record start: %s"
                     % meta.get("system_state_at_start"))
    lines.append("- data integrity: %s" % integrity)
    lines.append("")
    if motion and agg:
        lines.append("## MOTION RESULT — `%s`  (%d runs)" % (motion,
                                                             agg["n_runs"]))
        lines.append("")
        lines.append("| metric | mean ± std | worst run |")
        lines.append("|---|---|---|")
        for key, label, fmt in [
                ("mpjpe_mean", "mpjpe_l mean (mm)", "%.1f"),
                ("mpjpe_max", "mpjpe_l max (mm)", "%.1f"),
                ("ref_rmse_mean", "reference RMSE (rad)", "%.4f"),
                ("pd_rmse_mean", "PD RMSE (rad)", "%.4f"),
                ("tilt_max", "max abs tilt (deg, info)", "%.2f"),
                ("gyro_max", "max gyro (rad/s)", "%.3f"),
                ("eff_max", "max effort (Nm)", "%.2f"),
                ("cmd_step_max", "max cmd step (rad/20ms)", "%.4f")]:
            if (key + "_mean") in agg:
                lines.append("| %s | %s ± %s | %s |" % (
                    label, fmt % agg[key + "_mean"],
                    fmt % agg[key + "_std"], fmt % agg[key + "_worst"]))
        lines.append("")
        lines.append("")
    elif motion:
        lines.append("## MOTION RESULT — `%s`" % motion)
        lines.append("")
        lines.append("**No evaluable runs found** (no PLAY window matched the "
                     "clip — wrong motion played, or control never engaged).")
        lines.append("")
    lines.append("## Timeline")
    lines.append("")
    lines.append("| # | window | t0 (s) | dur (s) | clip (score) |")
    lines.append("|---|--------|--------|---------|--------------|")
    t_base = S["t"][0]
    for i, r in enumerate(results):
        clip = ""
        if r["metrics"].get("clip"):
            clip = "%s (%.2f)" % (r["metrics"]["clip"],
                                  r["metrics"].get("clip_score") or 0)
        lines.append("| %d | %s | %.1f | %.1f | %s |"
                     % (i, r["kind"], r["t0"] - t_base, r["dur"], clip))
    lines.append("")
    for i, r in enumerate(results):
        m = r["metrics"]
        lines.append("## %d. %s  (t=%.1f s, %.1f s)%s"
                     % (i, r["kind"], r["t0"] - t_base, r["dur"],
                        "" if not m.get("clip") else " — " + str(m["clip"])))
        lines.append("")
        lines.append("| metric | value | | metric | value |")
        lines.append("|---|---|---|---|---|")

        def f(k, fmt="%.4f"):
            v = m.get(k)
            return (fmt % v) if isinstance(v, float) else "—"

        lines.append("| mpjpe_l mean/max (mm) | %s / %s | | ref RMSE (rad) | %s |"
                     % (f("mpjpe_mean", "%.1f"), f("mpjpe_max", "%.1f"),
                        f("ref_rmse_mean")))
        lines.append("| PD RMSE (rad) | %s | | | |" % f("pd_rmse_mean"))
        lines.append("| PD err max | %s | | tilt mean/max (deg) | %s / %s |"
                     % (f("pd_err_max"), f("tilt_mean", "%.2f"),
                        f("tilt_max", "%.2f")))
        lines.append("| gyro max (rad/s) | %s | | dq max (rad/s) | %s |"
                     % (f("gyro_max", "%.3f"), f("dq_max", "%.3f")))
        lines.append("| effort max/RMS (Nm) | %s / %s | | cmd step mean/max "
                     "(rad/20ms) | %s / %s |"
                     % (f("eff_max", "%.2f"), f("eff_rms", "%.2f"),
                        f("cmd_step_mean"), f("cmd_step_max")))
        if m.get("pd_rmse_worst"):
            lines.append("")
            lines.append("worst PD joints: %s" % ", ".join(
                "%s=%.3f" % (n, v) for n, v in m["pd_rmse_worst"]))
        if m.get("ref_rmse_worst"):
            lines.append("worst ref joints: %s" % ", ".join(
                "%s=%.3f" % (n, v) for n, v in m["ref_rmse_worst"]))
        lines.append("")

    path = os.path.join(session_dir, "report.md")
    with open(path, "w") as fo:
        fo.write("\n".join(lines))
    return path


def aggregate_runs(results):
    """Mean/std over PLAY windows of the same clip — the per-motion metric."""
    runs = [r for r in results if r["kind"] == "PLAY"
            and isinstance(r["metrics"].get("ref_rmse_mean"), float)]
    if not runs:
        return None
    keys = ["mpjpe_mean", "mpjpe_max", "pd_rmse_mean", "ref_rmse_mean",
            "tilt_max", "gyro_max", "dq_max", "eff_max", "cmd_step_max",
            "duration_s"]
    agg = {"n_runs": len(runs)}
    for k in keys:
        vals = [r["metrics"][k] for r in runs
                if isinstance(r["metrics"].get(k), float)]
        if vals:
            agg[k + "_mean"] = float(np.mean(vals))
            agg[k + "_std"] = float(np.std(vals))
            agg[k + "_worst"] = float(np.max(vals))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--reference", default=None,
                    help="reference clip library (default: "
                         "gear_sonic_deploy/reference/ffmaster next to this script)")
    ap.add_argument("--motion", default=None,
                    help="clip folder name this recording is dedicated to "
                         "(overrides meta.json 'motion'). Every PLAY window is "
                         "evaluated against THIS clip only — recommended "
                         "one-motion-per-recording workflow. The match score "
                         "becomes a sanity check: a window that does not "
                         "resemble the clip is flagged, not misattributed.")
    ap.add_argument("--commands", default=None,
                    help="bridge command flight-log (bridge_cmd_*.csv). "
                         "Auto-discovered from the session/recordings dir "
                         "if omitted.")
    ap.add_argument("--mjcf", default=None,
                    help="FF Master MJCF for FK/mpjpe (default: training model "
                         "ffmaster_sonic_29dof.xml)")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--run", type=int, default=None,
                    help="which run for the per-DOF plots, 0..num_runs-1 "
                         "(default: last identified run)")
    args = ap.parse_args()

    ref = args.reference or os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "reference", "ffmaster"))

    S = load_session(args.session_dir, commands=args.commands)
    lib = load_reference_library(ref) if os.path.isdir(ref) else {}
    fk_model = None
    mjcf = args.mjcf or DEFAULT_MJCF
    if os.path.isfile(mjcf):
        fk_model = load_fk_model(mjcf)
        print("FK model: %s (%d bodies) -> mpjpe_l enabled" % (
            os.path.basename(mjcf), len(fk_model)))
    else:
        print("FK model not found (%s) — mpjpe_l skipped" % mjcf)

    motion = args.motion or S["meta"].get("motion")
    if motion:
        if motion not in lib:
            sys.exit("motion %r not found in reference library %s"
                     % (motion, ref))
        lib = {motion: lib[motion]}
        print("known-motion mode: evaluating all runs against %r" % motion)
    if fk_model is not None:
        lib["__fk_model__"] = fk_model
    print("loaded %d rows, %d reference clips" % (len(S["t"]), len(lib)))

    dt = np.diff(S["t"])
    integrity = ("%.0f Hz mean, max gap %.0f ms, %d gaps >100 ms"
                 % (1.0 / max(np.median(dt), 1e-9), np.max(dt) * 1e3,
                    int((dt > 0.1).sum())))

    min_run_s = 2.0
    if motion and motion in lib:
        min_run_s = max(2.0, THRESHOLDS["min_run_frac"] * len(lib[motion]) / CLIP_FPS)

    results = []
    for kind, i0, i1 in segment(S):
        if kind == "ENGAGED":
            for k2, a, b in split_hold_play(S, i0, i1):
                dur = S["t"][b - 1] - S["t"][a]
                if k2 == "PLAY" and dur < min_run_s:
                    # too short to be a run of the known clip: a reset snap /
                    # transition tail. Shown in the timeline, excluded from
                    # the aggregate, not verdict-graded.
                    m = window_metrics(S, a, b, None)
                    results.append(dict(kind="TRANS", t0=S["t"][a], dur=dur,
                                        a=a, b=b,
                                        metrics=m, verdict="—", checks=[]))
                    continue
                m = window_metrics(S, a, b, lib if k2 == "PLAY" else None)
                results.append(dict(kind=k2, t0=S["t"][a],
                                    dur=dur, a=a, b=b,
                                    metrics=m, verdict="", checks=[]))
        elif kind == "DAMP":
            if S["t"][i1 - 1] - S["t"][i0] > 0.5:
                results.append(dict(kind="DAMP", t0=S["t"][i0],
                                    dur=S["t"][i1 - 1] - S["t"][i0],
                                    a=i0, b=i1,
                                    metrics={}, verdict="—", checks=[]))

    agg = aggregate_runs(results) if motion else None
    path = write_report(args.session_dir, S, results, integrity,
                        motion=motion, agg=agg)
    payload = {"windows": [{k: v for k, v in r.items() if k != "checks"}
                           for r in results]}
    if agg:
        payload["motion"] = motion
        payload["aggregate"] = agg
    with open(os.path.join(args.session_dir, "metrics.json"), "w") as fo:
        json.dump(payload, fo, indent=2, default=str)
    print("report : %s" % path)
    if agg:
        print("motion : %s — %d runs, mpjpe_l %.1f ± %.1f mm, ref RMSE %.4f rad"
              % (motion, agg["n_runs"], agg.get("mpjpe_mean_mean", float("nan")),
                 agg.get("mpjpe_mean_std", 0),
                 agg.get("ref_rmse_mean_mean", float("nan"))))

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def shade(ax):
            for r in results:
                ax.axvspan(r["t0"] - S["t"][0], r["t0"] - S["t"][0] + r["dur"],
                           alpha=0.15,
                           color={"PLAY": "g", "HOLD": "b", "TRANS": "gray",
                                  "DAMP": "orange"}.get(r["kind"], "gray"))

        t = S["t"] - S["t"][0]

        # ---- overview: mpjpe_l + PD error ----------------------------------
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fkm = lib.get("__fk_model__")
        for r in results:
            m = r["metrics"]
            if (r["kind"] != "PLAY" or not m.get("clip")
                    or m["clip"] == "UNIDENTIFIED" or fkm is None):
                continue
            a, b = r["a"], r["b"]
            tw = S["t"][a:b]
            q50w = resample_to_clip_rate(tw, S["q"][a:b])
            clipq = lib[m["clip"]]["q"]
            lag = m.get("clip_lag_frames", 0)
            nn = min(len(clipq) - lag, len(q50w))
            mp = mpjpe_local_series(fkm, q50w[:nn], clipq[lag:lag + nn])
            tt = np.linspace(tw[0], tw[-1], len(q50w))[:nn] - S["t"][0]
            ax[0].plot(tt, mp, color="C0")
        ax[0].set_ylabel("mpjpe_l mm")
        ax[0].set_title("local MPJPE vs reference (root-relative FK, "
                        "PLAY windows)", fontsize=9)
        ax[1].plot(t, np.nanmean(np.abs(S["q"] - S["cq"]), axis=1))
        ax[1].set_ylabel("mean |q - cq| rad")
        ax[1].set_title("motor tracking: mean abs error across 29 joints, "
                        "state vs POLICY COMMAND (not reference)", fontsize=9)
        ax[1].set_xlabel("t s")
        for a_ in ax:
            shade(a_)
        fig.tight_layout()
        fig.savefig(os.path.join(args.session_dir, "overview.png"), dpi=110)
        print("plot   : %s" % os.path.join(args.session_dir, "overview.png"))

        # ---- per-DOF plots for one run --------------------------------------
        runs = [r for r in results
                if r["kind"] == "PLAY" and r["metrics"].get("clip")
                and r["metrics"]["clip"] != "UNIDENTIFIED"]
        if runs:
            n_run = (args.run if args.run is not None else len(runs) - 1)
            if not 0 <= n_run < len(runs):
                sys.exit("--run %d out of range (0..%d)" % (n_run, len(runs) - 1))
            r = runs[n_run]
            m = r["metrics"]
            a, b = r["a"], r["b"]
            tw = S["t"][a:b] - S["t"][a]
            clip = lib[m["clip"]]["q"]
            lag = m.get("clip_lag_frames", 0)
            q50 = resample_to_clip_rate(S["t"][a:b], S["q"][a:b])
            nref = min(len(clip) - lag, len(q50))
            tref = np.arange(nref) / CLIP_FPS * (tw[-1] / max(tw[-1], nref / CLIP_FPS))
            tref = np.linspace(0, tw[-1], nref)
            per_rmse = np.sqrt(np.nanmean(
                (q50[:nref] - clip[lag:lag + nref]) ** 2, axis=0))

            for which in ("angles", "errors"):
                fig, axes = plt.subplots(6, 5, figsize=(20, 16),
                                         sharex=True)
                for j in range(N):
                    ax_ = axes[j // 5][j % 5]
                    if which == "angles":
                        ax_.plot(tref, clip[lag:lag + nref, j], "k--",
                                 lw=1.0, label="reference")
                        ax_.plot(tw, S["cq"][a:b, j], "C1", lw=0.9,
                                 label="policy cmd")
                        ax_.plot(tw, S["q"][a:b, j], "C0", lw=0.9,
                                 label="state")
                    else:
                        q50j = np.interp(tref, tw, S["q"][a:b, j])
                        ax_.plot(tref, q50j - clip[lag:lag + nref, j], "C3",
                                 lw=0.9, label="q - ref")
                        ax_.plot(tw, S["q"][a:b, j] - S["cq"][a:b, j], "C2",
                                 lw=0.9, label="q - cmd")
                        ax_.axhline(0, color="k", lw=0.5)
                    ax_.set_title("%s  (rmse %.3f)" % (MJ_NAMES[j] if False
                                  else JOINT_NAMES[j], per_rmse[j]),
                                  fontsize=8)
                    ax_.tick_params(labelsize=7)
                axes[5][4].axis("off")
                handles, labels = axes[0][0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower right", fontsize=11)
                fig.suptitle("%s — run %d/%d (%s, score %.2f)  [rad vs s]"
                             % ("joint angles" if which == "angles"
                                else "tracking errors",
                                n_run, len(runs) - 1, m["clip"],
                                m.get("clip_score", 0)), fontsize=12)
                fig.tight_layout(rect=[0, 0.01, 1, 0.98])
                name = ("perdof_run%d.png" if which == "angles"
                        else "perdof_err_run%d.png") % n_run
                fig.savefig(os.path.join(args.session_dir, name), dpi=100)
                print("plot   : %s" % os.path.join(args.session_dir, name))


if __name__ == "__main__":
    main()
