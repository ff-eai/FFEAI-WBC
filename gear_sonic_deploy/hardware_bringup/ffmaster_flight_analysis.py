#!/usr/bin/env python3
"""Analyze a SONIC flight log: what did the policy SEE, and what did it DO?

Input: a directory produced by ffmaster_sonic.sh's flight recorder:
    policy_input.csv   994 floats per 20 ms tick (the decoder's exact input)
    target_motion.csv  body pos(3) + quat wxyz(4) + 29 target joints (MuJoCo)

Answers, per tick and at the first anomaly:
  - was the GRAVITY observation sane? (unit norm, pointing down)
  - were JOINT POSITIONS sane? (within limits, continuous, matching a real robot)
  - were VELOCITIES sane?
  - what did the policy COMMAND? (recovered from the last-actions history)
  - what was it TOLD to track? (target motion)
Verdict logic:
  insane obs  -> input pipeline fault (bridge/IMU/ordering) — fix upstream
  sane obs + insane actions -> policy/checkpoint fault — back to sim
  sane obs + sane actions but tracking violent target -> reference fault

994-dim layout (observation_config.yaml, verified from binary startup print):
  [  0: 64]  token_state
  [ 64: 94]  ang-vel history   10 frames x 3   (oldest first)
  [ 94:384]  joint-pos history 10 frames x 29  (IsaacLab order)
  [384:674]  joint-vel history 10 frames x 29
  [674:964]  last-actions hist 10 frames x 29
  [964:994]  gravity-dir hist  10 frames x 3   (body frame; standing ~ (0,0,-1))

Usage: python3 ffmaster_flight_analysis.py <flight_log_dir> [--ticks N]
"""
import argparse
import math
import os
import sys

import numpy as np

IL2MJ = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
         11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
MJ_NAMES = ["L_hip_p", "L_hip_r", "L_hip_y", "L_knee", "L_ank_p", "L_ank_r",
            "R_hip_p", "R_hip_r", "R_hip_y", "R_knee", "R_ank_p", "R_ank_r",
            "waist_y", "waist_p", "waist_r",
            "L_sho_p", "L_sho_r", "L_sho_y", "L_elb", "L_wr_y", "L_wr_p", "L_wr_r",
            "R_sho_p", "R_sho_r", "R_sho_y", "R_elb", "R_wr_y", "R_wr_p", "R_wr_r"]
# vendor MJCF limits, MuJoCo order (bridge.yaml clamp table, un-shrunk ±0.02)
LIM_LO = np.array([-2.704, -0.235, -1.684, 0.0, -0.803, -0.262, -2.704, -2.906,
                   -3.43, 0.0, -0.803, -0.262, -3.43, -0.314, -0.488, -3.08,
                   -0.061, -2.556, -2.356, -2.556, -0.558, -1.571, -3.08, -2.993,
                   -2.556, -2.356, -2.556, -0.558, -0.724])
LIM_HI = np.array([2.556, 2.906, 3.43, 2.407, 0.453, 0.262, 2.556, 0.235, 1.684,
                   2.407, 0.453, 0.262, 2.382, 0.314, 0.488, 2.04, 2.993, 2.556,
                   0.0, 2.556, 0.558, 0.724, 2.04, 0.061, 2.556, 0.0, 2.556,
                   0.558, 1.571])
DEFAULT_MJ = np.array([-0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669,
                       -0.363, 0, 0, 0, 0, 0.2, 0.2, 0, -0.3, 0, 0, 0,
                       0.2, -0.2, 0, -0.3, 0, 0, 0])
ACTION_SCALE = 0.25  # q_cmd = default + action*scale (policy_parameters_ffmaster.hpp)


def load_csv(path, ncol):
    rows = []
    with open(path) as f:
        for line in f:
            parts = [p for p in line.strip().split(",") if p != ""]
            if len(parts) >= ncol:
                rows.append([float(x) for x in parts[:ncol]])
    return np.array(rows)


def latest(hist_block, width):
    """Newest frame of an oldest-first 10-frame history."""
    return hist_block[..., -width:]


def il_to_mj(v):
    return v[..., IL2MJ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--ticks", type=int, default=0, help="analyze first N ticks only")
    a = ap.parse_args()

    obs = load_csv(os.path.join(a.logdir, "policy_input.csv"), 994)
    tgt_path = os.path.join(a.logdir, "target_motion.csv")
    tgt = load_csv(tgt_path, 36) if os.path.exists(tgt_path) else None
    if len(obs) == 0:
        sys.exit("no policy input rows — control never started")
    if a.ticks:
        obs = obs[:a.ticks]
    T = len(obs)
    print("ticks: %d (%.1f s at 50 Hz)   target rows: %s"
          % (T, T / 50.0, len(tgt) if tgt is not None else "none"))

    token = obs[:, 0:64]
    angv = latest(obs[:, 64:94], 3)
    q_il = latest(obs[:, 94:384], 29)
    dq_il = latest(obs[:, 384:674], 29)
    act_il = latest(obs[:, 674:964], 29)
    grav = latest(obs[:, 964:994], 3)

    q = il_to_mj(q_il)
    dq = il_to_mj(dq_il)
    act = il_to_mj(act_il)
    cmd = DEFAULT_MJ + act * ACTION_SCALE          # what the policy asked for

    print("\n--- OBSERVATION SANITY (the world as the policy saw it) ---")
    gnorm = np.linalg.norm(grav, axis=1)
    tilt = np.degrees(np.arccos(np.clip(-grav[:, 2] / np.maximum(gnorm, 1e-9), -1, 1)))
    print("gravity: |g| range [%.3f, %.3f] (want ~1)   gz range [%+.3f, %+.3f] (standing ~ -1)"
          % (gnorm.min(), gnorm.max(), grav[:, 2].min(), grav[:, 2].max()))
    print("tilt-from-gravity: first=%.1f°  max=%.1f°  last=%.1f°"
          % (tilt[0], tilt.max(), tilt[-1]))
    if grav[:, 2].max() > -0.5:
        print("  ** ANOMALY: gravity-z not pointing down (>-0.5) — IMU frame/sign fault upstream **")

    viol = (q < LIM_LO - 0.05) | (q > LIM_HI + 0.05)
    if viol.any():
        t0, j0 = np.argwhere(viol)[0]
        print("  ** ANOMALY: observed joint OUT OF PHYSICAL LIMITS: %s=%.3f at tick %d **"
              % (MJ_NAMES[j0], q[t0, j0], t0))
    else:
        print("observed joints: all within physical limits")
    jump = np.abs(np.diff(q, axis=0)).max(axis=1)
    if len(jump) and jump.max() > 0.3:
        tj = int(np.argmax(jump > 0.3))
        print("  ** ANOMALY: observed pose JUMPED %.2f rad in one tick at tick %d (%s) — "
              "sensor/transport glitch **"
              % (jump[tj], tj + 1, MJ_NAMES[int(np.argmax(np.abs(q[tj + 1] - q[tj])))]))
    else:
        print("observed pose continuity: max per-tick change %.3f rad (smooth)"
              % (jump.max() if len(jump) else 0))
    print("angvel max |w| = %.2f rad/s   joint-vel max |dq| = %.2f rad/s"
          % (np.abs(angv).max(), np.abs(dq).max()))
    if np.isnan(obs).any():
        print("  ** ANOMALY: NaNs in observations at tick %d **"
              % int(np.argwhere(np.isnan(obs))[0][0]))
    print("token: |t| range [%.2f, %.2f], per-tick change max %.3f"
          % (np.linalg.norm(token, axis=1).min(), np.linalg.norm(token, axis=1).max(),
             np.abs(np.diff(token, axis=0)).max() if T > 1 else 0))

    print("\n--- POLICY OUTPUT (recovered from last-actions history) ---")
    astep = np.abs(np.diff(act, axis=0)).max(axis=1) if T > 1 else np.zeros(1)
    amax_t = int(np.argmax(np.abs(act).max(axis=1)))
    print("action magnitude: first-tick max |a|=%.2f   worst tick %d: max |a|=%.2f"
          % (np.abs(act[0]).max(), amax_t, np.abs(act[amax_t]).max()))
    worst_j = np.argsort(-np.abs(act[amax_t]))[:5]
    print("worst-tick commanded pose (default + a*%.2f):" % ACTION_SCALE)
    for j in worst_j:
        print("   %-8s a=%+6.2f -> cmd %+7.3f  (limit [%+.2f,%+.2f], obs was %+7.3f)"
              % (MJ_NAMES[j], act[amax_t, j], cmd[amax_t, j],
                 LIM_LO[j], LIM_HI[j], q[amax_t, j]))
    if len(astep):
        print("action step: max per-tick change %.2f at tick %d"
              % (astep.max(), int(np.argmax(astep)) + 1))

    if tgt is not None and len(tgt):
        print("\n--- REFERENCE (what it was told to track) ---")
        tj = tgt[:, 7:36]  # already MuJoCo order (binary writes via isaaclab_to_mujoco)
        d0 = np.abs(tj[0] - q[0])
        wj = np.argsort(-d0)[:3]
        print("first-tick target-vs-observed gap: " + ", ".join(
            "%s %.2f rad" % (MJ_NAMES[j], d0[j]) for j in wj))
        tstep = np.abs(np.diff(tj, axis=0)).max() if len(tj) > 1 else 0
        print("target motion per-tick change max: %.3f rad (%s)"
              % (tstep, "static hold" if tstep < 0.01 else "moving"))

    print("\n--- VERDICT HINTS ---")
    obs_bad = (grav[:, 2].max() > -0.5) or viol.any() or np.isnan(obs).any() \
        or (len(jump) and jump.max() > 0.3)
    act_bad = np.abs(act).max() > 3.0
    if obs_bad:
        print("OBS INSANE -> input pipeline fault (bridge/IMU/ordering). Fix upstream first.")
    elif act_bad:
        print("OBS SANE but ACTIONS EXTREME -> policy/checkpoint fault. Reproduce in sim.")
    else:
        print("obs and actions within bounds — compare against a known-good log; "
              "the fault may be actuation-side or in fields not captured here.")


if __name__ == "__main__":
    main()
