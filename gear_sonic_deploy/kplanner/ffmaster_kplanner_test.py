#!/usr/bin/env python3
"""Closed-loop sim2sim test of the FF Master kinematic planner plug-in.

Starts the MuJoCo simulator and the unmodified ffmaster_deploy_onnx_ref with
--input-type zmq, stands the robot up, enables streaming (ENTER), then drives
the planner with a scripted command sequence while recording the simulator.

Reports per command segment: commanded vs achieved forward / lateral speed and
yaw rate, minimum pelvis height, falls, and the joint-tracking RMS between the
streamed reference and the simulated robot. Results go to
data/kplanner/test_runs/<UTC>/ (results.json, run.npz, deploy.log).

Run from the repo root with the sim venv (deploy binary must be built):
  .venv_sim/bin/python gear_sonic_deploy/kplanner/ffmaster_kplanner_test.py
  .venv_sim/bin/python gear_sonic_deploy/kplanner/ffmaster_kplanner_test.py --script my_script.json --onscreen
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ffmaster_kinematics import ISAAC_TO_MJ, REPO, yaw_from_quat_wxyz  # noqa: E402
import ffmaster_kplanner_stream as S  # noqa: E402

DEPLOY_DIR = os.path.join(REPO, "gear_sonic_deploy")
os.chdir(REPO)

DEPLOY_CMD = (
    "source scripts/setup_env.sh >/dev/null 2>&1; "
    "export LD_LIBRARY_PATH=$PWD/thirdparty/unitree_sdk2/thirdparty/lib/x86_64:$LD_LIBRARY_PATH; "
    "exec ./target/release/ffmaster_deploy_onnx_ref lo "
    "{decoder} {motion_dir} --obs-config {obs_config} --encoder-file {encoder} "
    "--input-type zmq --zmq-host localhost --output-type all --disable-crc-check"
)

DEFAULT_SCRIPT = [
    [3.0, 0.0, 0.0, 0.0],    # stand
    [6.0, 0.6, 0.0, 0.0],    # slow walk
    [6.0, 1.0, 0.0, 0.0],    # walk
    [6.0, 1.6, 0.0, 0.0],    # jog
    [5.0, 0.8, 0.0, 0.5],    # walk, turning left
    [5.0, 0.8, 0.0, -0.5],   # walk, turning right
    [4.0, 0.0, 0.0, 0.0],    # stop
]


class Session:
    def __init__(self, args, onscreen=False):
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory

        cfg = SimLoopConfig(wbc_version="ffmaster_sonic_model12", interface="sim", verbose=False)
        wbc = cfg.load_wbc_yaml()
        wbc["ENV_NAME"] = "default"
        wbc["PRINT_SCENE_INFORMATION"] = False
        self.sim = SimulatorFactory.create_simulator(config=wbc, env_name="default", onscreen=onscreen, offscreen=False)
        self.sim.start_as_thread()
        time.sleep(2.0)
        self.mj = self.sim.sim_env.mj_data
        # the simulator's fall auto-reset interrupts LowState and makes the binary exit; we judge falls ourselves
        self.sim.sim_env.fall_check_enabled = False
        cmd = DEPLOY_CMD.format(decoder=args.decoder, encoder=args.encoder, motion_dir=args.motion_dir,
                                obs_config=args.obs_config)
        self.log_path = os.path.join(args.out, "deploy.log")
        self.logf = open(self.log_path, "w")
        self.proc = subprocess.Popen(["bash", "-c", cmd], cwd=DEPLOY_DIR, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.ready = threading.Event()
        self.streaming = threading.Event()
        self.catchups = 0
        self.waiting_lines = 0
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            self.logf.write(line)
            self.logf.flush()
            if "Init Done" in line:
                self.ready.set()
            if "ZMQ STREAMING MODE: ENABLED" in line:
                self.streaming.set()
            if re.search(r"catch.?up", line, re.I) and "reset" in line.lower():
                self.catchups += 1
            if "completed and waiting following motion" in line:
                self.waiting_lines += 1

    def send(self, key):
        self.proc.stdin.write(key)
        self.proc.stdin.flush()

    def wait_ready(self, timeout=900):
        t0 = time.time()
        while not self.ready.is_set():
            if self.proc.poll() is not None:
                raise RuntimeError(f"deploy exited during init; see {self.log_path}")
            if time.time() - t0 > timeout:
                raise RuntimeError(f"deploy init timeout; see {self.log_path}")
            time.sleep(1)
        time.sleep(2)

    def start_and_stand(self):
        self.send("]")
        time.sleep(8)
        for tgt in np.linspace(1.0, 0.68, 16):
            self.sim.sim_env.elastic_band.point = np.array([0.0, 0.0, float(tgt)])
            time.sleep(0.15)
        self.sim.sim_env.elastic_band.enable = False
        time.sleep(3)

    def enable_streaming(self, timeout=10):
        self.send("\n")
        t0 = time.time()
        while not self.streaming.is_set() and time.time() - t0 < timeout:
            time.sleep(0.1)
        return self.streaming.is_set()

    def state(self):
        q = self.mj.qpos
        bji = self.sim.sim_env.body_joint_index
        joints = np.array([q[j + 6] for j in bji])
        return np.array(q[0:3]), np.array(q[3:7]), joints

    def close(self):
        try:
            self.send("O")
            time.sleep(1)
            self.proc.terminate()
        except Exception:
            pass
        self.sim._running = False


def run(args):
    os.makedirs(args.out, exist_ok=True)
    script = json.load(open(args.script)) if args.script else DEFAULT_SCRIPT
    s = Session(args, onscreen=args.onscreen)
    log = {}
    results = {"script": script, "segments": [], "fell": False}
    try:
        s.wait_ready()
        s.start_and_stand()
        z0 = float(s.mj.qpos[2])
        _, _, j_stand = s.state()
        print(f"[test] standing at z={z0:.3f}; enabling streaming")
        if not s.enable_streaming():
            raise RuntimeError("binary did not report ZMQ STREAMING MODE: ENABLED")
        time.sleep(0.5)
        st = S.build(args, log)
        rec = {"t": [], "pos": [], "quat": [], "joints": [], "cmd": []}
        t0 = time.monotonic()
        for _ in range(25):  # 0.5 s of pre-stream baseline
            p, q, j = s.state()
            rec["t"].append(time.monotonic() - t0)
            rec["pos"].append(p)
            rec["quat"].append(q)
            rec["joints"].append(j)
            rec["cmd"].append((0.0, 0.0, 0.0))
            time.sleep(0.02)
        st.start()
        t0 = st.t0
        for k in range(len(rec["t"])):
            rec["t"][k] -= (t0 - (t0 - 0.5))  # baseline samples get negative times
        first = np.array(log["joint_pos"][0])[ISAAC_TO_MJ]
        results["first_frame_vs_stand_deg"] = float(np.degrees(np.abs(first - j_stand).max()))
        print(f"[test] streaming; script {sum(x[0] for x in script):.0f} s; first reference frame vs standing pose: "
              f"max {results['first_frame_vs_stand_deg']:.1f} deg")
        sc = S.Script(script)
        period_s = args.period / S.FPS
        next_tick = t0
        next_sample = t0
        total = sc.total
        deploy_exit = None
        while True:
            now = time.monotonic()
            if s.proc.poll() is not None:
                deploy_exit = now - t0
                print(f"[test] deploy exited at t={deploy_exit:.2f} s; see {s.log_path}")
                break
            (vx, vy, wz), done = sc.command(now)
            st.set_command(vx, vy, wz)
            if now >= next_tick:
                st.tick()
                next_tick += period_s
                if next_tick < now:
                    next_tick = now + period_s
            if now >= next_sample:
                p, q, j = s.state()
                rec["t"].append(now - t0)
                rec["pos"].append(p)
                rec["quat"].append(q)
                rec["joints"].append(j)
                rec["cmd"].append((vx, vy, wz))
                next_sample += 1.0 / S.FPS
            if now - t0 > total + 0.5:
                break
            time.sleep(0.002)
        print(f"[test] done; {st.pub.messages} messages, stalls {st.stalls}, catch-up resets seen {s.catchups}, "
              f"head-hold lines {s.waiting_lines}")
        results["deploy_exit_s"] = deploy_exit
    finally:
        s.close()

    # ------------------------------------------------------------- analysis
    T = np.array(rec["t"])
    P = np.array(rec["pos"])
    Q = np.array(rec["quat"])          # MuJoCo qpos quaternion is w,x,y,z
    J = np.array(rec["joints"])        # MuJoCo joint order
    yaw = np.unwrap(yaw_from_quat_wxyz(Q))
    V = np.gradient(P[:, :2], T, axis=0)
    yaw_rate = np.gradient(yaw, T)
    roll_pitch = np.abs(np.degrees(np.arcsin(np.clip(2 * (Q[:, 0] * Q[:, 1] + Q[:, 2] * Q[:, 3]), -1, 1))))
    fell = bool((P[:, 2] < 0.35).any())
    results["fell"] = fell
    if fell:
        results["fall_t"] = float(T[np.argmax(P[:, 2] < 0.35)])
        print(f"[test] FELL at t={results['fall_t']:.2f} s after stream start")
    results["z_min"] = float(P[:, 2].min())
    t_acc = 0.0
    for d, vx, vy, wz in script:
        m = (T >= t_acc + d * 0.5) & (T < t_acc + d)   # second half of the segment
        if m.sum() > 5:
            fwd = np.stack([np.cos(yaw[m]), np.sin(yaw[m])], 1)
            lat = np.stack([-np.sin(yaw[m]), np.cos(yaw[m])], 1)
            seg = dict(cmd=[vx, vy, wz], fwd=float((V[m] * fwd).sum(1).mean()), lat=float((V[m] * lat).sum(1).mean()),
                       yaw_rate=float(yaw_rate[m].mean()), z_min=float(P[m, 2].min()),
                       z_mean=float(P[m, 2].mean()))
            results["segments"].append(seg)
            print(f"[test] cmd v=({vx:.1f},{vy:.1f}) w={wz:.1f} -> fwd {seg['fwd']:.2f} lat {seg['lat']:.2f} m/s, "
                  f"yaw {seg['yaw_rate']:.2f} rad/s, z_min {seg['z_min']:.3f}")
        t_acc += d
    # joint tracking RMS: reference frame i is played at t0 + i / 50 (head estimate); lag-search
    if log:
        idx = np.array(log["idx"])
        ref = np.array(log["joint_pos"])
        last = {}
        for i, r in zip(idx, ref):
            last[int(i)] = r
        ii = np.array(sorted(last))
        R = np.array([last[i] for i in ii])[:, ISAAC_TO_MJ]  # to MuJoCo order
        t_ref = (ii - ii[0]) / S.FPS + (log["t0"] - t0)
        best = (1e9, 0.0)
        for lag in np.arange(-0.2, 1.0, 0.02):
            k = np.clip(np.round((T - lag - t_ref[0]) * S.FPS).astype(int), 0, len(R) - 1)
            valid = T > 2.0
            e = J[valid] - R[k[valid]]
            r = float(np.sqrt((e ** 2).mean()))
            best = min(best, (r, lag))
        rms, lag = best
        k = np.clip(np.round((T - lag - t_ref[0]) * S.FPS).astype(int), 0, len(R) - 1)
        valid = T > 2.0
        e = J[valid] - R[k[valid]]
        legs = float(np.degrees(np.sqrt((e[:, :12] ** 2).mean())))
        arms = float(np.degrees(np.sqrt((e[:, 15:] ** 2).mean())))
        results["tracking_rms_deg"] = dict(all=float(np.degrees(rms)), legs=legs, arms=arms, lag_s=float(lag))
        print(f"[test] tracking RMS all {np.degrees(rms):.2f} deg (legs {legs:.2f}, arms {arms:.2f}) at lag {lag:.2f} s")
        results["planner"] = dict(messages=int(st.pub.messages), stalls=int(st.stalls), jumps=int(st.pl.jumps),
                                  catchup_resets=int(s.catchups), head_hold_lines=int(s.waiting_lines))
    results["fell"] = fell
    print(f"[test] fell: {fell}  z_min {results['z_min']:.3f}  max |roll/pitch| {roll_pitch.max():.1f} deg")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(args.out, "run.npz"), t=T, pos=P, quat=Q, joints=J, cmd=np.array(rec["cmd"]),
             **({"ref_" + k: np.asarray(v) for k, v in log.items()} if log else {}))
    print(f"[test] results in {args.out}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", help="JSON [[duration_s, vx, vy, yaw_rate], ...] (default: built-in walk/jog/turn/stop)")
    ap.add_argument("--onscreen", action="store_true", help="show the MuJoCo viewer")
    ap.add_argument("--out", default=os.path.join(REPO, "data/kplanner/test_runs", time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())))
    ap.add_argument("--motion-dir", default="reference/ffmaster_set8/")
    ap.add_argument("--decoder", default="policy/ffmaster/model_decoder.onnx")
    ap.add_argument("--encoder", default="policy/ffmaster/model_encoder.onnx")
    ap.add_argument("--obs-config", default="policy/ffmaster/observation_config.yaml")
    # planner options (same defaults as ffmaster_kplanner_stream.py)
    ap.add_argument("--db", default=S.DEFAULT_DB)
    ap.add_argument("--host", default="*")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--topic", default="pose")
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--lead", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=75)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--search-interval", type=int, default=5)
    ap.add_argument("--blend", type=float, default=0.25)
    ap.add_argument("--hysteresis", type=float, default=0.15)
    ap.add_argument("--min-play", type=int, default=20, help="frames a new clip plays before another voluntary switch")
    ap.add_argument("--max-speed", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-start-pose", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
