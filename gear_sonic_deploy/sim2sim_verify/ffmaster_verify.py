#!/usr/bin/env python3
"""FF Master sim2sim verification tool — the automated gates for a staged checkpoint.

Consolidates what used to be several scratchpad harnesses. Run from the repo
root with the sim venv:  .venv_sim/bin/python gear_sonic_deploy/sim2sim_verify/ffmaster_verify.py <mode>

Modes
  sweep    Gate 1 — play every reference motion, pass/fail per clip
  rms      tracking RMS (commanded reference vs achieved) on one motion
  stress   Gate 2 — repeated trials of one motion with the GUI viewer on.
           The viewer slows physics below real time while the deploy keeps
           commanding at wall-clock 50 Hz, so the reference plays effectively
           fast — a disturbance that discriminates thin-margin checkpoints
           (it caught the 40.1k jump regression that all clean metrics missed).
  motion   single-motion check with reference pelvis-height comparison

Two hard-won environment details are baked into the deploy launch:
  * the vendored CycloneDDS must precede any ROS2 CycloneDDS on
    LD_LIBRARY_PATH, or the binary heap-corrupts at DDS init
  * no NVIDIA compat libs (driver mismatch resolved 2026-08-10 reboot)

Pass criteria are derived per clip from its own reference trajectory, since
several motions legitimately end crouched or low (pick-up, deep reach).
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import threading
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repository root
DEPLOY_DIR = os.path.join(REPO, "gear_sonic_deploy")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)

# IsaacLab-order index of each MuJoCo slot; reference CSVs are IsaacLab order,
# sim qpos is MuJoCo order (FFMASTER_ISAACLAB_TO_MUJOCO_DOF, robots/ffmaster.py).
ISO2MUJ = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
           11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]

DEPLOY_CMD = (
    "source scripts/setup_env.sh >/dev/null 2>&1; "
    "export LD_LIBRARY_PATH=$PWD/thirdparty/unitree_sdk2/thirdparty/lib/x86_64:$LD_LIBRARY_PATH; "
    "exec ./target/release/ffmaster_deploy_onnx_ref lo "
    "{decoder} {motion_dir} --obs-config {obs_config} "
    "--encoder-file {encoder} --input-type keyboard --output-type all --disable-crc-check{extra}"
)


class Session:
    """A running MuJoCo sim + deploy binary pair."""

    def __init__(self, args, onscreen=False):
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory

        cfg = SimLoopConfig(wbc_version="ffmaster_sonic_model12", interface="sim", verbose=False)
        wbc = cfg.load_wbc_yaml()
        wbc["ENV_NAME"] = "default"
        wbc["PRINT_SCENE_INFORMATION"] = False
        self.sim = SimulatorFactory.create_simulator(
            config=wbc, env_name="default", onscreen=onscreen, offscreen=False)
        self.sim.start_as_thread()
        time.sleep(2.0)
        self.mj = self.sim.sim_env.mj_data
        self.motion_dir = args.motion_dir

        extra = f" --initial-encoder-mode {args.encoder_mode}" if args.encoder_mode is not None else ""
        cmd = DEPLOY_CMD.format(decoder=args.decoder, encoder=args.encoder,
                                motion_dir=args.motion_dir, extra=extra,
                                obs_config=args.obs_config)
        self.log_path = os.path.join(OUT_DIR, "last_deploy.log")
        self.logf = open(self.log_path, "w")
        self.proc = subprocess.Popen(["bash", "-c", cmd], cwd=DEPLOY_DIR,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.ready = threading.Event()
        self.order = []
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            self.logf.write(line)
            self.logf.flush()
            m = re.search(r"Motion '([^']+)' encode_mode set to", line)
            if m:
                self.order.append(m.group(1))
            if "Initialized keyboard input interface" in line:
                self.ready.set()

    def z(self):
        return float(self.mj.qpos[2])

    def joints_mujoco(self):
        bji = self.sim.sim_env.body_joint_index
        return np.array([self.mj.qpos[j + 6] for j in bji])

    def send(self, key):
        self.proc.stdin.write(key)
        self.proc.stdin.flush()

    def wait_ready(self, timeout=600):
        t0 = time.time()
        while not self.ready.is_set():
            if self.proc.poll() is not None:
                raise RuntimeError(f"deploy exited during init; see {self.log_path}")
            if time.time() - t0 > timeout:
                raise RuntimeError(f"deploy init timeout; see {self.log_path}")
            time.sleep(1)
        time.sleep(2)

    def start_and_stand(self):
        """']' then lower the band and release — equivalent to viewer key 9."""
        self.send("]")
        time.sleep(8)
        for tgt in np.linspace(1.0, 0.68, 16):
            self.sim.sim_env.elastic_band.point = np.array([0.0, 0.0, float(tgt)])
            time.sleep(0.15)
        self.sim.sim_env.elastic_band.enable = False
        time.sleep(3)

    def goto(self, name):
        idx = self.order.index(name)
        for _ in range(idx):
            self.send("N")
            time.sleep(0.15)
        time.sleep(1)

    def recover(self):
        self.sim.sim_env.elastic_band.point = np.array([0.0, 0.0, 0.70])
        self.sim.sim_env.elastic_band.enable = True
        time.sleep(3)
        self.sim.sim_env.elastic_band.enable = False
        time.sleep(2)

    def close(self):
        try:
            self.send("O")
            time.sleep(1)
            self.proc.terminate()
        except Exception:
            pass
        self.sim._running = False


def ref_z(motion_dir, name):
    p = os.path.join(DEPLOY_DIR, motion_dir, name, "body_pos.csv")
    return [float(r[2]) for r in list(csv.reader(open(p)))[1:]]


def ref_joints(motion_dir, name):
    p = os.path.join(DEPLOY_DIR, motion_dir, name, "joint_pos.csv")
    return np.array(list(csv.reader(open(p)))[1:], dtype=float)[:, ISO2MUJ]


def n_frames(motion_dir, name):
    with open(os.path.join(DEPLOY_DIR, motion_dir, name, "joint_pos.csv")) as f:
        return sum(1 for _ in f) - 1


def play(s, name, log_joints=False):
    """Play one clip; return (z_min, z_end, times, joints)."""
    dur = n_frames(s.motion_dir, name) / 50.0 + 1.0
    s.send("T")
    t0, zmin, ts, qs = time.time(), 1e9, [], []
    while time.time() - t0 < dur:
        if s.proc.poll() is not None:
            raise RuntimeError(f"deploy exited during {name}; see {s.log_path}")
        zmin = min(zmin, s.z())
        if log_joints:
            ts.append(time.time() - t0)
            qs.append(s.joints_mujoco())
        time.sleep(0.02)
    time.sleep(2.0)
    return zmin, s.z(), np.array(ts), (np.array(qs) if qs else None)


def verdict(motion_dir, name, zmin, zend):
    rz = ref_z(motion_dir, name)
    ok = zmin > max(0.21, min(rz) - 0.15) and zend > min(rz[-1], 0.55) - 0.12
    return ok, min(rz), rz[-1]


def cmd_sweep(args):
    s = Session(args)
    results = {}
    try:
        s.wait_ready()
        print(f"[sweep] {len(s.order)} motions in {args.motion_dir}")
        s.start_and_stand()
        print(f"[sweep] standing at z={s.z():.3f}")
        for i, name in enumerate(s.order):
            if i:
                s.send("N")
                time.sleep(1.0)
            zmin, zend, _, _ = play(s, name)
            ok, rmin, rend = verdict(args.motion_dir, name, zmin, zend)
            results[name] = dict(z_min=round(zmin, 3), z_end=round(zend, 3),
                                 ref_min=round(rmin, 3), ref_end=round(rend, 3), passed=ok)
            print(f"[sweep] {name}: z_min={zmin:.3f} z_end={zend:.3f} "
                  f"(ref {rmin:.2f}->{rend:.2f}) {'PASS' if ok else 'FAIL'}")
            if zend < min(rend, 0.55) - 0.12:      # genuine fall only
                s.recover()
                print(f"[sweep] recovered to z={s.z():.3f}")
    finally:
        s.close()
        with open(os.path.join(OUT_DIR, "last_sweep.json"), "w") as f:
            json.dump(results, f, indent=2)
        n = sum(1 for r in results.values() if r["passed"])
        print(f"[sweep] DONE: {n}/{len(results)} pass")


def cmd_rms(args):
    s = Session(args)
    try:
        s.wait_ready()
        s.start_and_stand()
        s.goto(args.motion)
        _, _, ts, qs = play(s, args.motion, log_joints=True)
        ref = ref_joints(args.motion_dir, args.motion)
        best = (1e9, 0.0)
        for lag in np.arange(0, 2, 0.04):          # absorb keypress latency
            idx = np.clip(((ts - lag) * 50).astype(int), 0, len(ref) - 1)
            v = ts >= lag
            if v.sum() < 100:
                continue
            r = float(np.sqrt(((qs[v] - ref[idx[v]]) ** 2).mean()))
            best = min(best, (r, lag))
        rms, lag = best
        idx = np.clip(((ts - lag) * 50).astype(int), 0, len(ref) - 1)
        v = ts >= lag
        e = qs[v] - ref[idx[v]]
        legs = np.degrees(np.sqrt((e[:, :12] ** 2).mean()))
        arms = np.degrees(np.sqrt((e[:, 15:] ** 2).mean()))
        print(f"[rms] {args.motion}: all-DoF {np.degrees(rms):.2f} deg, "
              f"legs {legs:.2f}, arms {arms:.2f}")
        np.savez(os.path.join(OUT_DIR, "last_rms_traj.npz"), t=ts, q=qs)
    finally:
        s.close()


def cmd_stress(args):
    s = Session(args, onscreen=True)
    falls = 0
    try:
        s.wait_ready()
        s.start_and_stand()
        s.goto(args.motion)
        for k in range(args.trials):
            zmin, zend, _, _ = play(s, args.motion)
            ok, rmin, rend = verdict(args.motion_dir, args.motion, zmin, zend)
            falls += not ok
            print(f"[stress] trial {k+1}: z_min={zmin:.3f} z_end={zend:.3f} "
                  f"{'ok' if ok else 'FALL'}")
            if zend < min(rend, 0.55) - 0.12:
                s.recover()
        print(f"[stress] RESULT: {falls}/{args.trials} falls")
    finally:
        s.close()


def cmd_motion(args):
    s = Session(args)
    try:
        s.wait_ready()
        s.start_and_stand()
        s.goto(args.motion)
        zmin, zend, _, _ = play(s, args.motion)
        ok, rmin, rend = verdict(args.motion_dir, args.motion, zmin, zend)
        print(f"[motion] {args.motion}: sim z_min={zmin:.3f} z_end={zend:.3f} | "
              f"ref {rmin:.2f}->{rend:.2f} | {'PASS' if ok else 'FAIL'}")
    finally:
        s.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["sweep", "rms", "stress", "motion"])
    p.add_argument("--motion", default="walking_quip_360_R_002__A428")
    p.add_argument("--motion-dir", default="reference/ffmaster_set8/")
    p.add_argument("--decoder", default="policy/ffmaster/model_decoder.onnx")
    p.add_argument("--encoder", default="policy/ffmaster/model_encoder.onnx")
    p.add_argument("--obs-config", default="policy/ffmaster/observation_config.yaml",
                   help="deploy observation config; the chingmu 2-encoder model "
                        "needs policy/ffmaster_chingmu/observation_config.yaml (910-wide encoder)")
    p.add_argument("--encoder-mode", type=int, default=None,
                   help="0=tracking 1=teleop 2=smpl (default: binary's default)")
    p.add_argument("--trials", type=int, default=4)
    args = p.parse_args()
    {"sweep": cmd_sweep, "rms": cmd_rms, "stress": cmd_stress, "motion": cmd_motion}[args.mode](args)


if __name__ == "__main__":
    main()
