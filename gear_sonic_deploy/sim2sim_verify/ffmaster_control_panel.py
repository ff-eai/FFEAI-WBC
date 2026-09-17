#!/usr/bin/env python3
"""FF Master sim2sim control panel — GUI replacement for the two-terminal keyboard flow.

Why a separate window: MuJoCo's passive viewer has no widget API, and the
controls are split across two processes — motion selection lives in the deploy
binary (keyboard stdin) while the elastic band, reset and fall-check live in
the simulator. This panel owns both: it runs the simulator in-process (so its
viewer window opens as usual) and launches the deploy binary as a child,
writing keys to its stdin.

Run from the repo root:
    .venv_sim/bin/python gear_sonic_deploy/sim2sim_verify/ffmaster_control_panel.py
    ... --motion-dir reference/ffmaster_set4/ --encoder-mode 2

Panel gives you: a motion dropdown (jumps by sending N/P), Play, Restart,
Start Policy, Stop Control, Drop Robot, Reset Robot, Fall-check toggle.
The MuJoCo window's own keys still work.
"""

import argparse
import os
import queue
import re
import subprocess
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repository root
DEPLOY_DIR = os.path.join(REPO, "gear_sonic_deploy")
os.chdir(REPO)

DEPLOY_CMD = (
    "source scripts/setup_env.sh >/dev/null 2>&1; "
    "export LD_LIBRARY_PATH=$PWD/thirdparty/unitree_sdk2/thirdparty/lib/x86_64:$LD_LIBRARY_PATH; "
    "exec ./target/release/ffmaster_deploy_onnx_ref lo "
    "{dec} {mdir} --obs-config {obs_config} "
    "--encoder-file {enc} --input-type keyboard --output-type all --disable-crc-check{extra}"
)


class Panel:
    def __init__(self, args):
        self.args = args
        self.msgs = queue.Queue()
        self.order = []          # motion names, in the deploy binary's own order
        self.cursor = 0          # deploy's current motion index
        self.ready = False
        self.replay = None
        self._start_sim()
        self._start_deploy()
        if args.with_replay:
            self._start_replay()
        self._build_ui()

    # ---------------- backend ----------------
    def _start_sim(self):
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory
        cfg = SimLoopConfig(wbc_version="ffmaster_sonic_model12", interface="sim", verbose=False)
        wbc = cfg.load_wbc_yaml()
        wbc["ENV_NAME"] = "default"
        wbc["PRINT_SCENE_INFORMATION"] = False
        self.sim = SimulatorFactory.create_simulator(
            config=wbc, env_name="default", onscreen=True, offscreen=False)
        self.sim.start_as_thread()

    def _start_deploy(self):
        extra = f" --initial-encoder-mode {self.args.encoder_mode}" if self.args.encoder_mode is not None else ""
        cmd = DEPLOY_CMD.format(dec=self.args.decoder, enc=self.args.encoder,
                                mdir=self.args.motion_dir, extra=extra,
                                obs_config=self.args.obs_config)
        self.proc = subprocess.Popen(["bash", "-c", cmd], cwd=DEPLOY_DIR,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._reader, daemon=True).start()

    def _start_replay(self):
        """Second window: kinematic playback of the same reference CSVs, so the
        commanded motion and the policy's execution can be watched together."""
        cmd = [os.path.join(REPO, ".venv_sim/bin/python"),
               os.path.join(REPO, "gear_sonic_deploy/sim2sim_verify/ffmaster_replay_viewer.py"),
               "--motion-dir", self.args.motion_dir]
        self.replay = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE,
                                       text=True, bufsize=1)

    def send_replay(self, line):
        if self.replay is None or self.replay.poll() is not None:
            return
        try:
            self.replay.stdin.write(line + "\n")
            self.replay.stdin.flush()
        except Exception:
            pass

    def _reader(self):
        for line in self.proc.stdout:
            m = re.search(r"Motion '([^']+)' encode_mode set to", line)
            if m:
                self.order.append(m.group(1))
            if "Initialized keyboard input interface" in line:
                self.msgs.put(("ready", None))
            m2 = re.search(r"Motion index: (\d+)", line)
            if m2:
                self.msgs.put(("idx", int(m2.group(1))))

    def send(self, key, label):
        if self.proc.poll() is not None:
            self.status.set("deploy process has exited")
            return
        try:
            self.proc.stdin.write(key)
            self.proc.stdin.flush()
            self.status.set(f"sent {label}")
        except Exception as e:
            self.status.set(f"send failed: {e}")

    # ---------------- sim-side actions ----------------
    def drop_robot(self):
        """Equivalent to viewer key '9' but ramped: lower the band, then release."""
        def run():
            band = self.sim.sim_env.elastic_band
            for t in np.linspace(1.0, 0.68, 16):
                band.point = np.array([0.0, 0.0, float(t)])
                threading.Event().wait(0.15)
            band.enable = False
            self.msgs.put(("status", "robot dropped (band released)"))
        threading.Thread(target=run, daemon=True).start()
        self.status.set("lowering band...")

    def lift_robot(self):
        band = self.sim.sim_env.elastic_band
        band.point = np.array([0.0, 0.0, 0.70])
        band.enable = True
        self.status.set("band re-enabled (robot suspended)")

    def reset_robot(self):
        import mujoco
        mujoco.mj_resetData(self.sim.sim_env.mj_model, self.sim.sim_env.mj_data)
        self.status.set("sim reset (Backspace equivalent)")

    def toggle_fall_check(self):
        env = self.sim.sim_env
        env.fall_check_enabled = not env.fall_check_enabled
        self.fall_btn.config(text=f"Fall auto-reset: {'ON' if env.fall_check_enabled else 'OFF'}")
        self.status.set(f"fall auto-reset {'enabled' if env.fall_check_enabled else 'disabled'}")

    # ---------------- motion selection ----------------
    def goto_selected(self):
        name = self.motion_var.get()
        if name not in self.order:
            self.status.set("motion list not loaded yet")
            return
        target = self.order.index(name)
        delta = target - self.cursor
        if delta == 0:
            self.status.set(f"already on {name}")
            return
        n_tot = len(self.order)
        fwd = (target - self.cursor) % n_tot          # deploy wraps at the ends
        bwd = (self.cursor - target) % n_tot
        key, n = ("N", fwd) if fwd <= bwd else ("P", bwd)
        def run():
            for _ in range(n):
                self.send(key, key)
                threading.Event().wait(0.06)
            self.cursor = target
            self.send_replay(f"pose {name}")
            self.msgs.put(("status", f"selected [{target}] {name}"))
        threading.Thread(target=run, daemon=True).start()
        self.status.set(f"navigating {n}x {key} (~{n*0.06:.0f}s) ...")

    def play(self):
        """Trigger policy and reference replay together."""
        self.send_replay(f"play {self.motion_var.get()}")
        self.send("T", "play")

    def jump_to(self, k):
        """Quick-select motion #k: the deploy binary jumps directly on digit
        keys 0-9 (keyboard_handler.hpp); folders with more than 10 motions
        only expose the first 10 here — use the dropdown for the rest."""
        if not self.order:
            self.status.set("motion list not loaded yet")
            return
        if k >= len(self.order):
            self.status.set(f"no motion #{k} — folder has only {len(self.order)}")
            return
        self.send(str(k), f"select #{k}")
        self.cursor = k
        self.motion_var.set(self.order[k])
        self.send_replay(f"pose {self.order[k]}")

    # ---------------- UI ----------------
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("FF Master sim2sim control")
        self.root.attributes("-topmost", True)
        self.status = tk.StringVar(value="starting deploy (TensorRT build on first run)...")

        pad = dict(padx=6, pady=4)
        f0 = ttk.LabelFrame(self.root, text="Motion")
        f0.grid(row=0, column=0, sticky="ew", **pad)
        self.motion_var = tk.StringVar()
        self.combo = ttk.Combobox(f0, textvariable=self.motion_var, width=54, state="readonly")
        self.combo.grid(row=0, column=0, columnspan=3, **pad)
        # selection alone updates the reference window; "Go to selected" also
        # walks the deploy binary's own cursor (which only moves via N/P)
        self.combo.bind("<<ComboboxSelected>>",
                        lambda _e: self.send_replay(f"pose {self.motion_var.get()}"))
        ttk.Button(f0, text="◀ Prev", command=lambda: self._step(-1)).grid(row=1, column=0, **pad)
        ttk.Button(f0, text="Go to selected", command=self.goto_selected).grid(row=1, column=1, **pad)
        ttk.Button(f0, text="Next ▶", command=lambda: self._step(1)).grid(row=1, column=2, **pad)
        fq = ttk.Frame(f0)
        fq.grid(row=2, column=0, columnspan=3, **pad)
        ttk.Label(fq, text="quick select:").pack(side="left", padx=(0, 4))
        for k in range(10):
            ttk.Button(fq, text=str(k), width=2,
                       command=lambda k=k: self.jump_to(k)).pack(side="left", padx=1)

        f1 = ttk.LabelFrame(self.root, text="Playback")
        f1.grid(row=1, column=0, sticky="ew", **pad)
        ttk.Button(f1, text="▶ Play  (T)", command=self.play).grid(row=0, column=0, **pad)
        ttk.Button(f1, text="↺ Restart  (R)", command=lambda: self.send("R", "restart")).grid(row=0, column=1, **pad)

        f2 = ttk.LabelFrame(self.root, text="Policy")
        f2.grid(row=2, column=0, sticky="ew", **pad)
        ttk.Button(f2, text="Start policy  (])", command=lambda: self.send("]", "start")).grid(row=0, column=0, **pad)
        ttk.Button(f2, text="Stop control  (O)", command=lambda: self.send("O", "stop")).grid(row=0, column=1, **pad)
        # heading delta: 15 deg (pi/12) per press, applied to the reference
        # heading — use at the initial pose to point the robot before playing
        ttk.Button(f2, text="⟲ Rotate left  (Q)",
                   command=lambda: self.send("q", "rotate left 15°")).grid(row=1, column=0, **pad)
        ttk.Button(f2, text="⟳ Rotate right  (E)",
                   command=lambda: self.send("e", "rotate right 15°")).grid(row=1, column=1, **pad)

        f3 = ttk.LabelFrame(self.root, text="Simulator")
        f3.grid(row=3, column=0, sticky="ew", **pad)
        ttk.Button(f3, text="Drop robot  (9)", command=self.drop_robot).grid(row=0, column=0, **pad)
        ttk.Button(f3, text="Suspend robot", command=self.lift_robot).grid(row=0, column=1, **pad)
        ttk.Button(f3, text="Reset  (Backspace)", command=self.reset_robot).grid(row=0, column=2, **pad)
        self.fall_btn = ttk.Button(f3, text="Fall auto-reset: ON", command=self.toggle_fall_check)
        self.fall_btn.grid(row=1, column=0, columnspan=3, sticky="ew", **pad)

        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w").grid(
            row=4, column=0, sticky="ew", padx=6, pady=(0, 6))

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.after(300, self._pump)

    def _step(self, d):
        self.send("N" if d > 0 else "P", "N" if d > 0 else "P")
        self.cursor = max(0, min(len(self.order) - 1, self.cursor + d)) if self.order else 0
        if self.order:
            self.motion_var.set(self.order[self.cursor])
            self.send_replay(f"pose {self.order[self.cursor]}")

    def _pump(self):
        try:
            while True:
                kind, val = self.msgs.get_nowait()
                if kind == "ready":
                    self.ready = True
                    self.combo["values"] = self.order
                    if self.order:
                        self.motion_var.set(self.order[0])
                    self.status.set(f"ready — {len(self.order)} motions loaded. "
                                    f"Start policy, then Drop robot, then Play.")
                elif kind == "idx":
                    self.cursor = val
                    if self.order and val < len(self.order):
                        self.motion_var.set(self.order[val])
                elif kind == "status":
                    self.status.set(val)
        except queue.Empty:
            pass
        if self.proc.poll() is not None and self.ready:
            self.status.set("deploy process exited")
        self.root.after(300, self._pump)

    def quit(self):
        """Order matters: stop the policy, kill the child, then wind the sim
        thread down and let it close the viewer before the interpreter exits —
        tearing the process down with the viewer still live segfaults."""
        try:
            self.send("O", "stop")
            threading.Event().wait(0.5)
            self.proc.terminate()
            self.proc.wait(timeout=5)
            if self.replay is not None:
                self.send_replay("quit")
                self.replay.terminate()
        except Exception:
            pass
        try:
            self.sim._running = False
            if getattr(self.sim, "sim_thread", None) is not None:
                self.sim.sim_thread.join(timeout=5)
            self.sim.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion-dir", default="reference/ffmaster_set8/")
    p.add_argument("--decoder", default="policy/ffmaster/model_decoder.onnx")
    p.add_argument("--encoder", default="policy/ffmaster/model_encoder.onnx")
    p.add_argument("--obs-config", default="policy/ffmaster/observation_config.yaml",
                   help="deploy observation config; the chingmu 2-encoder model "
                        "needs policy/ffmaster_chingmu/observation_config.yaml (910-wide encoder)")
    p.add_argument("--encoder-mode", type=int, default=None)
    p.add_argument("--with-replay", action="store_true",
                   help="open a second window replaying the reference motion kinematically")
    Panel(p.parse_args()).run()


if __name__ == "__main__":
    main()
