#!/usr/bin/env python3
"""Kinematic replay of FF Master reference motions — the 'what it should look like' window.

No physics, no policy: pelvis pose and joint angles are written straight into
mj_data.qpos from the reference CSVs and rendered with mj_forward. Pair it with
the policy sim2sim window to see reference vs execution side by side.

Reference CSV conventions handled here:
  joint_pos.csv  29 joints, IsaacLab order  -> permuted to MJCF/MuJoCo order
  body_pos.csv   body 0 = pelvis, world xyz
  body_quat.csv  body 0 = pelvis, world wxyz
  all at 50 fps

Standalone:
    .venv_sim/bin/python ffmaster_replay_viewer.py --motion-dir reference/ffmaster_set4/ \
        --motion salute_R_003__A405
Driven by the control panel: reads stdin lines
    play <motion_name>   |   pose <motion_name>   |   stop   |   quit
"""

import argparse
import csv
import os
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repository root
DEPLOY_DIR = os.path.join(REPO, "gear_sonic_deploy")
SCENE = os.path.join(REPO, "gear_sonic/data/assets/robot_description/mjcf/ffmaster_scene_29dof.xml")
# IsaacLab-order index for each MuJoCo joint slot (FFMASTER_ISAACLAB_TO_MUJOCO_DOF)
ISO2MUJ = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
           11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
FPS = 50.0


def load_motion(motion_dir, name):
    base = os.path.join(DEPLOY_DIR, motion_dir, name)
    rows = lambda f: list(csv.reader(open(os.path.join(base, f))))[1:]
    jp = np.array(rows("joint_pos.csv"), dtype=float)[:, ISO2MUJ]     # (T,29) MuJoCo order
    bp = np.array(rows("body_pos.csv"), dtype=float)[:, 0:3]          # pelvis xyz
    bq = np.array(rows("body_quat.csv"), dtype=float)[:, 0:4]         # pelvis wxyz
    n = min(len(jp), len(bp), len(bq))
    return jp[:n], bp[:n], bq[:n]


class Replay:
    def __init__(self, args):
        self.args = args
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        # Pure visualisation: no contacts, no constraint solver. Posing arbitrary
        # reference frames can push geoms deep into the floor, and mj_forward's
        # constraint allocation then blows up ("nefc under-allocation").
        self.model.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT
                                        | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        self.data = mujoco.MjData(self.model)
        self.motions = sorted(d for d in os.listdir(os.path.join(DEPLOY_DIR, args.motion_dir))
                              if os.path.isdir(os.path.join(DEPLOY_DIR, args.motion_dir, d)))
        self.cur = args.motion if args.motion in self.motions else (self.motions[0] if self.motions else None)
        self.frames = None
        self.frame_i = 0
        self.playing = False
        self.lock = threading.Lock()
        if self.cur:
            self.load(self.cur, pose_only=True)

    def load(self, name, pose_only=False):
        try:
            jp, bp, bq = load_motion(self.args.motion_dir, name)
        except Exception as e:
            print(f"[replay] cannot load {name}: {e}", flush=True)
            return
        with self.lock:
            self.cur, self.frames, self.frame_i = name, (jp, bp, bq), 0
            self.playing = not pose_only
        self.apply(0)
        print(f"[replay] {'posed' if pose_only else 'playing'} {name} ({len(jp)} frames)", flush=True)

    def apply(self, i):
        jp, bp, bq = self.frames
        i = min(i, len(jp) - 1)
        self.data.qpos[0:3] = bp[i]
        self.data.qpos[3:7] = bq[i]
        self.data.qpos[7:7 + 29] = jp[i]
        self.data.qvel[:] = 0
        # forward kinematics only — everything the renderer needs
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def stdin_loop(self):
        for line in sys.stdin:
            try:
                parts = line.strip().split(None, 1)
                if not parts:
                    continue
                cmd, arg = parts[0], (parts[1] if len(parts) > 1 else "")
                if cmd == "play":
                    self.load(arg or self.cur, pose_only=False)
                elif cmd == "pose":
                    self.load(arg or self.cur, pose_only=True)
                elif cmd == "stop":
                    with self.lock:
                        self.playing = False
                elif cmd == "quit":
                    os._exit(0)
            except Exception as e:      # keep accepting commands no matter what
                print(f"[replay] command failed ({line.strip()!r}): {e}", flush=True)

    def run(self):
        threading.Thread(target=self.stdin_loop, daemon=True).start()
        with mujoco.viewer.launch_passive(self.model, self.data,
                                          show_left_ui=False, show_right_ui=False) as v:
            v.cam.azimuth, v.cam.elevation, v.cam.distance = 120, -20, 3.0
            v.cam.lookat[:] = [0, 0, 0.6]
            next_t = time.time()
            while v.is_running():
                try:
                  with self.lock:
                    if self.playing and self.frames is not None:
                        self.frame_i += 1
                        if self.frame_i >= len(self.frames[0]):
                            self.frame_i = len(self.frames[0]) - 1
                            self.playing = False
                        self.apply(self.frame_i)
                except Exception as e:
                    print(f"[replay] frame error: {e}", flush=True)
                    self.playing = False
                v.sync()
                next_t += 1.0 / FPS
                time.sleep(max(0.0, next_t - time.time()))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion-dir", default="reference/ffmaster_set8/")
    p.add_argument("--motion", default=None)
    Replay(p.parse_args()).run()


if __name__ == "__main__":
    main()
