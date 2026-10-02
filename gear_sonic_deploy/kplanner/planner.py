"""Motion-matching kinematic planner for FF Master (runtime core, no I/O).

Given a velocity command (forward, lateral, yaw rate) it produces a 50 Hz
whole-body reference (IsaacLab-order joint positions, world root position and
w,x,y,z root quaternion) by playing back frames of the motion database built
with build_db.py and jumping between clips whenever another frame of the
library fits the commanded trajectory and the current pose better.

Pipeline per tick
  1. spring model: smooth the command into a predicted root path 0.2/0.4/0.6 s ahead
  2. query = [current root velocity, predicted path, predicted headings,
              current foot positions and velocities]  (heading frame)
  3. every `search_interval` frames (or on demand): nearest database frame;
     jump if it beats continuing the current clip by the hysteresis margin
  4. on a jump: re-anchor the new clip's root to the current world root
     (position and yaw are continuous by construction) and start an
     inertialization blend on joints and root height
  5. emit the current frame transformed into the planner's world frame
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffmaster_kinematics import (  # noqa: E402
    quat_mul_wxyz, rot2d, wrap_angle, yaw_quat_wxyz,
)


class MotionDB:
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.fps = float(z["fps"])
        self.joint_pos = z["joint_pos"]
        self.root_pos = z["root_pos"]
        self.root_quat = z["root_quat"]
        self.yaw = z["yaw"]
        self.foot_pos = z["foot_pos"]
        self.root_vel = z["root_vel"]
        self.feat = z["feat"]
        self.clip_id = z["clip_id"]
        self.frame = z["frame"]
        self.valid = z["valid"]
        self.clip_names = list(z["clip_names"])
        self.clip_start = z["clip_start"]
        self.clip_len = z["clip_len"]
        self.clip_speed = z["clip_speed"]
        self.feat_mean = z["feat_mean"]
        self.feat_std = z["feat_std"]
        self.weights = z["feat_weights"]
        self.lookahead = z["lookahead"]
        self.n = len(self.joint_pos)
        self.valid_idx = np.nonzero(self.valid)[0]
        # pre-normalised, weighted feature matrix of the searchable frames
        self._set_weights(self.weights)

    def _set_weights(self, w: np.ndarray):
        self.weights = np.asarray(w, np.float32)
        self.scale = (self.weights / self.feat_std).astype(np.float32)
        self.F = np.ascontiguousarray(((self.feat[self.valid_idx] - self.feat_mean) * self.scale).astype(np.float32))
        self.F_norm2 = np.einsum("ij,ij->i", self.F, self.F)

    def set_group_weights(self, root_vel=None, traj_pos=None, traj_dir=None, foot_pos=None, foot_vel=None):
        w = self.weights.copy()
        for sl, v in ((slice(0, 3), root_vel), (slice(3, 9), traj_pos), (slice(9, 15), traj_dir),
                      (slice(15, 21), foot_pos), (slice(21, 27), foot_vel)):
            if v is not None:
                w[sl] = v
        self._set_weights(w)

    def foot_vel_norm(self) -> np.ndarray:
        return np.stack([np.linalg.norm(self.feat[:, 21:24], axis=1), np.linalg.norm(self.feat[:, 24:27], axis=1)], 1)

    def index(self, clip: int, frame: int) -> int:
        return int(self.clip_start[clip] + frame)

    def cost(self, query: np.ndarray) -> np.ndarray:
        """Squared weighted distance of every searchable frame to the query."""
        q = ((query - self.feat_mean) * self.scale).astype(np.float32)
        return self.F_norm2 - 2.0 * (self.F @ q) + float(q @ q)


class KinematicPlanner:
    def __init__(self, db: MotionDB, tau_vel: float = 0.25, tau_yaw: float = 0.2,
                 search_interval: int = 5, blend_time: float = 0.25, hysteresis: float = 0.15,
                 same_clip_window: int = 12, max_speed: float = 2.5, max_yaw_rate: float = 1.5,
                 seed: int = 0, min_play: int = 20):
        self.db = db
        self.dt = 1.0 / db.fps
        self.tau_vel, self.tau_yaw = tau_vel, tau_yaw
        self.search_interval = search_interval
        self.blend_frames = max(1, int(round(blend_time * db.fps)))
        self.hysteresis = hysteresis
        self.same_clip_window = same_clip_window
        self.max_speed, self.max_yaw_rate = max_speed, max_yaw_rate
        # after a voluntary jump, play at least this many frames of the new clip before another
        # voluntary jump: motion matching must play through gait cycles, not re-pick every search
        self.min_play = min_play
        self.rng = np.random.default_rng(seed)
        # command (heading frame) and its smoothed "simulation" state
        self.cmd_vel = np.zeros(2)
        self.cmd_yaw_rate = 0.0
        self.sim_vel = np.zeros(2)
        self.sim_yaw_rate = 0.0
        # playback state
        self.clip = 0
        self.frame = 0
        self.anchor_world_xy = np.zeros(2)
        self.anchor_world_yaw = 0.0
        self.anchor_clip_xy = np.zeros(2)
        self.anchor_clip_yaw = 0.0
        self.blend_dq = np.zeros(29)
        self.blend_dz = 0.0
        self.blend_k = self.blend_frames  # frames elapsed since jump (>= blend_frames: no blend)
        self.since_search = 0
        self.tick = 0
        self.last_out = None
        self.jumps = 0
        self.last_cost = float("nan")
        self.force_search = False

    # ------------------------------------------------------------------ state
    def reset(self, world_xy=(0.0, 0.0), world_yaw: float = 0.0, standing: bool = True, still_s: float = 1.5):
        """Start at the given world pose on a frame that stays still for `still_s` seconds (or a random one)."""
        db = self.db
        if standing:
            n = int(still_s * db.fps)
            sp = np.linalg.norm(db.root_vel[:, :2], axis=1)
            fv = np.linalg.norm(db.foot_vel_norm(), axis=1) if hasattr(db, "foot_vel_norm") else None
            still = sp < 0.1
            # frames whose next n frames are all still (within the same clip)
            ok = np.zeros(db.n, bool)
            for c in range(len(db.clip_names)):
                s0, ln = int(db.clip_start[c]), int(db.clip_len[c])
                cs = np.concatenate([[0], np.cumsum(still[s0:s0 + ln])])
                for f in range(0, ln - n):
                    ok[s0 + f] = (cs[f + n] - cs[f]) == n
            ok &= db.valid
            if fv is not None:
                ok &= fv < 0.3
            cand = np.nonzero(ok)[0]
            if len(cand) == 0:
                cand = db.valid_idx[np.argsort(sp[db.valid_idx])[:200]]
            i = int(self.rng.choice(cand))
        else:
            i = int(self.rng.choice(db.valid_idx))
        self.clip, self.frame = int(db.clip_id[i]), int(db.frame[i])
        self._anchor(np.asarray(world_xy, float), float(world_yaw))
        self.blend_dq[:] = 0.0
        self.blend_dz = 0.0
        self.blend_k = self.blend_frames
        self.sim_vel[:] = 0.0
        self.sim_yaw_rate = 0.0
        self.since_search = 0
        self.tick = 0
        self.last_out = None
        self.jumps = 0

    def get_state(self) -> dict:
        return dict(clip=self.clip, frame=self.frame, awxy=self.anchor_world_xy.copy(),
                    awyaw=self.anchor_world_yaw, acxy=self.anchor_clip_xy.copy(), acyaw=self.anchor_clip_yaw,
                    dq=self.blend_dq.copy(), dz=self.blend_dz, bk=self.blend_k, ss=self.since_search,
                    tick=self.tick, sv=self.sim_vel.copy(), syr=self.sim_yaw_rate,
                    last_out=None if self.last_out is None else {k: np.copy(v) for k, v in self.last_out.items()},
                    jumps=self.jumps, rng=self.rng.bit_generator.state)

    def set_state(self, s: dict):
        self.clip, self.frame = s["clip"], s["frame"]
        self.anchor_world_xy, self.anchor_world_yaw = s["awxy"].copy(), s["awyaw"]
        self.anchor_clip_xy, self.anchor_clip_yaw = s["acxy"].copy(), s["acyaw"]
        self.blend_dq, self.blend_dz, self.blend_k = s["dq"].copy(), s["dz"], s["bk"]
        self.since_search, self.tick = s["ss"], s["tick"]
        self.sim_vel, self.sim_yaw_rate = s["sv"].copy(), s["syr"]
        self.last_out = None if s["last_out"] is None else {k: np.copy(v) for k, v in s["last_out"].items()}
        self.jumps = s["jumps"]
        self.rng.bit_generator.state = s["rng"]

    def set_command(self, vx: float, vy: float, yaw_rate: float):
        """Desired forward / lateral speed (m/s, heading frame) and yaw rate (rad/s)."""
        v = np.array([vx, vy], float)
        n = np.linalg.norm(v)
        if n > self.max_speed:
            v *= self.max_speed / n
        yr = float(np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))
        if np.linalg.norm(v - self.cmd_vel) > 0.3 or abs(yr - self.cmd_yaw_rate) > 0.3:
            self.force_search = True
        self.cmd_vel, self.cmd_yaw_rate = v, yr

    # --------------------------------------------------------------- helpers
    def _anchor(self, world_xy, world_yaw):
        i = self.db.index(self.clip, self.frame)
        self.anchor_world_xy = np.asarray(world_xy, float).copy()
        self.anchor_world_yaw = float(world_yaw)
        self.anchor_clip_xy = self.db.root_pos[i, :2].astype(float).copy()
        self.anchor_clip_yaw = float(self.db.yaw[i])

    def _world_root(self, i: int):
        """World xy / yaw / quat of database frame i under the current anchor."""
        dyaw = wrap_angle(self.anchor_world_yaw - self.anchor_clip_yaw)
        xy = self.anchor_world_xy + rot2d(dyaw) @ (self.db.root_pos[i, :2] - self.anchor_clip_xy)
        yaw = wrap_angle(self.db.yaw[i] + dyaw)
        quat = quat_mul_wxyz(yaw_quat_wxyz(dyaw), self.db.root_quat[i].astype(float))
        return xy, yaw, quat

    def _predict_path(self):
        """Predicted root xy and heading at the lookahead frames, in the current heading frame."""
        v, yr = self.sim_vel.copy(), self.sim_yaw_rate
        p, th = np.zeros(2), 0.0
        out_p, out_d = [], []
        ks = set(int(k) for k in self.db.lookahead)
        for k in range(1, max(ks) + 1):
            a = 1.0 - np.exp(-self.dt / self.tau_vel)
            v = v + (self.cmd_vel - v) * a
            yr = yr + (self.cmd_yaw_rate - yr) * (1.0 - np.exp(-self.dt / self.tau_yaw))
            th += yr * self.dt
            p = p + rot2d(th) @ v * self.dt
            if k in ks:
                out_p.append(p.copy())
                out_d.append(np.array([np.cos(th), np.sin(th)]))
        return np.concatenate(out_p), np.concatenate(out_d)

    def _query(self, i: int) -> np.ndarray:
        f = self.db.feat[i].astype(float).copy()
        pp, pd = self._predict_path()
        f[3:9], f[9:15] = pp, pd
        return f

    def _search(self, i_cur: int, must_jump: bool):
        db = self.db
        q = self._query(i_cur)
        cost = db.cost(q)
        j = int(np.argmin(cost))
        best = int(db.valid_idx[j])
        best_cost = float(cost[j])
        # cost of simply continuing
        cur_cost = float("inf")
        if db.valid[i_cur]:
            pos = np.searchsorted(db.valid_idx, i_cur)
            cur_cost = float(cost[pos])
        self.last_cost = best_cost
        same_clip = db.clip_id[best] == self.clip
        near = same_clip and abs(int(db.frame[best]) - self.frame) <= self.same_clip_window
        if near and not must_jump:
            return
        if not must_jump and self.blend_k < self.min_play:
            return
        if must_jump:
            if same_clip and int(db.frame[best]) <= self.frame:
                # exclude frames behind us in a clip that is running out
                mask = (db.clip_id[db.valid_idx] == self.clip) & (db.frame[db.valid_idx] <= self.frame)
                cost = np.where(mask, np.inf, cost)
                j = int(np.argmin(cost))
                best = int(db.valid_idx[j])
        elif not (best_cost < cur_cost * (1.0 - self.hysteresis)):
            return
        self._jump_to(best)

    def _jump_to(self, i_new: int):
        db = self.db
        i_old = db.index(self.clip, self.frame)
        xy, yaw, _ = self._world_root(i_old)
        # pose we would have emitted from the old clip (with its remaining blend)
        old_q = db.joint_pos[i_old].astype(float) + self._blend_scale() * self.blend_dq
        old_z = float(db.root_pos[i_old, 2]) + self._blend_scale() * self.blend_dz
        self.clip, self.frame = int(db.clip_id[i_new]), int(db.frame[i_new])
        self._anchor(xy, yaw)
        self.blend_dq = old_q - db.joint_pos[i_new].astype(float)
        self.blend_dz = old_z - float(db.root_pos[i_new, 2])
        self.blend_k = 0
        self.jumps += 1

    def _blend_scale(self) -> float:
        if self.blend_k >= self.blend_frames:
            return 0.0
        t = self.blend_k / self.blend_frames
        return float((1.0 - t) ** 2 * (1.0 + 2.0 * t))  # smooth decay 1 -> 0

    # ------------------------------------------------------------------ step
    def step(self) -> dict:
        """Advance one control tick and return the reference frame to stream."""
        db = self.db
        # smooth the command
        self.sim_vel += (self.cmd_vel - self.sim_vel) * (1.0 - np.exp(-self.dt / self.tau_vel))
        self.sim_yaw_rate += (self.cmd_yaw_rate - self.sim_yaw_rate) * (1.0 - np.exp(-self.dt / self.tau_yaw))
        i = db.index(self.clip, self.frame)
        remaining = int(db.clip_len[self.clip]) - self.frame
        must = remaining <= 15
        if must or self.force_search or self.since_search >= self.search_interval:
            self._search(i, must)
            self.since_search = 0
            self.force_search = False
            i = db.index(self.clip, self.frame)
        # emit
        s = self._blend_scale()
        xy, yaw, quat = self._world_root(i)
        joint = db.joint_pos[i].astype(float) + s * self.blend_dq
        z = float(db.root_pos[i, 2]) + s * self.blend_dz
        out = dict(joint_pos=joint, root_pos=np.array([xy[0], xy[1], z]), root_quat=quat, yaw=yaw,
                   clip=self.clip, frame=self.frame, tick=self.tick)
        # advance
        self.frame += 1
        self.blend_k += 1
        self.since_search += 1
        self.tick += 1
        self.last_out = out
        return out

    def describe(self) -> str:
        return (f"clip {self.db.clip_names[self.clip]} f{self.frame}/{int(self.db.clip_len[self.clip])} "
                f"jumps {self.jumps} cost {self.last_cost:.2f} cmd v=({self.cmd_vel[0]:.2f},{self.cmd_vel[1]:.2f}) "
                f"w={self.cmd_yaw_rate:.2f}")
