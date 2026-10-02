"""FF Master kinematic helpers shared by the planner tools.

Conventions (verified against gear_sonic/data_process/export_deploy_reference.py
output on the chingmu_ffmaster_loco set):
  * Motion-library PKLs: `dof` (T,29) in MuJoCo/MJCF joint order, `root_rot`
    (T,4) as x,y,z,w, `root_trans_offset` (T,3) metres, `fps` (30).
  * Deploy reference / ZMQ stream: joint_pos in IsaacLab order, body
    quaternions as w,x,y,z.
  * Foot bodies: left_ankle_roll_link / right_ankle_roll_link, pelvis = root.
"""

from __future__ import annotations

import ast
import os
import re

import mujoco
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MJCF = os.path.join(REPO, "gear_sonic/data/assets/robot_description/mjcf/ffmaster_sonic_29dof.xml")
ROBOT_PY = os.path.join(REPO, "gear_sonic/envs/manager_env/robots/ffmaster.py")

CONTROL_FPS = 50.0
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _table(name: str) -> list[int]:
    src = open(ROBOT_PY).read()
    m = re.search(name + r"\s*=\s*(\[[^\]]*\])", src, re.S)
    return ast.literal_eval(re.sub(r"#[^\n]*", "", m.group(1)))


# data_mj = data_isaac[ISAAC_TO_MJ];  data_isaac = data_mj[MJ_TO_ISAAC]
ISAAC_TO_MJ = np.array(_table("FFMASTER_ISAACLAB_TO_MUJOCO_DOF"))
MJ_TO_ISAAC = np.array(_table("FFMASTER_MUJOCO_TO_ISAACLAB_DOF"))


class FFMasterModel:
    """MuJoCo model of FF Master for forward kinematics and joint metadata."""

    def __init__(self, xml: str = MJCF):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joint_names_mj = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)
        ]
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES]
        self.joint_axis_mj = np.array([m.jnt_axis[j] for j in range(1, m.njnt)])
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, m.nbody)]
        swap = lambda nm: ("right_" + nm[5:]) if nm.startswith("left_") else (("left_" + nm[6:]) if nm.startswith("right_") else nm)  # noqa: E731
        self.body_perm = np.array([names.index(swap(nm)) for nm in names])

    def fk(self, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, dof_mj: np.ndarray):
        """Body positions of pelvis and both feet for one frame."""
        d = self.data
        d.qpos[0:3] = root_pos
        d.qpos[3:7] = root_quat_wxyz
        d.qpos[7:] = dof_mj
        mujoco.mj_kinematics(self.model, d)
        return d.xpos[self.pelvis].copy(), d.xpos[self.feet[0]].copy(), d.xpos[self.feet[1]].copy()

    def fk_all(self, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, dof_mj: np.ndarray):
        """Positions (B,3) and rotation matrices (B,3,3) of every body except world."""
        d = self.data
        d.qpos[0:3] = root_pos
        d.qpos[3:7] = root_quat_wxyz
        d.qpos[7:] = dof_mj
        mujoco.mj_kinematics(self.model, d)
        return d.xpos[1:].copy(), d.xmat[1:].reshape(-1, 3, 3).copy()

    @staticmethod
    def mirror_bodies(pos: np.ndarray, rot: np.ndarray, body_perm: np.ndarray):
        """Reflect a set of body poses through the x-z plane and swap left/right bodies."""
        M = np.diag([1.0, -1.0, 1.0])
        return pos[body_perm] * np.array([1.0, -1.0, 1.0]), M @ rot[body_perm] @ M

    def derive_mirror_map(self, angle: float = 0.3, tol: float = 5e-3):
        """Find the left/right joint mirror map by forward kinematics.

        For each joint j, apply `angle` on j alone and measure how every body
        moved relative to the rest pose (position delta and world-frame
        rotation increment). Reflect those increments through the x-z plane
        with left/right bodies swapped, and find the (partner, sign) that
        reproduces them. Increments make the test independent of how the
        left/right link frames are defined in the MJCF; orientations are needed
        because leaf joints (ankle roll, wrists) move no body origin.
        Returns (perm, sign, max_err) in MuJoCo joint order or None.
        """
        n = len(self.joint_names_mj)
        root = np.zeros(3)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        M = np.diag([1.0, -1.0, 1.0])
        rest_p, rest_r = self.fk_all(root, quat, np.zeros(n))

        def increments(q):
            p, r = self.fk_all(root, quat, q)
            return p - rest_p, r @ np.transpose(rest_r, (0, 2, 1))

        def reflect(dp, dr):
            return dp[self.body_perm] * np.array([1.0, -1.0, 1.0]), M @ dr[self.body_perm] @ M

        def err_between(a, b):
            return max(np.abs(a[0] - b[0]).max(), np.abs(a[1] - b[1]).max())

        probes = []
        for j in range(n):
            q = np.zeros(n)
            q[j] = angle
            probes.append(increments(q))
        perm, sign, max_err = np.zeros(n, int), np.zeros(n), 0.0
        for j in range(n):
            target = reflect(*probes[j])
            best = None
            for k in range(n):
                for s_ in (1.0, -1.0):
                    q = np.zeros(n)
                    q[k] = s_ * angle
                    e = err_between(increments(q), target)
                    if best is None or e < best[0]:
                        best = (e, k, s_)
            e, k, s_ = best
            if e > tol:
                return None
            perm[j], sign[j] = k, s_
            max_err = max(max_err, e)
        return perm, sign, max_err


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def yaw_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """Heading of the body x axis projected on the ground plane."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    fx = 1.0 - 2.0 * (y * y + z * z)
    fy = 2.0 * (x * y + w * z)
    return np.arctan2(fy, fx)


def yaw_quat_wxyz(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def rot2d(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def resample_clip(dof_mj: np.ndarray, root_pos: np.ndarray, root_quat_xyzw: np.ndarray,
                  fps_in: float, fps_out: float = CONTROL_FPS):
    """Linear resample of joints/root position and slerp of the root quaternion."""
    from scipy.spatial.transform import Rotation as R, Slerp

    T = len(dof_mj)
    t_in = np.arange(T) / fps_in
    n_out = int(np.floor((T - 1) / fps_in * fps_out)) + 1
    t_out = np.arange(n_out) / fps_out
    dof = np.stack([np.interp(t_out, t_in, dof_mj[:, j]) for j in range(dof_mj.shape[1])], axis=1)
    pos = np.stack([np.interp(t_out, t_in, root_pos[:, j]) for j in range(3)], axis=1)
    rots = R.from_quat(root_quat_xyzw)
    quat_xyzw = Slerp(t_in, rots)(t_out).as_quat()
    return dof.astype(np.float32), pos.astype(np.float32), quat_xyzw_to_wxyz(quat_xyzw).astype(np.float32)


def finite_diff(x: np.ndarray, fps: float) -> np.ndarray:
    """Forward difference like the deploy binary's ResampleGeneratedSequence50Hz."""
    v = np.empty_like(x)
    v[:-1] = (x[1:] - x[:-1]) * fps
    v[-1] = v[-2] if len(x) > 1 else 0.0
    return v
