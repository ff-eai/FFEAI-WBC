"""Export motion-lib PKLs into the deploy-PKL format consumed by
gear_sonic_deploy/reference/convert_motions.py.

The deploy stack plays reference motions from per-motion CSV folders
(joint_pos.csv etc. in IsaacLab DOF order at 50 fps, body_* CSVs for the 14
tracked bodies, metadata.txt with the tracked bodies' IsaacLab indexes).
convert_motions.py serializes those folders from a joblib dict ("deploy PKL");
this script produces that dict from ordinary motion-lib PKLs
(root_trans_offset / pose_aa / dof / root_rot / fps) by loading them through
MotionLibRobot, which also resamples to the 50 fps the 50 Hz control loop
expects.

Example (FF Master):
    python gear_sonic/data_process/export_deploy_reference.py \
        --robot ffmaster \
        --motion_dir data/ffmaster_motions/deploy_export_selection \
        --output data/ffmaster_motions/ffmaster_deploy_ref.pkl
    python3 gear_sonic_deploy/reference/convert_motions.py \
        data/ffmaster_motions/ffmaster_deploy_ref.pkl gear_sonic_deploy/reference/ffmaster
"""

import argparse
import ast
from pathlib import Path

import easydict
import joblib
import numpy as np

ROBOT_SPECS = {
    "g1": {
        "asset_file": "g1_29dof_rev_1_0.xml",
        "mapping_module": "gear_sonic.envs.manager_env.robots.g1",
        "mapping_attr": "G1_ISAACLAB_TO_MUJOCO_MAPPING",
        # commands.motion.body_names (gear_sonic/config/manager_env/commands/terms/motion.yaml)
        "body_names": [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    },
    "ffmaster": {
        "asset_file": "ffmaster_sonic_29dof.xml",
        "mapping_module": "gear_sonic.envs.manager_env.robots.ffmaster",
        "mapping_attr": "FFMASTER_ISAACLAB_TO_MUJOCO_MAPPING",
        # body_names override from config/exp/manager/universal_token/all_modes/sonic_ffmaster.yaml
        # (FF Master hand body is *_wrist_roll_link; G1's is *_wrist_yaw_link)
        "body_names": [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_roll_link",
        ],
    },
}


def load_mapping_without_isaaclab(module_name: str, mapping_attr: str) -> dict:
    """Extract the *_ISAACLAB_TO_MUJOCO_MAPPING dict from a robots/<r>.py module
    without importing it (the module imports isaaclab, which needs the Isaac Sim
    runtime). The mapping and the arrays it references are static literals, so
    they are recovered from the module's AST."""
    path = Path(__file__).resolve().parents[1] / (
        module_name.split("gear_sonic.", 1)[1].replace(".", "/") + ".py"
    )
    tree = ast.parse(path.read_text())
    literals = {}
    mapping_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id == mapping_attr:
                mapping_node = node.value
            else:
                try:
                    literals[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    if mapping_node is None:
        raise KeyError(f"{mapping_attr} not found in {path}")
    mapping = {}
    for key, value in zip(mapping_node.keys, mapping_node.values):
        mapping[ast.literal_eval(key)] = literals[value.id]
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=sorted(ROBOT_SPECS), default="ffmaster")
    parser.add_argument(
        "--motion_dir",
        required=True,
        help="Directory of motion-lib PKLs (globbed recursively) or a single PKL",
    )
    parser.add_argument("--output", required=True, help="Output deploy PKL path")
    parser.add_argument(
        "--target_fps",
        type=int,
        default=50,
        help="Playback fps of the deploy control loop (one frame per tick)",
    )
    parser.add_argument(
        "--smpl_dir",
        default=None,
        help="Directory of SMPL PKLs keyed <motion_key>.pkl (e.g. data/smpl_filtered). "
        "Adds a per-frame-canonicalized smpl_joint array so the deploy SMPL encoder "
        "mode can be used (the C++ gatherer copies the CSV raw, so the trainer's "
        "quat_apply(quat_inv(q_smpl_root), joints) is baked in here).",
    )
    args = parser.parse_args()

    spec = ROBOT_SPECS[args.robot]
    mapping = load_mapping_without_isaaclab(spec["mapping_module"], spec["mapping_attr"])
    body_idx = [mapping["isaaclab_joints"].index(n) for n in spec["body_names"]]
    print(f"{args.robot} tracked-body IsaacLab indexes: {body_idx}")

    from gear_sonic.utils.motion_lib import motion_lib_robot

    cfg = easydict.EasyDict(
        {
            "motion_file": args.motion_dir,
            "asset": {
                "assetRoot": "gear_sonic/data/assets/robot_description/mjcf/",
                "assetFileName": spec["asset_file"],
                "urdfFileName": "",
            },
            "extend_config": [],
            "target_fps": args.target_fps,
            "multi_thread": False,
            "smpl_motion_file": args.smpl_dir if args.smpl_dir else "dummy",
            "mujoco_to_isaaclab_dof": mapping["mujoco_to_isaaclab_dof"],
            "isaaclab_to_mujoco_dof": mapping["isaaclab_to_mujoco_dof"],
            "mujoco_to_isaaclab_body": mapping["mujoco_to_isaaclab_body"],
            "isaaclab_to_mujoco_body": mapping["isaaclab_to_mujoco_body"],
            "body_indexes_data": body_idx,
        }
    )
    lib = motion_lib_robot.MotionLibRobot(cfg, num_envs=1, device="cpu")
    lib.load_motions_for_training()

    starts = lib.length_starts.cpu().numpy()
    nfr = lib._motion_num_frames.cpu().numpy()

    def canonicalized_smpl_joints(s, num_frames):
        """Trainer-equivalent SMPL joints in the SMPL-root local frame.

        Replicates commands.py smpl_root_quat_w_multi_future (y-up -> z-up
        premultiply, remove_smpl_base_rot postmultiply) followed by
        observations.py smpl_joints_multi_future_local's per-frame
        quat_apply(quat_inv(q), joints). The joints tensor itself is used raw
        (root-relative, y-up) exactly as training does.
        """
        from scipy.spatial.transform import Rotation as R

        pose_aa = lib._motion_smpl_poses[s : s + num_frames].cpu().numpy()  # (n,72)
        joints = lib._motion_smpl_joints[s : s + num_frames].cpu().numpy()  # (n,24,3)
        q_root = R.from_rotvec(pose_aa[:, :3])
        rx90 = R.from_euler("x", 90, degrees=True)
        base = R.from_quat([0.5, 0.5, 0.5, 0.5])  # xyzw == wxyz for this value
        q = rx90 * q_root * base.inv()
        n = joints.shape[0]
        q_expanded = R.from_quat(np.repeat(q.as_quat(), 24, axis=0))
        local = q_expanded.inv().apply(joints.reshape(n * 24, 3)).reshape(n, 72)
        return local.astype(np.float32)

    out = {}
    for i, key in enumerate(lib.curr_motion_keys):
        s, num_frames = int(starts[i]), int(nfr[i])
        out[str(key)] = {
            "joint_pos": lib.dof_pos[s : s + num_frames].cpu().numpy().astype(np.float32),
            "joint_vel": lib.dof_vel[s : s + num_frames].cpu().numpy().astype(np.float32),
            "body_pos_w": lib.body_pos_w[s : s + num_frames].cpu().numpy().astype(np.float32),
            "body_quat_w": lib.body_quat_w[s : s + num_frames].cpu().numpy().astype(np.float32),
            "body_lin_vel_w": lib.body_lin_vel_w[s : s + num_frames].cpu().numpy().astype(np.float32),
            "body_ang_vel_w": lib.body_ang_vel_w[s : s + num_frames].cpu().numpy().astype(np.float32),
            "_body_indexes": np.array(body_idx, dtype=np.int64),
            "time_step_total": np.int64(num_frames),
        }
        has_smpl = bool(args.smpl_dir) and bool(lib.motion_has_smpl[i])
        if has_smpl:
            out[str(key)]["smpl_joint"] = canonicalized_smpl_joints(s, num_frames)
        print(
            f"  {key}: {num_frames} frames @ {args.target_fps} fps"
            + (" +smpl" if has_smpl else "")
        )

    joblib.dump(out, args.output, compress=True)
    print(f"Wrote {len(out)} motions to {args.output}")


if __name__ == "__main__":
    main()
