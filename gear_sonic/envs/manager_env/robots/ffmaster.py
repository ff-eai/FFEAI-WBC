# FF Master robot configuration (29 DOF: head_yaw/head_pitch locked).
#
# Assets: ffmaster_sonic_29dof.{urdf,xml} — generated copies of the vendor
# ffmaster assets with sensor/base fixed links stripped (all massless), head
# joints fixed (URDF) / removed (MJCF), and the URDF pelvis inertial injected
# into the MJCF. Originals are preserved untouched.
#
# Gains, default pose, and effort limits come from the vendor's deployed RL stack
# (aimdk SDK: extra/ffmaster_rl_deploy/ffmaster_rl_deploy_controller/config/motion_control.yaml
# and ffmaster_rl_deploy_mujoco .../model_info/ffmaster.xml):
#   - kp/kd: vendor rl_config values, validated on real hardware at 50 Hz.
#   - armature 0.03: the value the vendor's own MuJoCo deploy sim uses for every
#     joint (no per-motor rotor inertia is published anywhere).
#   - effort limits: vendor ctrlrange (118 not 120 for the big motors; wrist
#     pitch/roll clamped to 2.2 even though the joint spec is 4.8).
#   - standing pose: vendor default_dof_pos; height 0.641 computed by MuJoCo FK
#     (lowest foot mesh vertex at that pose) plus small clearance.
#
# Ordering arrays derived from the URDF/MJCF trees with the BFS rule that
# exactly reproduces the hand-written G1 and H2 tables, and cross-checked
# against the vendor RL obs sequence (identical). The IsaacLab body list is a
# PREDICTION for merge_fixed_joints=False imports — verify against
# articulation.body_names on first spawn, since G1/H2 only validate the
# merge_fixed_joints=True path.

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

# 32 bodies: 30 articulated + head_yaw/head_pitch links welded to torso.
# Order verified against articulation.body_names from an actual Isaac Lab spawn
# (2026-07-21): under merge_fixed_joints=False, fixed-joint children come BEFORE
# movable children at the same tree depth (head_yaw at 12, head_pitch at 17).
FFMASTER_ISAACLAB_JOINTS = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "head_yaw_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "head_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
]

# 29 DOF (head excluded). Same values as G1's map: the 29-DOF kinematic trees
# are topologically identical.
FFMASTER_ISAACLAB_TO_MUJOCO_DOF = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]

FFMASTER_MUJOCO_TO_ISAACLAB_DOF = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]

FFMASTER_ISAACLAB_TO_MUJOCO_BODY = [
    0,
    1,
    4,
    7,
    10,
    15,
    20,
    2,
    5,
    8,
    11,
    16,
    21,
    3,
    6,
    9,
    13,
    18,
    22,
    24,
    26,
    28,
    30,
    14,
    19,
    23,
    25,
    27,
    29,
    31,
    12,
    17,
]

FFMASTER_MUJOCO_TO_ISAACLAB_BODY = [
    0,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    4,
    10,
    30,
    16,
    23,
    5,
    11,
    31,
    17,
    24,
    6,
    12,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
    22,
    29,
]

FFMASTER_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": FFMASTER_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": FFMASTER_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": FFMASTER_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": FFMASTER_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": FFMASTER_MUJOCO_TO_ISAACLAB_BODY,
}

FFMASTER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        # Keep the fixed head links as articulation bodies so the IsaacLab body
        # set matches the MJCF (head bodies welded there). The URDF copy has no
        # other fixed joints left.
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        # Sphere-foot variant: foot mesh collision replaced by the 12-sphere sole
        # cluster copied verbatim from the MJCF (vendor deploy-sim geometry), so
        # Isaac training and MuJoCo sim2sim share the same foot contact regime.
        # Mesh-foot original kept alongside as ffmaster_sonic_29dof.urdf.
        asset_path=f"{ASSET_DIR}/robot_description/urdf/ffmaster/ffmaster_sonic_29dof_spherefeet.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.645),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": -0.3,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=118.0,
            velocity_limit_sim=11.936,
            stiffness={
                ".*_hip_pitch_joint": 120.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_yaw_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_pitch_joint": 5.0,
                ".*_hip_roll_joint": 4.0,
                ".*_hip_yaw_joint": 4.0,
                ".*_knee_joint": 5.0,
            },
            armature=0.03,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 36.0,
                ".*_ankle_roll_joint": 24.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 13.087,
                ".*_ankle_roll_joint": 15.077,
            },
            stiffness=40.0,
            damping=2.0,
            armature=0.03,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_pitch_joint", "waist_roll_joint"],
            effort_limit_sim=48.0,
            velocity_limit_sim=13.088,
            stiffness=200.0,
            damping=2.0,
            armature=0.03,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=118.0,
            velocity_limit_sim=11.936,
            stiffness=40.1792,
            damping=2.5579,
            armature=0.03,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
                ".*_wrist_yaw_joint": 24.0,
                ".*_wrist_pitch_joint": 2.2,
                ".*_wrist_roll_joint": 2.2,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 13.088,
                ".*_shoulder_roll_joint": 13.088,
                ".*_shoulder_yaw_joint": 15.077,
                ".*_elbow_joint": 15.077,
                ".*_wrist_yaw_joint": 15.077,
                ".*_wrist_pitch_joint": 4.188,
                ".*_wrist_roll_joint": 4.188,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 50.0,
                ".*_shoulder_roll_joint": 50.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
                ".*_wrist_yaw_joint": 20.0,
                ".*_wrist_pitch_joint": 20.0,
                ".*_wrist_roll_joint": 20.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 3.0,
                ".*_shoulder_roll_joint": 3.0,
                ".*_shoulder_yaw_joint": 3.0,
                ".*_elbow_joint": 3.0,
                ".*_wrist_yaw_joint": 2.0,
                ".*_wrist_pitch_joint": 2.0,
                ".*_wrist_roll_joint": 2.0,
            },
            armature=0.03,
        ),
    },
)

FFMASTER_ACTION_SCALE = {}
for a in FFMASTER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            FFMASTER_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
