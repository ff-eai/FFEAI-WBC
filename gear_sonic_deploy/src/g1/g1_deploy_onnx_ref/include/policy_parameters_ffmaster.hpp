/**
 * @file policy_parameters_ffmaster.hpp
 * @brief Motor constants, PD gains, joint mappings, action scales, and default
 *        standing angles for the FF Master 29-DOF policy.
 *
 * Selected by building with -DROBOT_FFMASTER (see policy_parameters.hpp dispatch).
 *
 * ## Joint Ordering
 *
 * MuJoCo/hardware order follows the FF Master MJCF kinematic tree
 * (gear_sonic/data/assets/robot_description/mjcf/ffmaster_sonic_29dof.xml):
 *   0-5   left leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
 *   6-11  right leg
 *   12-14 waist     (yaw, pitch, roll)          <-- G1 is yaw, roll, pitch
 *   15-21 left arm  (sh_pitch, sh_roll, sh_yaw, elbow, wrist_yaw, wrist_pitch, wrist_roll)
 *   22-28 right arm                              <-- G1 wrists are roll, pitch, yaw
 *
 * The 29-DOF FF Master kinematic tree is topologically identical to G1's, so the
 * isaaclab<->mujoco permutation arrays are numerically identical to the G1
 * tables (verified against FFMASTER_ISAACLAB_TO_MUJOCO_DOF in
 * gear_sonic/envs/manager_env/robots/ffmaster.py); only per-slot joint semantics
 * differ (waist and wrist axis order).
 *
 * ## Gains
 *
 * kp/kd are the vendor aimdk RL gains used verbatim by the Isaac Lab
 * training run (FFMASTER_CFG actuator stiffness/damping in robots/ffmaster.py) — the
 * deploy PD must match the training actuator model.
 *
 * ## Action Scaling
 *
 * action_scale = 0.25 x effort_limit / stiffness per joint, matching
 * FFMASTER_ACTION_SCALE used by training (robots/ffmaster.py).
 * Final joint target: target = action x action_scale + default_angle.
 */

#ifndef POLICY_PARAMETERS_FFMASTER_HPP
#define POLICY_PARAMETERS_FFMASTER_HPP

#include <array>
#include <vector>

const double ONE_DEGREE = 0.0174533;  ///< One degree in radians.

// FF Master vendor effort limits (MJCF ctrlrange, aimdk deploy values)
const double FFMASTER_EFFORT_HIP_KNEE = 118.0;
const double FFMASTER_EFFORT_WAIST_PR = 48.0;
const double FFMASTER_EFFORT_ANKLE_P = 36.0;
const double FFMASTER_EFFORT_ANKLE_R = 24.0;
const double FFMASTER_EFFORT_SHOULDER_P_R = 36.0;
const double FFMASTER_EFFORT_SHOULDER_Y_ELBOW = 24.0;
const double FFMASTER_EFFORT_WRIST_YAW = 24.0;
const double FFMASTER_EFFORT_WRIST_PR = 2.2;  // deploy clamp (joint spec is 4.8)

// FF Master vendor stiffness (kp) per joint group
const double FFMASTER_KP_HIP_PITCH = 120.0;
const double FFMASTER_KP_HIP_ROLL_YAW = 100.0;
const double FFMASTER_KP_KNEE = 150.0;
const double FFMASTER_KP_ANKLE = 40.0;
const double FFMASTER_KP_WAIST_YAW = 40.1792;
const double FFMASTER_KP_WAIST_PR = 200.0;
const double FFMASTER_KP_SHOULDER_ELBOW = 50.0;
const double FFMASTER_KP_WRIST = 20.0;

// FF Master vendor damping (kd) per joint group
const double FFMASTER_KD_HIP_PITCH = 5.0;
const double FFMASTER_KD_HIP_ROLL_YAW = 4.0;
const double FFMASTER_KD_KNEE = 5.0;
const double FFMASTER_KD_ANKLE = 2.0;
const double FFMASTER_KD_WAIST_YAW = 2.5579;
const double FFMASTER_KD_WAIST_PR = 2.0;
const double FFMASTER_KD_SHOULDER_ELBOW = 3.0;
const double FFMASTER_KD_WRIST = 2.0;

// VR5Point index (isaaclab BODY index into the FF Master 32-body list, robots/ffmaster.py
// FFMASTER_ISAACLAB_JOINTS): left wrist_roll_link, right wrist_roll_link, pelvis,
// left ankle_roll_link, right ankle_roll_link.
// Matches training reward_point_body in sonic_ffmaster.yaml (pelvis + wrist_roll +
// ankle_roll). Note the welded head links occupy indices 12 and 17.
const std::array<int, 5> vr_5point_index = {30, 31, 0, 20, 21};

// VR3Point index (isaaclab BODY index): left wrist_roll_link,
// right wrist_roll_link, torso_link — matches vr_3point_body in sonic_ffmaster.yaml.
const std::array<int, 3> vr_3point_index = {30, 31, 9};

// Upper body joint index (mujoco order: waist y/p/r, L arm, R arm)
const std::vector<int> upper_body_joint_mujoco_order_in_isaaclab_index = { 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28};
const std::vector<int> upper_body_joint_mujoco_order_in_mujoco_index = { 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};

// upper body joint index (isaaclab order)
const std::vector<int> upper_body_joint_isaaclab_order_in_isaaclab_index = { 2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
const std::vector<int> upper_body_joint_isaaclab_order_in_mujoco_index = { 12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};

// wrist joint index (mujoco order: L yaw/pitch/roll, R yaw/pitch/roll)
const std::vector<int> wrist_joint_mujoco_order_in_isaaclab_index = {23, 25, 27, 24, 26, 28};
const std::vector<int> wrist_joint_mujoco_order_in_mujoco_index = {19, 20, 21, 26, 27, 28};

// wrist joint index (isaaclab order: L/R yaw, L/R pitch, L/R roll)
const std::vector<int> wrist_joint_isaaclab_order_in_isaaclab_index = {23, 24, 25, 26, 27, 28};
const std::vector<int> wrist_joint_isaaclab_order_in_mujoco_index = {19, 26, 20, 27, 21, 28};

// lower body joint index (mujoco order)
const std::vector<int> lower_body_joint_mujoco_order_in_isaaclab_index = {0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18};
const std::vector<int> lower_body_joint_mujoco_order_in_mujoco_index = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

// lower body joint index (isaaclab order)
const std::vector<int> lower_body_joint_isaaclab_order_in_isaaclab_index = {0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18};
const std::vector<int> lower_body_joint_isaaclab_order_in_mujoco_index = {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};

// Joint mapping arrays (mujoco order in isaaclab index) — identical to G1
// (topologically identical 29-DOF trees), verified against
// FFMASTER_ISAACLAB_TO_MUJOCO_DOF / FFMASTER_MUJOCO_TO_ISAACLAB_DOF in robots/ffmaster.py.
const std::array<int, 29> isaaclab_to_mujoco = {0,  3,  6,  9,  13, 17, 1,  4,  7,  10, 14, 18, 2,  5, 8,
                                                11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28};
// Joint mapping arrays  (isaaclab order in mujoco index)
const std::array<int, 29> mujoco_to_isaaclab = {0,  6,  12, 1,  7,  13, 2,  8,  14, 3,  9,  15, 22, 4, 10,
                                                16, 23, 5,  11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};

// Action scaling parameters (FF Master MuJoCo order)
// Computed using: action_scale = 0.25 * effort_limit / stiffness
// Matches FFMASTER_ACTION_SCALE from gear_sonic/envs/manager_env/robots/ffmaster.py
const std::array<double, 29> g1_action_scale = {
    0.25 * FFMASTER_EFFORT_HIP_KNEE / FFMASTER_KP_HIP_PITCH,          // left_hip_pitch_joint
    0.25 * FFMASTER_EFFORT_HIP_KNEE / FFMASTER_KP_HIP_ROLL_YAW,       // left_hip_roll_joint
    0.25 * FFMASTER_EFFORT_HIP_KNEE / FFMASTER_KP_HIP_ROLL_YAW,       // left_hip_yaw_joint
    0.25 * FFMASTER_EFFORT_HIP_KNEE / FFMASTER_KP_KNEE,               // left_knee_joint
    0.25 * FFMASTER_EFFORT_ANKLE_P / FFMASTER_KP_ANKLE,               // left_ankle_pitch_joint
    0.25 * FFMASTER_EFFORT_ANKLE_R / FFMASTER_KP_ANKLE,               // left_ankle_roll_joint
    0.25 * FFMASTER_EFFORT_HIP_KNEE / FFMASTER_KP_HIP_PITCH,          // right_hip_pitch_joint
    0.25 * FFMASTER_EFFORT_HIP_KNEE / FFMASTER_KP_HIP_ROLL_YAW,       // right_hip_roll_joint
    0.25 * FFMASTER_EFFORT_HIP_KNEE / FFMASTER_KP_HIP_ROLL_YAW,       // right_hip_yaw_joint
    0.25 * FFMASTER_EFFORT_HIP_KNEE / FFMASTER_KP_KNEE,               // right_knee_joint
    0.25 * FFMASTER_EFFORT_ANKLE_P / FFMASTER_KP_ANKLE,               // right_ankle_pitch_joint
    0.25 * FFMASTER_EFFORT_ANKLE_R / FFMASTER_KP_ANKLE,               // right_ankle_roll_joint
    0.25 * FFMASTER_EFFORT_HIP_KNEE / FFMASTER_KP_WAIST_YAW,          // waist_yaw_joint
    0.25 * FFMASTER_EFFORT_WAIST_PR / FFMASTER_KP_WAIST_PR,           // waist_pitch_joint
    0.25 * FFMASTER_EFFORT_WAIST_PR / FFMASTER_KP_WAIST_PR,           // waist_roll_joint
    0.25 * FFMASTER_EFFORT_SHOULDER_P_R / FFMASTER_KP_SHOULDER_ELBOW, // left_shoulder_pitch_joint
    0.25 * FFMASTER_EFFORT_SHOULDER_P_R / FFMASTER_KP_SHOULDER_ELBOW, // left_shoulder_roll_joint
    0.25 * FFMASTER_EFFORT_SHOULDER_Y_ELBOW / FFMASTER_KP_SHOULDER_ELBOW, // left_shoulder_yaw_joint
    0.25 * FFMASTER_EFFORT_SHOULDER_Y_ELBOW / FFMASTER_KP_SHOULDER_ELBOW, // left_elbow_joint
    0.25 * FFMASTER_EFFORT_WRIST_YAW / FFMASTER_KP_WRIST,             // left_wrist_yaw_joint
    0.25 * FFMASTER_EFFORT_WRIST_PR / FFMASTER_KP_WRIST,              // left_wrist_pitch_joint
    0.25 * FFMASTER_EFFORT_WRIST_PR / FFMASTER_KP_WRIST,              // left_wrist_roll_joint
    0.25 * FFMASTER_EFFORT_SHOULDER_P_R / FFMASTER_KP_SHOULDER_ELBOW, // right_shoulder_pitch_joint
    0.25 * FFMASTER_EFFORT_SHOULDER_P_R / FFMASTER_KP_SHOULDER_ELBOW, // right_shoulder_roll_joint
    0.25 * FFMASTER_EFFORT_SHOULDER_Y_ELBOW / FFMASTER_KP_SHOULDER_ELBOW, // right_shoulder_yaw_joint
    0.25 * FFMASTER_EFFORT_SHOULDER_Y_ELBOW / FFMASTER_KP_SHOULDER_ELBOW, // right_elbow_joint
    0.25 * FFMASTER_EFFORT_WRIST_YAW / FFMASTER_KP_WRIST,             // right_wrist_yaw_joint
    0.25 * FFMASTER_EFFORT_WRIST_PR / FFMASTER_KP_WRIST,              // right_wrist_pitch_joint
    0.25 * FFMASTER_EFFORT_WRIST_PR / FFMASTER_KP_WRIST,              // right_wrist_roll_joint
};

// PD control gains - Position gains (Kp), FF Master MuJoCo order
const std::array<float, 29> kps = {
    FFMASTER_KP_HIP_PITCH,    // left_hip_pitch_joint
    FFMASTER_KP_HIP_ROLL_YAW, // left_hip_roll_joint
    FFMASTER_KP_HIP_ROLL_YAW, // left_hip_yaw_joint
    FFMASTER_KP_KNEE,         // left_knee_joint
    FFMASTER_KP_ANKLE,        // left_ankle_pitch_joint
    FFMASTER_KP_ANKLE,        // left_ankle_roll_joint
    FFMASTER_KP_HIP_PITCH,    // right_hip_pitch_joint
    FFMASTER_KP_HIP_ROLL_YAW, // right_hip_roll_joint
    FFMASTER_KP_HIP_ROLL_YAW, // right_hip_yaw_joint
    FFMASTER_KP_KNEE,         // right_knee_joint
    FFMASTER_KP_ANKLE,        // right_ankle_pitch_joint
    FFMASTER_KP_ANKLE,        // right_ankle_roll_joint
    FFMASTER_KP_WAIST_YAW,    // waist_yaw_joint
    FFMASTER_KP_WAIST_PR,     // waist_pitch_joint
    FFMASTER_KP_WAIST_PR,     // waist_roll_joint
    FFMASTER_KP_SHOULDER_ELBOW, // left_shoulder_pitch_joint
    FFMASTER_KP_SHOULDER_ELBOW, // left_shoulder_roll_joint
    FFMASTER_KP_SHOULDER_ELBOW, // left_shoulder_yaw_joint
    FFMASTER_KP_SHOULDER_ELBOW, // left_elbow_joint
    FFMASTER_KP_WRIST,        // left_wrist_yaw_joint
    FFMASTER_KP_WRIST,        // left_wrist_pitch_joint
    FFMASTER_KP_WRIST,        // left_wrist_roll_joint
    FFMASTER_KP_SHOULDER_ELBOW, // right_shoulder_pitch_joint
    FFMASTER_KP_SHOULDER_ELBOW, // right_shoulder_roll_joint
    FFMASTER_KP_SHOULDER_ELBOW, // right_shoulder_yaw_joint
    FFMASTER_KP_SHOULDER_ELBOW, // right_elbow_joint
    FFMASTER_KP_WRIST,        // right_wrist_yaw_joint
    FFMASTER_KP_WRIST,        // right_wrist_pitch_joint
    FFMASTER_KP_WRIST,        // right_wrist_roll_joint
};

// PD control gains - Derivative gains (Kd), FF Master MuJoCo order
const std::array<float, 29> kds = {
    FFMASTER_KD_HIP_PITCH,    // left_hip_pitch_joint
    FFMASTER_KD_HIP_ROLL_YAW, // left_hip_roll_joint
    FFMASTER_KD_HIP_ROLL_YAW, // left_hip_yaw_joint
    FFMASTER_KD_KNEE,         // left_knee_joint
    FFMASTER_KD_ANKLE,        // left_ankle_pitch_joint
    FFMASTER_KD_ANKLE,        // left_ankle_roll_joint
    FFMASTER_KD_HIP_PITCH,    // right_hip_pitch_joint
    FFMASTER_KD_HIP_ROLL_YAW, // right_hip_roll_joint
    FFMASTER_KD_HIP_ROLL_YAW, // right_hip_yaw_joint
    FFMASTER_KD_KNEE,         // right_knee_joint
    FFMASTER_KD_ANKLE,        // right_ankle_pitch_joint
    FFMASTER_KD_ANKLE,        // right_ankle_roll_joint
    FFMASTER_KD_WAIST_YAW,    // waist_yaw_joint
    FFMASTER_KD_WAIST_PR,     // waist_pitch_joint
    FFMASTER_KD_WAIST_PR,     // waist_roll_joint
    FFMASTER_KD_SHOULDER_ELBOW, // left_shoulder_pitch_joint
    FFMASTER_KD_SHOULDER_ELBOW, // left_shoulder_roll_joint
    FFMASTER_KD_SHOULDER_ELBOW, // left_shoulder_yaw_joint
    FFMASTER_KD_SHOULDER_ELBOW, // left_elbow_joint
    FFMASTER_KD_WRIST,        // left_wrist_yaw_joint
    FFMASTER_KD_WRIST,        // left_wrist_pitch_joint
    FFMASTER_KD_WRIST,        // left_wrist_roll_joint
    FFMASTER_KD_SHOULDER_ELBOW, // right_shoulder_pitch_joint
    FFMASTER_KD_SHOULDER_ELBOW, // right_shoulder_roll_joint
    FFMASTER_KD_SHOULDER_ELBOW, // right_shoulder_yaw_joint
    FFMASTER_KD_SHOULDER_ELBOW, // right_elbow_joint
    FFMASTER_KD_WRIST,        // right_wrist_yaw_joint
    FFMASTER_KD_WRIST,        // right_wrist_pitch_joint
    FFMASTER_KD_WRIST,        // right_wrist_roll_joint
};

// Default joint angles (standing pose) — FF Master training init pose
// (FFMASTER_CFG.init_state in robots/ffmaster.py; aimdk vendor default_dof_pos).
// NOTE: FF Master elbow default is -0.3 (elbow range [-2.3556, 0]), not G1's +0.6.
const std::array<double, 29> default_angles = {
    -0.312, // left_hip_pitch_joint
    0.0,    // left_hip_roll_joint
    0.0,    // left_hip_yaw_joint
    0.669,  // left_knee_joint
    -0.363, // left_ankle_pitch_joint
    0.0,    // left_ankle_roll_joint
    -0.312, // right_hip_pitch_joint
    0.0,    // right_hip_roll_joint
    0.0,    // right_hip_yaw_joint
    0.669,  // right_knee_joint
    -0.363, // right_ankle_pitch_joint
    0.0,    // right_ankle_roll_joint
    0.0,    // waist_yaw_joint
    0.0,    // waist_pitch_joint
    0.0,    // waist_roll_joint
    0.2,    // left_shoulder_pitch_joint
    0.2,    // left_shoulder_roll_joint
    0.0,    // left_shoulder_yaw_joint
    -0.3,   // left_elbow_joint
    0.0,    // left_wrist_yaw_joint
    0.0,    // left_wrist_pitch_joint
    0.0,    // left_wrist_roll_joint
    0.2,    // right_shoulder_pitch_joint
    -0.2,   // right_shoulder_roll_joint
    0.0,    // right_shoulder_yaw_joint
    -0.3,   // right_elbow_joint
    0.0,    // right_wrist_yaw_joint
    0.0,    // right_wrist_pitch_joint
    0.0,    // right_wrist_roll_joint
};

#endif // POLICY_PARAMETERS_FFMASTER_HPP
