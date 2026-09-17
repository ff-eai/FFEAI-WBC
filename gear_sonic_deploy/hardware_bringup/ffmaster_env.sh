#!/usr/bin/env bash
# FF Master deployment environment for the robot dev board (PC2 / 10.0.1.41).
# Firmware v1.0.x  (board flashed 2026-08-12; previous version was v0.9.6).
#
# Source this before any ros2 / colcon / build command on the robot.
# It is kept in the repo deliberately: a firmware flash wipes $HOME, and this
# file was lost in the v1.0 flash.
#
# Changes vs the v0.9.6 version of this file:
#   * python packages moved to  local/lib/python3.10/dist-packages
#     (was lib/python3.10/site-packages) — this is what broke `ros2 service call`
#     with "The passed service type is invalid" after the flash.
#   * aimdk_msgs no longer needs building by us: /agibot/software/common now
#     ships share/ + lib/ + cmake/, so it works directly as an AMENT/CMAKE
#     prefix. (On v0.9.6 the shipped SDK did not match the firmware and we had
#     to build our own copy.)
#   * JointState gained `uint16 error_code` and lost coil/motor temps — anything
#     built against v0.9.6 messages MUST be rebuilt.

source /opt/ros/humble/setup.bash

# ---- robot-native aimdk messages (ground truth for the running firmware) ----
# Two prefixes, deliberately:
#   common/            = what the RUNNING vendor stack uses. Ships msg defs +
#                        runtime .so + python, but NO cmake package config.
#   aimdk/.../prebuilt = the SDK shipped with this firmware. Has cmake config +
#                        C++ headers + libs, so it is what we COMPILE against.
# Verified 2026-08-12: JointState.msg and JointCommand.msg are md5-identical
# between the two, so compiling against prebuilt is consistent with the runtime.
# (On v0.9.6 they did NOT match, which is why we used to build our own copy.)
# Re-check this md5 equality after any future firmware flash before trusting it.
export AIMDK_PREBUILT=/agibot/software/aimdk/src/aimdk_msgs/prebuilt_aarch64
export AMENT_PREFIX_PATH=$AIMDK_PREBUILT:/agibot/software/common:$AMENT_PREFIX_PATH
export CMAKE_PREFIX_PATH=$AIMDK_PREBUILT:/agibot/software/common:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/agibot/software/common/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/agibot/software/common/local/lib/python3.10/dist-packages:$PYTHONPATH

# ---- ONNX Runtime ----
# The deploy binary is LINKED against libonnxruntime.so (used by the locomotion
# planner). It will not start without it, even though the SONIC encoder/decoder
# run through TensorRT, not ORT. cmake locates it via onnxruntime_ROOT.
export onnxruntime_ROOT=$HOME/.local/onnxruntime
export LD_LIBRARY_PATH=$onnxruntime_ROOT/lib:$LD_LIBRARY_PATH

# ---- msgpack headers for the deploy binary build ----
# msgpack is header-only and is NOT installed system-wide on the robot (we ship
# it in thirdparty_headers/ precisely to avoid a system install).
#   CPATH             -> the compiler finds the headers
#   CMAKE_INCLUDE_PATH-> cmake's find_path() finds them. REQUIRED: the project's
#                        find_path(MSGPACK_INCLUDE_DIR msgpack.hpp ...) searches
#                        only /usr/include and /usr/local/include and ignores
#                        CPATH, so without this the configure step dies with
#                        "msgpack not found. Install libmsgpack-dev".
export CPATH=$HOME/sonic_deployment/thirdparty_headers:$CPATH
export CMAKE_INCLUDE_PATH=$HOME/sonic_deployment/thirdparty_headers:$CMAKE_INCLUDE_PATH

# ---- quieten FastDDS discovery spam (errors still shown) ----
export FASTDDS_LOG_LEVEL=Error

# ---- system state helpers (v1.0: Develop_MC replaces `aima em stop-app mc`) ----
# Valid states: Ready | Develop_Nav | Develop_Audio_Linux | Develop_Audio_ROS | Develop_MC
#   ffmaster_state          -> print current state
#   ffmaster_state <Name>   -> migrate to <Name>   (CHANGES ROBOT STATE — gantry first)
ffmaster_state() {
  if [ -z "$1" ]; then
    ros2 service call /aimdk_5Fmsgs/srv/GetSystemState aimdk_msgs/srv/GetSystemState "{}" \
      2>/dev/null | grep -oE "cur_state='[^']*'"
  else
    ros2 service call /aimdk_5Fmsgs/srv/MigrateSystemState \
      aimdk_msgs/srv/MigrateSystemState "{state: '$1'}" 2>/dev/null | tail -3
  fi
}

# ⚠ DO NOT source this file in the terminal that runs ffmaster_deploy_onnx_ref.
# Sourcing ROS puts ROS's libddsc.so.0 ahead of the unitree SDK's CycloneDDS and
# causes heap corruption. The binary gets its own LD_LIBRARY_PATH instead.

echo "ffmaster_env: ROS humble | aimdk_msgs=/agibot/software/common | ORT=$onnxruntime_ROOT"
