# sonic_ffmaster_dummy

Single-joint (left elbow) dummy controller. Purpose: validate the aimdk ROS2
transport, the Develop_MC handoff, and the safety skeleton (bus-free gate,
staleness watchdog, damping-on-exit) with the minimum possible energy — before
the real SONIC deploy node reuses this exact skeleton.

Plan/context: `docs/agent_outputs/claude/2026-08-06_dummy_controller_plan.md`.

## What it does

- Subscribes `/aima/hal/joint/arm/state` only.
- Refuses to create its command publisher until **all four**
  `/aima/hal/joint/*/command` topics have zero publishers AND
  `/aima/mc/common/state` has zero publishers (i.e. Develop_MC is active).
  Both are DDS **graph** queries — no subscription, works identically in sim.
- Then: capture pose → 2 s hold (elbow kp 25, others kp 0/kd 1) →
  4 sine cycles ±0.2 rad on the elbow → damping exit.
- Aborts to the damp path on: stale state (>50 ms), foreign command publisher,
  MC state topic reappearing, Ctrl-C/SIGTERM, exception.
- Publishes the **full 14-element arm array** every tick, never other groups,
  never `/aima/mc/common/state`.

## Build — robot (.41, aarch64)

Against the robot's own firmware-matched messages (never a bundled SDK):

```bash
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH=/agibot/software/common:$AMENT_PREFIX_PATH   # or: source ~/sonic_deployment/ffmaster_env.sh
cd ~/sonic_deployment/ws            # ws/src/sonic_ffmaster_dummy = this package
colcon build --packages-select sonic_ffmaster_dummy
source install/setup.bash
ros2 run sonic_ffmaster_dummy dummy_elbow <path>/dummy.yaml
```

## Build — laptop (x86, sim rehearsal)

Against the v1.0 SDK x86_64 prebuilt (JointCommand identical across versions;
we only read version-stable JointState fields):

```bash
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH=$HOME/aimdk-aarch64-a424add7-artifacts/extra/ffmaster_rl_deploy/aimdk_msgs/prebuilt_x86_64:$AMENT_PREFIX_PATH
colcon build --packages-select sonic_ffmaster_dummy
```

Rehearse against the vendor MuJoCo sim (`ffmaster_rl_deploy_mujoco/bin/start_sim.sh -s`,
model `0: lx2501_3_t2d5`), which serves the identical `/aima/hal/*` topics.
Required rehearsal cases: normal run; kill the sim mid-sine (staleness → damp);
Ctrl-C mid-sine (damp before shutdown — watch the last messages on
`ros2 topic echo /aima/hal/joint/arm/command`).

## Safety notes (hardware)

- Only under Develop_MC, robot gantried, entered from PASSIVE.
- Gamepad soft-stops are DEAD while native MC is disabled — stop chain is this
  node's damp path, then battery e-stop, then gantry.
- kp on the test joint is 25 (config), deliberately below the production 50.
