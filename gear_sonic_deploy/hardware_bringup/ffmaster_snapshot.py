#!/usr/bin/env python3
"""One-shot FF Master state snapshot: system state, bus ownership, IMU, all 29 joints.

Usage (on the robot, after `source ~/sonic_deployment/ffmaster_env.sh`):
    python3 ~/sonic_deployment/ffmaster_snapshot.py "BEFORE"

Reads only — never publishes. Safe at any time.

The SYSTEM block exists because `Develop_MC` was observed reverting to
`Business` on its own (2026-08-13), taking the command topics back with it.
A snapshot without it can look perfectly healthy while the MC has quietly
resumed ownership of the bus.
"""
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from aimdk_msgs.msg import JointStateArray
from sensor_msgs.msg import Imu

LEG = ["L_hip_p", "L_hip_r", "L_hip_y", "L_knee", "L_ank_p", "L_ank_r",
       "R_hip_p", "R_hip_r", "R_hip_y", "R_knee", "R_ank_p", "R_ank_r"]
WAIST = ["waist_yaw", "waist_pitch", "waist_roll"]
ARM = ["L_sho_p", "L_sho_r", "L_sho_y", "L_elb", "L_wr_y", "L_wr_p", "L_wr_r",
       "R_sho_p", "R_sho_r", "R_sho_y", "R_elb", "R_wr_y", "R_wr_p", "R_wr_r"]
CMD_TOPICS = ["leg", "waist", "arm", "head"]
MC_STATE_TOPIC = "/aima/mc/common/state"

QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                 durability=DurabilityPolicy.VOLATILE,
                 history=HistoryPolicy.KEEP_LAST)


def system_state(node, timeout=6.0):
    """Current SM state, or a marker string. Never raises."""
    try:
        from aimdk_msgs.srv import GetSystemState
        cli = node.create_client(GetSystemState,
                                 "/aimdk_5Fmsgs/srv/GetSystemState")
        if not cli.wait_for_service(timeout_sec=timeout):
            return "<service unavailable>"
        fut = cli.call_async(GetSystemState.Request())
        rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
        return fut.result().cur_state if fut.result() else "<no response>"
    except Exception as exc:                                # noqa: BLE001
        return "<error: %s>" % exc


def main():
    rclpy.init()
    node = Node("ffmaster_snapshot")

    # Query the SM state FIRST, before any subscription exists. The joint state
    # topics arrive at ~1 kHz each and saturate this single-threaded executor;
    # if the service call is made afterwards, spin_until_future_complete never
    # gets to the response and it reports "<no response>" on a healthy robot.
    state = system_state(node)

    data = {"leg": None, "waist": None, "arm": None, "imu": None}
    count = {"leg": 0, "imu": 0}

    def make_cb(key):
        def cb(msg):
            data[key] = msg
            if key in count:
                count[key] += 1
        return cb

    node.create_subscription(JointStateArray, "/aima/hal/joint/leg/state",
                             make_cb("leg"), QOS)
    node.create_subscription(JointStateArray, "/aima/hal/joint/waist/state",
                             make_cb("waist"), QOS)
    node.create_subscription(JointStateArray, "/aima/hal/joint/arm/state",
                             make_cb("arm"), QOS)
    node.create_subscription(Imu, "/aima/hal/imu/torso/state",
                             make_cb("imu"), QOS)

    window = 6.0
    start = time.time()
    while time.time() - start < window:
        rclpy.spin_once(node, timeout_sec=0.05)

    label = sys.argv[1] if len(sys.argv) > 1 else "SNAPSHOT"
    print("=" * 62)
    print(" %s" % label)
    print("=" * 62)

    # ---- system state + bus ownership -------------------------------------
    cmd_pubs = {t: len(node.get_publishers_info_by_topic(
        "/aima/hal/joint/%s/command" % t)) for t in CMD_TOPICS}
    mc_pubs = len(node.get_publishers_info_by_topic(MC_STATE_TOPIC))
    total_cmd = sum(cmd_pubs.values())

    print(" SYSTEM  state=%s" % state)
    print(" BUS     cmd publishers: %s   (total %d)"
          % ("  ".join("%s=%d" % (t, cmd_pubs[t]) for t in CMD_TOPICS),
             total_cmd))
    print(" BUS     %s publishers: %d" % (MC_STATE_TOPIC, mc_pubs))
    if total_cmd == 0 and mc_pubs == 0:
        print("         -> bus FREE (MC silent) — ours to command")
    elif state == "Develop_MC":
        print("         -> ** state is Develop_MC but publishers exist — "
              "investigate before commanding **")
    else:
        print("         -> bus OWNED by the MC — normal, not ours to command")

    # ---- IMU ---------------------------------------------------------------
    if data["imu"]:
        o = data["imu"].orientation
        x, y, z, w = o.x, o.y, o.z, o.w
        gz = 1 - 2 * (x * x + y * y)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, abs(gz)))))
        roll = math.degrees(math.atan2(2 * (w * x + y * z),
                                       1 - 2 * (x * x + y * y)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0,
                                                     2 * (w * y - z * x)))))
        print(" PELVIS  tilt=%6.2f deg   roll=%7.2f  pitch=%7.2f"
              % (tilt, roll, pitch))
        g = data["imu"].angular_velocity
        print(" GYRO    x=%+.4f y=%+.4f z=%+.4f rad/s" % (g.x, g.y, g.z))
    else:
        print(" PELVIS  *** NO IMU DATA ***")

    print(" RATES   leg=%.0f Hz  imu=%.0f Hz  (over %.0f s; python-limited,"
          " true rate is ~1 kHz)" % (count["leg"] / window,
                                     count["imu"] / window, window))

    # ---- joints ------------------------------------------------------------
    max_vel = 0.0
    max_eff = 0.0
    for key, names in (("leg", LEG), ("waist", WAIST), ("arm", ARM)):
        msg = data[key]
        if not msg:
            print(" %-5s *** NO DATA ***" % key)
            continue
        for i, name in enumerate(names):
            j = msg.joints[i]
            max_vel = max(max_vel, abs(j.velocity))
            max_eff = max(max_eff, abs(j.effort))
            print("   %-11s pos=%+8.4f  vel=%+7.4f  eff=%+8.3f"
                  % (name, j.position, j.velocity, j.effort))
    print(" MAX |vel|=%.4f rad/s   MAX |eff|=%.3f Nm" % (max_vel, max_eff))
    print("         (|vel| ~0.012 is encoder LSB = robot is still)")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
