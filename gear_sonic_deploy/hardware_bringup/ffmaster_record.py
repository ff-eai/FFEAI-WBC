#!/usr/bin/env python3
"""FF Master session recorder — bus-side, read-only, run alongside any test session.

Subscribes to the 29 policy joints (STATES ONLY) and the pelvis IMU, and
writes one wide CSV row per leg-state message (~500 Hz achievable from Python;
the true bus rate is ~1 kHz — sufficient for 50 Hz policy evaluation with 10x
oversampling).

Usage (robot, own terminal, any time — it never publishes):
    source ~/sonic_deployment/ffmaster_env.sh
    python3 ~/sonic_deployment/ffmaster_record.py --note "074000 ffmaster_gantry salute+wave"
    ... run the session ...
    Ctrl-C to stop; prints the session directory.

Output: ~/sonic_deployment/recordings/<UTC timestamp>/
    bus.csv    one row per leg-state message (columns in meta.json)
    meta.json  session metadata + column documentation

Column layout (all joints in FF Master MuJoCo order: leg 0-11, waist 12-14, arm 15-28):
    t                    epoch seconds (time.time() at leg-state callback)
    q00..q28             measured joint position  [rad]
    dq00..dq28           measured joint velocity  [rad/s]
    eff00..eff28         measured joint effort    [Nm]
    imu_qw..imu_qz       pelvis orientation quaternion (wxyz)
    gyr_x..gyr_z         pelvis angular velocity [rad/s]
    acc_x..acc_z         pelvis linear acceleration [m/s^2]
    tilt_deg             pelvis tilt from vertical [deg] (derived, convenience)

Commands are deliberately NOT subscribed (that registered the recorder inside
the bridge's DDS writers — the 2026-08-13 incident). They come from the
bridge's own flight-log (recordings/bridge_cmd_*.csv, same epoch clock);
ffmaster_eval.py merges the two files by timestamp.
"""
import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone

# Self-hardening BEFORE DDS comes up:
#  - CPU affinity away from the bridge's pinned cores (6,7)
#  - isolated FastDDS profile (develop0+lo, SHM off) if available
try:
    os_cores = set(range(os.cpu_count() or 8)) - {6, 7}
    os.sched_setaffinity(0, os_cores)
except (OSError, AttributeError):
    pass
_ISO = os.path.expanduser(
    "~/sonic_deployment/gear_sonic_deploy/src/ffmaster/sonic_ffmaster_bridge/config/"
    "fastdds_robot_isolated.xml")
if os.path.isfile(_ISO):
    os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", _ISO)

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from aimdk_msgs.msg import JointStateArray
from sensor_msgs.msg import Imu

QOS = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                 durability=DurabilityPolicy.VOLATILE,
                 history=HistoryPolicy.KEEP_LAST)

GROUPS = (("leg", 0, 12), ("waist", 12, 3), ("arm", 15, 14))
N = 29


def system_state_once(node, timeout=8.0):
    """One state query BEFORE any subscription exists (1 kHz subs starve the
    single-thread executor and the service reply never lands)."""
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


class Recorder(Node):
    def __init__(self, out_dir, note, motion=None):
        super().__init__("ffmaster_recorder")
        self.out_dir = out_dir
        self.note = note
        self.motion = motion

        self.sys_state = system_state_once(self)

        self.q = [float("nan")] * N
        self.dq = [float("nan")] * N
        self.eff = [float("nan")] * N
        self.imu = [float("nan")] * 10          # qw qx qy qz gx gy gz ax ay az
        self.rows = 0
        self.t_first = None
        self.t_last_wall = None
        self.gap_max = 0.0

        os.makedirs(out_dir, exist_ok=True)
        self.csv_file = open(os.path.join(out_dir, "bus.csv"), "w", newline="")
        self.writer = csv.writer(self.csv_file)
        hdr = (["t"]
               + ["q%02d" % i for i in range(N)]
               + ["dq%02d" % i for i in range(N)]
               + ["eff%02d" % i for i in range(N)]
               + ["imu_qw", "imu_qx", "imu_qy", "imu_qz",
                  "gyr_x", "gyr_y", "gyr_z", "acc_x", "acc_y", "acc_z",
                  "tilt_deg"])
        self.writer.writerow(hdr)
        self.header = hdr

        # state subs — leg is the row clock (writes the row), waist/arm update
        for name, base, count in GROUPS:
            topic = "/aima/hal/joint/%s/state" % name
            is_clock = (name == "leg")
            self.create_subscription(
                JointStateArray, topic,
                self._mk_state_cb(base, count, is_clock), QOS)
        # NO command subscriptions — commands come from the bridge's own
        # flight-log (bridge_cmd_*.csv). Subscribing here would register this
        # recorder inside the bridge's DDS writers (2026-08-13 incident).
        # pelvis IMU
        self.create_subscription(Imu, "/aima/hal/imu/torso/state",
                                 self._imu_cb, QOS)

    def _mk_state_cb(self, base, count, is_clock):
        def cb(msg):
            if len(msg.joints) != count:
                return
            for i in range(count):
                j = msg.joints[i]
                self.q[base + i] = j.position
                self.dq[base + i] = j.velocity
                self.eff[base + i] = j.effort
            if is_clock:
                self._write_row()
        return cb

    def _imu_cb(self, msg):
        o, g, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
        self.imu = [o.w, o.x, o.y, o.z, g.x, g.y, g.z, a.x, a.y, a.z]

    def _write_row(self):
        now = time.time()
        if self.t_first is None:
            self.t_first = now
        if self.t_last_wall is not None:
            self.gap_max = max(self.gap_max, now - self.t_last_wall)
        self.t_last_wall = now
        qw, qx, qy, qz = self.imu[0], self.imu[1], self.imu[2], self.imu[3]
        gz_body = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt = (math.degrees(math.acos(max(-1.0, min(1.0, abs(gz_body)))))
                if not math.isnan(qw) else float("nan"))
        self.writer.writerow(
            ["%.6f" % now]
            + ["%.6f" % v for v in self.q]
            + ["%.6f" % v for v in self.dq]
            + ["%.4f" % v for v in self.eff]
            + ["%.7f" % v for v in self.imu]
            + ["%.3f" % tilt])
        self.rows += 1
        if self.rows % 5000 == 0:
            dur = now - self.t_first
            print("  %d rows, %.1f s, %.0f Hz, max gap %.0f ms"
                  % (self.rows, dur, self.rows / max(dur, 1e-9),
                     self.gap_max * 1e3), flush=True)

    def finalize(self):
        self.csv_file.close()
        dur = ((self.t_last_wall - self.t_first)
               if (self.t_first and self.t_last_wall) else 0.0)
        meta = {
            "started_utc": datetime.fromtimestamp(
                self.t_first or time.time(), tz=timezone.utc).isoformat(),
            "duration_s": round(dur, 3),
            "rows": self.rows,
            "mean_rate_hz": round(self.rows / dur, 1) if dur > 0 else None,
            "max_gap_ms": round(self.gap_max * 1e3, 1),
            "system_state_at_start": self.sys_state,
            "note": self.note,
            "motion": self.motion,
            "joint_order": "FF Master MuJoCo (leg 0-11, waist 12-14, arm 15-28)",
            "columns": self.header,
            "format": "bus_v2_states_only — commands live in the bridge'\''s "
                      "bridge_cmd_*.csv; ffmaster_eval.py merges by timestamp",
        }
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print("\nsession dir : %s" % self.out_dir)
        print("rows        : %d  (%.1f s @ %.0f Hz, max gap %.0f ms)"
              % (self.rows, dur, self.rows / max(dur, 1e-9),
                 self.gap_max * 1e3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None,
                    help="session dir (default: ~/sonic_deployment/recordings/<ts>[_motion])")
    ap.add_argument("--note", default="",
                    help="free-text session note (checkpoint, motion list, rig)")
    ap.add_argument("--motion", default=None,
                    help="reference clip folder name for a one-motion recording "
                         "(e.g. salute_R_003__A405). Stored in meta.json; "
                         "ffmaster_eval.py then evaluates every run in this recording "
                         "against that clip — no identification, no ambiguity. "
                         "RECOMMENDED workflow: one recording per motion.")
    args = ap.parse_args()

    suffix = ("_" + args.motion) if args.motion else ""
    out = args.out or os.path.expanduser(
        "~/sonic_deployment/recordings/%s%s"
        % (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ"), suffix))

    rclpy.init()
    rec = Recorder(out, args.note, args.motion)
    print("recording to %s   (system state at start: %s%s)"
          % (out, rec.sys_state,
             (", motion: " + args.motion) if args.motion else ""))
    print("Ctrl-C to stop.", flush=True)

    def stop(_sig, _frm):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        rclpy.spin(rec)
    except KeyboardInterrupt:
        pass
    finally:
        rec.finalize()
        rec.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
