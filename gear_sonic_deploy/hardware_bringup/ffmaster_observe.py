#!/usr/bin/env python3
"""FF Master bring-up observer — READ ONLY. Publishes nothing, ever.

IMPORTANT: the robot HAL is INDEX-ordered, not name-keyed. Per the vendor doc
(Interface/control_mod/joint_control.html): JointCommand.name is "optional" and
JointState.name is "currently unused" — expect empty name strings on the wire.
Everything here therefore keys on (group_topic, array_index); names are shown
only as decoration when present.

Documented HAL array order (leg + waist + arm concatenated) is identical to our
policy's MuJoCo/hardware order, so a verified match means zero permutation is
needed in the adapter. This tool's job is to verify that, not assume it.

Modes:
  inventory   topics, array lengths, rates, active command publishers
  watch       live table of joint positions by index
  fingerprint hand-move one joint at a time; prints which (group,index) moved
  imu         gravity/frame check for every IMU topic
  record      log everything to CSV (use while cycling vendor modes)

Usage (on the robot PC):
  python3 ffmaster_observe.py inventory
  python3 ffmaster_observe.py fingerprint --trials 8
  python3 ffmaster_observe.py record --seconds 240 --out mode_cycle.csv
"""

import argparse
import csv
import math
import sys
import time
from collections import defaultdict, deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# Documented HAL ordering, per group (joint_control.html). The waist row in the
# vendor doc reads "wrist_yaw/pitch/roll" which is a copy-paste typo for waist.
EXPECTED = {
    "leg": ["L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
            "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll"],
    "waist": ["waist_yaw", "waist_pitch", "waist_roll"],
    "arm": ["L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow", "L_wrist_yaw", "L_wrist_pitch", "L_wrist_roll",
            "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow", "R_wrist_yaw", "R_wrist_pitch", "R_wrist_roll"],
    "head": ["head_yaw", "head_pitch"],
}
# Our policy's 29-DOF order == leg + waist + arm in the above ordering.
POLICY_CONCAT = ["leg", "waist", "arm"]


def sensor_qos(depth=10):
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST, depth=depth)


def latched_qos(depth=1):
    # HAL state topics are documented TRANSIENT_LOCAL; try this if BEST_EFFORT
    # yields nothing (a latched first sample can also be stale — check stamps).
    return QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      history=HistoryPolicy.KEEP_LAST, depth=depth)


def group_of(topic):
    for g in ("leg", "waist", "arm", "head"):
        if f"/{g}/" in topic:
            return g
    return topic.strip("/").replace("/", "_")


class Observer(Node):
    def __init__(self, args):
        super().__init__("ffmaster_observer")
        self.args = args
        self.pos = {}    # group -> list[float] by index
        self.vel = {}
        self.eff = {}
        self.names = {}  # group -> list[str] (often empty strings)
        self.err = {}    # group -> DomainErrorState value
        self.imu = {}
        self.stamps = defaultdict(lambda: deque(maxlen=100))
        self.cmd_count = defaultdict(int)
        self.subs = []
        self._discover()

    def _discover(self):
        time.sleep(1.5)
        from aimdk_msgs.msg import JointStateArray, JointCommandArray

        self.state_topics, self.imu_topics, self.command_topics = [], [], []
        for name, types in self.get_topic_names_and_types():
            t = types[0] if types else ""
            if "JointStateArray" in t:
                self.state_topics.append((name, t))
            elif "JointCommandArray" in t:
                self.command_topics.append((name, t))
            elif t.endswith("/Imu"):
                self.imu_topics.append((name, t))

        for name, _ in self.state_topics:
            for qos in (sensor_qos(), latched_qos()):
                self.subs.append(self.create_subscription(
                    JointStateArray, name, lambda m, n=name: self._on_state(m, n), qos))
        for name, _ in self.command_topics:
            self.subs.append(self.create_subscription(
                JointCommandArray, name, lambda m, n=name: self._on_cmd(m, n), sensor_qos()))
        for name, ttype in self.imu_topics:
            cls = self._imp(ttype)
            if cls:
                self.subs.append(self.create_subscription(
                    cls, name, lambda m, n=name: self._on_imu(m, n), sensor_qos()))

    def _imp(self, ttype):
        try:
            pkg, _, cls = ttype.split("/")
            return getattr(__import__(f"{pkg}.msg", fromlist=[cls]), cls)
        except Exception as e:
            self.get_logger().warn(f"cannot import {ttype}: {e}")
            return None

    def _on_state(self, msg, topic):
        g = group_of(topic)
        self.stamps[topic].append(time.time())
        self.pos[g] = [j.position for j in msg.joints]
        self.vel[g] = [j.velocity for j in msg.joints]
        self.eff[g] = [j.effort for j in msg.joints]
        self.names[g] = [j.name for j in msg.joints]
        st = getattr(msg, "state", None)
        self.err[g] = getattr(st, "value", None) if st is not None else None

    def _on_cmd(self, msg, topic):
        self.stamps[topic].append(time.time())
        self.cmd_count[topic] += 1

    def _on_imu(self, msg, topic):
        self.stamps[topic].append(time.time())
        o, w, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
        self.imu[topic] = {
            "quat_wxyz": (o.w, o.x, o.y, o.z),   # ROS msg is xyzw fields; policy wants wxyz
            "gyro": (w.x, w.y, w.z),
            "accel": (a.x, a.y, a.z),
            "frame": msg.header.frame_id,
        }

    def rate(self, topic):
        ts = self.stamps[topic]
        return 0.0 if len(ts) < 2 else (len(ts) - 1) / max(1e-6, ts[-1] - ts[0])

    def spin(self, seconds):
        end = time.time() + seconds
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def flat_policy_vector(self):
        """leg+waist+arm concatenated == our 29-DOF MuJoCo order."""
        out = []
        for g in POLICY_CONCAT:
            out.extend(self.pos.get(g, []))
        return out


DOMAIN_ERR = {1: "Damping", 2: "PowerOff", 3: "Disabled", 4: "CommunicationFailed"}


def cmd_inventory(obs):
    obs.spin(4.0)
    print("\n=== JOINT STATE TOPICS ===")
    for name, _ in obs.state_topics:
        g = group_of(name)
        n = len(obs.pos.get(g, []))
        exp = len(EXPECTED.get(g, []))
        flag = "OK" if exp and n == exp else (f"EXPECTED {exp}" if exp else "")
        e = obs.err.get(g)
        estr = f" err={DOMAIN_ERR.get(e, e)}" if e else ""
        print(f"  {name:42s} len={n:3d} {flag:14s} {obs.rate(name):7.1f} Hz{estr}")
        nm = obs.names.get(g, [])
        if any(nm):
            print(f"      names present: {nm}")
        else:
            print("      names EMPTY on the wire (expected — index is the contract)")

    print("\n=== COMMAND TOPICS (who is driving the robot right now) ===")
    for name, _ in obs.command_topics:
        c = obs.cmd_count[name]
        print(f"  {name:42s} {obs.rate(name):7.1f} Hz  "
              f"{'ACTIVE PUBLISHER' if c else 'silent'}")

    print("\n=== IMU TOPICS ===")
    for name, _ in obs.imu_topics:
        d = obs.imu.get(name, {})
        print(f"  {name:42s} {obs.rate(name):7.1f} Hz  frame_id={d.get('frame','?')}")

    print("\n=== POLICY VECTOR (leg+waist+arm concat, expect 29) ===")
    v = obs.flat_policy_vector()
    print(f"  length = {len(v)}  {'OK' if len(v) == 29 else 'MISMATCH — adapter must not assume 29'}")
    idx = 0
    for g in POLICY_CONCAT:
        for i, val in enumerate(obs.pos.get(g, [])):
            lbl = EXPECTED[g][i] if i < len(EXPECTED.get(g, [])) else "?"
            print(f"  [{idx:2d}] {g:5s}[{i:2d}] {lbl:14s} {val:+.4f}")
            idx += 1


def cmd_watch(obs, args):
    end = time.time() + args.seconds
    while rclpy.ok() and time.time() < end:
        obs.spin(0.4)
        print("\033[2J\033[H", end="")
        for g in ("leg", "waist", "arm", "head"):
            for i, p in enumerate(obs.pos.get(g, [])):
                lbl = EXPECTED[g][i] if i < len(EXPECTED.get(g, [])) else "?"
                print(f"  {g:5s}[{i:2d}] {lbl:14s} q={p:+.4f} "
                      f"dq={obs.vel[g][i]:+.4f} tau={obs.eff[g][i]:+.2f}")
        for t, d in obs.imu.items():
            print(f"  {t}: quat={tuple(round(x,4) for x in d['quat_wxyz'])}")


def cmd_fingerprint(obs, args):
    print("ZERO-TORQUE / PASSIVE mode only. Move ONE joint by hand when prompted.")
    print("Records (group, index) — the HAL contract — not names.\n")
    results = []
    for trial in range(args.trials):
        input(f"[{trial+1}/{args.trials}] hold still, press Enter, then move ONE joint...")
        obs.spin(0.6)
        base = {g: list(v) for g, v in obs.pos.items()}
        print("   ... move it now (3 s)")
        obs.spin(3.0)
        deltas = []
        for g, cur in obs.pos.items():
            for i, val in enumerate(cur):
                if i < len(base.get(g, [])):
                    deltas.append((abs(val - base[g][i]), g, i, val - base[g][i]))
        deltas.sort(reverse=True)
        if not deltas:
            print("   no data")
            continue
        d, g, i, signed = deltas[0]
        lbl = EXPECTED[g][i] if i < len(EXPECTED.get(g, [])) else "?"
        direction = "POSITIVE" if signed > 0 else "NEGATIVE"
        print(f"   MOVED: {g}[{i}]  expected-to-be '{lbl}'  delta {signed:+.3f} rad ({direction})")
        for d2, g2, i2, s2 in deltas[1:3]:
            if d2 > 0.01:
                print(f"     also moved: {g2}[{i2}] {s2:+.3f}")
        note = input("   which joint did you actually move? ")
        results.append((g, i, lbl, round(signed, 4), note.strip()))
    print("\n=== FINGERPRINT RESULT ===")
    print(f"{'group[idx]':14s} {'doc-expects':16s} {'delta':>8s}  you-moved")
    for g, i, lbl, s, note in results:
        match = "OK " if note and note.lower().replace(" ", "_") in lbl.lower() else "?? "
        print(f"{match}{g}[{i}]".ljust(14) + f" {lbl:16s} {s:+8.3f}  {note}")
    print("\nAny '??' row needs a second look before we ever publish commands.")


def cmd_imu(obs, args):
    print("IMU FRAME CHECK — robot upright and still.\n")
    obs.spin(4.0)
    for topic, d in obs.imu.items():
        w, x, y, z = d["quat_wxyz"]
        ax, ay, az = d["accel"]
        gx_, gy_, gz_ = d["gyro"]
        mag = math.sqrt(ax * ax + ay * ay + az * az)
        # body-frame gravity direction from quaternion (policy convention)
        gx = 2 * (x * z - w * y)
        gy = 2 * (y * z + w * x)
        gz = 1 - 2 * (x * x + y * y)
        print(f"{topic}   frame_id={d['frame']}")
        print(f"  quat(wxyz)   = ({w:+.4f}, {x:+.4f}, {y:+.4f}, {z:+.4f})")
        print(f"  accel        = ({ax:+.3f}, {ay:+.3f}, {az:+.3f})  |a|={mag:.2f}"
              f"  {'OK' if 9.0 < mag < 10.6 else '<-- CHECK UNITS/FRAME'}")
        print(f"  gyro         = ({gx_:+.4f}, {gy_:+.4f}, {gz_:+.4f})"
              f"  {'OK (still)' if max(abs(gx_),abs(gy_),abs(gz_)) < 0.05 else '<-- not still?'}")
        print(f"  gravity_dir  = ({-gx:+.3f}, {-gy:+.3f}, {-gz:+.3f})   want ~(0,0,-1)")
        roll = math.degrees(math.atan2(gy, gz))
        pitch = math.degrees(math.atan2(-gx, math.sqrt(gy * gy + gz * gz)))
        print(f"  implied tilt : roll={roll:+.2f} deg  pitch={pitch:+.2f} deg"
              f"  {'OK' if abs(roll) < 1.0 and abs(pitch) < 1.0 else '<-- mounting offset?'}\n")
    print("NEXT: bend the robot at the WAIST (pelvis stays level, torso tilts).")
    print("  the topic that does NOT move is the PELVIS IMU -> that is the one our policy needs.")
    print("  vendor docs say /aima/hal/imu/torso/state is the pelvis one. Verify it.")


def cmd_record(obs, args):
    obs.spin(1.5)
    groups = [g for g in ("leg", "waist", "arm", "head") if g in obs.pos]
    imu_t = sorted(obs.imu)
    # t_epoch is absolute wall-clock so this log can be aligned against
    # bus_ownership.csv, which is written by a separate process with its own t0.
    hdr = ["t_epoch", "t"]
    for g in groups:
        n = len(obs.pos[g])
        hdr += [f"q_{g}{i}" for i in range(n)] + [f"dq_{g}{i}" for i in range(n)] \
             + [f"tau_{g}{i}" for i in range(n)] + [f"err_{g}"]
    for t in imu_t:
        hdr += [f"{t}_{k}" for k in ("qw", "qx", "qy", "qz", "gx", "gy", "gz", "ax", "ay", "az")]
    hdr += [f"cmdcount_{t}" for t, _ in obs.command_topics]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        t0 = time.time()
        print(f"recording {args.seconds}s to {args.out} — cycle vendor modes now (Ctrl-C to stop early)")
        try:
            while rclpy.ok() and time.time() - t0 < args.seconds:
                obs.spin(0.02)
                now = time.time()
                row = [round(now, 4), round(now - t0, 4)]
                for g in groups:
                    row += obs.pos.get(g, []) + obs.vel.get(g, []) + obs.eff.get(g, [])
                    row += [obs.err.get(g)]
                for t in imu_t:
                    d = obs.imu.get(t, {})
                    row += list(d.get("quat_wxyz", (float("nan"),) * 4))
                    row += list(d.get("gyro", (float("nan"),) * 3))
                    row += list(d.get("accel", (float("nan"),) * 3))
                row += [obs.cmd_count[t] for t, _ in obs.command_topics]
                w.writerow(row)
        except KeyboardInterrupt:
            pass
    print(f"\nwrote {args.out}")
    print("command-topic totals (nonzero = something was driving the robot):")
    for t, _ in obs.command_topics:
        print(f"  {t}: {obs.cmd_count[t]}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["inventory", "watch", "fingerprint", "imu", "record"])
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--out", default="ffmaster_observe_log.csv")
    args = p.parse_args()

    rclpy.init()
    obs = Observer(args)
    try:
        {"inventory": lambda: cmd_inventory(obs),
         "watch": lambda: cmd_watch(obs, args),
         "fingerprint": lambda: cmd_fingerprint(obs, args),
         "imu": lambda: cmd_imu(obs, args),
         "record": lambda: cmd_record(obs, args)}[args.mode]()
    except KeyboardInterrupt:
        pass
    finally:
        obs.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
