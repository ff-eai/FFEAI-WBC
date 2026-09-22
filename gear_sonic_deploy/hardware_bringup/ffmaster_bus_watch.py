#!/usr/bin/env python3
"""FF Master command-bus ownership watcher — READ ONLY. Publishes nothing, ever.

Answers the one question that gates the whole adapter design:

    when the vendor motion controller is put into PASSIVE / DAMPING / JOINT,
    does it STOP publishing to /aima/hal/joint/*/command ?

If it does not, our policy node can never be the sole commander: the HAL
(`hal_ethercat_ffmaster`) subscribes BEST_EFFORT with no arbitration, so two publishers
means last-sample-wins at ~500 Hz.

Method: publisher counts come from the DDS *graph* (get_publishers_info_by_topic),
not from subscribing to the data. That is exact, costs nothing, and cannot be
skewed by Python executor saturation the way ffmaster_observe.py's Hz column is.

Mode comes from /aima/mc/common/state (McCommonState), a continuously published
topic — more robust than polling the GetMcAction service, which the vendor's own
docs warn is unreliable cross-host.

Usage (on the robot, after `source ~/ffmaster_env.sh`):
    python3 ffmaster_bus_watch.py 2>/dev/null
    python3 ffmaster_bus_watch.py --out bus_ownership.csv --seconds 300 2>/dev/null

Then cycle modes from the app/controller and watch the table.
"""

import argparse
import csv
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

COMMAND_TOPICS = [
    "/aima/hal/joint/leg/command",
    "/aima/hal/joint/waist/command",
    "/aima/hal/joint/arm/command",
    "/aima/hal/joint/head/command",
]
MC_STATE_TOPIC = "/aima/mc/common/state"

STATUS = {0: "IDLE", 100: "RUNNING", 200: "TRANSITION"}


def best_effort(depth=10):
    return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST, depth=depth)


class BusWatch(Node):
    def __init__(self):
        super().__init__("ffmaster_bus_watch")
        self.mode = "?"
        self.status = None
        self.input_source = ""
        self.mode_changed_at = None

        from aimdk_msgs.msg import McCommonState
        self.create_subscription(McCommonState, MC_STATE_TOPIC,
                                 self._on_mc, best_effort())

    def _on_mc(self, msg):
        info = msg.action_info
        desc = info.action_desc or "?"
        if desc != self.mode:
            self.mode_changed_at = time.time()
        self.mode = desc
        self.status = getattr(info.status, "value", None)
        src = getattr(msg, "input_source", None)
        self.input_source = getattr(src, "name", "") if src is not None else ""

    def publishers(self, topic):
        """Exact publisher list from the DDS graph — no data subscription needed."""
        try:
            return [e.node_name for e in self.get_publishers_info_by_topic(topic)]
        except Exception:
            return []

    def snapshot(self):
        return {t: self.publishers(t) for t in COMMAND_TOPICS}


def fmt_pubs(pubs):
    if not pubs:
        return "\033[32mNONE\033[0m"          # green: bus is free
    return "\033[33m%d (%s)\033[0m" % (len(pubs), ",".join(pubs))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=600.0)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--out", default=None, help="optional CSV log")
    args = p.parse_args()

    rclpy.init()
    node = BusWatch()

    writer = fh = None
    if args.out:
        fh = open(args.out, "w", newline="")
        writer = csv.writer(fh)
        # t_epoch is absolute wall-clock so this log can be aligned against
        # mode_cycle.csv, which is written by a different process with its own t0.
        writer.writerow(["t_epoch", "t", "mode", "status", "input_source"]
                        + [f"npub_{t.split('/')[-2]}" for t in COMMAND_TOPICS]
                        + [f"pubs_{t.split('/')[-2]}" for t in COMMAND_TOPICS])

    print("READ-ONLY. Cycle vendor modes now; this publishes nothing.")
    print("Waiting for DDS discovery to settle...", flush=True)

    # Discovery is not instantaneous: for the first ~2 s the graph legitimately
    # reports zero publishers on every topic. Reporting that as "bus is free"
    # would be a dangerous false positive, so settle first and refuse to judge
    # until McCommonState has actually been received.
    settle_end = time.time() + 3.0
    while rclpy.ok() and time.time() < settle_end:
        rclpy.spin_once(node, timeout_sec=0.05)

    print()
    print(f"{'time':>7}  {'mode':<20} {'status':<11} " +
          "  ".join(f"{t.split('/')[-2]:>6}" for t in COMMAND_TOPICS))
    print("-" * 92)

    t0 = time.time()
    last_key = None
    try:
        while rclpy.ok() and time.time() - t0 < args.seconds:
            deadline = time.time() + args.interval
            while time.time() < deadline and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)

            # On Ctrl-C, rclpy tears down the graph cache before this loop exits,
            # so get_publishers_info_by_topic() returns [] for every topic. That is
            # a shutdown artifact, NOT the MC releasing the bus — reporting it as
            # "ALL COMMAND TOPICS FREE" would be a dangerous false positive.
            if not rclpy.ok():
                break

            snap = node.snapshot()
            key = (node.mode, node.status,
                   tuple(tuple(sorted(v)) for v in snap.values()))

            if key != last_key:                     # only print on change
                last_key = key
                t = time.time() - t0
                st = STATUS.get(node.status, node.status)
                cells = "  ".join(fmt_pubs(snap[c]).rjust(6 + 9) for c in COMMAND_TOPICS)
                src = f"  src='{node.input_source}'" if node.input_source else ""
                print(f"{t:7.1f}  {node.mode:<20} {str(st):<11} {cells}{src}")

                # Only trustworthy once we are actually receiving McCommonState;
                # otherwise "no publishers" just means discovery hasn't run yet.
                free = [c for c in COMMAND_TOPICS if not snap[c]]
                if len(free) == len(COMMAND_TOPICS) and node.mode != "?":
                    print("         \033[32m>>> ALL COMMAND TOPICS FREE — "
                          f"MC released the bus in {node.mode} <<<\033[0m")
                elif node.mode == "?":
                    print("         (mode unknown — McCommonState not yet received; "
                          "publisher counts not yet meaningful)")

            if writer:
                now = time.time()
                writer.writerow(
                    [round(now, 3), round(now - t0, 3), node.mode, node.status,
                     node.input_source]
                    + [len(snap[c]) for c in COMMAND_TOPICS]
                    + ["|".join(snap[c]) for c in COMMAND_TOPICS])
    except KeyboardInterrupt:
        pass
    finally:
        if fh:
            fh.close()
            print(f"\nwrote {args.out}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
