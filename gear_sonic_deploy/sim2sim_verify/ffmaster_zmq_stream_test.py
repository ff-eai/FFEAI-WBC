#!/usr/bin/env python3
"""Interactive ZMQ motion streamer for ffmaster_deploy_onnx_ref.

Loads a FOLDER of exported reference motions (the 9-CSV layout, e.g.
reference/ffmaster/) and streams the selected motion on the ZMQ "pose" topic at
50 Hz, in the packed-message wire format expected by
ZMQPackedMessageSubscriber:

    [topic bytes][1280-byte null-padded JSON header][little-endian binary fields]

Publisher-side keys (mirror the deploy binary's own layout):
    0-9     select one of the first 10 motions
    n / p   next / previous motion (selection stops any current playback)
    t       stream the selected motion from its first frame
    r       restart the current motion from its first frame
    space   pause / resume publishing (robot holds pose while paused)
    l       toggle looping of the current motion
    q       quit (robot holds last pose)

Protocols (one per publisher session -- the deploy side latches the version;
toggle ENTER there to reset the latch before switching):
    v1  joint_pos + joint_vel + body_quat + frame_index      -> encoder mode 0
    v3  v1 fields + smpl_joints [N,24,3] + smpl_pose [N,21,3] -> encoder mode 2
        (smpl_joint.csv is pre-canonicalized at export; smpl_pose is zero:
         the merger requires the field but no encoder observation consumes it)

Frame indices increase monotonically across motion switches and restarts, so
the merger splices every change as a stream continuation. NOTE: switching or
restarting mid-play makes the robot snap toward the new reference pose --
fine in sim, treat with care on hardware.

Run order:
    Terminal 1:  source .venv_sim/bin/activate
                 python gear_sonic/scripts/run_sim_loop.py --wbc-version ffmaster_sonic_model12
    Terminal 2:  cd gear_sonic_deploy
                 bash deploy.sh --robot ffmaster --input-type zmq --zmq-host localhost sim
                 then:  ]  (start) -> drop robot in MuJoCo -> ENTER (streaming on)
    Terminal 3:  source .venv_sim/bin/activate
                 python gear_sonic_deploy/sim2sim_verify/ffmaster_zmq_stream_test.py --protocol 1
"""

import argparse
import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import zmq

HEADER_SIZE = 1280  # must match ZMQPackedMessageSubscriber::HEADER_SIZE
CONTROL_HZ = 50.0   # deploy control rate; reference CSVs are 1 row per tick


# --------------------------------------------------------------------------
# Motion loading
# --------------------------------------------------------------------------
class MotionLibrary:
    """Lazily loads motion directories from a reference set folder."""

    def __init__(self, ref_dir: Path, protocol: int):
        self.protocol = protocol
        self.dirs = sorted(
            d for d in ref_dir.iterdir() if (d / "joint_pos.csv").exists()
        )
        if not self.dirs:
            sys.exit(f"error: no motion dirs with joint_pos.csv under {ref_dir}")
        self._cache: dict[int, dict] = {}

    def name(self, i: int) -> str:
        return self.dirs[i].name

    def __len__(self) -> int:
        return len(self.dirs)

    def load(self, i: int):
        """Returns the data dict for motion i, or None if unusable."""
        if i in self._cache:
            return self._cache[i]
        d = self.dirs[i]

        def csv(name):
            return np.loadtxt(d / name, delimiter=",", skiprows=1, dtype=np.float32)

        data = {
            "joint_pos": csv("joint_pos.csv"),
            "joint_vel": csv("joint_vel.csv"),
            "body_quat": csv("body_quat.csv")[:, 0:4],  # root wxyz only
        }
        if self.protocol == 3:
            smpl_file = d / "smpl_joint.csv"
            if not smpl_file.exists():
                print(f"[stream] '{d.name}' has no smpl_joint.csv -- unusable in v3")
                return None
            smpl = csv("smpl_joint.csv")
            if smpl.shape[1] != 72:
                print(f"[stream] '{d.name}': smpl_joint.csv has {smpl.shape[1]} cols, expected 72")
                return None
            data["smpl_joints"] = smpl.reshape(-1, 24, 3)
            pose_file = d / "smpl_pose.csv"
            if pose_file.exists():
                data["smpl_pose"] = np.loadtxt(
                    pose_file, delimiter=",", skiprows=1, dtype=np.float32
                ).reshape(-1, 21, 3)
            else:
                data["smpl_pose"] = np.zeros(
                    (smpl.shape[0], 21, 3), dtype=np.float32
                )
        T = data["joint_pos"].shape[0]
        for k, v in data.items():
            if v.shape[0] != T:
                print(f"[stream] '{d.name}': frame-count mismatch on {k} -- skipping")
                return None
        self._cache[i] = data
        return data


# --------------------------------------------------------------------------
# Wire format
# --------------------------------------------------------------------------
def pack_message(topic: bytes, version: int, fields: list) -> bytes:
    """fields: list of (name, np.ndarray float32/int64/uint8), order preserved."""
    dtype_map = {np.dtype(np.float32): "f32", np.dtype(np.int64): "i64",
                 np.dtype(np.uint8): "u8"}
    header = {
        "v": version,
        "endian": "le",
        "count": int(fields[0][1].shape[0]),
        "fields": [
            {"name": name, "dtype": dtype_map[arr.dtype], "shape": list(arr.shape)}
            for name, arr in fields
        ],
    }
    header_json = json.dumps(header).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        sys.exit(f"error: header too large ({len(header_json)} > {HEADER_SIZE})")
    header_bytes = header_json + b"\x00" * (HEADER_SIZE - len(header_json))
    payload = b"".join(arr.tobytes() for _, arr in fields)
    return topic + header_bytes + payload


# --------------------------------------------------------------------------
# Interactive player
# --------------------------------------------------------------------------
HELP = ("keys: 0-9 select | n/p next/prev | t play | r restart | "
        "space pause | l loop | q quit")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", default=None,
                    help="motion set folder (default: <repo>/reference/ffmaster)")
    ap.add_argument("--motion", default=None,
                    help="motion name to preselect (default: first in folder)")
    ap.add_argument("--protocol", type=int, choices=[1, 3], default=1,
                    help="1 = joint tracking (mode 0), 3 = SMPL (mode 2)")
    ap.add_argument("--host", default="*", help="bind address (default all interfaces)")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--topic", default="pose")
    ap.add_argument("--chunk", type=int, default=10, help="frames per message")
    ap.add_argument("--no-catch-up", action="store_true",
                    help="catch_up=0: strict real-time, no playback fast-forward")
    args = ap.parse_args()

    deploy_root = Path(__file__).resolve().parent.parent  # gear_sonic_deploy/
    ref_dir = Path(args.ref_dir) if args.ref_dir else deploy_root / "reference" / "ffmaster"
    if not ref_dir.is_dir():
        sys.exit(f"error: ref dir not found: {ref_dir}")

    lib = MotionLibrary(ref_dir, args.protocol)
    print(f"[stream] {len(lib)} motions in {ref_dir} "
          f"(protocol v{args.protocol} -> encoder mode {0 if args.protocol == 1 else 2})")
    for i, d in enumerate(lib.dirs):
        hotkey = str(i) if i < 10 else "  "
        print(f"  [{hotkey:>2}] {d.name}")

    sel = 0
    if args.motion is not None:
        names = [d.name for d in lib.dirs]
        if args.motion not in names:
            sys.exit(f"error: motion '{args.motion}' not in {ref_dir}")
        sel = names.index(args.motion)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://{args.host}:{args.port}")
    print(f"[stream] bound tcp://{args.host}:{args.port}, topic '{args.topic}'")
    print("[stream] deploy side: --input-type zmq, then ']' start, ENTER = streaming on")
    print(f"[stream] {HELP}")
    print(f"[stream] selected [{sel}] {lib.name(sel)} -- press 't' to stream")
    time.sleep(0.5)  # let the subscriber connect

    topic = args.topic.encode()
    catch_up = np.array([0 if args.no_catch_up else 1], dtype=np.uint8)
    chunk = args.chunk

    playing = False
    paused = False
    loop = False
    data = None          # current motion arrays
    T = 0                # current motion length
    ptr = 0              # next local frame to send
    global_frame = 0     # monotonic across everything
    next_send = time.monotonic()

    def start(from_msg):
        nonlocal playing, paused, data, T, ptr, next_send
        data = lib.load(sel)
        if data is None:
            print(f"[stream] cannot stream [{sel}] {lib.name(sel)}")
            playing = False
            return
        T = data["joint_pos"].shape[0]
        ptr = 0
        playing, paused = True, False
        next_send = time.monotonic()
        print(f"[stream] {from_msg} [{sel}] {lib.name(sel)} "
              f"({T} frames, {T / CONTROL_HZ:.1f} s)")

    def select_motion(new_sel):
        nonlocal sel, playing
        if playing:
            playing = False
            print("[stream] playback stopped (robot holds pose)")
        sel = new_sel
        print(f"[stream] selected [{sel}] {lib.name(sel)} -- press 't' to stream")

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            timeout = max(0.0, next_send - time.monotonic()) if (playing and not paused) else 0.05
            ready, _, _ = select.select([sys.stdin], [], [], min(timeout, 0.05))
            if ready:
                key = sys.stdin.read(1)
                if key == "q":
                    print("\n[stream] quit (robot holds last pose; 'O' on deploy side stops control)")
                    break
                elif key.isdigit():
                    idx = int(key)
                    if idx < len(lib):
                        select_motion(idx)
                    else:
                        print(f"[stream] no motion [{idx}]")
                elif key == "n":
                    select_motion((sel + 1) % len(lib))
                elif key == "p":
                    select_motion((sel - 1) % len(lib))
                elif key == "t":
                    if not playing:
                        start("streaming")
                    else:
                        print("[stream] already playing ('r' restarts)")
                elif key == "r":
                    start("restarting")
                elif key == " ":
                    if playing:
                        paused = not paused
                        if paused:
                            print("[stream] PAUSED (robot holds pose)")
                        else:
                            next_send = time.monotonic()
                            print("[stream] resumed")
                elif key == "l":
                    loop = not loop
                    print(f"[stream] loop {'ON' if loop else 'OFF'}")

            if playing and not paused and time.monotonic() >= next_send:
                end = min(ptr + chunk, T)
                n = end - ptr
                idx = np.arange(global_frame, global_frame + n, dtype=np.int64)
                fields = [
                    ("joint_pos", data["joint_pos"][ptr:end]),
                    ("joint_vel", data["joint_vel"][ptr:end]),
                    ("body_quat", data["body_quat"][ptr:end]),
                    ("frame_index", idx),
                    ("catch_up", catch_up),
                ]
                if args.protocol == 3:
                    fields.insert(3, ("smpl_joints", data["smpl_joints"][ptr:end]))
                    fields.insert(4, ("smpl_pose", data["smpl_pose"][ptr:end]))
                pub.send(pack_message(topic, args.protocol, fields))
                global_frame += n
                ptr = end
                next_send += n / CONTROL_HZ

                if ptr % 100 < chunk:
                    print(f"[stream] {lib.name(sel)}  {ptr / CONTROL_HZ:5.1f}s / "
                          f"{T / CONTROL_HZ:.1f}s")
                if ptr >= T:
                    if loop:
                        ptr = 0
                        print(f"[stream] looping {lib.name(sel)}")
                    else:
                        playing = False
                        print(f"[stream] {lib.name(sel)} done -- robot holds last pose. {HELP}")
    except KeyboardInterrupt:
        print("\n[stream] interrupted (robot holds last pose)")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        pub.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
