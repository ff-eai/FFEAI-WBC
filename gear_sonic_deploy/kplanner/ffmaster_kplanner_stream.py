#!/usr/bin/env python3
"""FF Master kinematic planner: ZMQ plug-in for the unmodified ffmaster_deploy_onnx_ref.

Turns keyboard or scripted velocity commands into a continuously
re-planned whole-body reference (motion matching over FF Master locomotion
clips, see planner.py) and streams it on the binary's ZMQ "pose" topic in the
protocol-v1 wire format that the binary already accepts (--input-type zmq or
manager). The binary is not changed.

Streaming contract (from the deploy binary's streamed-motion merger):
  * frames are indexed by a global 50 Hz counter; a chunk that starts after the
    playback head and extends past the buffered end REPLACES the buffered
    future -> re-planning is native
  * the head advances one frame per control tick while >= 47 frames are
    buffered ahead of it, so the planner keeps `horizon` frames (1.5 s) ahead
  * only the root quaternion's relative yaw and absolute roll/pitch matter;
    root position is not consumed in tracking mode

Run order (sim example, three terminals from the repo root):
  T1: .venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py --wbc-version ffmaster_sonic_model12
  T2: cd gear_sonic_deploy && bash deploy.sh --robot ffmaster --input-type zmq sim
      then  ]  (start)  ->  9 in the MuJoCo window (release)  ->  ENTER (streaming on)
  T3: .venv_sim/bin/python gear_sonic_deploy/kplanner/ffmaster_kplanner_stream.py --input keyboard
      then  g  (go), and drive with w/s a/d q/e, 1/2/3 gait presets, space = stop, x = quit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import sys
import termios
import threading
import time
import tty

import numpy as np
import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffmaster_kinematics import MJ_TO_ISAAC, REPO  # noqa: E402
from planner import KinematicPlanner, MotionDB  # noqa: E402

HEADER_SIZE = 1280  # must match ZMQPackedMessageSubscriber::HEADER_SIZE
FPS = 50.0
DEFAULT_DB = os.path.join(REPO, "data/kplanner/ffmaster_loco_db.npz")
POLICY_PARAMS = os.path.join(REPO, "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters_ffmaster.hpp")

KEY_HELP = ("g go (start streaming) | HOLD w/s forward/backward, a/d left/right, q/e turn left/right "
            "(arrows = wasd, combine freely) | 1/2/3 forward speed walk 0.8 / jog 1.5 / run 2.2 | "
            "space stop | r restart stream | x quit")


# ----------------------------------------------------------------------------- wire format
def pack_message(topic: bytes, version: int, fields: list) -> bytes:
    """Same layout as sim2sim_verify/ffmaster_zmq_stream_test.py: topic + 1280-byte header + payload."""
    dtype_map = {np.dtype(np.float32): "f32", np.dtype(np.int64): "i64", np.dtype(np.uint8): "u8"}
    header = {
        "v": version,
        "endian": "le",
        "count": int(fields[0][1].shape[0]),
        "fields": [{"name": n, "dtype": dtype_map[a.dtype], "shape": list(a.shape)} for n, a in fields],
    }
    hj = json.dumps(header).encode("utf-8")
    if len(hj) > HEADER_SIZE:
        raise ValueError(f"header too large ({len(hj)} > {HEADER_SIZE})")
    return topic + hj + b"\x00" * (HEADER_SIZE - len(hj)) + b"".join(a.tobytes() for _, a in fields)


class ReferencePublisher:
    def __init__(self, host="*", port=5556, topic="pose"):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://{host}:{port}")
        self.topic = topic.encode()
        self.catch_up = np.array([1], dtype=np.uint8)
        self.messages = 0

    def send(self, idx0: int, joint_pos: np.ndarray, joint_vel: np.ndarray, root_quat: np.ndarray):
        n = len(joint_pos)
        fields = [
            ("joint_pos", np.ascontiguousarray(joint_pos, dtype=np.float32)),
            ("joint_vel", np.ascontiguousarray(joint_vel, dtype=np.float32)),
            ("body_quat", np.ascontiguousarray(root_quat, dtype=np.float32)),
            ("frame_index", np.arange(idx0, idx0 + n, dtype=np.int64)),
            ("catch_up", self.catch_up),
        ]
        self.sock.send(pack_message(self.topic, 1, fields))
        self.messages += 1


def held_pose_isaac(motion_dir: str) -> np.ndarray | None:
    """Frame 0 of the first preloaded reference clip: the pose the binary holds before streaming starts."""
    d = os.path.join(REPO, "gear_sonic_deploy", motion_dir) if not os.path.isabs(motion_dir) else motion_dir
    if not os.path.isdir(d):
        return None
    clips = sorted(x for x in os.listdir(d) if os.path.isfile(os.path.join(d, x, "joint_pos.csv")))
    if not clips:
        return None
    with open(os.path.join(d, clips[0], "joint_pos.csv")) as f:
        f.readline()
        return np.array([float(v) for v in f.readline().strip().split(",")])


def default_standing_pose_isaac() -> np.ndarray:
    """The binary's default_angles (standing pose, MuJoCo order) in IsaacLab order."""
    src = open(POLICY_PARAMS).read()
    m = re.search(r"default_angles\s*=\s*\{([^}]*)\}", src, re.S)
    vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", re.sub(r"//[^\n]*", "", m.group(1)))]
    assert len(vals) == 29, len(vals)
    return np.array(vals)[MJ_TO_ISAAC]


# ----------------------------------------------------------------------------- streaming planner
class PlanStreamer:
    """Keeps a re-plannable horizon of reference frames ahead of the binary's playback head."""

    def __init__(self, planner: KinematicPlanner, pub: ReferencePublisher, period_frames=5, lead=15,
                 horizon=75, start_pose: np.ndarray | None = None, start_blend_s=1.0, log=None, overlap=10):
        self.pl, self.pub = planner, pub
        self.period, self.lead, self.horizon = period_frames, lead, horizon
        # every append re-sends the previous `overlap` frames: the binary keeps only the latest
        # message per 10 ms input cycle, so a lost message must not leave a gap (gap => catch-up snap)
        self.overlap = overlap
        self.start_pose, self.start_blend_s = start_pose, start_blend_s
        self.frames: dict[int, dict] = {}
        self.states: dict[int, dict] = {}
        self.next_idx = 0
        self.published_end = -1
        self.t0 = None
        self.dirty = False
        self.cmd = (0.0, 0.0, 0.0)
        self.log = log
        self.stalls = 0

    # -- generation
    def _gen_until(self, end: int):
        while self.next_idx <= end:
            self.states[self.next_idx] = self.pl.get_state()
            o = self.pl.step()
            o["cmd"] = self.cmd
            self.frames[self.next_idx] = o
            self.next_idx += 1

    def _regen_from(self, r: int, end: int):
        self.pl.set_state(self.states[r])
        for k in [k for k in self.frames if k >= r]:
            self.frames.pop(k, None)
            self.states.pop(k, None)
        self.next_idx = r
        self._gen_until(end)

    def _publish(self, a: int, b: int):
        """Publish frames a..b inclusive; velocities need frame b+1 to exist."""
        idx = list(range(a, b + 1))
        jp = np.stack([self.frames[i]["joint_pos"] for i in idx])
        jn = np.stack([self.frames[i + 1]["joint_pos"] for i in idx])
        jv = (jn - jp) * FPS
        rq = np.stack([self.frames[i]["root_quat"] for i in idx])
        self.pub.send(a, jp, jv, rq)
        if self.log is not None:
            t_now = time.monotonic()
            est = self.est_head() if self.t0 is not None else -1
            for k, i in enumerate(idx):
                self.log.setdefault("msg", []).append(self.pub.messages)
                self.log.setdefault("est_head", []).append(est)
                f = self.frames[i]
                self.log.setdefault("idx", []).append(i)
                self.log.setdefault("joint_pos", []).append(jp[k])
                self.log.setdefault("joint_vel", []).append(jv[k])
                self.log.setdefault("root_pos", []).append(f["root_pos"])
                self.log.setdefault("root_quat", []).append(rq[k])
                self.log.setdefault("cmd", []).append(f["cmd"])
                self.log.setdefault("clip", []).append(f["clip"])
                self.log.setdefault("frame", []).append(f["frame"])
                self.log.setdefault("t_pub", []).append(t_now)

    # -- public API
    def start(self):
        self.pl.reset()
        if self.start_pose is not None:
            # blend from the robot's default standing pose into the first database frame
            i = self.pl.db.index(self.pl.clip, self.pl.frame)
            self.pl.blend_dq = self.start_pose - self.pl.db.joint_pos[i].astype(float)
            self.pl.blend_dz = 0.0
            self.pl.blend_k = 0
            saved = self.pl.blend_frames
            self.pl.blend_frames = max(saved, int(self.start_blend_s * FPS))
        self.frames.clear()
        self.states.clear()
        self.next_idx = 0
        self._gen_until(self.horizon + 1)
        if self.start_pose is not None:
            self.pl.blend_frames = saved
        self._publish(0, self.horizon)
        self.published_end = self.horizon
        self.t0 = time.monotonic()
        if self.log is not None:
            self.log["t0"] = self.t0

    def est_head(self) -> int:
        return int((time.monotonic() - self.t0) * FPS)

    def set_command(self, vx, vy, wz):
        if (vx, vy, wz) != self.cmd:
            self.cmd = (float(vx), float(vy), float(wz))
            self.pl.set_command(*self.cmd)
            self.dirty = True

    def tick(self):
        """Call every `period_frames` / FPS seconds after start()."""
        h = self.est_head()
        end = h + self.horizon
        if end <= self.published_end:
            return
        if h + self.lead > self.published_end + 1:
            # we fell behind (process stall): the head may have run out of frames
            self.stalls += 1
        if self.dirty:
            r = max(h + self.lead, self.next_idx - self.horizon)
            r = min(r, self.published_end + 1)   # no gap allowed
            r = max(r, min(self.states) if self.states else r)
            self._regen_from(r, end + 1)
            self._publish(r, end)
            self.dirty = False
        else:
            self._gen_until(end + 1)
            a = max(self.published_end + 1 - self.overlap, h + 1, min(self.frames))
            self._publish(a, end)
        self.published_end = end
        for k in [k for k in self.frames if k < h - 10]:
            self.frames.pop(k, None)
            self.states.pop(k, None)

    def status(self) -> str:
        return (f"head~{self.est_head()} published..{self.published_end} msgs {self.pub.messages} "
                f"stalls {self.stalls} | {self.pl.describe()}")


# ----------------------------------------------------------------------------- inputs
class Keyboard:
    """Momentary (press-and-hold) keyboard control through a raw terminal.

    Terminals deliver no key-up events, so a key counts as held while its
    auto-repeat keeps arriving: after the first press it stays held for
    `first_hold` s (bridges the OS auto-repeat delay, 500 ms by default), and
    while repeating it stays held for `repeat_hold` s after the last repeat.
    Release latency is therefore ~0.2 s during a hold and ~0.6 s for a tap.
    Arrow keys are aliases of w/a/s/d.
    """

    ARROWS = {"A": "w", "B": "s", "D": "a", "C": "d"}

    def __init__(self, first_hold=0.65, repeat_hold=0.2):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        os.set_blocking(self.fd, False)
        termios.tcflush(self.fd, termios.TCIFLUSH)  # drop anything typed before we started
        self.first_hold, self.repeat_hold = first_hold, repeat_hold
        self.last_seen: dict[str, float] = {}
        self.press_start: dict[str, float] = {}
        self.pending = b""

    def poll(self):
        """Return the one-shot keys pressed since the last call and update hold state."""
        now = time.monotonic()
        try:
            data = os.read(self.fd, 256)
        except BlockingIOError:
            data = b""
        buf = self.pending + data
        keys = []
        i = 0
        while i < len(buf):
            c = buf[i:i + 1]
            if c == b"\x1b":
                if len(buf) - i < 3:
                    break  # incomplete escape sequence, keep for next poll
                if buf[i + 1:i + 2] == b"[" and buf[i + 2:i + 3].decode(errors="ignore") in self.ARROWS:
                    keys.append(self.ARROWS[buf[i + 2:i + 3].decode()])
                    i += 3
                    continue
                i += 1
                continue
            keys.append(c.decode(errors="ignore").lower())
            i += 1
        self.pending = buf[i:]
        for k in keys:
            if k in "wasdqe":
                if now - self.last_seen.get(k, -1e9) > self.first_hold:
                    self.press_start[k] = now
                self.last_seen[k] = now
        return keys

    def held(self, k: str) -> bool:
        now = time.monotonic()
        t = self.last_seen.get(k)
        if t is None:
            return False
        window = self.first_hold if now - self.press_start.get(k, now) < self.first_hold else self.repeat_hold
        return now - t < window

    def release_all(self):
        self.last_seen.clear()
        self.press_start.clear()
        self.pending = b""
        try:
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except Exception:
            pass

    def close(self):
        os.set_blocking(self.fd, True)
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class X11Keys:
    """True simultaneous key state from the X server (XQueryKeymap via ctypes).

    Terminals only auto-repeat the most recently pressed key, so held-key
    combinations cannot be read from stdin. On an X11 session the server's key
    bitmap gives the real down/up state of every key at once. Keys are read
    regardless of which window has focus.
    """

    KEYS = "wasdqe"
    ARROWS = {"Up": "w", "Down": "s", "Left": "a", "Right": "d"}

    def __init__(self):
        import ctypes as C
        self.C = C
        x = C.CDLL("libX11.so.6")
        x.XOpenDisplay.restype = C.c_void_p
        x.XOpenDisplay.argtypes = [C.c_char_p]
        x.XQueryKeymap.argtypes = [C.c_void_p, C.c_char * 32]
        x.XStringToKeysym.argtypes = [C.c_char_p]
        x.XStringToKeysym.restype = C.c_ulong
        x.XKeysymToKeycode.argtypes = [C.c_void_p, C.c_ulong]
        x.XKeysymToKeycode.restype = C.c_ubyte
        self.x = x
        self.d = x.XOpenDisplay(None)
        if not self.d:
            raise RuntimeError("no X display")
        self.km = (C.c_char * 32)()
        self.codes = {k: [x.XKeysymToKeycode(self.d, x.XStringToKeysym(k.encode()))] for k in self.KEYS}
        for name, k in self.ARROWS.items():
            self.codes[k].append(x.XKeysymToKeycode(self.d, x.XStringToKeysym(name.encode())))
        self.state = {k: False for k in self.KEYS}

    def poll(self):
        self.x.XQueryKeymap(self.d, self.km)
        b = bytes(self.km)
        for k, codes in self.codes.items():
            self.state[k] = any(b[kc // 8] & (1 << (kc % 8)) for kc in codes if kc)
        return self.state

    def held(self, k):
        return self.state.get(k, False)

    def release_all(self):
        pass


class KeyCommand:
    """Maps held keys to a velocity command. Speeds in m/s and rad/s."""

    def __init__(self, fwd=0.8, back=0.4, lat=0.4, yaw=0.8):
        self.fwd, self.back, self.lat, self.yaw = fwd, back, lat, yaw

    def command(self, kb):
        vx = (self.fwd if kb.held("w") else 0.0) - (self.back if kb.held("s") else 0.0)
        vy = (self.lat if kb.held("a") else 0.0) - (self.lat if kb.held("d") else 0.0)
        wz = (self.yaw if kb.held("q") else 0.0) - (self.yaw if kb.held("e") else 0.0)
        return vx, vy, wz

    def held_string(self, kb):
        return "".join(k for k in "wasdqe" if kb.held(k)) or "-"


class Script:
    """[[duration_s, vx, vy, yaw_rate], ...]; holds the last command when exhausted."""

    def __init__(self, seq):
        self.seq = [(float(d), float(vx), float(vy), float(wz)) for d, vx, vy, wz in seq]
        self.t_start = None

    @property
    def total(self):
        return sum(s[0] for s in self.seq)

    def command(self, now):
        if self.t_start is None:
            self.t_start = now
        t = now - self.t_start
        for d, vx, vy, wz in self.seq:
            if t < d:
                return (vx, vy, wz), False
            t -= d
        return self.seq[-1][1:], True


# ----------------------------------------------------------------------------- main
def build(args, log=None):
    db = MotionDB(args.db)
    pl = KinematicPlanner(db, search_interval=args.search_interval, blend_time=args.blend,
                          hysteresis=args.hysteresis, max_speed=args.max_speed, seed=args.seed,
                          min_play=args.min_play)
    pub = ReferencePublisher(args.host, args.port, args.topic)
    start_pose = None
    if not args.no_start_pose:
        start_pose = held_pose_isaac(args.motion_dir)
        if start_pose is None:
            start_pose = default_standing_pose_isaac()
    st = PlanStreamer(pl, pub, period_frames=args.period, lead=args.lead, horizon=args.horizon,
                      start_pose=start_pose, log=log, overlap=args.overlap)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--input", choices=["keyboard", "script"], default="keyboard")
    ap.add_argument("--script", help="JSON file: [[duration_s, vx, vy, yaw_rate], ...]")
    ap.add_argument("--host", default="*")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--topic", default="pose")
    ap.add_argument("--period", type=int, default=5, help="re-plan period in frames (5 = 100 ms)")
    ap.add_argument("--lead", type=int, default=15, help="first re-planned frame is this many frames after the head")
    ap.add_argument("--horizon", type=int, default=75, help="frames kept ahead of the head (>= 47 required)")
    ap.add_argument("--overlap", type=int, default=10, help="frames re-sent with every append (tolerates lost messages)")
    ap.add_argument("--search-interval", type=int, default=5)
    ap.add_argument("--blend", type=float, default=0.25)
    ap.add_argument("--hysteresis", type=float, default=0.15)
    ap.add_argument("--min-play", type=int, default=20, help="frames a new clip plays before another voluntary switch")
    ap.add_argument("--max-speed", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-start-pose", action="store_true", help="do not blend from the held pose")
    ap.add_argument("--motion-dir", default="reference/ffmaster_set8/",
                    help="the binary's preloaded motion set; frame 0 of its first clip is the pose held before streaming")
    ap.add_argument("--walk-speed", type=float, default=0.8, help="forward speed while w is held (1/2/3 change it)")
    ap.add_argument("--back-speed", type=float, default=0.7,
                    help="backward speed while s is held (backward-walking clips are 0.3-0.9 m/s; too slow a target matches standing)")
    ap.add_argument("--keys", choices=["auto", "x11", "stdin"], default="auto",
                    help="movement-key source: X11 key state (combinations, default when DISPLAY is set) or terminal auto-repeat")
    ap.add_argument("--lat-speed", type=float, default=0.4, help="lateral speed while a/d is held")
    ap.add_argument("--yaw-rate", type=float, default=0.8, help="yaw rate while q/e is held (rad/s)")
    ap.add_argument("--first-hold", type=float, default=0.65, help="seconds a key counts as held after its first press")
    ap.add_argument("--repeat-hold", type=float, default=0.2, help="seconds a key counts as held after an auto-repeat")
    ap.add_argument("--autostart", action="store_true", help="start streaming immediately (script mode default)")
    ap.add_argument("--log", help="write the streamed reference to this .npz")
    args = ap.parse_args()
    if args.horizon < 50:
        sys.exit("--horizon must be >= 50 (binary holds the head unless 47 frames are buffered)")

    log = {} if args.log else None
    st = build(args, log)
    print(f"[kplanner] db {args.db}: {st.pl.db.n} frames, {len(st.pl.db.clip_names)} clips; "
          f"bound tcp://{args.host}:{args.port} topic '{args.topic}'")
    print("[kplanner] binary side: --input-type zmq  ->  ]  ->  release robot  ->  ENTER (streaming on)")
    print(f"[kplanner] keys: {KEY_HELP}")

    kb = script = None
    keys = None  # object with held(k): X11 key state when available, else the stdin repeat emulation
    keycmd = KeyCommand(fwd=args.walk_speed, back=args.back_speed, lat=args.lat_speed, yaw=args.yaw_rate)
    if args.input == "keyboard":
        kb = Keyboard(first_hold=args.first_hold, repeat_hold=args.repeat_hold)
        keys = kb
        if args.keys == "x11" or (args.keys == "auto" and os.environ.get("DISPLAY")):
            try:
                keys = X11Keys()
                print("[kplanner] movement keys: X11 key state (true press/release, combinations work; "
                      "keys are read whatever window is focused)")
            except Exception as e:  # noqa: BLE001
                print(f"[kplanner] X11 key state unavailable ({e}); falling back to terminal auto-repeat "
                      "(one movement key at a time)")
        else:
            print("[kplanner] movement keys: terminal auto-repeat (one movement key at a time)")
    if args.input == "script":
        if not args.script:
            sys.exit("--input script needs --script FILE")
        script = Script(json.load(open(args.script)))
        args.autostart = True
    time.sleep(0.5)
    running = args.autostart
    if running:
        st.start()
        print("[kplanner] streaming started")
    vx = vy = wz = 0.0
    period_s = args.period / FPS
    next_tick = time.monotonic()
    last_status = 0.0
    try:
        while True:
            now = time.monotonic()
            if kb is not None:
                for k in kb.poll():
                    if k == "x":
                        raise KeyboardInterrupt
                    if k == "g" and not running:
                        kb.release_all()
                        st.start()
                        running = True
                        next_tick = time.monotonic()
                        print("[kplanner] streaming started (standing); hold w/a/s/d/q/e to move")
                    elif k == "r" and running:
                        kb.release_all()
                        st.set_command(0, 0, 0)
                        st.start()
                        next_tick = time.monotonic()
                        print("[kplanner] stream restarted (binary re-anchors)")
                    elif k == " ":
                        kb.release_all()
                    elif k == "1":
                        keycmd.fwd = 0.8
                        print("[kplanner] forward speed: walk 0.8 m/s")
                    elif k == "2":
                        keycmd.fwd = 1.5
                        print("[kplanner] forward speed: jog 1.5 m/s")
                    elif k == "3":
                        keycmd.fwd = 2.2
                        print("[kplanner] forward speed: run 2.2 m/s")
                if keys is not kb:
                    keys.poll()
                vx, vy, wz = keycmd.command(keys) if running else (0.0, 0.0, 0.0)
            done = False
            if script is not None:
                (vx, vy, wz), done = script.command(now)
            if running:
                st.set_command(round(vx, 3), round(vy, 3), round(wz, 3))
                if now >= next_tick:
                    st.tick()
                    next_tick += period_s
                    if next_tick < now:
                        next_tick = now + period_s
                if now - last_status > 1.0:
                    held = keycmd.held_string(keys) if keys is not None else "-"
                    print(f"[kplanner] keys {held} cmd v=({vx:+.2f},{vy:+.2f}) w={wz:+.2f} | {st.status()}")
                    last_status = now
                if done and script is not None and now - script.t_start > script.total + 1.0:
                    break
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        if kb is not None:
            kb.close()
        if log is not None and log:
            np.savez(args.log, **{k: np.asarray(v) for k, v in log.items()})
            print(f"[kplanner] log written to {args.log}")
        print("[kplanner] bye")


if __name__ == "__main__":
    main()
