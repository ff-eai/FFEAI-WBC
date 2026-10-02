#!/usr/bin/env python3
"""Open-loop check of the planner's reference for a set of commands (no simulator).

For each command the planner runs 2 s standing then `--dur` seconds with the
command, and the second half of that window is scored on the reference itself:
  root velocity / yaw rate vs the command      (does it go where asked)
  leg joint range                              (is it actually stepping; a smeared
                                                pose gives < 10 deg while the root still moves)
  foot skate: fraction of frames where BOTH feet move > 0.2 m/s in the world
                                               (a real gait keeps one foot planted)
  clip switches per second
Usage: .venv_sim/bin/python gear_sonic_deploy/kplanner/eval_openloop.py [--db PATH]
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffmaster_kinematics import ISAAC_TO_MJ, FFMasterModel  # noqa: E402
from planner import KinematicPlanner, MotionDB  # noqa: E402

COMMANDS = [(0, 0, 0), (0.8, 0, 0), (1.5, 0, 0), (2.2, 0, 0), (-0.7, 0, 0), (0, 0.4, 0), (0, -0.4, 0),
            (0, 0, 0.8), (0, 0, -0.8), (0.8, 0, 0.8), (0.8, 0.4, 0), (0.8, -0.4, 0), (-0.7, 0.4, 0), (0.8, -0.4, 0.8)]


def score(db, model, cmd, dur, **kw):
    pl = KinematicPlanner(db, **kw)
    pl.reset()
    outs = []
    for _ in range(100):
        outs.append(pl.step())
    j0 = pl.jumps
    pl.set_command(*cmd)
    for _ in range(int(dur * 50)):
        outs.append(pl.step())
    o = outs[100 + int(dur * 25):]
    P = np.array([x["root_pos"] for x in o]); Y = np.array([x["yaw"] for x in o]); J = np.array([x["joint_pos"] for x in o])
    v = np.gradient(P[:, :2], 1 / 50, axis=0)
    fwd = float((v[:, 0] * np.cos(Y) + v[:, 1] * np.sin(Y)).mean()); lat = float((-v[:, 0] * np.sin(Y) + v[:, 1] * np.cos(Y)).mean())
    yr = float(np.gradient(np.unwrap(Y), 1 / 50).mean())
    feet = np.array([model.fk(x["root_pos"], x["root_quat"], x["joint_pos"][ISAAC_TO_MJ])[1:] for x in o])
    fs = np.linalg.norm(np.gradient(feet, 1 / 50, axis=0)[..., :2], axis=2)
    skate = float(((fs[:, 0] > 0.2) & (fs[:, 1] > 0.2)).mean())
    legs = float(np.degrees(J.max(0) - J.min(0))[:12].mean())
    return dict(fwd=fwd, lat=lat, yaw=yr, legs=legs, skate=skate, jumps_s=(pl.jumps - j0) / dur)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data/kplanner/ffmaster_loco_db.npz"))
    ap.add_argument("--dur", type=float, default=8.0)
    ap.add_argument("--min-play", type=int, default=20)
    ap.add_argument("--hysteresis", type=float, default=0.15)
    ap.add_argument("--search-interval", type=int, default=5)
    args = ap.parse_args()
    db = MotionDB(args.db); model = FFMasterModel()
    print(f"{'command (vx,vy,wz)':22s} {'root vel':>14s} {'yaw':>6s} {'legs':>6s} {'skate':>6s} {'jumps/s':>8s}  verdict")
    for c in COMMANDS:
        r = score(db, model, c, args.dur, min_play=args.min_play, hysteresis=args.hysteresis, search_interval=args.search_interval)
        moving = np.hypot(c[0], c[1]) > 0.05 or abs(c[2]) > 0.05
        verr = np.hypot(r["fwd"] - c[0], r["lat"] - c[1]); yerr = abs(r["yaw"] - c[2])
        bad = []
        if verr > 0.25: bad.append("speed")
        if yerr > 0.3: bad.append("yaw")
        if moving and r["legs"] < 12: bad.append("NO STEPPING")
        if moving and r["skate"] > 0.5: bad.append("skating")
        if not moving and r["legs"] > 15: bad.append("moving while stop")
        print(f"({c[0]:+.1f},{c[1]:+.1f},{c[2]:+.1f})          ({r['fwd']:+.2f},{r['lat']:+.2f}) {r['yaw']:+.2f} {r['legs']:5.0f}° {r['skate']:5.2f} {r['jumps_s']:7.1f}   {'ok' if not bad else ', '.join(bad)}")


if __name__ == "__main__":
    main()
