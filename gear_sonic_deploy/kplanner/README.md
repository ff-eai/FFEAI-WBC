# FF Master kinematic planner (ZMQ plug-in)

Keyboard-driven locomotion for the FF Master SONIC policy without changing the
deploy binary. The planner is an external process that turns a velocity command
(forward, lateral, yaw rate) into a continuously re-planned whole-body reference
and streams it into `ffmaster_deploy_onnx_ref` on the ZMQ `pose` topic the binary
already consumes (`--input-type zmq` or `manager`). The tracking policy does the
balancing; the planner only decides what human-like motion to follow.

```
keyboard / script                      ffmaster_kplanner_stream.py                    ffmaster_deploy_onnx_ref (unchanged)
 vx, vy, yaw rate  ──────────►  spring model ─► query ─► nearest DB frame ─► clip     ──ZMQ pose v1──►  streamed-motion merger
                                playback + inertialization blend ─► 50 Hz reference                    ─► encoder (mode 0) ─► decoder ─► motors
                                (joint_pos, joint_vel, root quat), 1.5 s ahead of the head
```

## Files

| File | Role |
|---|---|
| `build_db.py` | builds the motion-matching database from motion-library PKLs (FF Master retarget of ChingMU MotionDecode locomotion clips) |
| `planner.py` | runtime core: command smoothing, feature query, nearest-neighbour search, clip switching with inertialization |
| `ffmaster_kplanner_stream.py` | the plug-in: input handling, re-plannable horizon, ZMQ publisher |
| `ffmaster_kplanner_test.py` | closed-loop sim2sim test: simulator + binary + planner, scripted commands, metrics |
| `ffmaster_kinematics.py` | joint-order tables, MuJoCo forward kinematics, quaternion helpers, resampling |
| `eval_openloop.py` | open-loop check of the reference for a set of commands: achieved velocity, leg joint range (is it stepping), foot skating, switches per second |

The database `data/kplanner/ffmaster_loco_db.npz` is installed by
`download_ffmaster_assets.sh` (bundle `ffmaster_planner_db`, skipped with
`--exclude-planner-db`); it is not in git. The section below is only needed to
rebuild or extend it.

## How it works

**Database.** Every frame of the selected clips is resampled to the 50 Hz control
rate and stored with a 27-number feature vector expressed in that frame's heading
frame: root velocity, root position 0.2 / 0.4 / 0.6 s ahead, heading direction at
the same times, and both feet's position and velocity relative to the root
(MuJoCo forward kinematics on the FF Master MJCF). Left/right mirrored copies are
added using a joint mirror map derived automatically by forward kinematics.

**Command to path.** The commanded velocity and yaw rate are smoothed with a
first-order lag (time constants 0.25 s and 0.2 s) and integrated 0.6 s ahead to
give the same trajectory features as the database.

**Search and switch.** Every 0.1 s the query (predicted path + current pose
features) is compared with every searchable database frame by a weighted squared
distance (one matrix-vector product, about 6 ms for 600k frames). The planner
jumps to the best frame only if it beats continuing the current clip by 15 %, at
least 0.4 s have played since the last switch (`--min-play`), and the current
clip is not about to run out (then it must switch). On a jump the new clip is
re-anchored so root position and yaw stay continuous, and the joint difference is
faded out over 0.25 s (inertialization). Pose (foot) feature weights must stay
comparable to the trajectory weights: with trajectory-heavy weights the matcher
re-picks at every search and the blend degenerates into one static pose dragged
along the path. `eval_openloop.py` checks for that.

**Streaming.** Frames carry a global 50 Hz index. The binary plays one frame per
control tick as long as at least 47 frames are buffered beyond the playback head,
and a chunk that starts after the head and extends past the buffered end replaces
the buffered future. The planner keeps 75 frames (1.5 s) ahead, and when the
command changes it regenerates the future from 15 frames (0.3 s) after the
estimated head, so a stick input takes effect after about 0.3 to 0.4 s. Each
message re-sends the previous 10 frames (`--overlap`) because the binary keeps
only the latest message per 10 ms and a lost message would otherwise leave a gap
and trigger a catch-up snap. Only the root quaternion's relative yaw and its
roll/pitch matter to the policy; the binary re-anchors yaw to the robot when
streaming starts.

## Build or extend the database

Sim venv, repository root. The lab data installed by `download_ffmaster_assets.sh`
contains 25 locomotion clips. The database used for the results in this folder
(348 clips plus mirrors, 213 min) adds about 170 locomotion clips from the
training corpus (converted PKLs, not part of the release) and backward-walking
and lateral-walking takes fetched from the public dataset (categories
`1.3.1.Normal_Walking`, of which 294 takes are backward walking at 0.4-0.6 m/s,
and `1.4.6.Lateral_Walking`); `build_db.py` takes any number of `--pkl-dir`
arguments with converted clips, so the commands below reproduce a database of the
same kind from public data alone.

```bash
# optional: more locomotion clips from the public dataset (Isaac Lab env for the converter)
python -c "from huggingface_hub import snapshot_download as d; d('zhiyangrobot/g1-ffmaster-retarget', repo_type='dataset', local_dir='data/chingmu_csv', allow_patterns=['csv/ff_master/1.3.Basic_Gait_Category/1.3.1.Normal_Walking/*', 'csv/ff_master/1.4.Constrained_Gait_Category/1.4.6.Lateral_Walking/*'])"
python tools/chingmu/stage_chingmu.py --root data/chingmu_csv/csv/ff_master --stage data/chingmu/stage
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py --robot ffmaster --input data/chingmu/stage --output data/chingmu/pkl --individual --fps 30 --fps_source 120 --num_workers 12
python tools/chingmu/trim_pkl_last_frame.py --pkl-dir data/chingmu/pkl

# the database (sim venv)
.venv_sim/bin/python gear_sonic_deploy/kplanner/build_db.py \
    --pkl-dir data/ffmaster_motions/lab_train/chingmu_stage_loco \
    --pkl-dir data/chingmu/pkl/1.4.6.Lateral_Walking \
    --pkl-dir data/chingmu/pkl/1.3.1.Normal_Walking \
    --out data/kplanner/ffmaster_loco_db.npz
```

Clips with a per-tick joint step above 18° (solver jumps) and jump/climb
transitions are skipped, and left/right mirrored copies are added
(`--no-mirror` disables them). Keep the database around 0.5 to 1 million
frames: the search is one matrix-vector product over all searchable frames
(about 6 ms for 600k frames on a workstation CPU) and runs up to twelve times
per re-plan.

## Drive the robot in simulation

Three terminals from the repository root (deploy binary built, assets downloaded).

```bash
# T1: simulator
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py --wbc-version ffmaster_sonic_model12

# T2: policy with the ZMQ input
cd gear_sonic_deploy && bash deploy.sh --robot ffmaster --input-type zmq sim
#   ]      start control
#   9      in the MuJoCo window: release the robot
#   ENTER  streaming on  ("ZMQ STREAMING MODE: ENABLED")

# T3: planner
.venv_sim/bin/python gear_sonic_deploy/kplanner/ffmaster_kplanner_stream.py --input keyboard
#   g            go: start streaming; the robot keeps standing until a movement key is held
#   hold w / s   walk forward / backward          hold a / d   step left / right
#   hold q / e   turn left / right                arrows       same as w a s d
#   keys combine while held (w+a = forward-left, w+q = walk in a curve); release = stop
#   1 / 2 / 3    forward speed while w is held: walk 0.8 / jog 1.5 / run 2.2 m/s
#   space        stop            r  restart the stream            x  quit
```

Keys are momentary. On an X11 desktop the planner reads the real key state from the X server, so any combination of held keys works (w+a, w+q, ...) and keys are read whatever window is focused. Over ssh or without a display it falls back to the terminal auto-repeat, where a key counts as held
while its auto-repeat keeps arriving: release is noticed within about 0.2 s
during a hold and about 0.6 s after a short tap (`--first-hold`, `--repeat-hold`).
Speeds: `--walk-speed 0.8 --back-speed 0.7 --lat-speed 0.4 --yaw-rate 0.8`.
Keys typed before `g` are discarded. The corpus has backward walking only at
0.4-0.6 m/s, so `s` walks backward at that speed in every speed mode; there is
no backward jog or run data. Diagonal strafing (w+a) has no human takes either,
so the nearest match is a curve or a slow strafe.

A gamepad input is planned; the planner core is input-agnostic (it only needs
forward speed, lateral speed and yaw rate).

Under `--input-type manager` (the default of `deploy.sh`) press `#` in T2 to
switch to the ZMQ sub-interface before `]` and ENTER.

## Automated test

```bash
.venv_sim/bin/python gear_sonic_deploy/kplanner/ffmaster_kplanner_test.py            # built-in walk/jog/turn/stop script
.venv_sim/bin/python gear_sonic_deploy/kplanner/ffmaster_kplanner_test.py --onscreen  # with the MuJoCo viewer
.venv_sim/bin/python gear_sonic_deploy/kplanner/ffmaster_kplanner_test.py --script cmds.json
```

`cmds.json` is `[[duration_s, vx, vy, yaw_rate], ...]`. The test reports, per
segment, commanded vs achieved forward/lateral speed and yaw rate, minimum pelvis
height, falls, and the joint-tracking RMS between the streamed reference and the
simulated robot. Output: `data/kplanner/test_runs/<UTC>/{results.json, run.npz, deploy.log}`.

## On the robot

Run the planner on a laptop with a desktop session (key combinations need the X
key state) and let the binary on the Orin subscribe to it over the LAN:

```bash
# laptop, repository root: publishes on tcp://<laptop-ip>:5556
.venv_sim/bin/python gear_sonic_deploy/kplanner/ffmaster_kplanner_stream.py --input keyboard

# Orin: third argument (or env ZMQ_HOST) = host of the reference stream
~/sonic_deployment/ffmaster_sonic.sh ffmaster_set8 050000 <laptop-ip>
```

Then `#`, `]`, ENTER on the Orin policy terminal and `g` on the laptop. The Orin
needs the checkpoint pair under `~/sonic_deployment/checkpoints/` and the updated
`ffmaster_sonic.sh`. Test in simulation first, and keep the gantry procedure of
the lab handout.
