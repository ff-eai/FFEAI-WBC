# Lab — Learned Whole-Body Control on FF Master with SONIC

## Challenge

Run a trained humanoid controller in simulation, train one yourself, evaluate it, carry it into a second simulator, and run it on the real FF Master.

## Learning Goals

- Explain what the SONIC policy computes from a reference motion and the robot's state.
- Launch training, read its progress signals, and save checkpoints.
- Compare two checkpoints with success rate and tracking error rather than reward.
- Explain why sim2sim is stronger evidence than the training simulator alone.
- Engage, run and stop the policy on hardware following the safety procedure.

## Background

A humanoid cannot be driven by replaying recorded joint angles: it topples within a step, because nothing reacts to the robot tipping over.

SONIC is a neural network trained to solve this. Fifty times a second it receives a window of the **reference motion** (what we want over the next second) and the robot's **recent state** (joint angles and speeds, gravity direction, its last commands), and outputs 29 **joint-position targets** that a PD controller on each motor tracks.

```
reference motion (next ~1 s) + robot state (last 0.2 s)
                     │
                     ▼
                   SONIC  (neural network, 50 Hz)
                     │
                     ▼
           29 joint-position targets → PD control → FF Master
```

It was trained in Isaac Lab, a GPU simulator that steps thousands of copies of FF Master at once. Each copy is an **environment**; each step gives the policy an **observation**, receives an **action**, and produces a **reward** (higher when tracking is good; severe failures such as falling terminate the episode). A batch of this experience from all environments is a **rollout**; after each rollout, **PPO** adjusts the network so higher-reward actions become more likely; one rollout plus one update is an **iteration**; the weights saved every N iterations are a **checkpoint**. The training reward only says how the policy scored while practising. Whether it works is answered by **evaluation** (fixed clips, success and position error), by **sim2sim** (the exported network driving a different physics engine, MuJoCo, through the same C++ program that runs on the robot), and finally by the robot.

```
Isaac Lab training → checkpoint → evaluation → ONNX export → MuJoCo sim2sim → FF Master
```

Conventions: commands run from `~/FFEAI-WBC`; `conda activate sonic` is the Isaac Lab environment; `.venv_sim/bin/python` is the MuJoCo environment. The robot is FF Master; in code it is `ffmaster`.

## Before You Start

- A prepared workstation (`instructor_setup.md`): repository, both environments, the built binary, the released model, robot meshes, reference sets and lab data.
- Part D additionally needs the robot, the gantry and the instructor.

## Part A — Run the released policy in simulation

```bash
cd ~/FFEAI-WBC
.venv_sim/bin/python gear_sonic_deploy/sim2sim_verify/ffmaster_control_panel.py \
    --motion-dir reference/ffmaster_set8/ --with-replay
```

Three windows open: the panel, the simulator with FF Master hanging from an elastic band, and a second window that will play the reference. In the panel: **Start policy** (`]`) and wait for the log; **Drop robot** (`9`), the robot should stand at about 0.65 m pelvis height; select slot `00` (waving) and **Play** (`t`). Both windows play: the reference window shows what was asked, the policy window what the robot did. Repeat with slot `06` (walking) and `08` (basketball throw).

Close the panel, then measure joint-tracking RMS: the error between the clip's reference joint angles and the joint angles the simulated robot actually reached (the tool starts its own simulator and policy):

```bash
V=".venv_sim/bin/python gear_sonic_deploy/sim2sim_verify/ffmaster_verify.py"
$V rms --motion-dir reference/ffmaster_set8/ --motion 00_SII_Waving_00069
$V rms --motion-dir reference/ffmaster_set8/ --motion 06_BG_Normal_Walking_00547
```

Each ends with `[rms] <clip>: all-DoF X deg, legs Y, arms Z` (degrees). Record both.

## Part B — Train

For lab time constraints, you will fine-tune the released SONIC checkpoint rather than train the policy from random initialization: 50 clips, 2,048 environments, one GPU, 1,000 iterations. Pick a team tag.

```bash
conda activate sonic
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_ffmaster \
    +checkpoint=sonic_ffmaster/model_step_006000.pt \
    num_envs=2048 headless=True exp_var=lab_teamA \
    use_wandb=true wandb.wandb_project=sonic-labs \
    callbacks.model_save.save_frequency=200 \
    ++algo.config.num_learning_iterations=1000 \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/ffmaster_motions/lab_train
```

Start-up takes a few minutes; then a panel refreshes every iteration. Note the `Logging Directory` line; that folder receives your checkpoints (`model_step_000200.pt` … `model_step_001000.pt`). Record `Learning iteration`, `Computation … steps/s` and the `Mean episode …` reward lines at iterations 1, 100 and 500. In W&B plot `Episode_Reward/tracking_vr_5point_local` (1.0 = perfect body tracking) and `Episode_Reward/time_out` (fraction of episodes that finished their clip).

One iteration takes a few seconds; the run needs about one to two hours. Leave it running and continue when it finishes.

## Part C — Evaluate, export, sim2sim

Evaluate the earliest and the last checkpoint on the ten held-back clips:

```bash
RUN=logs_rl/TRL_FFMaster/manager/universal_token/all_modes/sonic_ffmaster_lab_teamA-<timestamp>
for STEP in 000200 001000; do
python gear_sonic/eval_agent_trl.py +checkpoint=$RUN/model_step_$STEP.pt +headless=True \
    ++eval_callbacks=im_eval ++run_eval_loop=False ++num_envs=10 \
    ++eval_output_dir=results/eval_$STEP \
    "+manager_env/terminations=tracking/eval" \
    "++manager_env.terminations.ee_body_pos.params.body_names=[left_ankle_roll_link,right_ankle_roll_link,left_wrist_roll_link,right_wrist_roll_link]" \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/ffmaster_motions/lab_eval \
    ++manager_env.commands.motion.motion_lib_cfg.multi_thread=false
done
python - <<'EOF'
import json
for step in ("000200", "001000"):
    d = json.load(open(f"results/eval_{step}/metrics_eval.json")); m = d["eval/all_metrics_dict"]
    print(f"\n{step}: success {d['eval/success/success_rate']:.2f}  mpjpe_l {d['eval/all/mpjpe_l']:.1f} mm  mpjpe_g {d['eval/all/mpjpe_g']:.1f} mm")
    for k, t, e in zip(m["motion_keys"], m["terminated"], m["mpjpe_l"]):
        print(f"  {k:45s} {'FAIL' if t else 'ok  '} {e:7.1f} mm")
EOF
```

`mpjpe_l` is body-position error relative to the pelvis (shape of the motion), `mpjpe_g` in the world (also counts drifting away). `FAIL` means the robot drifted more than 0.25 m or tilted more than 1 rad and the episode was cut.

Export both checkpoints and run them in MuJoCo through the C++ runtime (use absolute paths; the sweep reports a pelvis-height pass/fail per clip, `rms` the joint tracking):

```bash
for STEP in 000200 001000; do
python gear_sonic/eval_agent_trl.py +checkpoint=$RUN/model_step_$STEP.pt +headless=True ++num_envs=1 +export_onnx_only=true
E=$PWD/$RUN/exported/model_step_${STEP}_encoder.onnx; D=$PWD/$RUN/exported/model_step_${STEP}_decoder.onnx
$V sweep --motion-dir reference/ffmaster_set8/ --encoder $E --decoder $D
cp gear_sonic_deploy/sim2sim_verify/last_sweep.json results/eval_$STEP/sweep.json
$V rms --motion-dir reference/ffmaster_set8/ --motion 06_BG_Normal_Walking_00547 --encoder $E --decoder $D
done
```

Fill in: checkpoint × (Isaac success, Isaac mpjpe_l, MuJoCo sweep n/10, MuJoCo walk RMS all/legs/arms), plus a row for the released model from Part A.

## Part D — Run on the real FF Master

**Rules.** Run the steps in the order below and complete each check before the next step. Everyone not at a keyboard stays outside the robot's reach. Only the released model runs on the robot.

**Rig.** Robot on the gantry with the line slack so the feet carry its weight; knee joints under load. A hanging robot is outside the training distribution and will flail.

**Stop and abort.** Anything unexpected: `Ctrl-C` in T-B; the bridge switches the motors to damping within a quarter second and the robot goes soft onto the gantry. The robot has no e-stop and its own controller has no command timeout, so the bridge's damping and the gantry are the only protection. Never stop the bridge before the policy.

On the Orin, three shells:

```
T-op:  source ~/sonic_deployment/ffmaster_env.sh
       bash  ~/sonic_deployment/ffmaster_env_gate.sh
       python3 ~/sonic_deployment/ffmaster_snapshot.py "PRE-SESSION"   # state Business, 4 command publishers, MC publisher 1, robot still
       python3 ~/sonic_deployment/ffmaster_migrate.py Develop_MC        # ~7 s, no motion
       python3 ~/sonic_deployment/ffmaster_snapshot.py "POST-MIGRATE"  # 0 command publishers, MC publisher 0 → bus FREE
T-A:   ~/sonic_deployment/ffmaster_bridge.sh                            # wait for: bus check 3/3 free → ACTIVE
T-B:   ~/sonic_deployment/ffmaster_sonic.sh chingmu_ffmaster_gantry_small 006000   # Found 3 motion folders … Init Done
T-op:  python3 ~/sonic_deployment/ffmaster_snapshot.py "PRE-ENGAGE"    # the five gates below
T-B:   ]  (wait 10–20 s)   p / n select, read the name aloud   t play   r frame 0   o stop   Ctrl-C abort
T-A:   Ctrl-C                                                           # damping drain → FINISHED
T-op:  python3 ~/sonic_deployment/ffmaster_migrate.py Ready             # robot controller returns
       python3 ~/sonic_deployment/ffmaster_snapshot.py "POST-SESSION"
```

**Mode switch check.** `ffmaster_migrate.py Develop_MC` hands the joints from the robot's own motion controller (MC) to the bridge; nothing may command the robot until that has verifiably happened. The migrate output must end with a `t+Ns state : Develop_MC` line, and the `POST-MIGRATE` snapshot must show `SYSTEM state=Develop_MC`, `cmd publishers … (total 0)`, `mc/common/state publishers: 0` and `-> bus FREE (MC silent)`. `bus OWNED by the MC` means the switch did not take: wait a few seconds, snapshot again, re-run the migrate. `state is Develop_MC but publishers exist` means the MC is still talking on the bus: do not start the bridge, snapshot again until it is silent. `Develop_MC` has been seen reverting to `Business` on its own, so the bridge checks the bus three times before going `ACTIVE` and drops to damping if the MC reappears. After `migrate Ready` at the end, the `POST-SESSION` snapshot must show the MC publisher back (`bus OWNED by the MC`); never leave the robot in `Develop_MC`.

Pre-engage gates, all five before `]`: pelvis tilt under 3°; robot still (max joint velocity ≈ 0.01 rad/s); feet loaded (knee efforts −4 N·m or more negative); bus ours (`mc/common/state` publishers = 0); pose near default (hip −0.31, knee 0.67–0.76, elbow −0.31 rad). If one fails, re-rig and snapshot again.

Run the waving clip three times, then bowing. Record the session from a fourth shell started before `]`:

```bash
source ~/sonic_deployment/ffmaster_env.sh
python3 ~/sonic_deployment/ffmaster_record.py --note "lab 006000 gantry_small"     # Ctrl-C prints the session dir
```

Locomotion needs the robot off the gantry in open floor space with a spotter; the instructor decides.

Back on the workstation, with the session directory and the matching `bridge_cmd_*.csv` copied next to it:

```bash
python3 gear_sonic_deploy/hardware_bringup/ffmaster_eval.py <session_copy> \
    --reference gear_sonic_deploy/reference/chingmu_ffmaster_gantry_small --motion SII_Waving_00449 --plots
$V rms --motion-dir reference/chingmu_ffmaster_gantry_small/ --motion SII_Waving_00449
```

Compare the hardware `reference RMSE (rad)` (× 57.3 for degrees) with the MuJoCo RMS for the same clip, and read the per-joint plots.

## Check-off / Expected Result

- A: FF Master stands after the drop and plays slots `00`, `06`, `08` without falling; two RMS lines recorded.
- B: run reaches iteration 1,000; five numbered checkpoints exist; two W&B plots.
- C: two `metrics_eval.json` and two `sweep.json`; the comparison table with a one-paragraph verdict.
- D: `POST-MIGRATE` snapshot showing `bus FREE`; pre-engage snapshot with all five gates passing; waving played three times without an abort; a session directory with an evaluation report; `POST-SESSION` snapshot with the MC back.

## Questions

1. In one sentence: what does the network compute at each step, from what inputs? Why can it not be replaced by replaying the reference angles?
2. The tracking reward rose during your run. What else could raise it without the robot moving more like the reference, and which Part C number rules that out?
3. On hardware, was the remaining error mostly "motors did not follow the targets" (PD RMSE) or "the policy did not ask for the reference" (reference RMSE)? Name two physical effects the simulator did not model.

## What to Submit

- Part A RMS table; Part B panel values at three iterations and the two reward plots; Part C comparison table and verdict; Part D snapshots (post-migrate, pre-engage, post-session) and one per-joint plot with the largest-error joints marked.
- Answers to the three questions.

## Troubleshooting

| Problem | Likely cause | What to check |
|---|---|---|
| Panel log repeats `LowState is not available` | policy and simulator not talking | close everything, make sure no other simulator or `ffmaster_verify` is running, restart |
| Robot collapses when dropped | policy not started before the drop | Start policy, wait for the log, then Drop |
| Training stops at start with a motion-library error | wrong `motion_file` | `ls data/ffmaster_motions/lab_train` shows `.pkl` files |
| Evaluation hangs after loading / no `metrics_eval.json` | `++run_eval_loop=False` or `++eval_output_dir` missing | copy the command exactly |
| `ffmaster_verify` fails at init with a dimension error | ONNX pair from a different encoder set | use a `sonic_ffmaster` checkpoint; the observation config in `policy/ffmaster/` matches it |
| Bridge stays at `bus NOT free` | robot's own controller still owns the joints | snapshot shows `bus OWNED by the MC`; run `ffmaster_migrate.py Develop_MC` again and re-check |
| Robot flails after `]` | hanging, not standing | `Ctrl-C` at once; lower the robot until knee efforts are negative |
