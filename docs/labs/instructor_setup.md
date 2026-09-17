# Instructor Setup

Prepare each workstation before the first session (about half a day for the first one, an hour per additional one) and the robot side before Part D.

## Workstation

NVIDIA GPU with 24 GB or more, Ubuntu 22.04, CUDA 12.x.

```bash
git clone -b ffmaster_sonic https://github.com/ff-eai/FFEAI-WBC.git ~/FFEAI-WBC && cd ~/FFEAI-WBC && git lfs pull
bash download_ffmaster_assets.sh            # model -> policy/ffmaster + sonic_ffmaster/, robot meshes, reference sets, lab data
```

| Environment | Used by | Install |
|---|---|---|
| `sonic` conda env: Isaac Lab 2.3.x + `pip install -e "gear_sonic/[training]"` (Python 3.11) | Parts B and C | NVIDIA [Installation (Training)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html); name the env `sonic` so `conda activate sonic` in the handout works |
| `.venv_sim` | Parts A and C | `bash install_scripts/install_mujoco_sim.sh` |
| Deploy build (TensorRT 10.13 x86_64, `just`) | Parts A and C | `cd gear_sonic_deploy && ./scripts/install_deps.sh && source scripts/setup_env.sh && HAS_ROS2=0 just build` → `target/release/ffmaster_deploy_onnx_ref` (`HAS_ROS2=0` avoids a ROS 2 input-handler build that fails on workstations with ROS 2 installed) |

The download script places the released model where every tool expects it: ONNX pair and `observation_config.yaml` in `gear_sonic_deploy/policy/ffmaster/`, PyTorch checkpoint and config in `sonic_ffmaster/`. Reference sets go to `gear_sonic_deploy/reference/`, the lab PKLs to `data/ffmaster_motions/{lab_train,lab_eval}` (50 gesture and locomotion clips; the ten `ffmaster_set8` clips). Start the control panel once so the TensorRT engines are compiled and cached next to the ONNX files.

Weights & Biases: create a class project (the handout uses `sonic-labs`) and run `wandb login` on each workstation, or have students add `WANDB_MODE=offline` and sync later.

Timing to measure before class on the class GPU: seconds per iteration of the Part B command (expect one to two hours for 1,000 iterations at 2,048 environments on a 48 GB card) and the duration of one Part C evaluation on `lab_eval` (a few minutes plus Isaac Sim start-up). Consider pre-running one Part B run so a team whose run fails still has checkpoints.

Pre-class checklist:

- [ ] `ffmaster_control_panel.py --motion-dir reference/ffmaster_set8/ --with-replay`: policy starts, robot stands after Drop, slot `00` plays
- [ ] `ffmaster_verify.py sweep --motion-dir reference/ffmaster_set8/` runs to `DONE`
- [ ] Part B command runs 20 iterations with `++algo.config.num_learning_iterations=20 callbacks.model_save.save_frequency=10` and writes a checkpoint
- [ ] Part C evaluation and export commands produce `metrics_eval.json` and the ONNX pair; `ffmaster_verify.py sweep --encoder … --decoder …` accepts them

## Robot side

Prerequisites: FF Master firmware exposing the `Develop_MC` system state and the ROS 2 HAL topics; Orin with JetPack 6, ROS 2 Humble, TensorRT 10.7; ONNX Runtime 1.16.3 aarch64 under `~/.local/onnxruntime`; the robot's message package at `/agibot/software/common`. Gantry, clear floor, a laptop with SSH to the Orin.

`~/sonic_deployment` is not a folder in the repository or in the release. `make_robot_bundle.sh` builds it on the workstation from the repository's `gear_sonic_deploy/` tree, the released ONNX pair, the reference sets and the msgpack headers, and packs it as one archive that unpacks to `~/sonic_deployment` on the Orin.

```bash
# on the workstation, after download_ffmaster_assets.sh and git lfs pull
cd ~/FFEAI-WBC && bash gear_sonic_deploy/hardware_bringup/make_robot_bundle.sh     # -> robot_bundle/sonic_deployment_006000.tar.gz (~120 MB)
scp robot_bundle/sonic_deployment_006000.tar.gz <orin>:~
ssh <orin> 'tar xzf sonic_deployment_006000.tar.gz'                                # unpacks to ~/sonic_deployment
```

The archive holds the session scripts at its root, the `gear_sonic_deploy/` tree without build outputs and G1 files, `checkpoints/model_step_006000_{encoder,decoder}.onnx`, every reference set found under `gear_sonic_deploy/reference/`, the msgpack headers in `thirdparty_headers/` (the robot has no `libmsgpack-dev`; `ffmaster_env.sh` puts that folder on the include path) and the FF Master MJCF for `ffmaster_eval.py`. `--step` and `--onnx-dir` bundle a different exported pair, `--sets` limits the reference sets, `--msgpack` uses a local msgpack-c copy when the workstation is offline; `-h` lists them all. Unpacking over an existing `~/sonic_deployment` keeps `ws/` and `build/`; rebuild after a code change.

On the Orin:

```bash
source ~/sonic_deployment/ffmaster_env.sh
mkdir -p ~/sonic_deployment/ws/src && cd ~/sonic_deployment/ws/src
ln -sfn ~/sonic_deployment/gear_sonic_deploy/src/ffmaster/sonic_ffmaster_bridge . && ln -sfn ~/sonic_deployment/gear_sonic_deploy/src/ffmaster/sonic_ffmaster_dummy .
cd ~/sonic_deployment/ws && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
cd ~/sonic_deployment/gear_sonic_deploy/build && cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release && cmake --build . -j6
```

If the googletest download fails on the lab network, fetch googletest 1.14.0 once and pass `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<path>`. A firmware flash wipes `$HOME`; redo this section after one. After every robot reboot run `~/sonic_deployment/ffmaster_dds_guard.sh`.

Before the first class on a unit: run `ffmaster_observe.py inventory`, `fingerprint --trials 8` and `imu` (read-only) to confirm joint order, signs and the pelvis IMU topic; run the single-joint `sonic_ffmaster_dummy` node on the gantry to exercise the bridge and damping path; run Part D yourself with `chingmu_ffmaster_gantry_small` so the TensorRT engines are baked (first start, about three minutes); decide whether the class runs locomotion off the gantry, and if so run `chingmu_ffmaster_loco_small` yourself first with a spotter.

The students run the session themselves in the order of the handout; the instructor supervises and decides whether locomotion runs off the gantry. The abort is `Ctrl-C` in T-B; the robot has no e-stop, so the bridge's damping and the gantry are the only protection.

## Notes for developers

- `ffmaster_control_panel.py`, `ffmaster_replay_viewer.py` and `ffmaster_verify.py` derive the repository root from their own location and pass `policy/ffmaster/observation_config.yaml` to the binary; a model with a different encoder set needs that file swapped.
- `ffmaster_verify.py` overwrites `last_sweep.json` / `last_rms_traj.npz` on every run; copy results out.
- The motion library keys clips by file name; duplicate stems overwrite each other (`tools/chingmu/stage_chingmu.py` handles this for the full dataset). MotionDecode CSVs carry a corrupt final row; `tools/chingmu/trim_pkl_last_frame.py` removes the last converted frame of every clip.
- `+checkpoint=` loads weights only (fresh optimizer, iteration counter from zero); `++resume=True` restores the full trainer state.
- Above roughly 8k environments per GPU pass `++manager_env.config.gpu_max_rigid_patch_count=655360`, or PhysX drops foot contacts silently.
- On the robot, start the policy only through `ffmaster_sonic.sh`; it checks with `ldd` that the binary resolves the SDK's CycloneDDS.
- Wrist columns of the retargeted dataset are zero; the policy holds the wrists at the default pose, so wrist error in hardware reports is expected.
