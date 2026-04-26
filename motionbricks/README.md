# MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-76B900.svg" alt="License"></a>
  <a href="https://nvlabs.github.io/motionbricks"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="docs/motion_representation.md"><img src="https://img.shields.io/badge/docs-online-green.svg" alt="Documentation"></a>
</p>

<p align="center">
  <img src="assets/teaser_motion_bricks_three_rows.jpg" alt="MotionBricks teaser" width="100%">
</p>

MotionBricks is a real-time generative framework that transforms interactive motion control for animation and robotics. By combining a large-scale latent backbone with intuitive "smart primitives," it delivers high-quality, zero-shot motion synthesis at 15,000 FPS, allowing users to effortlessly build complex animations and robotic movements like assembling bricks.

## Contents

- [News & Roadmap](#news--roadmap)
- [Results](#results)
- [Setup](#setup)
- [Interactive Demo: Quick Start](#interactive-demo-quick-start)
- [Training](#training)
- [Motion Representation and Custom Datasets](#motion-representation-and-custom-datasets)
- [Related Work](#related-work)
- [Project Structure](#project-structure)
- [Known Issues](#known-issues)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## News & Roadmap

### News

- **2026-04-27** — Initial public release: interactive demo, pretrained checkpoints (VQVAE · pose · root), synthetic training code, motion-representation docs, and GIF gallery.

### Roadmap

- [ ] Full training pipeline inside [GR00T Whole-Body Control](https://github.com/NVlabs/GR00T-WholeBodyControl)'s GEAR-SONIC pipeline — targeted for approximately one month out; reproducibility experiments are already in flight.

## Results

See the [project page](https://nvlabs.github.io/motionbricks) for the full uncut demos and comparison videos. Short clips below are GIFs (muted, ~10 s each).

### Teasers

| Animation | Robotics |
| :---: | :---: |
| ![Animation teaser](assets/gifs/teaser_animation.gif) | ![Robotics teaser](assets/gifs/teaser_robotics.gif) |

### Smart Locomotion — Single Styles

| Zombie | Injured leg |
| :---: | :---: |
| ![Zombie](assets/gifs/loco_zombie.gif) | ![Injured leg](assets/gifs/loco_injured_leg.gif) |
| **Injured torso** | **Skipping** |
| ![Injured torso](assets/gifs/loco_injured_torso.gif) | ![Skipping](assets/gifs/loco_skipping.gif) |
| **Strafing** | **Crouch strafing** |
| ![Strafing](assets/gifs/loco_strafing.gif) | ![Crouch strafing](assets/gifs/loco_crouch_strafing.gif) |

### Smart Locomotion — Mixture of Styles

| Freestyle | Idle · Walk · Jog · Run |
| :---: | :---: |
| ![Freestyle](assets/gifs/loco_freestyle.gif) | ![Idle / walk / jog / run](assets/gifs/loco_idle_walk_jog_run.gif) |

### Smart Objects

| Pick up sword | Falling |
| :---: | :---: |
| ![Pick up sword](assets/gifs/obj_pickup_sword.gif) | ![Falling](assets/gifs/obj_falling.gif) |
| **Jump over bench** | **Sitting** |
| ![Jump over bench](assets/gifs/obj_jump_bench.gif) | ![Sitting](assets/gifs/obj_sitting.gif) |
| **Interactive authoring** | |
| ![Interactive authoring](assets/gifs/obj_interactive_authoring.gif) | |

## Setup

**Requirements:** Python 3.10+, a CUDA-capable GPU, [Git LFS](https://git-lfs.com/).

### Clone the repository

Pretrained checkpoints and mesh assets are tracked with Git LFS, so install LFS before cloning:

```bash
git lfs install
```

If you only need source code (for example, to train on your own data), you can skip downloading the ~2.3 GB pretrained checkpoints:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <REPO_URL>
cd motionbricks
```

If you want the checkpoints for the interactive demo, clone normally:

```bash
git clone <REPO_URL>
cd motionbricks
```

After cloning, verify that checkpoint files were downloaded (not tiny Git LFS pointer files):

```bash
ls -lh out/G1-clip.ckpt                                     # ~7.5 MB
ls -lh out/motionbricks_vqvae/version_1/checkpoints/*.ckpt  # ~273 MB
ls -lh out/motionbricks_pose/version_1/checkpoints/*.ckpt   # ~1.6 GB
ls -lh out/motionbricks_root/version_1/checkpoints/*.ckpt   # ~391 MB
```

If these files are unexpectedly small (around 1 KB), they are LFS pointers. Run `git lfs pull` to fetch the actual checkpoints.


### Install dependencies

```bash
# Create environment
conda create -n motionbricks python=3.10 -y
conda activate motionbricks

# Install dependencies
pip install -e .

# Linux only: needed for keyboard input and MuJoCo key-grab workaround
pip install pynput python-xlib
```

## Interactive Demo: Quick Start

```bash
DISPLAY=:1 python scripts/interactive_demo_g1.py
```

This launches the MuJoCo viewer with the G1 robot. Use your keyboard to control it in real time. Hold the left mouse button and drag to change the camera look-at direction.

<p align="center">
  <img src="assets/gifs/interactive_demo.gif" alt="Interactive demo screencast" width="480">
</p>

### Movement Controls

| Key | Action |
|-----|--------|
| `W` | Move forward |
| `A` | Move left |
| `S` | Move backward |
| `D` | Move right |

The movement direction is relative to the camera. Rotate the camera by right-clicking and dragging in the MuJoCo viewer.

### Motion Styles

| Key | Style |
|-----|-------|
| `V` | Slow walk |
| `Z` | Hand crawling |
| `X` | Walk boxing |
| `B` | Elbow crawling |
| `R` | Stealth walk |
| `T` | Injured walk |
| `C` | Walk stealth (crouched) |
| `E` | Happy dance walk |
| `F` | Zombie walk |
| `G` | Gun walk |
| `Q` | Scared walk |

Note: crawling modes (`Z` hand crawling and `B` elbow crawling) currently do not support side-only directions.

Without pressing a style key, the default locomotion is: **idle** (no movement keys), **walk** (WASD pressed).

## Training

Training scripts are provided for all three model components. The scripts use synthetic data by default and load model configs from the saved checkpoints in `out/`. The full motion datasets are available at <https://bones.studio/datasets>.

**Full release status:** A full release — a model fully embedded in [GR00T whole-body control](https://github.com/NVlabs/GR00T-WholeBodyControl)'s robotics formulation, along with the complete training pipeline — is targeted for approximately one month out. Reproducibility experiments are already in flight; please check back for updates.

```bash
# Train the VQVAE (motion tokenizer)
python scripts/train_vqvae.py

# Train the pose model (requires pretrained VQVAE checkpoint)
python scripts/train_pose.py

# Train the root model (no VQVAE needed)
python scripts/train_root.py
```

### Dataset

The datasets used to train the pretrained checkpoints can be downloaded at <https://bones.studio/datasets>. All current training scripts default to **synthetic data** (see `motionbricks/data/synthetic_dataset.py`) so that the full training pipeline can be verified end-to-end without the real dataset.

## Motion Representation and Custom Datasets

For details on the motion feature representation, skeleton system, coordinate conventions, normalization, and feature computation pipeline, see [docs/motion_representation.md](docs/motion_representation.md).

For a step-by-step guide to training MotionBricks on your own motion data and adapting it to a new robot, see [docs/adding_your_own_dataset.md](docs/adding_your_own_dataset.md).

## Related Work

**Kimodo** — A sibling project focused on offline motion generation, complementary to MotionBricks' real-time runtime.

[Project page](https://research.nvidia.com/labs/sil/projects/kimodo/) · [GitHub](https://github.com/nv-tlabs/kimodo)

<p align="center">
  <img src="assets/gifs/kimodo_teaser.gif" alt="Kimodo teaser" width="480">
</p>

**GEAR-SONIC** — Together with MotionBricks, GEAR-SONIC anchors NVIDIA's GR00T Whole-Body Control initiative.

[Project page](https://nvlabs.github.io/GEAR-SONIC/) · [GitHub](https://github.com/NVlabs/GR00T-WholeBodyControl)

<p align="center">
  <img src="assets/gifs/sonic_teaser.gif" alt="GEAR-SONIC teaser" width="480">
</p>

**BONES-SEED Dataset** — MotionBricks' training corpus — 350k production-grade mocap clips from real human actors and actresses.

[Dataset page](https://huggingface.co/datasets/bones-studio/seed)

<p align="center">
  <img src="assets/gifs/bones_seed_teaser.gif" alt="BONES-SEED teaser" width="480">
</p>

**SOMA Retargeter** — The Newton-based solver that retargets SOMA capture onto the G1, producing MotionBricks' training data.

[GitHub](https://github.com/NVIDIA/soma-retargeter)

<p align="center">
  <img src="assets/gifs/soma_retargeter_teaser.gif" alt="SOMA Retargeter teaser" width="480">
</p>

## Project Structure

```
motionbricks/
  assets/skeletons/g1/     # MuJoCo XMLs and STL meshes
  motionbricks/            # Python package
  scripts/
    interactive_demo_g1.py # Interactive demo
    train_vqvae.py         # VQVAE training
    train_pose.py          # Pose model training
    train_root.py          # Root model training
  out/                     # Pre-trained checkpoints (Git LFS)
    G1-clip.ckpt
    motionbricks_vqvae/
    motionbricks_pose/
    motionbricks_root/
  setup.py
```

## Known Issues

- **Linux/X11 only:** The keyboard key-grab workaround requires X11 (`python-xlib`). On Wayland, macOS, or Windows, some MuJoCo keyboard shortcuts may conflict with the controller keys. Keep the **terminal focused** (not the MuJoCo window) as a workaround.
- **`PYTORCH_JIT=0` disables key grabs:** Running with `PYTORCH_JIT=0` interferes with the X11 key-grab workaround. If you need `PYTORCH_JIT=0`, keep the terminal focused instead.
- The `pynput` package is required for keyboard input on Linux/macOS. On Windows, the `keyboard` package is used instead.

## Citation

BibTeX coming soon.

## License

Source code in this repository is licensed under **Apache 2.0**. Pretrained model weights are licensed under the **NVIDIA Open Model License**, which permits commercial use with attribution subject to the trustworthy AI requirements. See the `legal/` folder for the full terms.

## Contact

For questions and feedback, please reach out at **`gear-wbc@nvidia.com`**.
