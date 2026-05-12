#!/bin/bash
# install_ros.sh
# Sets up the `teleop_ros` conda env with RoboStack ROS 2 Humble for the
# Isaac Teleop / CloudXR ROS bridge. Pinned to Python 3.10 to compose with
# .venv_teleop (created by install_pico.sh).
#
# Usage:  bash install_scripts/install_ros.sh   (run from repo root)

set -e

ENV_NAME="${1:-teleop_ros}"
PY_VERSION="3.10"

# Source conda's shell hooks so `conda activate` works in a non-interactive script.
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "♻️  Reusing existing conda env: $ENV_NAME"
else
    echo "🆕 Creating conda env '$ENV_NAME' with Python $PY_VERSION..."
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi
conda activate "$ENV_NAME"

echo "🔄 Cleaning up incomplete or cached packages..."
conda clean --packages --tarballs --yes

echo "🔧 Adding RoboStack and conda-forge channels to the current environment..."
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging

# Optional: remove defaults to avoid conflicts (ignore error if not present)
echo "⚙️  Removing 'defaults' channel if present..."
conda config --env --remove channels defaults || true

echo "📦 Installing ROS 2 Humble Desktop from RoboStack..."
conda install -y ros-humble-desktop

echo "✅ Sourcing ROS environment from current conda env..."
source "$CONDA_PREFIX/setup.bash"

# Add ROS setup to bashrc if not already present
SETUP_LINE="source \"\$CONDA_PREFIX/setup.bash\" && export ROS_LOCALHOST_ONLY=1"
if ! grep -q "$SETUP_LINE" ~/.bashrc; then
    echo "📝 Adding ROS setup to ~/.bashrc..."
    echo "$SETUP_LINE" >> ~/.bashrc
    echo "✅ Added ROS setup to ~/.bashrc"
else
    echo "ℹ️ ROS setup already exists in ~/.bashrc"
fi

echo "🧪 Verifying rclpy import..."
python -c "import rclpy; print('✅ rclpy imported')"
