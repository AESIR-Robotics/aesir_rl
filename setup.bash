#!/usr/bin/env bash
# setup.bash — installs all system and Python dependencies for the Aesir RL stack.
# Assumes ROS (Humble or Jazzy) is already installed.
# Usage:
#   bash setup.bash           # auto-detect distro
#   bash setup.bash humble    # force Humble
#   bash setup.bash jazzy     # force Jazzy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}/workspace"
WORKSPACE_SRC="${WORKSPACE_ROOT}/src"
VENV_DIR="${SCRIPT_DIR}/.venv"

# --------------------------------------------------------------------------- #
#  Distro detection
# --------------------------------------------------------------------------- #
detect_distro() {
    if [[ -n "${ROS_DISTRO:-}" ]]; then
        echo "$ROS_DISTRO"
        return
    fi
    local ubuntu_ver
    ubuntu_ver=$(lsb_release -rs 2>/dev/null || echo "0")
    case "$ubuntu_ver" in
        22.04) echo "humble" ;;
        24.04) echo "jazzy"  ;;
        *)
            echo "[ERROR] Cannot auto-detect ROS distro from Ubuntu $ubuntu_ver." >&2
            echo "        Pass the distro explicitly: bash setup.bash humble|jazzy" >&2
            exit 1
            ;;
    esac
}

DISTRO="${1:-$(detect_distro)}"

if [[ "$DISTRO" != "humble" && "$DISTRO" != "jazzy" ]]; then
    echo "[ERROR] Unsupported distro '$DISTRO'. Only 'humble' and 'jazzy' are supported." >&2
    exit 1
fi

ROS_PREFIX="ros-${DISTRO}"
echo "==> Setting up for ROS 2 ${DISTRO^}"

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
apt_install() {
    sudo apt-get install -y --no-install-recommends "$@"
}

# Install an apt package only if it exists in the cache; print a warning otherwise.
apt_install_optional() {
    local pkg="$1"
    if apt-cache show "$pkg" &>/dev/null; then
        apt_install "$pkg"
    else
        echo "[WARN] apt package '$pkg' not found — skipping."
    fi
}

# --------------------------------------------------------------------------- #
#  1. System build tools + rosdep
# --------------------------------------------------------------------------- #
echo ""
echo "--- [1/5] System build tools ---"
sudo apt-get update -qq
apt_install \
    python3-pip \
    python3-venv \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    python3-argcomplete \
    build-essential \
    cmake \
    git \
    lsb-release \
    curl

# --------------------------------------------------------------------------- #
#  2. ROS core + messages
# --------------------------------------------------------------------------- #
echo ""
echo "--- [2/5] ROS core and message packages ---"
apt_install \
    "${ROS_PREFIX}-rclpy" \
    "${ROS_PREFIX}-rclcpp" \
    "${ROS_PREFIX}-std-msgs" \
    "${ROS_PREFIX}-geometry-msgs" \
    "${ROS_PREFIX}-sensor-msgs" \
    "${ROS_PREFIX}-nav-msgs" \
    "${ROS_PREFIX}-rosgraph-msgs" \
    "${ROS_PREFIX}-trajectory-msgs" \
    "${ROS_PREFIX}-builtin-interfaces" \
    "${ROS_PREFIX}-tf2-ros" \
    "${ROS_PREFIX}-robot-state-publisher" \
    "${ROS_PREFIX}-joint-state-publisher" \
    "${ROS_PREFIX}-joint-state-publisher-gui" \
    "${ROS_PREFIX}-xacro" \
    "${ROS_PREFIX}-launch-ros" \
    "${ROS_PREFIX}-rviz2" \
    "${ROS_PREFIX}-rviz-common" \
    "${ROS_PREFIX}-rviz-default-plugins"

# --------------------------------------------------------------------------- #
#  3. ros2_control + controllers
# --------------------------------------------------------------------------- #
echo ""
echo "--- [3/5] ros2_control + controllers ---"
apt_install \
    "${ROS_PREFIX}-ros2-control" \
    "${ROS_PREFIX}-ros2-controllers" \
    "${ROS_PREFIX}-controller-manager" \
    "${ROS_PREFIX}-joint-state-broadcaster" \
    "${ROS_PREFIX}-joint-trajectory-controller" \
    "${ROS_PREFIX}-velocity-controllers" \
    "${ROS_PREFIX}-forward-command-controller" \
    "${ROS_PREFIX}-diff-drive-controller"

# --------------------------------------------------------------------------- #
#  4. MoveIt 2 + Servo
#     warehouse-ros-mongo is optional (only needed for MoveIt scene persistence
#     via the warehouse_db launch). It is NOT in the standard Ubuntu apt repos
#     for Humble — skip it gracefully.
# --------------------------------------------------------------------------- #
echo ""
echo "--- [4/5] MoveIt 2 ---"
apt_install \
    "${ROS_PREFIX}-moveit" \
    "${ROS_PREFIX}-moveit-ros-move-group" \
    "${ROS_PREFIX}-moveit-kinematics" \
    "${ROS_PREFIX}-moveit-planners" \
    "${ROS_PREFIX}-moveit-simple-controller-manager" \
    "${ROS_PREFIX}-moveit-configs-utils" \
    "${ROS_PREFIX}-moveit-ros-visualization" \
    "${ROS_PREFIX}-moveit-ros-warehouse" \
    "${ROS_PREFIX}-moveit-setup-assistant" \
    "${ROS_PREFIX}-moveit-servo"

# warehouse-ros-mongo: optional, not always in apt
apt_install_optional "${ROS_PREFIX}-warehouse-ros-mongo"

# mujoco_ros2_control: available in apt since late 2024
apt_install_optional "${ROS_PREFIX}-mujoco-ros2-control"
if ! apt-cache show "${ROS_PREFIX}-mujoco-ros2-control" &>/dev/null; then
    echo ""
    echo "       To build from source:"
    echo "         git clone https://github.com/moveit/mujoco_ros2_control.git \\"
    echo "               --branch ${DISTRO}  <workspace>/src/"
    echo "         colcon build --packages-select mujoco_ros2_control"
fi

# --------------------------------------------------------------------------- #
#  5. Python venv + packages
#     The venv uses --system-site-packages so rclpy and all ROS Python
#     libraries installed by apt remain visible inside it.
# --------------------------------------------------------------------------- #
echo ""
echo "--- [5/5] Python venv + packages (${VENV_DIR}) ---"

# Source ROS before creating the venv so the system site-packages include
# the correct ROS Python path from the start.
# shellcheck source=/dev/null
source "/opt/ros/${DISTRO}/setup.bash"

if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo "    Created venv at ${VENV_DIR}"
else
    echo "    Reusing existing venv at ${VENV_DIR}"
fi

# All pip installs go into the venv from here on.
VENV_PIP="${VENV_DIR}/bin/pip"

# Pin setuptools<80 to stay compatible with colcon-core's declared constraint.
"$VENV_PIP" install --upgrade pip "setuptools<80" wheel

# --ignore-installed forces pip to install into the venv even when the same
# package exists in the system site-packages (e.g. Ubuntu's torch 1.8 stub).
PIP_INSTALL="${VENV_PIP} install --ignore-installed"

# PyTorch installation.
# - CPU path: goes directly to PyPI to avoid depending on download.pytorch.org.
# - CUDA path: tries the PyTorch CDN (CUDA wheels are not on PyPI); falls back
#   to PyPI CPU build if the CDN is unreachable (firewall / air-gapped env).
install_torch() {
    local index_url="${1:-pypi}"
    if [[ "$index_url" == "pypi" ]]; then
        echo "    Installing torch>=2.0 from PyPI..."
        $PIP_INSTALL "torch>=2.0" "torchvision>=0.15"
    else
        echo "    Installing torch>=2.0 from ${index_url} ..."
        $PIP_INSTALL "torch>=2.0" "torchvision>=0.15" --index-url "$index_url"
    fi
}

if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
               | head -1 || echo "")
    if python3 -c \
        "v='${CUDA_VER}'; major=int(v.split('.')[0]) if v else 0; exit(0 if major>=525 else 1)" \
        2>/dev/null; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi
    echo "    NVIDIA GPU detected — attempting CUDA torch (${TORCH_INDEX})"
    install_torch "$TORCH_INDEX" \
        || { echo "    CDN unreachable, falling back to PyPI CPU build..."; install_torch pypi; }
else
    echo "    No GPU detected — installing CPU torch from PyPI"
    install_torch pypi
fi

# Remaining Python dependencies
$PIP_INSTALL \
    numpy \
    mujoco \
    mediapy \
    matplotlib \
    wandb

# --------------------------------------------------------------------------- #
#  rosdep — install remaining declared deps for the workspace
# --------------------------------------------------------------------------- #
echo ""
echo "--- rosdep install ---"

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init
fi
rosdep update --rosdistro "$DISTRO"
rosdep install \
    --from-paths "$WORKSPACE_SRC" \
    --ignore-src \
    --rosdistro "$DISTRO" \
    -y \
    --skip-keys "mujoco_ros2_control python3-torch warehouse_ros_mongo"

# --------------------------------------------------------------------------- #
#  Build the workspace (using the venv's Python)
# --------------------------------------------------------------------------- #
echo ""
echo "--- colcon build ---"
cd "$WORKSPACE_ROOT"
VENV_PYTHON="${VENV_DIR}/bin/python3"
colcon build \
    --symlink-install \
    --cmake-args \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        "-DPYTHON_EXECUTABLE=${VENV_PYTHON}"

# --------------------------------------------------------------------------- #
echo ""
echo "==> Setup complete."
echo ""
echo "    Activate the environment with:"
echo "      source /opt/ros/${DISTRO}/setup.bash"
echo "      source ${VENV_DIR}/bin/activate"
echo "      source ${WORKSPACE_ROOT}/install/setup.bash"
