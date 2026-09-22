#!/usr/bin/env bash
# Build the robot-side deployment folder for FF Master's Orin as one archive.
#
#   bash gear_sonic_deploy/hardware_bringup/make_robot_bundle.sh [options]
#
# Run on the workstation after `bash download_ffmaster_assets.sh` (the released
# model) or with --onnx-dir pointing at your own exported encoder/decoder pair.
# Writes <out>/sonic_deployment/ and <out>/sonic_deployment_<step>.tar.gz; the
# archive unpacks to ~/sonic_deployment on the Orin:
#
#   sonic_deployment/
#     ffmaster_*.sh, ffmaster_*.py          session scripts (from hardware_bringup/)
#     gear_sonic_deploy/                    binary sources, ROS 2 packages, SDK libraries,
#                                           policy/ffmaster/observation_config.yaml,
#                                           reference/<set>/ motion lists
#     checkpoints/model_step_<step>_{encoder,decoder}.onnx
#     thirdparty_headers/                   msgpack headers for the binary build
#     gear_sonic/data/assets/robot_description/mjcf/ffmaster_sonic_29dof.xml
#     recordings/                           sessions land here
#
# Options
#   --step NNNNNN     checkpoint step used in the file names (default 006000)
#   --onnx-dir DIR    folder with model_encoder.onnx + model_decoder.onnx (default
#                     gear_sonic_deploy/policy/ffmaster) or a training export folder
#                     with model_step_<step>_{encoder,decoder}.onnx
#   --sets a,b,c      reference sets to include (default: every folder in
#                     gear_sonic_deploy/reference/)
#   --msgpack PATH    msgpack-c source tarball or include/ folder to use instead of
#                     downloading cpp-3.3.0 from GitHub
#   --out DIR         output folder (default robot_bundle/ in the repository)
#   --no-archive      stage the folder only, skip the tar.gz
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY="$REPO/gear_sonic_deploy"
STEP="006000"; ONNX_DIR="$DEPLOY/policy/ffmaster"; SETS=""; MSGPACK=""; OUT="$REPO/robot_bundle"; ARCHIVE=1
MSGPACK_URL="https://github.com/msgpack/msgpack-c/releases/download/cpp-3.3.0/msgpack-3.3.0.tar.gz"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step)       STEP="$2"; shift ;;
    --onnx-dir)   ONNX_DIR="$2"; shift ;;
    --sets)       SETS="$2"; shift ;;
    --msgpack)    MSGPACK="$2"; shift ;;
    --out)        OUT="$2"; shift ;;
    --no-archive) ARCHIVE=0 ;;
    -h|--help)    sed -n 2,31p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done
STEP="${STEP#model_step_}"
die() { echo "ERROR: $*" >&2; exit 1; }

# --- inputs -------------------------------------------------------------------
if [[ -f "$ONNX_DIR/model_encoder.onnx" && -f "$ONNX_DIR/model_decoder.onnx" ]]; then
  ENC="$ONNX_DIR/model_encoder.onnx"; DEC="$ONNX_DIR/model_decoder.onnx"
elif [[ -f "$ONNX_DIR/model_step_${STEP}_encoder.onnx" && -f "$ONNX_DIR/model_step_${STEP}_decoder.onnx" ]]; then
  ENC="$ONNX_DIR/model_step_${STEP}_encoder.onnx"; DEC="$ONNX_DIR/model_step_${STEP}_decoder.onnx"
else
  die "no encoder/decoder pair in $ONNX_DIR (run 'bash download_ffmaster_assets.sh --model' or pass --onnx-dir)"
fi
for f in "$ENC" "$DEC" "$DEPLOY"/thirdparty/unitree_sdk2/lib/aarch64/*.a "$DEPLOY"/thirdparty/unitree_sdk2/thirdparty/lib/aarch64/*.so*; do
  [[ -f "$f" ]] || die "missing $f"
  [[ "$(head -c 7 "$f")" == "version" ]] && die "$f is a Git LFS pointer; run 'git lfs pull' first"
done
MJCF="$REPO/gear_sonic/data/assets/robot_description/mjcf/ffmaster_sonic_29dof.xml"
[[ -f "$MJCF" ]] || die "missing $MJCF"
[[ -f "$DEPLOY/policy/ffmaster/observation_config.yaml" ]] || die "missing policy/ffmaster/observation_config.yaml"
if [[ -z "$SETS" ]]; then
  SETS="$(cd "$DEPLOY/reference" && ls -d */ 2>/dev/null | sed 's#/$##' | grep -v __pycache__ | tr '\n' ',' | sed 's/,$//')"
  [[ -n "$SETS" ]] || die "no reference sets in gear_sonic_deploy/reference/ (run 'bash download_ffmaster_assets.sh --reference')"
fi

# --- stage --------------------------------------------------------------------
STAGE="$OUT/sonic_deployment"
rm -rf "$STAGE"; mkdir -p "$STAGE/checkpoints" "$STAGE/thirdparty_headers" "$STAGE/recordings" \
  "$STAGE/gear_sonic/data/assets/robot_description/mjcf"

rsync -a --delete \
  --exclude /build/ --exclude /target/ --exclude /g1/ --exclude /planner/ --exclude /policy/release/ \
  --exclude '*.onnx' --exclude '*.trt' --exclude '__pycache__/' --exclude 'sim2sim_verify/last_*' \
  --exclude 'reference/*/' \
  "$DEPLOY/" "$STAGE/gear_sonic_deploy/"
IFS=',' read -r -a SET_ARR <<< "$SETS"
for s in "${SET_ARR[@]}"; do
  [[ -d "$DEPLOY/reference/$s" ]] || die "reference set not found: gear_sonic_deploy/reference/$s"
  rsync -a --exclude '__pycache__/' "$DEPLOY/reference/$s/" "$STAGE/gear_sonic_deploy/reference/$s/"
done

cp "$DEPLOY"/hardware_bringup/ffmaster_*.sh "$DEPLOY"/hardware_bringup/ffmaster_*.py "$STAGE/"
cp "$ENC" "$STAGE/checkpoints/model_step_${STEP}_encoder.onnx"
cp "$DEC" "$STAGE/checkpoints/model_step_${STEP}_decoder.onnx"
cp "$MJCF" "$STAGE/gear_sonic/data/assets/robot_description/mjcf/"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
if [[ -z "$MSGPACK" ]]; then
  echo "downloading msgpack headers: $MSGPACK_URL"
  curl -fL --progress-bar -o "$TMP/msgpack.tar.gz" "$MSGPACK_URL"
  MSGPACK="$TMP/msgpack.tar.gz"
fi
if [[ -d "$MSGPACK" ]]; then
  cp -r "$MSGPACK"/. "$STAGE/thirdparty_headers/"
else
  mkdir -p "$TMP/mp" && tar xzf "$MSGPACK" -C "$TMP/mp"
  INC="$(find "$TMP/mp" -maxdepth 2 -type d -name include | head -1)"
  [[ -n "$INC" ]] || die "no include/ folder inside $MSGPACK"
  cp -r "$INC"/. "$STAGE/thirdparty_headers/"
fi
[[ -f "$STAGE/thirdparty_headers/msgpack.hpp" ]] || die "msgpack.hpp not found in thirdparty_headers/"

# --- archive ------------------------------------------------------------------
echo "staged: $STAGE  ($(du -sh "$STAGE" | cut -f1))"
echo "  checkpoints : model_step_${STEP}_{encoder,decoder}.onnx"
echo "  reference   : $SETS"
if [[ $ARCHIVE -eq 1 ]]; then
  TGZ="$OUT/sonic_deployment_${STEP}.tar.gz"
  tar czf "$TGZ" -C "$OUT" sonic_deployment
  echo "archive: $TGZ  ($(du -sh "$TGZ" | cut -f1))"
  cat <<MSG

Next, from the workstation (an existing ws/ and build/ on the Orin are kept):
  scp $TGZ run@<orin>:~
  ssh run@<orin> 'tar xzf $(basename "$TGZ")'        # unpacks to ~/sonic_deployment
then continue with "Robot side", step 3 "Build the bridge and the binary", in docs/labs/instructor_setup.md.
MSG
fi
