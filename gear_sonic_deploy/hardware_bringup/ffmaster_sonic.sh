#!/usr/bin/env bash
# T-B wrapper: start ffmaster_deploy_onnx_ref (the SONIC policy binary).
# (2026-08-13 procedure doc, step 4 — one command instead of six.)
#
# Usage:
#   ./ffmaster_sonic.sh                      # default gantry list, default checkpoint step
#   ./ffmaster_sonic.sh ffmaster_set8              # other motion list, default checkpoint
#   ./ffmaster_sonic.sh ffmaster_set8 006000       # list + checkpoint step
#
#   arg 1  motion list  — name under reference/ (or a full path). Default: chingmu_ffmaster_gantry_small
#   arg 2  checkpoint   — step number under checkpoints/ (or "model_step_NNNNNN").
#                         Default: 006000 (the released ffmaster_sonic_v0.1 checkpoint)
#
# After start, VERIFY in the log:
#   "Found N motion folders"  — N must match the list you asked for
#                               (chingmu_ffmaster_gantry_small=3, chingmu_ffmaster_loco_small=5, ffmaster_set8=10)
#   both engines load from cache in seconds, then "Init Done"
# Keys: ] engage | o stop | p/n motion | t play | r restart | Ctrl-C = ABORT
#
# Works from any shell — ROS sourced or not (the binary RPATH-pins its own
# CycloneDDS; the script verifies the actual library resolution before start).

SONIC=~/sonic_deployment
DEPLOY="$SONIC/gear_sonic_deploy"

# ---- arguments --------------------------------------------------------------
MOTIONS="${1:-chingmu_ffmaster_gantry_small}"
CKPT="${2:-006000}"

case "$MOTIONS" in
  */*) MOTION_DIR="$MOTIONS" ;;                       # path given
  *)   MOTION_DIR="reference/$MOTIONS/" ;;            # list name given
esac
case "$CKPT" in
  model_step_*) STEP="$CKPT" ;;
  *)            STEP="model_step_$CKPT" ;;
esac

DECODER="$SONIC/checkpoints/${STEP}_decoder.onnx"
ENCODER="$SONIC/checkpoints/${STEP}_encoder.onnx"

# ---- validation --------------------------------------------------------------
cd "$DEPLOY"
if [ ! -d "$MOTION_DIR" ]; then
  echo "ERROR: motion list not found: $DEPLOY/$MOTION_DIR"
  echo "       available:"; ls -d reference/*/ | sed 's/^/         /'
  exit 1
fi
for f in "$DECODER" "$ENCODER"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: checkpoint file not found: $f"
    echo "       available:"; ls "$SONIC/checkpoints/" | sed 's/^/         /'
    exit 1
  fi
done
N_MOTIONS=$(ls -d "$MOTION_DIR"/*/ 2>/dev/null | wc -l)
if ! ls "$SONIC/checkpoints/" | grep -q "policy_${STEP}_decoder.trt"; then
  echo "NOTE: no cached TRT engine for $STEP — first start will BAKE (~minutes)."
fi

echo "=== ffmaster_deploy_onnx_ref | $STEP | $MOTION_DIR ($N_MOTIONS motions) | domain 77 ==="
echo "=== expect: 'Found $N_MOTIONS motion folders' ... 'Init Done'. ] to engage, Ctrl-C to ABORT ==="

# ---- environment + start -----------------------------------------------------
export FFMASTER_DDS_DOMAIN=77   # must match ffmaster_bridge.sh
export LD_LIBRARY_PATH="$HOME/.local/onnxruntime/lib:$DEPLOY/thirdparty/unitree_sdk2/thirdparty/lib/aarch64:${LD_LIBRARY_PATH:-}"

# guard: verify the binary will load the SDK's CycloneDDS, not a foreign one.
# (Historical x86-workstation hazard: ROS's libddsc.so.0 shadowing the SDK's
# caused heap corruption. On this Orin the binary's RPATH pins the SDK copy
# and ROS ships no libddsc, so a sourced ROS env is harmless — checked here
# against the REAL resolution rather than proxies like $ROS_DISTRO.)
DDSC=$(ldd ./target/release/ffmaster_deploy_onnx_ref 2>/dev/null \
       | awk '/libddsc\.so/ {print $3; exit}')
case "$DDSC" in
  "$DEPLOY"/thirdparty/*) : ;;   # correct: SDK copy
  *)
    echo "ERROR: binary would load a NON-SDK CycloneDDS: ${DDSC:-<unresolved>}"
    echo "       expected it under $DEPLOY/thirdparty/. Refusing to start"
    echo "       (foreign libddsc caused heap corruption on the workstation)."
    exit 1 ;;
esac

# Always-on flight recorder: per-tick policy inputs (994-dim obs) and the
# tracked reference. Tiny CSVs; the ONLY way to diagnose a bad engage after
# the fact — behavioral debugging without these cost us 2026-08-15.
LOGDIR="$SONIC/flight_logs/$(date -u +%Y%m%d_%H%M%SZ)_${STEP}"
mkdir -p "$LOGDIR"
echo "=== flight recorder: $LOGDIR ==="

exec ./target/release/ffmaster_deploy_onnx_ref lo \
  "$DECODER" "$MOTION_DIR" \
  --obs-config policy/ffmaster/observation_config.yaml \
  --encoder-file "$ENCODER" \
  --policy-input-logfile "$LOGDIR/policy_input.csv" \
  --target-motion-logfile "$LOGDIR/target_motion.csv" \
  --input-type manager --output-type all --zmq-host localhost --disable-crc-check
