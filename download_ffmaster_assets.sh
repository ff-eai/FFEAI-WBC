#!/usr/bin/env bash
# Download the FF Master assets that are not kept in git and place them where the
# code expects them. Run from the repository root.
#
#   bash download_ffmaster_assets.sh                 # everything, from the release URL
#   bash download_ffmaster_assets.sh --model         # only the policy model
#   bash download_ffmaster_assets.sh --meshes        # only the robot meshes (MJCF/URDF)
#   bash download_ffmaster_assets.sh --reference     # only the reference motion sets
#   bash download_ffmaster_assets.sh --lab-data      # only the lab training/eval PKLs
#   bash download_ffmaster_assets.sh --planner-db    # only the kinematic planner database
#   bash download_ffmaster_assets.sh --exclude-planner-db   # everything except the planner database
#   bash download_ffmaster_assets.sh --from-dir DIR  # use a local copy of the bundles
#
# Bundles (tar.gz, also accepted as unpacked folders under --from-dir):
#   ffmaster_sonic_v0.2      -> gear_sonic_deploy/policy/ffmaster/{model_*.onnx, observation_config.yaml}
#                               sonic_ffmaster/{model_step_050000.pt, config.yaml}
#   ffmaster_robot_meshes    -> gear_sonic/data/assets/robot_description/{mjcf,urdf/ffmaster}/meshes/
#   ffmaster_reference_sets  -> gear_sonic_deploy/reference/<set>/
#   ffmaster_lab_data        -> data/ffmaster_motions/{lab_train,lab_eval}/
#   ffmaster_planner_db      -> data/kplanner/ffmaster_loco_db.npz (motion database of gear_sonic_deploy/kplanner)
#
# Override the download location with FFMASTER_ASSET_URL=<base url> or --url <base url>.
set -euo pipefail

URL="${FFMASTER_ASSET_URL:-https://github.com/ff-eai/FFEAI-WBC/releases/download/assets-v0.2}"
MODEL_BUNDLE="ffmaster_sonic_v0.2"
FROM_DIR=""
WANT_MODEL=0; WANT_MESH=0; WANT_REF=0; WANT_LAB=0; WANT_DB=0; EXCLUDE_DB=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)     WANT_MODEL=1 ;;
    --meshes)    WANT_MESH=1 ;;
    --reference) WANT_REF=1 ;;
    --lab-data)  WANT_LAB=1 ;;
    --planner-db) WANT_DB=1 ;;
    --exclude-planner-db) EXCLUDE_DB=1 ;;
    --from-dir)  FROM_DIR="$2"; shift ;;
    --url)       URL="$2"; shift ;;
    -h|--help)   sed -n 2,25p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done
if [[ $WANT_MODEL -eq 0 && $WANT_MESH -eq 0 && $WANT_REF -eq 0 && $WANT_LAB -eq 0 && $WANT_DB -eq 0 ]]; then
  WANT_MODEL=1; WANT_MESH=1; WANT_REF=1; WANT_LAB=1; WANT_DB=1
fi
if [[ $EXCLUDE_DB -eq 1 ]]; then WANT_DB=0; fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# fetch <bundle>: leaves the unpacked folder at $TMP/<bundle>
fetch() {
  local name="$1"
  if [[ -n "$FROM_DIR" ]]; then
    if [[ -d "$FROM_DIR/$name" ]]; then cp -r "$FROM_DIR/$name" "$TMP/"; return; fi
    if [[ -f "$FROM_DIR/$name.tar.gz" ]]; then tar xzf "$FROM_DIR/$name.tar.gz" -C "$TMP"; return; fi
    echo "ERROR: neither $FROM_DIR/$name/ nor $FROM_DIR/$name.tar.gz exists" >&2; exit 1
  fi
  echo "downloading $URL/$name.tar.gz"
  curl -fL --progress-bar -o "$TMP/$name.tar.gz" "$URL/$name.tar.gz"
  tar xzf "$TMP/$name.tar.gz" -C "$TMP"
}

if [[ $WANT_MODEL -eq 1 ]]; then
  fetch "$MODEL_BUNDLE"
  mkdir -p "$ROOT/gear_sonic_deploy/policy/ffmaster" "$ROOT/sonic_ffmaster"
  rm -f "$ROOT"/gear_sonic_deploy/policy/ffmaster/*.trt   # engines are baked per ONNX; stale ones are rebuilt anyway
  cp "$TMP/$MODEL_BUNDLE"/model_*.onnx "$ROOT/gear_sonic_deploy/policy/ffmaster/"
  cp "$TMP/$MODEL_BUNDLE"/observation_config.yaml "$ROOT/gear_sonic_deploy/policy/ffmaster/"
  cp "$TMP/$MODEL_BUNDLE"/model_step_*.pt "$TMP/$MODEL_BUNDLE"/config.yaml "$ROOT/sonic_ffmaster/"
  echo "model ($MODEL_BUNDLE) -> gear_sonic_deploy/policy/ffmaster/ and sonic_ffmaster/"
fi
if [[ $WANT_MESH -eq 1 ]]; then
  fetch ffmaster_robot_meshes
  RD="$ROOT/gear_sonic/data/assets/robot_description"
  mkdir -p "$RD/mjcf/meshes" "$RD/urdf/ffmaster/meshes"
  cp "$TMP"/ffmaster_robot_meshes/meshes/*.STL "$RD/mjcf/meshes/"
  cp "$TMP"/ffmaster_robot_meshes/meshes/*.STL "$RD/urdf/ffmaster/meshes/"
  echo "meshes -> gear_sonic/data/assets/robot_description/{mjcf,urdf/ffmaster}/meshes/ ($(ls "$TMP"/ffmaster_robot_meshes/meshes | wc -l) files)"
fi
if [[ $WANT_REF -eq 1 ]]; then
  fetch ffmaster_reference_sets
  mkdir -p "$ROOT/gear_sonic_deploy/reference"
  cp -r "$TMP"/ffmaster_reference_sets/* "$ROOT/gear_sonic_deploy/reference/"
  echo "reference sets -> gear_sonic_deploy/reference/: $(ls "$TMP"/ffmaster_reference_sets | tr '\n' ' ')"
fi
if [[ $WANT_LAB -eq 1 ]]; then
  fetch ffmaster_lab_data
  mkdir -p "$ROOT/data/ffmaster_motions"
  cp -r "$TMP"/ffmaster_lab_data/* "$ROOT/data/ffmaster_motions/"
  echo "lab data -> data/ffmaster_motions/: $(ls "$TMP"/ffmaster_lab_data | tr '\n' ' ')"
fi
if [[ $WANT_DB -eq 1 ]]; then
  fetch ffmaster_planner_db
  mkdir -p "$ROOT/data/kplanner"
  cp "$TMP"/ffmaster_planner_db/ffmaster_loco_db.npz "$ROOT/data/kplanner/"
  echo "planner database -> data/kplanner/ffmaster_loco_db.npz"
fi
echo "done"
