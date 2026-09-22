#!/usr/bin/env bash
# T-A wrapper: start sonic_ffmaster_bridge with the validated session environment.
# (2026-08-13 procedure doc, step 3 — one command instead of five.)
#
# Usage:            ./ffmaster_bridge.sh
# Stop / abort:     Ctrl-C   (bridge damps and exits cleanly)
#
# Prerequisite: robot in Develop_MC (ffmaster_migrate.py Develop_MC), otherwise the
# gate will correctly report "bus NOT free" forever — that is not a bug.

SONIC=~/sonic_deployment

source "$SONIC/ffmaster_env.sh"

# bridge <-> binary loopback link lives on DDS domain 77 (vendor stack
# exhausts domain-0 participant slots on the Orin). Must match ffmaster_sonic.sh.
export FFMASTER_DDS_DOMAIN=77

# SHM off (FastDDS /dev/shm pool rot) + DDS confined to develop0 (lab-WiFi
# publishers invisible). The bridge does NOT set this itself.
export FASTRTPS_DEFAULT_PROFILES_FILE="$SONIC/gear_sonic_deploy/src/ffmaster/sonic_ffmaster_bridge/config/fastdds_robot_isolated.xml"

source "$SONIC/ws/install/setup.bash"

# Always-on command flight-log: every command the bridge forwards (or damp it
# publishes), timestamped with the same epoch clock as ffmaster_record.py — the
# evaluator merges the two by time. Self-logged to disk so the recorder never
# needs to subscribe to the command topics (= never registers inside this
# process's DDS writers; see the 2026-08-13 recorder incident).
mkdir -p "$SONIC/recordings"
CMDLOG="$SONIC/recordings/bridge_cmd_$(date -u +%Y%m%d_%H%M%SZ).csv"

echo "=== sonic_ffmaster_bridge | domain 77 | isolated profile | cores 6,7 ==="
echo "=== command flight-log: $CMDLOG ==="
echo "=== waiting for: bus check 3/3 free -> ACTIVE. Ctrl-C to damp-exit. ==="

taskset -c 6,7 "$SONIC/ws/install/sonic_ffmaster_bridge/lib/sonic_ffmaster_bridge/bridge" \
  "$SONIC/gear_sonic_deploy/src/ffmaster/sonic_ffmaster_bridge/config/bridge.yaml" \
  "$CMDLOG" 2>&1 \
  | grep --line-buffered -v "\[Warning\]"
