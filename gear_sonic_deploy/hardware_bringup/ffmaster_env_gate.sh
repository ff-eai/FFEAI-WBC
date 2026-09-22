#!/usr/bin/env bash
# FF Master environment gate — "is the robot in a known-good state for a session?"
# Run FIRST, before anything else. Read-only except nothing. Prints PASS/WARN
# per check and a final verdict. Born 2026-08-15 after a day of debugging a
# contested environment (packages reinstalled mid-session, firewall order
# changed, unknown daemons) — this makes drift VISIBLE before it costs a session.
PASS=0; WARN=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
warn() { echo "  WARN  $1"; WARN=$((WARN+1)); }

echo "== ffmaster_env_gate $(date -u +%FT%TZ) (uptime: $(uptime -p)) =="

# 1. firewall order: cn-block may exist, but develop0/lo traffic must not walk it
FIRST3=$(sudo -n iptables -L INPUT -n --line-numbers 2>/dev/null | sed -n '3,5p')
CNPOS=$(sudo -n iptables -L INPUT -n --line-numbers 2>/dev/null | awk '/cn-block/{print $1; exit}')
DEVPOS=$(sudo -n iptables -L INPUT -n -v --line-numbers 2>/dev/null | awk '$0~/develop0/ && $0~/ACCEPT/{print $1; exit}')
LOPOS=$(sudo -n iptables -L INPUT -n -v --line-numbers 2>/dev/null | awk '$0~/ lo / && $0~/ACCEPT/{print $1; exit}')
if [ -z "$CNPOS" ]; then ok "cn-block absent"
elif [ -n "$DEVPOS" ] && [ "$DEVPOS" -lt "$CNPOS" ] && [ -n "$LOPOS" ] && [ "$LOPOS" -lt "$CNPOS" ]; then
  ok "cn-block present but develop0+lo exempted above it"
else
  warn "cn-block at INPUT position ${CNPOS} WITHOUT develop0/lo exemption above it -> per-packet latency tax on robot control (LowState age 40-50ms era)"
fi

# 2. wifi DDS block (guard against lab-WiFi DDS participants crashing FastDDS)
if sudo -n iptables -C INPUT -i wifi0 -p udp --dport 7400:7599 -j DROP 2>/dev/null; then
  ok "wifi0 DDS block active"
else
  warn "wifi0 DDS block MISSING -> run ~/sonic_deployment/ffmaster_dds_guard.sh (lab DDS churn can segfault ROS processes)"
fi

# 3. package drift since the last known inventory
SNAP=~/sonic_deployment/.pkg_inventory
dpkg -l 2>/dev/null | grep -E "bot-ota|bot-uploader|system-config |ff-robot|ffrobot" | awk '{print $1,$2,$3}' > /tmp/pkg_now
if [ -f "$SNAP" ]; then
  if diff -q "$SNAP" /tmp/pkg_now >/dev/null 2>&1; then ok "FF package set unchanged since last gate"
  else warn "FF PACKAGES CHANGED since last gate:"; diff "$SNAP" /tmp/pkg_now | sed 's/^/        /'; fi
else warn "no package baseline yet — recording one now"; fi
cp /tmp/pkg_now "$SNAP"

# 4. unknown DDS/middleware daemons running?
Z=$(pgrep -af "zenoh|ffrobot" 2>/dev/null | grep -v pgrep | head -2)
[ -z "$Z" ] && ok "no zenoh/ffrobot daemons running" || warn "middleware daemons running: $Z"

# 5. vendor stack up + stream healthy
source ~/sonic_deployment/ffmaster_env.sh >/dev/null 2>&1
RATE=$(timeout 15 ros2 topic hz --window 300 /aima/hal/joint/leg/state 2>/dev/null | grep -oE "average rate: [0-9.]+" | head -1 | grep -oE "[0-9.]+")
if [ -n "$RATE" ] && [ "${RATE%.*}" -ge 400 ]; then ok "leg/state at ${RATE} Hz"
elif [ -n "$RATE" ]; then warn "leg/state only ${RATE} Hz (want >400 as seen by CLI; true rate 1 kHz)"
else warn "leg/state NOT FLOWING — vendor stack down or booting; do NOT migrate states until it flows"; fi

# 6. shm pool + load
NSHM=$(ls /dev/shm 2>/dev/null | grep -c fastrtps)
[ "$NSHM" -lt 90 ] && ok "shm pool: $NSHM fastrtps segments" || warn "shm pool large ($NSHM) — consider reboot"
LOAD=$(cut -d" " -f1 /proc/loadavg)
[ "${LOAD%.*}" -lt 6 ] && ok "load $LOAD" || warn "load $LOAD high"

# 7. other admins logged in
OTHERS=$(who | awk '{print $NF}' | sort -u | wc -l)
who | sed 's/^/        /'
[ "$OTHERS" -le 1 ] && ok "single operator source" || warn "$OTHERS distinct login sources — coordinate before changing anything"

echo
echo "== verdict: $PASS pass, $WARN warn =="
[ "$WARN" -eq 0 ] && echo "ENVIRONMENT KNOWN-GOOD — proceed with session" \
                  || echo "RESOLVE WARNINGS FIRST — or knowingly accept them"
