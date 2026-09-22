#!/usr/bin/env bash
# Idempotently (re)apply the wifi0 DDS block. Runtime-only rule — a reboot
# clears it, so run this at every session start (or after any reboot).
#
# Why: lab machines on the shared WiFi (for example a workstation on the lab WiFi,
# running an ffmaster_rl_deploy example nobody owns) flood ROS2 domain 0. Their
# flapping participants trigger a FastDDS defect on the robot: any ROS process
# (bridge, tools, a 30-line subscriber) can segfault or stall while digesting
# the churn (observed: probe SIGSEGV with a lab-workstation GUID in
# the crash log, on a system with no other load and no firewall).
# Blocking UDP 7400-7599 on wifi0 stops DISCOVERY, which stops everything —
# no discovery, no connections, no churn. Robot-internal DDS (develop0, lo)
# and SSH/TCP are untouched. Trade-off: no ros2-over-WiFi INTO the robot
# (SSH in and run on-board instead).
if sudo -n iptables -C INPUT -i wifi0 -p udp --dport 7400:7599 -j DROP 2>/dev/null; then
  echo "dds-guard: already active"
else
  sudo -n iptables -I INPUT 1 -i wifi0 -p udp --dport 7400:7599 -j DROP \
    && echo "dds-guard: applied" || echo "dds-guard: FAILED (needs sudo)"
fi
sudo -n iptables -L INPUT -n -v 2>/dev/null | sed -n "3p" | awk '{print "dds-guard: dropped so far:", $1, "packets"}'
