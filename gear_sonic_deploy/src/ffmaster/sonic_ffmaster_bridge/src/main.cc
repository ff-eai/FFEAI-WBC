// sonic_ffmaster_bridge — main. Same signal-safe exit ordering as sonic_ffmaster_dummy:
// our handler sets a flag, the loop drives the damp path to completion, and
// only then does rclcpp shut down. Unitree ChannelFactory is initialized once
// here, BEFORE the node (its publishers live on this DDS instance).

#include <rclcpp/rclcpp.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

#include <atomic>
#include <csignal>

#include "sonic_ffmaster_bridge/bridge_node.h"

namespace {
std::atomic<bool> g_stop{false};
void OnSignal(int) { g_stop = true; }
}  // namespace

int main(int argc, char** argv) {
  rclcpp::InitOptions init_opts;
  rclcpp::init(argc, argv, init_opts, rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);

  std::string config_path = "config/bridge.yaml";
  if (argc > 1) config_path = argv[1];

  int rc = 0;
  try {
    auto cfg = sonic_ffmaster_bridge::Config::Load(config_path);
    // argv[2] (optional): command flight-log CSV. Per-session, so it lives on
    // the command line rather than in the frozen production yaml.
    if (argc > 2) cfg.cmd_log_path = argv[2];
    // Unitree DDS on the configured interface (loopback by default). Domain is
    // env-overridable via FFMASTER_DDS_DOMAIN — the SAME variable the deploy binary
    // reads, so both ends of the loopback link switch together (default 0;
    // 77 on the robot, where the vendor stack exhausts domain-0 indices).
    int32_t dds_domain = 0;
    if (const char* d = std::getenv("FFMASTER_DDS_DOMAIN")) {
      dds_domain = std::atoi(d);
      fprintf(stderr, "[INFO] FFMASTER_DDS_DOMAIN set: unitree side on DDS domain %d\n", dds_domain);
    }
    unitree::robot::ChannelFactory::Instance()->Init(dds_domain, cfg.network_interface);

    auto node = std::make_shared<sonic_ffmaster_bridge::BridgeNode>(cfg);
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);

    bool damp_requested = false;
    while (rclcpp::ok() && !node->finished()) {
      if (g_stop && !damp_requested) {
        node->RequestDampExit("operator stop (signal)");
        damp_requested = true;   // keep spinning: damp must reach the wire
      }
      exec.spin_some(std::chrono::milliseconds(2));
    }
    RCLCPP_INFO(node->get_logger(), "exited via damp path; shutting down");
  } catch (const std::exception& e) {
    fprintf(stderr, "fatal: %s\n", e.what());
    rc = 1;
  }

  rclcpp::shutdown();
  return rc;
}
