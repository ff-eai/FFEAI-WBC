// sonic_ffmaster_dummy — main.
//
// Exit-ordering is the whole point of this file. The vendor example calls its
// damping routine AFTER rclcpp::shutdown() is already in flight, so the last
// full-stiffness command stays latched on the wire (audited hazard). Here:
//
//   1. rclcpp is initialized with SignalHandlerOptions::None — Ctrl-C does NOT
//      tear down the context behind our back.
//   2. Our own SIGINT/SIGTERM handler only sets an atomic flag.
//   3. The main loop sees the flag, asks the node for a damp exit, and KEEPS
//      SPINNING until the damp messages have actually been published.
//   4. Only then rclcpp::shutdown().

#include <rclcpp/rclcpp.hpp>

#include <atomic>
#include <csignal>

#include "sonic_ffmaster_dummy/dummy_node.h"

namespace {
std::atomic<bool> g_stop{false};
void OnSignal(int) { g_stop = true; }
}  // namespace

int main(int argc, char** argv) {
  rclcpp::InitOptions init_opts;
  rclcpp::init(argc, argv, init_opts, rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);

  std::string config_path = "config/dummy.yaml";
  if (argc > 1) config_path = argv[1];

  int rc = 0;
  try {
    auto cfg = sonic_ffmaster_dummy::Config::Load(config_path);
    auto node = std::make_shared<sonic_ffmaster_dummy::DummyNode>(cfg);
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);

    bool damp_requested = false;
    while (rclcpp::ok() && !node->finished()) {
      if (g_stop && !damp_requested) {
        node->RequestDampExit("operator stop (signal)");
        damp_requested = true;   // keep spinning: damp messages must go out
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
