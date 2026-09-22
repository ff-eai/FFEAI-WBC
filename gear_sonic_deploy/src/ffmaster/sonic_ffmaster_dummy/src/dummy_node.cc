#include "sonic_ffmaster_dummy/dummy_node.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>

namespace sonic_ffmaster_dummy {

namespace {
// HAL publishers are BEST_EFFORT; a RELIABLE endpoint would not match at all
// (verified on hardware 2026-08-04). Commands likewise go out BEST_EFFORT.
rclcpp::QoS SensorQos() {
  return rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
}
}  // namespace

Config Config::Load(const std::string& yaml_path) {
  YAML::Node y = YAML::LoadFile(yaml_path);
  Config c;
  const auto t = y["topics"];
  c.arm_state_topic   = t["arm_state"].as<std::string>();
  c.arm_command_topic = t["arm_command"].as<std::string>();
  c.mc_state_topic    = t["mc_state"].as<std::string>();
  for (const auto& n : t["all_commands"]) c.all_command_topics.push_back(n.as<std::string>());

  c.rate_hz  = y["control"]["rate_hz"].as<double>();
  c.engage_s = y["control"]["engage_s"].as<double>();

  const auto j = y["test_joint"];
  c.test_index = j["index"].as<int>();
  c.amplitude  = j["amplitude_rad"].as<double>();
  c.period_s   = j["period_s"].as<double>();
  c.cycles     = j["cycles"].as<int>();
  c.kp_test    = j["kp"].as<double>();
  c.kd_test    = j["kd"].as<double>();

  c.kp_limp = y["limp_joints"]["kp"].as<double>();
  c.kd_limp = y["limp_joints"]["kd"].as<double>();

  const auto s = y["safety"];
  c.clamp_rad          = s["clamp_rad"].as<double>();
  c.state_stale_ms     = s["state_stale_ms"].as<double>();
  c.discovery_settle_s = s["discovery_settle_s"].as<double>();
  c.bus_free_checks    = s["bus_free_checks"].as<int>();
  c.check_period_s  = s["check_period_s"].as<double>();
  c.exit_damp_kd    = s["exit_damp_kd"].as<double>();
  c.exit_damp_msgs  = s["exit_damp_msgs"].as<int>();

  for (const auto& n : y["arm_joint_names"]) c.joint_names.push_back(n.as<std::string>());
  if (c.joint_names.size() != DummyNode::kNumArmJoints)
    throw std::runtime_error("arm_joint_names must have exactly 14 entries");
  if (c.test_index < 0 || c.test_index >= DummyNode::kNumArmJoints)
    throw std::runtime_error("test_joint.index out of range");
  return c;
}

const char* DummyNode::Name(State s) {
  switch (s) {
    case State::WAIT_STATE:     return "WAIT_STATE";
    case State::CHECK_BUS_FREE: return "CHECK_BUS_FREE";
    case State::CAPTURE_POSE:   return "CAPTURE_POSE";
    case State::ENGAGE:         return "ENGAGE";
    case State::SINE:           return "SINE";
    case State::DAMP_EXIT:      return "DAMP_EXIT";
    case State::FINISHED:       return "FINISHED";
  }
  return "?";
}

DummyNode::DummyNode(const Config& cfg) : Node("sonic_ffmaster_dummy"), cfg_(cfg) {
  node_start_ = std::chrono::steady_clock::now();
  state_sub_ = create_subscription<aimdk_msgs::msg::JointStateArray>(
      cfg_.arm_state_topic, SensorQos(),
      std::bind(&DummyNode::OnArmState, this, std::placeholders::_1));

  // NOTE: cmd_pub_ is intentionally NOT created here. Creating it would put a
  // publisher endpoint on the bus before the gate has passed; we only create
  // it once CHECK_BUS_FREE has proven the MC is silenced.

  const auto tick_period =
      std::chrono::duration<double>(1.0 / cfg_.rate_hz);
  tick_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(tick_period),
      std::bind(&DummyNode::Tick, this));

  bus_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(cfg_.check_period_s)),
      std::bind(&DummyNode::BusCheck, this));

  RCLCPP_INFO(get_logger(),
              "dummy up. test joint arm[%d] (%s), kp=%.1f amp=%.2f rad. "
              "Waiting for state, then bus-free gate.",
              cfg_.test_index, cfg_.joint_names[cfg_.test_index].c_str(),
              cfg_.kp_test, cfg_.amplitude);
}

void DummyNode::OnArmState(aimdk_msgs::msg::JointStateArray::SharedPtr msg) {
  if (msg->joints.size() != kNumArmJoints) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "arm state len=%zu != 14 — ignoring", msg->joints.size());
    return;
  }
  for (int i = 0; i < kNumArmJoints; ++i) q_now_[i] = msg->joints[i].position;
  last_state_rx_ = std::chrono::steady_clock::now();
  if (!have_state_) first_state_rx_ = last_state_rx_;
  have_state_ = true;
}

int DummyNode::ForeignPublishers(const std::string& topic) const {
  int n = 0;
  for (const auto& info : get_publishers_info_by_topic(topic))
    if (info.node_name() != get_name()) ++n;
  return n;
}

bool DummyNode::McAlive() const {
  return !get_publishers_info_by_topic(cfg_.mc_state_topic).empty();
}

void DummyNode::BusCheck() {
  // Pre-gate: count a streak of fully-free checks.
  if (state_ == State::CHECK_BUS_FREE) {
    // A fresh DDS graph legitimately reports 0 publishers until discovery
    // completes. Wall-clock-from-start is the WRONG anchor: on the real robot
    // (5 network interfaces) the live MC was still invisible 6 s after node
    // start — 1 s past a 5 s settle (negative tests, 2026-08-06). Anchor the
    // settle to the FIRST RECEIVED STATE message instead: that is positive
    // evidence discovery is delivering, not just time passing.
    if (!have_state_) return;  // nothing discovered yet — never count
    const double since_first_state =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - first_state_rx_).count();
    if (since_first_state < cfg_.discovery_settle_s) {
      RCLCPP_INFO(get_logger(),
                  "discovery settling (%.1f/%.1fs after first state) — not counting",
                  since_first_state, cfg_.discovery_settle_s);
      return;
    }
    int foreign = 0;
    for (const auto& t : cfg_.all_command_topics) foreign += ForeignPublishers(t);
    const bool mc = McAlive();
    if (foreign == 0 && !mc) {
      ++bus_free_streak_;
      RCLCPP_INFO(get_logger(), "bus check %d/%d: free",
                  bus_free_streak_, cfg_.bus_free_checks);
    } else {
      bus_free_streak_ = 0;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
          "bus NOT free: %d foreign command publisher(s), mc_state %s — is "
          "Develop_MC active?", foreign, mc ? "ALIVE" : "silent");
    }
    return;
  }
  // Post-gate: any foreign commander or MC revival is an abort.
  if (state_ == State::ENGAGE || state_ == State::SINE) {
    if (ForeignPublishers(cfg_.arm_command_topic) > 0) {
      bus_alarm_ = true;
      RequestDampExit("foreign publisher appeared on arm command topic");
    } else if (McAlive()) {
      bus_alarm_ = true;
      RequestDampExit("MC state topic publisher reappeared (MC restarted?)");
    }
  }
}

void DummyNode::SwitchTo(State s, const char* why) {
  RCLCPP_INFO(get_logger(), "%s -> %s (%s)", Name(state_), Name(s), why);
  state_ = s;
  t_in_state_ = 0.0;
}

void DummyNode::RequestDampExit(const char* reason) {
  if (state_ == State::DAMP_EXIT || state_ == State::FINISHED) return;
  RCLCPP_WARN(get_logger(), "DAMP EXIT: %s", reason);
  damp_msgs_left_ = cfg_.exit_damp_msgs;
  SwitchTo(State::DAMP_EXIT, reason);
}

void DummyNode::PublishCommand(double test_pos, double kp, double kd) {
  aimdk_msgs::msg::JointCommandArray cmd;
  cmd.header.stamp = now();
  cmd.header.sequence = seq_++;
  cmd.joints.resize(kNumArmJoints);
  for (int i = 0; i < kNumArmJoints; ++i) {
    auto& jc = cmd.joints[i];
    jc.name = cfg_.joint_names[i];       // informational; index is the contract
    jc.velocity = 0.0;
    jc.effort = 0.0;
    if (i == cfg_.test_index) {
      jc.position  = test_pos;
      jc.stiffness = kp;
      jc.damping   = kd;
    } else {
      jc.position  = q_hold_[i];         // harmless with kp=0; kept meaningful
      jc.stiffness = cfg_.kp_limp;
      jc.damping   = cfg_.kd_limp;
    }
  }
  cmd_pub_->publish(cmd);
}

void DummyNode::PublishDamp() {
  aimdk_msgs::msg::JointCommandArray cmd;
  cmd.header.stamp = now();
  cmd.header.sequence = seq_++;
  cmd.joints.resize(kNumArmJoints);
  for (int i = 0; i < kNumArmJoints; ++i) {
    auto& jc = cmd.joints[i];
    jc.name = cfg_.joint_names[i];
    jc.position  = q_now_[i];   // current pose; irrelevant at kp=0
    jc.velocity  = 0.0;
    jc.effort    = 0.0;
    jc.stiffness = 0.0;
    jc.damping   = cfg_.exit_damp_kd;
  }
  cmd_pub_->publish(cmd);
}

void DummyNode::Tick() {
  const double dt = 1.0 / cfg_.rate_hz;
  t_in_state_ += dt;

  // Staleness watchdog — armed as soon as we command anything.
  if (state_ == State::ENGAGE || state_ == State::SINE) {
    const double age_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - last_state_rx_).count();
    if (age_ms > cfg_.state_stale_ms) {
      // Log the measured age: ~60-100 ms => executor scheduling hiccup
      // (graph query blocking the thread); hundreds of ms => real outage.
      RCLCPP_WARN(get_logger(), "state age at trip: %.1f ms (limit %.0f)",
                  age_ms, cfg_.state_stale_ms);
      RequestDampExit("arm state stale");
      // fall through to DAMP_EXIT handling below on next tick
    }
  }

  switch (state_) {
    case State::WAIT_STATE:
      if (have_state_) SwitchTo(State::CHECK_BUS_FREE, "arm state received");
      break;

    case State::CHECK_BUS_FREE:
      if (bus_free_streak_ >= cfg_.bus_free_checks) {
        cmd_pub_ = create_publisher<aimdk_msgs::msg::JointCommandArray>(
            cfg_.arm_command_topic, SensorQos());
        SwitchTo(State::CAPTURE_POSE, "bus free — publisher created");
      }
      break;

    case State::CAPTURE_POSE:
      q_hold_ = q_now_;
      center_ = q_hold_[cfg_.test_index];
      RCLCPP_INFO(get_logger(), "captured pose; test joint center = %+.4f rad", center_);
      SwitchTo(State::ENGAGE, "pose captured");
      break;

    case State::ENGAGE:
      // Hold captured pose. Pose error ~0 at engage, so stiffness snap is safe
      // (paper: lerp position, snap stiffness — here there is nothing to lerp).
      PublishCommand(center_, cfg_.kp_test, cfg_.kd_test);
      if (t_in_state_ >= cfg_.engage_s) SwitchTo(State::SINE, "engage hold done");
      break;

    case State::SINE: {
      const double total = cfg_.cycles * cfg_.period_s;
      double target = center_ +
          cfg_.amplitude * std::sin(2.0 * M_PI * t_in_state_ / cfg_.period_s);
      // Hard clamp — belt and braces; also the pattern the real node keeps.
      target = std::clamp(target, center_ - cfg_.clamp_rad, center_ + cfg_.clamp_rad);
      PublishCommand(target, cfg_.kp_test, cfg_.kd_test);
      if (t_in_state_ >= total) RequestDampExit("test complete");
      break;
    }

    case State::DAMP_EXIT:
      if (!cmd_pub_) {           // aborted before the gate — nothing to damp
        SwitchTo(State::FINISHED, "no publisher was created");
        break;
      }
      if (damp_msgs_left_ > 0) {
        PublishDamp();
        --damp_msgs_left_;
      } else {
        SwitchTo(State::FINISHED, "damp drained");
      }
      break;

    case State::FINISHED:
      break;
  }
}

}  // namespace sonic_ffmaster_dummy
