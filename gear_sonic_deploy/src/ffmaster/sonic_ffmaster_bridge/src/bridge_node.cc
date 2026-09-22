#include "sonic_ffmaster_bridge/bridge_node.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstring>   // memcpy/memcmp for GID handling

namespace sonic_ffmaster_bridge {

using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;
using unitree_hg::msg::dds_::IMUState_;
using unitree_hg::msg::dds_::HandState_;

namespace {
constexpr const char* kLowStateTopic = "rt/lowstate";
constexpr const char* kLowCmdTopic   = "rt/lowcmd";
constexpr const char* kSecImuTopic   = "rt/secondary_imu";
constexpr const char* kLHandTopic    = "rt/dex3/left/state";
constexpr const char* kRHandTopic    = "rt/dex3/right/state";

rclcpp::QoS SensorQos() {
  return rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile();
}

template <size_t N>
std::array<double, N> LoadArray(const YAML::Node& n) {
  std::array<double, N> out{};
  if (n.size() != N) throw std::runtime_error("yaml array size mismatch");
  for (size_t i = 0; i < N; ++i) out[i] = n[i].as<double>();
  return out;
}
}  // namespace

Config Config::Load(const std::string& yaml_path) {
  YAML::Node y = YAML::LoadFile(yaml_path);
  Config c;
  const auto a = y["aimdk"];
  c.leg_state   = a["state_topics"]["leg"].as<std::string>();
  c.waist_state = a["state_topics"]["waist"].as<std::string>();
  c.arm_state   = a["state_topics"]["arm"].as<std::string>();
  c.leg_cmd   = a["command_topics"]["leg"].as<std::string>();
  c.waist_cmd = a["command_topics"]["waist"].as<std::string>();
  c.arm_cmd   = a["command_topics"]["arm"].as<std::string>();
  c.head_cmd  = a["command_topics"]["head"].as<std::string>();
  c.pelvis_imu = a["pelvis_imu"].as<std::string>();
  c.chest_imu  = a["chest_imu"].as<std::string>();
  c.mc_state   = a["mc_state"].as<std::string>();

  c.network_interface = y["unitree"]["network_interface"].as<std::string>();

  const auto r = y["rates"];
  c.lowstate_hz   = r["lowstate_hz"].as<double>();
  c.forward_hz    = r["forward_hz"].as<double>();
  c.hand_state_hz = r["hand_state_hz"].as<double>();

  const auto s = y["safety"];
  c.discovery_settle_s = s["discovery_settle_s"].as<double>();
  c.bus_free_checks    = s["bus_free_checks"].as<int>();
  c.check_period_s     = s["check_period_s"].as<double>();
  c.state_stale_ms     = s["state_stale_ms"].as<double>();
  c.state_lost_exit_ms = s["state_lost_exit_ms"].as<double>();
  c.lowcmd_stale_ms    = s["lowcmd_stale_ms"].as<double>();
  c.exit_damp_kd       = s["exit_damp_kd"].as<double>();
  c.exit_damp_msgs     = s["exit_damp_msgs"].as<int>();
  c.clamp_lo           = LoadArray<kNumMotors>(s["clamp_lo"]);
  c.clamp_hi           = LoadArray<kNumMotors>(s["clamp_hi"]);
  c.kp_cap             = LoadArray<kNumMotors>(s["kp_cap"]);
  c.kd_cap             = s["kd_cap"].as<double>();
  c.target_lpf_hz      = s["target_lpf_hz"].as<double>();

  c.default_angles = LoadArray<kNumMotors>(y["default_angles"]);
  for (const auto& n : y["joint_names"]["leg"])   c.names_leg.push_back(n.as<std::string>());
  for (const auto& n : y["joint_names"]["waist"]) c.names_waist.push_back(n.as<std::string>());
  for (const auto& n : y["joint_names"]["arm"])   c.names_arm.push_back(n.as<std::string>());
  for (const auto& n : y["joint_names"]["head"])  c.names_head.push_back(n.as<std::string>());
  if (c.names_leg.size() != kLeg || c.names_waist.size() != kWaist ||
      c.names_arm.size() != kArm || c.names_head.size() != kHead)
    throw std::runtime_error("joint_names group sizes must be 12/3/14/2");
  return c;
}

const char* BridgeNode::Name(State s) {
  switch (s) {
    case State::WAIT_STATE:     return "WAIT_STATE";
    case State::CHECK_BUS_FREE: return "CHECK_BUS_FREE";
    case State::ACTIVE:         return "ACTIVE";
    case State::DAMP_EXIT:      return "DAMP_EXIT";
    case State::FINISHED:       return "FINISHED";
  }
  return "?";
}

BridgeNode::BridgeNode(const Config& cfg) : Node("sonic_ffmaster_bridge"), cfg_(cfg) {
  using std::placeholders::_1;

  // ---- aimdk side: state in (always), commands out (created post-gate) ----
  sub_leg_ = create_subscription<aimdk_msgs::msg::JointStateArray>(
      cfg_.leg_state, SensorQos(),
      [this](aimdk_msgs::msg::JointStateArray::SharedPtr m) { OnGroupState(*m, 0, kLeg); });
  sub_waist_ = create_subscription<aimdk_msgs::msg::JointStateArray>(
      cfg_.waist_state, SensorQos(),
      [this](aimdk_msgs::msg::JointStateArray::SharedPtr m) { OnGroupState(*m, kLeg, kWaist); });
  sub_arm_ = create_subscription<aimdk_msgs::msg::JointStateArray>(
      cfg_.arm_state, SensorQos(),
      [this](aimdk_msgs::msg::JointStateArray::SharedPtr m) { OnGroupState(*m, kLeg + kWaist, kArm); });
  sub_pelvis_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      cfg_.pelvis_imu, SensorQos(), std::bind(&BridgeNode::OnPelvisImu, this, _1));
  sub_chest_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      cfg_.chest_imu, SensorQos(), std::bind(&BridgeNode::OnChestImu, this, _1));

  // ---- unitree side: loopback only, safe to start immediately ----
  lowstate_pub_.reset(new unitree::robot::ChannelPublisher<LowState_>(kLowStateTopic));
  lowstate_pub_->InitChannel();
  secondary_imu_pub_.reset(new unitree::robot::ChannelPublisher<IMUState_>(kSecImuTopic));
  secondary_imu_pub_->InitChannel();
  lhand_pub_.reset(new unitree::robot::ChannelPublisher<HandState_>(kLHandTopic));
  lhand_pub_->InitChannel();
  rhand_pub_.reset(new unitree::robot::ChannelPublisher<HandState_>(kRHandTopic));
  rhand_pub_->InitChannel();
  lowcmd_sub_.reset(new unitree::robot::ChannelSubscriber<LowCmd_>(kLowCmdTopic));
  lowcmd_sub_->InitChannel(std::bind(&BridgeNode::OnLowCmd, this, _1), 1);

  auto period = [](double hz) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / hz));
  };
  lowstate_timer_ = create_wall_timer(period(cfg_.lowstate_hz),
                                      std::bind(&BridgeNode::PublishLowState, this));
  forward_timer_  = create_wall_timer(period(cfg_.forward_hz),
                                      std::bind(&BridgeNode::ForwardTick, this));
  hand_timer_     = create_wall_timer(period(cfg_.hand_state_hz),
                                      std::bind(&BridgeNode::PublishHandStates, this));
  bus_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(cfg_.check_period_s)),
      std::bind(&BridgeNode::BusCheck, this));

  // Blocking graph queries live on their own thread (see header).
  graph_thread_ = std::thread(&BridgeNode::GraphPollLoop, this);

  // Optional command flight-log (argv[2] via cfg.cmd_log_path).
  if (!cfg_.cmd_log_path.empty()) {
    cmdlog_file_.open(cfg_.cmd_log_path, std::ios::out | std::ios::trunc);
    if (cmdlog_file_) {
      cmdlog_file_ << "t";
      for (int i = 0; i < kNumMotors; ++i) cmdlog_file_ << ",cq" << (i < 10 ? "0" : "") << i;
      for (int i = 0; i < kNumMotors; ++i) cmdlog_file_ << ",ckp" << (i < 10 ? "0" : "") << i;
      for (int i = 0; i < kNumMotors; ++i) cmdlog_file_ << ",ckd" << (i < 10 ? "0" : "") << i;
      cmdlog_file_ << "\n";
      cmdlog_run_ = true;
      cmdlog_thread_ = std::thread(&BridgeNode::CmdLogLoop, this);
      RCLCPP_INFO(get_logger(), "command flight-log: %s", cfg_.cmd_log_path.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "command flight-log: cannot open %s — logging DISABLED",
                  cfg_.cmd_log_path.c_str());
    }
  }

  RCLCPP_INFO(get_logger(),
              "bridge up. unitree iface=%s  lowstate@%.0fHz  forward@%.0fHz  "
              "clamp=joint-limits lpf=%.1fHz. State flow starts now; command "
              "flow is gated on a free bus.",
              cfg_.network_interface.c_str(), cfg_.lowstate_hz, cfg_.forward_hz,
              cfg_.target_lpf_hz);
}

BridgeNode::~BridgeNode() {
  graph_thread_run_ = false;
  if (graph_thread_.joinable()) graph_thread_.join();
  if (cmdlog_run_) {
    cmdlog_run_ = false;
    if (cmdlog_thread_.joinable()) cmdlog_thread_.join();
    // final drain (logger thread exits without a last pass)
    std::lock_guard<std::mutex> lk(cmdlog_mutex_);
    for (const auto& line : cmdlog_buf_) cmdlog_file_ << line;
    cmdlog_file_.flush();
  }
}

void BridgeNode::CmdLogTick(const std::array<double, kNumMotors>& q,
                            const std::array<double, kNumMotors>& kp,
                            const std::array<double, kNumMotors>& kd) {
  if (!cmdlog_run_) return;
  // epoch seconds via CLOCK_REALTIME — same clock as the recorder's
  // time.time(), so the evaluator can merge the two files by timestamp.
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  char buf[2048];
  int n = snprintf(buf, sizeof(buf), "%.6f", ts.tv_sec + ts.tv_nsec * 1e-9);
  for (int i = 0; i < kNumMotors; ++i)
    n += snprintf(buf + n, sizeof(buf) - n, ",%.5f", q[i]);
  for (int i = 0; i < kNumMotors; ++i)
    n += snprintf(buf + n, sizeof(buf) - n, ",%.3f", kp[i]);
  for (int i = 0; i < kNumMotors; ++i)
    n += snprintf(buf + n, sizeof(buf) - n, ",%.3f", kd[i]);
  n += snprintf(buf + n, sizeof(buf) - n, "\n");
  std::lock_guard<std::mutex> lk(cmdlog_mutex_);
  cmdlog_buf_.emplace_back(buf, n);
}

void BridgeNode::CmdLogLoop() {
  std::vector<std::string> drain;
  while (cmdlog_run_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    {
      std::lock_guard<std::mutex> lk(cmdlog_mutex_);
      drain.swap(cmdlog_buf_);
    }
    for (const auto& line : drain) cmdlog_file_ << line;
    cmdlog_file_.flush();
    drain.clear();
  }
}

void BridgeNode::GraphPollLoop() {
  // rclcpp's NodeGraph accessors are thread-safe; this thread absorbs their
  // blocking cost so the executor thread never stalls.
  const auto period = std::chrono::duration<double>(cfg_.check_period_s);
  while (graph_thread_run_) {
    int all = ForeignPublishers(cfg_.leg_cmd) + ForeignPublishers(cfg_.waist_cmd) +
              ForeignPublishers(cfg_.arm_cmd) + ForeignPublishers(cfg_.head_cmd);
    int armleg = ForeignPublishers(cfg_.arm_cmd) + ForeignPublishers(cfg_.leg_cmd);
    bool mc = McAlive();
    cached_foreign_all_ = all;
    cached_foreign_armleg_ = armleg;
    cached_mc_alive_ = mc;
    std::this_thread::sleep_for(
        std::chrono::duration_cast<std::chrono::milliseconds>(period));
  }
}

// ---------------------------------------------------------------- aimdk in
void BridgeNode::OnGroupState(const aimdk_msgs::msg::JointStateArray& msg,
                              int base, int expect) {
  if (static_cast<int>(msg.joints.size()) != expect) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "group state len=%zu != %d — ignoring", msg.joints.size(), expect);
    return;
  }
  std::lock_guard<std::mutex> lk(state_mutex_);
  for (int i = 0; i < expect; ++i) {
    q_[base + i]   = msg.joints[i].position;
    dq_[base + i]  = msg.joints[i].velocity;
    tau_[base + i] = msg.joints[i].effort;
  }
  group_seen_[base == 0 ? 0 : (base == kLeg ? 1 : 2)] = true;
  last_state_rx_ = std::chrono::steady_clock::now();
  if (!have_full_state_ && group_seen_[0] && group_seen_[1] && group_seen_[2] && imu_seen_) {
    first_state_rx_ = last_state_rx_;
    have_full_state_ = true;
  }
}

void BridgeNode::OnPelvisImu(sensor_msgs::msg::Imu::SharedPtr msg) {
  std::lock_guard<std::mutex> lk(state_mutex_);
  // ROS msg is xyzw fields; policy/lowstate wants wxyz.
  pelvis_quat_wxyz_ = {msg->orientation.w, msg->orientation.x,
                       msg->orientation.y, msg->orientation.z};
  pelvis_gyro_  = {msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z};
  pelvis_accel_ = {msg->linear_acceleration.x, msg->linear_acceleration.y,
                   msg->linear_acceleration.z};
  imu_seen_ = true;
}

void BridgeNode::OnChestImu(sensor_msgs::msg::Imu::SharedPtr msg) {
  std::lock_guard<std::mutex> lk(state_mutex_);
  chest_quat_wxyz_ = {msg->orientation.w, msg->orientation.x,
                      msg->orientation.y, msg->orientation.z};
  chest_gyro_  = {msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z};
  chest_accel_ = {msg->linear_acceleration.x, msg->linear_acceleration.y,
                  msg->linear_acceleration.z};
}

// ------------------------------------------------------------- unitree out
void BridgeNode::PublishLowState() {
  if (!have_full_state_) return;   // binary's own checks handle absence
  LowState_ ls{};
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    for (int i = 0; i < kNumMotors; ++i) {
      ls.motor_state()[i].q()       = static_cast<float>(q_[i]);
      ls.motor_state()[i].dq()      = static_cast<float>(dq_[i]);
      ls.motor_state()[i].tau_est() = static_cast<float>(tau_[i]);
      ls.motor_state()[i].motorstate() = 0;
      ls.motor_state()[i].temperature()[0] = 0;
      ls.motor_state()[i].temperature()[1] = 0;
    }
    for (int i = 0; i < 4; ++i)
      ls.imu_state().quaternion()[i] = static_cast<float>(pelvis_quat_wxyz_[i]);
    for (int i = 0; i < 3; ++i) {
      ls.imu_state().gyroscope()[i]     = static_cast<float>(pelvis_gyro_[i]);
      ls.imu_state().accelerometer()[i] = static_cast<float>(pelvis_accel_[i]);
    }
  }
  ls.tick() = static_cast<uint32_t>(tick_ms_);
  tick_ms_ += static_cast<uint64_t>(1000.0 / cfg_.lowstate_hz);
  lowstate_pub_->Write(ls);

  IMUState_ sec{};
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    for (int i = 0; i < 4; ++i)
      sec.quaternion()[i] = static_cast<float>(chest_quat_wxyz_[i]);
    for (int i = 0; i < 3; ++i) {
      sec.gyroscope()[i]     = static_cast<float>(chest_gyro_[i]);
      sec.accelerometer()[i] = static_cast<float>(chest_accel_[i]);
    }
  }
  secondary_imu_pub_->Write(sec);
}

void BridgeNode::PublishHandStates() {
  HandState_ hs{};   // zeros — hands are not policy-controlled on FF Master
  // CRITICAL: motor_state is an IDL sequence (default EMPTY). The deploy
  // binary indexes motor_state()[0..6] unguarded in GatherRobotStateToLogger —
  // an empty sequence segfaults it (observed: SIGSEGV on the control thread,
  // 2026-08-07). The vendor python sim bridge pre-sizes it; so must we.
  hs.motor_state().resize(7);
  lhand_pub_->Write(hs);
  rhand_pub_->Write(hs);
}

void BridgeNode::OnLowCmd(const void* msg) {
  std::lock_guard<std::mutex> lk(lowcmd_mutex_);
  lowcmd_ = *static_cast<const LowCmd_*>(msg);
  last_lowcmd_rx_ = std::chrono::steady_clock::now();
  lowcmd_seen_ = true;
}

// ---------------------------------------------------------- command path
void BridgeNode::ForwardTick() {
  const auto now = std::chrono::steady_clock::now();

  // FSM bookkeeping first
  switch (state_) {
    case State::WAIT_STATE:
      if (have_full_state_) SwitchTo(State::CHECK_BUS_FREE, "full aimdk state received");
      return;

    case State::CHECK_BUS_FREE:
      if (bus_free_streak_ >= cfg_.bus_free_checks) {
        pub_leg_   = create_publisher<aimdk_msgs::msg::JointCommandArray>(cfg_.leg_cmd, SensorQos());
        pub_waist_ = create_publisher<aimdk_msgs::msg::JointCommandArray>(cfg_.waist_cmd, SensorQos());
        pub_arm_   = create_publisher<aimdk_msgs::msg::JointCommandArray>(cfg_.arm_cmd, SensorQos());
        pub_head_  = create_publisher<aimdk_msgs::msg::JointCommandArray>(cfg_.head_cmd, SensorQos());
        RecordOwnPublisherGids();
        // The graph-poll thread may have sampled the graph in the instant
        // between create_publisher() and RecordOwnPublisherGids() — that
        // sample would count our own endpoints as foreign. Invalidate the
        // cache so the ACTIVE tripwire never acts on it; the poll thread
        // refreshes with a GID-aware count within check_period_s.
        cached_foreign_all_ = -1;
        cached_foreign_armleg_ = -1;
        SwitchTo(State::ACTIVE, "bus free — command publishers created");
        RCLCPP_INFO(get_logger(),
                    "ACTIVE: waiting for rt/lowcmd from ffmaster_deploy_onnx_ref. "
                    "Nothing is forwarded until the binary commands.");
      }
      return;

    case State::DAMP_EXIT:
      if (!pub_leg_) { SwitchTo(State::FINISHED, "no publishers were created"); return; }
      if (damp_msgs_left_ > 0) { PublishDamp(); --damp_msgs_left_; }
      else SwitchTo(State::FINISHED, "damp drained");
      return;

    case State::FINISHED:
      return;

    case State::ACTIVE:
      break;   // fall through to forwarding below
  }

  // ACTIVE: state-stream watchdog — damp-and-recover, like the lowcmd path.
  // Transient scheduling stalls on a loaded Orin must not kill the session;
  // during a gap we command damping, and forwarding resumes when the stream
  // does. Only a SUSTAINED loss (real outage) exits.
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    const double age_ms = std::chrono::duration<double, std::milli>(now - last_state_rx_).count();
    if (age_ms > cfg_.state_lost_exit_ms) {
      RCLCPP_WARN(get_logger(), "state LOST: age %.0f ms (limit %.0f) — exiting", age_ms, cfg_.state_lost_exit_ms);
      RequestDampExit("aimdk state lost");
      return;
    }
    if (age_ms > cfg_.state_stale_ms) {
      if (!state_stale_latched_) {
        RCLCPP_WARN(get_logger(), "state stale (%.0f ms) — holding DAMP until stream resumes", age_ms);
        state_stale_latched_ = true;
        lpf_init_ = false;   // re-seed shaping on resume
      }
      PublishDamp();
      return;
    }
    if (state_stale_latched_) {
      RCLCPP_INFO(get_logger(), "state stream resumed — forwarding again");
      state_stale_latched_ = false;
    }
  }

  if (!lowcmd_seen_) return;   // binary not commanding yet — publish nothing

  LowCmd_ cmd;
  double cmd_age_ms;
  {
    std::lock_guard<std::mutex> lk(lowcmd_mutex_);
    cmd = lowcmd_;
    cmd_age_ms = std::chrono::duration<double, std::milli>(now - last_lowcmd_rx_).count();
  }

  if (cmd_age_ms > cfg_.lowcmd_stale_ms) {
    // Binary stopped commanding (exit/crash). Keep the robot damped and stay
    // alive: the operator may restart the binary; a fresh lowcmd resumes flow.
    if (!lowcmd_stale_latched_) {
      RCLCPP_WARN(get_logger(),
                  "rt/lowcmd stale (%.0f ms) — holding DAMP until the binary commands again",
                  cmd_age_ms);
      lowcmd_stale_latched_ = true;
      lpf_init_ = false;   // re-seed the filter on resume
    }
    PublishDamp();
    return;
  }
  if (lowcmd_stale_latched_) {
    RCLCPP_INFO(get_logger(), "rt/lowcmd resumed — forwarding again");
    lowcmd_stale_latched_ = false;
  }

  // Translate + safety-shape the 29 motor commands
  std::array<double, kNumMotors> q{}, dq{}, tau{}, kp{}, kd{};
  const double dt = 1.0 / cfg_.forward_hz;
  const double alpha = (cfg_.target_lpf_hz > 0.0)
      ? std::clamp(2.0 * M_PI * cfg_.target_lpf_hz * dt /
                   (1.0 + 2.0 * M_PI * cfg_.target_lpf_hz * dt), 0.0, 1.0)
      : 1.0;
  for (int i = 0; i < kNumMotors; ++i) {
    double target = static_cast<double>(cmd.motor_cmd()[i].q());
    // absolute per-joint clamp = physical joint limits (vendor MJCF, -0.02 rad)
    const double clamped = std::clamp(target, cfg_.clamp_lo[i], cfg_.clamp_hi[i]);
    if (clamped != target && !clamp_warned_[i]) {
      RCLCPP_WARN(get_logger(), "clamping motor %d (%.3f -> %.3f) — logged once per joint",
                  i, target, clamped);
      clamp_warned_[i] = true;
    }
    target = clamped;
    // first-order low-pass (post-clamp, pre-bus — what we log is what ships)
    if (!lpf_init_) lpf_q_[i] = target;
    lpf_q_[i] += alpha * (target - lpf_q_[i]);
    q[i]   = lpf_q_[i];
    dq[i]  = static_cast<double>(cmd.motor_cmd()[i].dq());
    tau[i] = static_cast<double>(cmd.motor_cmd()[i].tau());
    kp[i]  = std::clamp(static_cast<double>(cmd.motor_cmd()[i].kp()), 0.0, cfg_.kp_cap[i]);
    kd[i]  = std::clamp(static_cast<double>(cmd.motor_cmd()[i].kd()), 0.0, cfg_.kd_cap);
  }
  lpf_init_ = true;

  PublishAimdkCommands(q, dq, tau, kp, kd);
}

void BridgeNode::PublishAimdkCommands(const std::array<double, kNumMotors>& q,
                                      const std::array<double, kNumMotors>& dq,
                                      const std::array<double, kNumMotors>& tau,
                                      const std::array<double, kNumMotors>& kp,
                                      const std::array<double, kNumMotors>& kd) {
  auto fill = [&](aimdk_msgs::msg::JointCommandArray& m, int base, int n,
                  const std::vector<std::string>& names) {
    m.header.stamp = this->now();
    m.header.sequence = seq_;
    m.joints.resize(n);
    for (int i = 0; i < n; ++i) {
      auto& jc = m.joints[i];
      jc.name      = names[i];
      jc.position  = q[base + i];
      jc.velocity  = dq[base + i];
      jc.effort    = tau[base + i];
      jc.stiffness = kp[base + i];
      jc.damping   = kd[base + i];
    }
  };
  aimdk_msgs::msg::JointCommandArray leg, waist, arm, head;
  fill(leg, 0, kLeg, cfg_.names_leg);
  fill(waist, kLeg, kWaist, cfg_.names_waist);
  fill(arm, kLeg + kWaist, kArm, cfg_.names_arm);
  // Head: policy-untouched. Full 2-array (empty-array semantics unknown), limp.
  head.header.stamp = this->now();
  head.header.sequence = seq_;
  head.joints.resize(kHead);
  for (int i = 0; i < kHead; ++i) {
    head.joints[i].name = cfg_.names_head[i];
    head.joints[i].position = 0.0;
    head.joints[i].velocity = 0.0;
    head.joints[i].effort = 0.0;
    head.joints[i].stiffness = 0.0;
    head.joints[i].damping = 1.0;
  }
  ++seq_;
  pub_leg_->publish(leg);
  pub_waist_->publish(waist);
  pub_arm_->publish(arm);
  pub_head_->publish(head);
  CmdLogTick(q, kp, kd);   // damp is visible in the log as kp==0
}

void BridgeNode::PublishDamp() {
  std::array<double, kNumMotors> q{}, dq{}, tau{}, kp{}, kd{};
  {
    std::lock_guard<std::mutex> lk(state_mutex_);
    q = q_;   // current pose; irrelevant at kp=0
  }
  kd.fill(cfg_.exit_damp_kd);
  PublishAimdkCommands(q, dq, tau, kp, kd);
}

// ------------------------------------------------------------ gate / trips
std::string BridgeNode::GidToHex(const EndpointGid& gid) {
  static const char* kHex = "0123456789abcdef";
  std::string s;
  s.reserve(3 * 12);
  for (size_t i = 0; i < 12 && i < gid.size(); ++i) {   // 12 bytes = GUID prefix
    if (i) s.push_back('.');
    s.push_back(kHex[(gid[i] >> 4) & 0xF]);
    s.push_back(kHex[gid[i] & 0xF]);
  }
  return s;
}

void BridgeNode::RecordOwnPublisherGids() {
  std::lock_guard<std::mutex> lk(gid_mutex_);
  own_gids_.clear();
  const rclcpp::PublisherBase* pubs[] = {
      pub_leg_.get(), pub_waist_.get(), pub_arm_.get(), pub_head_.get()};
  for (const auto* p : pubs) {
    if (!p) continue;
    EndpointGid g{};
    std::memcpy(g.data(), p->get_gid().data, RMW_GID_STORAGE_SIZE);
    own_gids_.push_back(g);
  }
  RCLCPP_INFO(get_logger(), "recorded %zu own publisher GIDs (first=%s)",
              own_gids_.size(),
              own_gids_.empty() ? "-" : GidToHex(own_gids_.front()).c_str());
}

bool BridgeNode::IsOwnEndpoint(const EndpointGid& gid) const {
  std::lock_guard<std::mutex> lk(gid_mutex_);
  for (const auto& own : own_gids_)
    if (std::memcmp(own.data(), gid.data(), RMW_GID_STORAGE_SIZE) == 0)
      return true;
  return false;
}

int BridgeNode::ForeignPublishers(const std::string& topic) const {
  int n = 0;
  for (const auto& info : get_publishers_info_by_topic(topic)) {
    // EXACT GID match against our own four publishers — never node name.
    // See the rationale block in bridge_node.h. Anything not proven to be
    // ours counts as foreign.
    if (IsOwnEndpoint(info.endpoint_gid())) continue;
    ++n;
    // Only name the offender once we have publishers of our own, i.e. once
    // we are ACTIVE. Before that, foreign publishers are expected (the MC
    // legitimately owns the bus) and logging them would be noise.
    bool have_own;
    { std::lock_guard<std::mutex> lk(gid_mutex_); have_own = !own_gids_.empty(); }
    // Plain WARN (not _THROTTLE — that needs a mutable clock and this method
    // is const). Fires only when we are ACTIVE and a genuinely foreign
    // publisher is present: worst case 4 lines/s during a real incident.
    if (have_own) {
      RCLCPP_WARN(get_logger(),
                  "foreign publisher on %s: node='%s' gid=%s", topic.c_str(),
                  info.node_name().c_str(), GidToHex(info.endpoint_gid()).c_str());
    }
  }
  return n;
}

bool BridgeNode::McAlive() const {
  return !get_publishers_info_by_topic(cfg_.mc_state).empty();
}

void BridgeNode::BusCheck() {
  if (state_ == State::CHECK_BUS_FREE) {
    if (!have_full_state_) return;
    const double since_first =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - first_state_rx_).count();
    if (since_first < cfg_.discovery_settle_s) {
      RCLCPP_INFO(get_logger(), "discovery settling (%.1f/%.1fs) — not counting",
                  since_first, cfg_.discovery_settle_s);
      return;
    }
    const int foreign = cached_foreign_all_;
    const bool mc = cached_mc_alive_;
    if (foreign < 0) return;   // graph thread has no data yet — do not count
    if (foreign == 0 && !mc) {
      ++bus_free_streak_;
      RCLCPP_INFO(get_logger(), "bus check %d/%d: free", bus_free_streak_, cfg_.bus_free_checks);
    } else {
      bus_free_streak_ = 0;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                           "bus NOT free: %d foreign command publisher(s), mc_state %s",
                           foreign, mc ? "ALIVE" : "silent");
    }
    return;
  }
  if (state_ == State::ACTIVE) {
    if (cached_foreign_armleg_ > 0) {
      RequestDampExit("foreign publisher appeared on a command topic");
    } else if (cached_mc_alive_) {
      RequestDampExit("MC state publisher reappeared (MC restarted?)");
    }
  }
}

void BridgeNode::SwitchTo(State s, const char* why) {
  RCLCPP_INFO(get_logger(), "%s -> %s (%s)", Name(state_), Name(s), why);
  state_ = s;
}

void BridgeNode::RequestDampExit(const char* reason) {
  if (state_ == State::DAMP_EXIT || state_ == State::FINISHED) return;
  RCLCPP_WARN(get_logger(), "DAMP EXIT: %s", reason);
  damp_msgs_left_ = cfg_.exit_damp_msgs;
  SwitchTo(State::DAMP_EXIT, reason);
}

}  // namespace sonic_ffmaster_bridge
