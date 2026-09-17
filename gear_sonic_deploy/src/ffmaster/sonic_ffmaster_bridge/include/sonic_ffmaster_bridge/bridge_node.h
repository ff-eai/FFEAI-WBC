#pragma once
// aimdk <-> Unitree-DDS bridge for running ffmaster_deploy_onnx_ref unmodified.
//
// Two faces, one process:
//   aimdk (rclcpp): subscribe leg/waist/arm state + pelvis/chest IMU; publish
//     leg/waist/arm/head command arrays — but ONLY after the bus-free gate.
//   unitree (unitree_sdk2, loopback): publish rt/lowstate (motors 0..28 in FF Master
//     MuJoCo order, pelvis IMU wxyz) + rt/secondary_imu + zero dex3 states;
//     subscribe rt/lowcmd from the deploy binary.
//
// State flow starts immediately (loopback only — harmless). Command flow is
// gated exactly like sonic_ffmaster_dummy: 4 command topics + mc state must be
// publisher-free for N consecutive checks after discovery settles.
//
// FSM:  WAIT_STATE -> CHECK_BUS_FREE -> ACTIVE -> DAMP_EXIT -> FINISHED
// Aborts (-> DAMP_EXIT): aimdk state stale; foreign command publisher; MC
// state publisher reappears; SIGINT/SIGTERM; exception.
// lowcmd stale while ACTIVE (binary crashed/exited): damp continuously,
// stay alive (operator may restart the binary), abort only on other trips.

#include <rclcpp/rclcpp.hpp>
#include <aimdk_msgs/msg/joint_state_array.hpp>
#include <aimdk_msgs/msg/joint_command_array.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/HandState_.hpp>

#include <rmw/types.h>   // RMW_GID_STORAGE_SIZE

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace sonic_ffmaster_bridge {

constexpr int kNumMotors = 29;   // FF Master policy DoF, MuJoCo order 0..28
constexpr int kLeg = 12, kWaist = 3, kArm = 14, kHead = 2;

struct Config {
  // aimdk
  std::string leg_state, waist_state, arm_state;
  std::string leg_cmd, waist_cmd, arm_cmd, head_cmd;
  std::string pelvis_imu, chest_imu, mc_state;
  // unitree
  std::string network_interface{"lo"};
  // rates
  double lowstate_hz{500.0}, forward_hz{500.0}, hand_state_hz{50.0};
  // safety
  double discovery_settle_s{5.0};
  int    bus_free_checks{3};
  double check_period_s{1.0};
  double state_stale_ms{300.0};
  double state_lost_exit_ms{2000.0};
  double lowcmd_stale_ms{250.0};
  double exit_damp_kd{2.0};
  int    exit_damp_msgs{100};
  std::array<double, kNumMotors> clamp_lo{}, clamp_hi{};   // absolute joint limits
  std::array<double, kNumMotors> kp_cap{};
  double kd_cap{8.0};
  double target_lpf_hz{8.0};
  std::array<double, kNumMotors> default_angles{};
  std::vector<std::string> names_leg, names_waist, names_arm, names_head;
  // Optional command flight-log (CSV path, set from argv[2] — NOT yaml: it is
  // per-session, the yaml is the frozen production config). Empty = disabled.
  // Purpose: the offline evaluator needs the command stream, and subscribing
  // to the command topics from a recorder registers that recorder inside THIS
  // process's DDS writers (the 2026-08-13 recorder incident). Self-logging to
  // disk decouples them completely.
  std::string cmd_log_path;

  static Config Load(const std::string& yaml_path);
};

class BridgeNode : public rclcpp::Node {
 public:
  explicit BridgeNode(const Config& cfg);
  ~BridgeNode() override;
  bool finished() const { return state_ == State::FINISHED; }
  void RequestDampExit(const char* reason);

 private:
  enum class State { WAIT_STATE, CHECK_BUS_FREE, ACTIVE, DAMP_EXIT, FINISHED };
  static const char* Name(State s);

  // aimdk-side callbacks
  void OnGroupState(const aimdk_msgs::msg::JointStateArray& msg, int base, int expect);
  void OnPelvisImu(sensor_msgs::msg::Imu::SharedPtr msg);
  void OnChestImu(sensor_msgs::msg::Imu::SharedPtr msg);

  // unitree-side
  void OnLowCmd(const void* msg);           // rt/lowcmd from the binary
  void PublishLowState();                   // lowstate_hz timer
  void PublishHandStates();                 // hand_state_hz timer

  // command path (forward_hz timer)
  void ForwardTick();
  void PublishAimdkCommands(const std::array<double, kNumMotors>& q,
                            const std::array<double, kNumMotors>& dq,
                            const std::array<double, kNumMotors>& tau,
                            const std::array<double, kNumMotors>& kp,
                            const std::array<double, kNumMotors>& kd);
  void PublishDamp();

  // gate / monitors. Graph queries BLOCK (tens to hundreds of ms on a busy
  // graph, e.g. right after a robot power cycle) — so they run on a dedicated
  // thread that caches results; the executor-side BusCheck timer only reads
  // the cache and can never starve state callbacks / the staleness watchdog.
  void BusCheck();                     // executor timer: cache readers only
  void GraphPollLoop();                // dedicated thread: the blocking queries
  int  ForeignPublishers(const std::string& topic) const;
  bool McAlive() const;
  void SwitchTo(State s, const char* why);

  // ---- own-endpoint identification -------------------------------------
  // "Is this publisher me?" is answered by DDS GID, never by node name.
  //
  // Node names are NOT carried in the endpoint. rmw announces them out of
  // band, so between creating a publisher and that announcement propagating,
  // get_publishers_info_by_topic() reports the endpoint with node_name
  // "_NODE_NAME_UNKNOWN_". The previous check (`info.node_name() != get_name()`)
  // therefore counted the bridge's OWN four publishers as foreign and damped
  // out ~1 s after reaching ACTIVE. Observed repeatedly 2026-08-11/13, and
  // confirmed by a graph dump showing the bridge listed twice on every command
  // topic — once as `sonic_ffmaster_bridge`, once as `_NODE_NAME_UNKNOWN_` — with the
  // MC verifiably stopped and nothing else on the bus.
  //
  // GID matching is EXACT (full RMW_GID_STORAGE_SIZE byte compare against the
  // GIDs of the four publishers we created). It is deliberately not a
  // heuristic: no name matching, no prefix matching, no "looks like ours"
  // fallback. Anything we cannot prove is one of our own four endpoints counts
  // as foreign. Before the publishers exist own_gids_ is empty, so during
  // CHECK_BUS_FREE every publisher counts as foreign — which is what the gate
  // requires.
  //
  // This is strictly safer than the name check, which had a real false-NEGATIVE
  // hole: a leftover bridge process from an earlier run publishes under the
  // same node name and would have been silently accepted as "me". Its GID
  // differs, so it is now correctly flagged.
  using EndpointGid = std::array<uint8_t, RMW_GID_STORAGE_SIZE>;
  bool IsOwnEndpoint(const EndpointGid& gid) const;
  void RecordOwnPublisherGids();       // call right after creating the 4 pubs
  static std::string GidToHex(const EndpointGid& gid);

  Config cfg_;
  State  state_{State::WAIT_STATE};

  // ---- aimdk side ----
  rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr sub_leg_, sub_waist_, sub_arm_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_pelvis_imu_, sub_chest_imu_;
  rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr pub_leg_, pub_waist_, pub_arm_, pub_head_;
  rclcpp::TimerBase::SharedPtr lowstate_timer_, forward_timer_, bus_timer_, hand_timer_;

  std::mutex state_mutex_;
  std::array<double, kNumMotors> q_{}, dq_{}, tau_{};
  std::array<bool, 3> group_seen_{};            // leg, waist, arm
  std::array<double, 4> pelvis_quat_wxyz_{1,0,0,0};
  std::array<double, 3> pelvis_gyro_{}, pelvis_accel_{};
  std::array<double, 4> chest_quat_wxyz_{1,0,0,0};
  std::array<double, 3> chest_gyro_{}, chest_accel_{};
  bool imu_seen_{false};
  std::chrono::steady_clock::time_point last_state_rx_{}, first_state_rx_{};
  bool have_full_state_{false};

  // ---- unitree side ----
  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowState_>  lowstate_pub_;
  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::IMUState_>  secondary_imu_pub_;
  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandState_> lhand_pub_, rhand_pub_;
  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowCmd_>   lowcmd_sub_;

  std::mutex lowcmd_mutex_;
  unitree_hg::msg::dds_::LowCmd_ lowcmd_{};
  std::atomic<bool> lowcmd_seen_{false};
  std::chrono::steady_clock::time_point last_lowcmd_rx_{};

  // command shaping
  std::array<double, kNumMotors> lpf_q_{};       // low-pass state
  bool lpf_init_{false};
  bool clamp_warned_[kNumMotors] = {false};

  // ---- command flight-log (optional; see Config::cmd_log_path) ----------
  // Control path only FORMATS into a memory buffer (a few µs); a dedicated
  // thread drains to disk every ~0.5 s. No file I/O ever on the control tick.
  void CmdLogTick(const std::array<double, kNumMotors>& q,
                  const std::array<double, kNumMotors>& kp,
                  const std::array<double, kNumMotors>& kd);
  void CmdLogLoop();
  std::ofstream cmdlog_file_;
  std::thread cmdlog_thread_;
  std::atomic<bool> cmdlog_run_{false};
  std::mutex cmdlog_mutex_;
  std::vector<std::string> cmdlog_buf_;

  // own publisher GIDs (written once on the executor thread at ACTIVE, read
  // continuously by the graph-poll thread — hence the mutex).
  mutable std::mutex gid_mutex_;
  std::vector<EndpointGid> own_gids_;
  std::atomic<bool> gid_selftest_done_{false};

  // graph-poll thread + cached results
  std::thread graph_thread_;
  std::atomic<bool> graph_thread_run_{true};
  std::atomic<int>  cached_foreign_all_{-1};   // sum over 4 cmd topics; -1 = no data yet
  std::atomic<int>  cached_foreign_armleg_{-1};// arm+leg only (post-gate tripwire)
  std::atomic<bool> cached_mc_alive_{true};

  int      bus_free_streak_{0};
  uint32_t seq_{0};
  int      damp_msgs_left_{0};
  uint64_t tick_ms_{0};
  bool     lowcmd_stale_latched_{false};
  bool     state_stale_latched_{false};
};

}  // namespace sonic_ffmaster_bridge
