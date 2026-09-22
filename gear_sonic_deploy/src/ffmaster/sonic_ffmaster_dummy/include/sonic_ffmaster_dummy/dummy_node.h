#pragma once
// Dummy single-joint controller — the transport + safety skeleton that the
// real SONIC deploy node will reuse. Publishes ONLY the arm command topic.
//
// FSM:
//   WAIT_STATE      wait for a full 14-joint arm state
//   CHECK_BUS_FREE  all 4 command topics must have 0 publishers AND the MC
//                   state topic must have 0 publishers, for N consecutive
//                   checks (Develop_MC active). We create our publisher only
//                   after this gate — before that we cannot even command.
//   CAPTURE_POSE    latch current arm positions as the hold pose
//   ENGAGE          hold captured pose, test joint at kp; others limp
//   SINE            test joint tracks center + A*sin(2*pi*t/T)
//   DAMP_EXIT       kp=0, kd=exit_damp_kd on all 14 joints, then finished
//
// Abort conditions (any state after CAPTURE): arm state older than
// state_stale_ms; a foreign publisher appears on the arm command topic; any
// publisher appears on the MC state topic (MC restarted). All aborts land in
// DAMP_EXIT, never a hard stop.

#include <rclcpp/rclcpp.hpp>
#include <aimdk_msgs/msg/joint_state_array.hpp>
#include <aimdk_msgs/msg/joint_command_array.hpp>

#include <array>
#include <chrono>
#include <string>
#include <vector>

namespace sonic_ffmaster_dummy {

struct Config {
  // topics
  std::string arm_state_topic, arm_command_topic, mc_state_topic;
  std::vector<std::string> all_command_topics;
  // control
  double rate_hz{500.0}, engage_s{2.0};
  // test joint
  int    test_index{3};
  double amplitude{0.2}, period_s{8.0};
  int    cycles{4};
  double kp_test{25.0}, kd_test{2.0};
  // limp joints
  double kp_limp{0.0}, kd_limp{1.0};
  // safety
  double clamp_rad{0.5};
  double state_stale_ms{50.0};
  double discovery_settle_s{5.0};
  int    bus_free_checks{3};
  double check_period_s{0.5};
  double exit_damp_kd{2.0};
  int    exit_damp_msgs{50};
  std::vector<std::string> joint_names;

  static Config Load(const std::string& yaml_path);
};

class DummyNode : public rclcpp::Node {
 public:
  static constexpr int kNumArmJoints = 14;

  explicit DummyNode(const Config& cfg);

  // Main loop calls this; true once DAMP_EXIT has fully drained.
  bool finished() const { return state_ == State::FINISHED; }

  // Called by main() on SIGINT/exception: force the damp path from wherever
  // we are. Safe to call repeatedly. Publishes nothing if the publisher was
  // never created (pre-gate abort).
  void RequestDampExit(const char* reason);

 private:
  enum class State { WAIT_STATE, CHECK_BUS_FREE, CAPTURE_POSE, ENGAGE, SINE,
                     DAMP_EXIT, FINISHED };
  static const char* Name(State s);

  void Tick();                       // rate_hz timer
  void OnArmState(aimdk_msgs::msg::JointStateArray::SharedPtr msg);

  // graph queries (typeless — work against sim and both msg versions)
  int  ForeignPublishers(const std::string& topic) const;
  bool McAlive() const;
  void BusCheck();                   // check_period_s timer

  void PublishCommand(double test_pos, double kp, double kd);
  void PublishDamp();
  void SwitchTo(State s, const char* why);

  Config cfg_;
  State  state_{State::WAIT_STATE};

  rclcpp::Subscription<aimdk_msgs::msg::JointStateArray>::SharedPtr state_sub_;
  rclcpp::Publisher<aimdk_msgs::msg::JointCommandArray>::SharedPtr  cmd_pub_;  // created post-gate
  rclcpp::TimerBase::SharedPtr tick_timer_, bus_timer_;

  std::array<double, kNumArmJoints> q_now_{};       // latest measured positions
  std::array<double, kNumArmJoints> q_hold_{};      // captured pose
  std::chrono::steady_clock::time_point last_state_rx_{};
  std::chrono::steady_clock::time_point first_state_rx_{};
  bool have_state_{false};

  int      bus_free_streak_{0};
  bool     bus_alarm_{false};
  std::chrono::steady_clock::time_point node_start_{};
  uint32_t seq_{0};
  int      damp_msgs_left_{0};
  double   t_in_state_{0.0};        // seconds since entering current state
  double   center_{0.0};            // sine center = captured test-joint pos
};

}  // namespace sonic_ffmaster_dummy
