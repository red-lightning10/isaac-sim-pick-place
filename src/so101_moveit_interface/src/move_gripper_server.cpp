/**
 * MoveGripperServer - ROS 2 service to command gripper joint position (rad).
 *
 * Uses MoveIt gripper planning group. Position 0.2618 rad ~= 15 deg open.
 */
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <so101_interfaces/srv/move_gripper.hpp>

class MoveGripperServer : public rclcpp::Node
{
public:
  MoveGripperServer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : Node("move_gripper_server", options)
  {
    planning_group_ = this->declare_parameter<std::string>("planning_group", "gripper");
    velocity_scaling_ = this->declare_parameter<double>("velocity_scaling", 0.5);
    acceleration_scaling_ = this->declare_parameter<double>("acceleration_scaling", 0.5);
    debug_ = this->declare_parameter<bool>("debug", false);

    service_ = this->create_service<so101_interfaces::srv::MoveGripper>(
      "/move_gripper",
      std::bind(&MoveGripperServer::handle_move_gripper, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_WARN(this->get_logger(), "Gripper Move Initiated.");
  }

private:
  void handle_move_gripper(
    const std::shared_ptr<so101_interfaces::srv::MoveGripper::Request> request,
    std::shared_ptr<so101_interfaces::srv::MoveGripper::Response> response)
  {
    if (debug_) {
      RCLCPP_INFO(this->get_logger(), "Received move gripper request: position=%.4f rad", request->position);
    }

    try {
      using moveit::planning_interface::MoveGroupInterface;
      MoveGroupInterface move_group_interface(shared_from_this(), planning_group_);
      move_group_interface.setMaxVelocityScalingFactor(velocity_scaling_);
      move_group_interface.setMaxAccelerationScalingFactor(acceleration_scaling_);

      move_group_interface.setJointValueTarget({{"gripper", request->position}});

      moveit::planning_interface::MoveGroupInterface::Plan plan;
      auto result = move_group_interface.plan(plan);

      if (result != moveit::core::MoveItErrorCode::SUCCESS) {
        response->success = false;
        response->message = "Planning failed";
        RCLCPP_ERROR(this->get_logger(), "Gripper planning failed");
        return;
      }

      auto exec_result = move_group_interface.execute(plan);
      if (exec_result != moveit::core::MoveItErrorCode::SUCCESS) {
        response->success = false;
        response->message = "Execution failed";
        RCLCPP_ERROR(this->get_logger(), "Gripper execution failed");
        return;
      }

      response->success = true;
      response->message = "Gripper moved successfully";
      if (debug_) {
        RCLCPP_INFO(this->get_logger(), "Gripper moved to %.4f rad", request->position);
      }

    } catch (const std::exception& e) {
      response->success = false;
      response->message = std::string("Exception: ") + e.what();
      RCLCPP_ERROR(this->get_logger(), "%s", e.what());
    }
  }

  rclcpp::Service<so101_interfaces::srv::MoveGripper>::SharedPtr service_;
  std::string planning_group_;
  double velocity_scaling_;
  double acceleration_scaling_;
  bool debug_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveGripperServer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
