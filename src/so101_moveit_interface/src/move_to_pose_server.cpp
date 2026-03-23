/**
 * MoveToPoseServer - ROS 2 service to move arm end-effector to a target pose.
 *
 * Uses Cartesian path when feasible; falls back to joint-space planning.
 * Optional cartesian_fraction truncates path to stop short of target (for grasping).
 */
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <so101_interfaces/srv/move_to_pose.hpp>

class MoveToPoseServer : public rclcpp::Node
{
public:
  MoveToPoseServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("move_to_pose_server", options)
  {
    planning_group_ = this->declare_parameter<std::string>("planning_group", "arm");
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
    velocity_scaling_ = this->declare_parameter<double>("velocity_scaling", 0.1);
    acceleration_scaling_ = this->declare_parameter<double>("acceleration_scaling", 0.1);
    cartesian_fraction_ = this->declare_parameter<double>("cartesian_fraction", 0.95);
    debug_ = this->declare_parameter<bool>("debug", false);

    callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    RCLCPP_WARN(this->get_logger(), "Move to Pose Initiated.");
  }

  void init()
  {
    move_group_interface_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), planning_group_);
    move_group_interface_->setPoseReferenceFrame(base_frame_);
    move_group_interface_->setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_interface_->setMaxAccelerationScalingFactor(acceleration_scaling_);

    service_ = this->create_service<so101_interfaces::srv::MoveToPose>(
      "/move_to_pose",
      std::bind(&MoveToPoseServer::handle_move_to_pose, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default,
      callback_group_);
      
  }

private:
  void handle_move_to_pose(
    const std::shared_ptr<so101_interfaces::srv::MoveToPose::Request> request,
    std::shared_ptr<so101_interfaces::srv::MoveToPose::Response> response)
  {
    try {

      move_group_interface_->setEndEffectorLink("gripper_frame_link");

      if (debug_) {
        RCLCPP_INFO(this->get_logger(), "Target pose: (%.3f, %.3f, %.3f)",
          request->target_pose.pose.position.x, request->target_pose.pose.position.y, request->target_pose.pose.position.z);
        RCLCPP_INFO(this->get_logger(), "Target pose orientation: (%.3f, %.3f, %.3f, %.3f)",
          request->target_pose.pose.orientation.x, request->target_pose.pose.orientation.y,
          request->target_pose.pose.orientation.z, request->target_pose.pose.orientation.w);
      }

      std::vector<geometry_msgs::msg::Pose> waypoints = {request->target_pose.pose};
      moveit_msgs::msg::RobotTrajectory trajectory;
      
      double fraction = move_group_interface_->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

      if (fraction >= 0.95) {
        // Truncate trajectory to cartesian_fraction when use_cartesian_fraction=true (stop short for grasp)
        if (request->use_cartesian_fraction && cartesian_fraction_ < 1.0 && !trajectory.joint_trajectory.points.empty()) {
          auto& points = trajectory.joint_trajectory.points;
          size_t total = points.size();
          size_t keep = std::max(size_t(1), static_cast<size_t>(total * cartesian_fraction_));
          points.resize(keep);
          if (debug_) {
            RCLCPP_INFO(this->get_logger(), "Executing %.0f%% of Cartesian path (%zu/%zu waypoints)",
              cartesian_fraction_ * 100.0, keep, total);
          }
        }
        auto result = move_group_interface_->execute(trajectory);
        response->success = (result == moveit::core::MoveItErrorCode::SUCCESS);
        response->message = response->success ? "Successfully moved via Cartesian path" : "Cartesian path execution failed";
      } else {
        // Cartesian path failed; fall back to joint-space planning
        move_group_interface_->setPoseTarget(request->target_pose.pose);
        moveit::planning_interface::MoveGroupInterface::Plan plan;

        if (static_cast<bool>(move_group_interface_->plan(plan))) {
          auto result = move_group_interface_->execute(plan);
          response->success = (result == moveit::core::MoveItErrorCode::SUCCESS);
          response->message = response->success ? "Successfully moved via joint-space planning" : "Joint-space execution failed";
        } else {
          response->success = false;
          response->message = "Both Cartesian and joint-space planning failed";
        }
      }
    }
    catch (const std::exception & e) {
      response->success = false;
      response->message = std::string("Exception: ") + e.what();
      RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
    }
  }

  std::string planning_group_;
  std::string base_frame_;
  double velocity_scaling_;
  double acceleration_scaling_;
  double cartesian_fraction_;
  bool debug_;

  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::Service<so101_interfaces::srv::MoveToPose>::SharedPtr service_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_interface_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveToPoseServer>();
  node->init();
  
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  
  rclcpp::shutdown();
  return 0;
}