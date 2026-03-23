/**
 * SpawnCubeServer - ROS 2 services for dynamic obstacles in MoveIt planning scene (bonus).
 *
 * /spawn_cube: add box at (x,y,z) with given size. /delete_cube: remove by object_name.
 * MoveIt avoids spawned objects during planning.
 */
#include <set>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <so101_interfaces/srv/spawn_cube.hpp>
#include <so101_interfaces/srv/delete_cube.hpp>

class SpawnCubeServer : public rclcpp::Node
{
public:
  SpawnCubeServer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
  : Node("spawn_cube_server", options),
    planning_scene_interface_("")
  {
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
    debug_ = this->declare_parameter<bool>("debug", false);

    spawn_service_ = this->create_service<so101_interfaces::srv::SpawnCube>(
      "/spawn_cube",
      std::bind(&SpawnCubeServer::handle_spawn_cube, this,
                std::placeholders::_1, std::placeholders::_2));

    delete_service_ = this->create_service<so101_interfaces::srv::DeleteCube>(
      "/delete_cube",
      std::bind(&SpawnCubeServer::handle_delete_cube, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_WARN(this->get_logger(), "Spawn Cube Initiated.");
  }

private:
  /** Generate unique name cube_0, cube_1, ... */
  std::string make_object_name()
  {
    size_t idx = 0;
    std::string name;
    do {
      name = "cube_" + std::to_string(idx);
      idx++;
    } while (spawned_objects_.find(name) != spawned_objects_.end());
    return name;
  }

  /** Build CollisionObject for a box at (x,y,z) with side length size. */
  moveit_msgs::msg::CollisionObject make_collision_object(
    const std::string& id, double x, double y, double z, double size)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = base_frame_;
    obj.header.stamp = this->now();
    obj.id = id;
    obj.operation = obj.ADD;

    shape_msgs::msg::SolidPrimitive box;
    box.type = box.BOX;
    box.dimensions = {size, size, size};
    obj.primitives.push_back(box);

    geometry_msgs::msg::Pose pose;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = z;
    pose.orientation.w = 1.0;
    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = 0.0;
    obj.primitive_poses.push_back(pose);

    return obj;
  }

  void handle_spawn_cube(
    const std::shared_ptr<so101_interfaces::srv::SpawnCube::Request> request,
    std::shared_ptr<so101_interfaces::srv::SpawnCube::Response> response)
  {
    if (request->size <= 0.0) {
      response->success = false;
      response->message = "Cube size must be positive";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    std::string obj_name = make_object_name();
    moveit_msgs::msg::CollisionObject collision_object = make_collision_object(
      obj_name, request->x, request->y, request->z, request->size);

    std::vector<moveit_msgs::msg::CollisionObject> objects = {collision_object};
    bool success = planning_scene_interface_.applyCollisionObjects(objects);

    if (success) {
      spawned_objects_.insert(obj_name);
      response->success = true;
      response->object_name = obj_name;
      response->message = "Spawned " + obj_name + " at (" +
        std::to_string(request->x) + ", " + std::to_string(request->y) + ", " +
        std::to_string(request->z) + ") size=" + std::to_string(request->size);
      if (debug_) {
        RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
      }
    } else {
      response->success = false;
      response->object_name = "";
      response->message = "Failed to apply collision object to planning scene";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
    }
  }

  void handle_delete_cube(
    const std::shared_ptr<so101_interfaces::srv::DeleteCube::Request> request,
    std::shared_ptr<so101_interfaces::srv::DeleteCube::Response> response)
  {
    const std::string& name = request->object_name;
    if (name.empty()) {
      response->success = false;
      response->message = "Object name cannot be empty";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    if (spawned_objects_.find(name) == spawned_objects_.end()) {
      response->success = false;
      response->message = "Object '" + name + "' not found (not spawned by this server)";
      RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    moveit_msgs::msg::CollisionObject remove_obj;
    remove_obj.id = name;
    remove_obj.operation = remove_obj.REMOVE;
    std::vector<moveit_msgs::msg::CollisionObject> objects = {remove_obj};
    bool success = planning_scene_interface_.applyCollisionObjects(objects);

    if (success) {
      spawned_objects_.erase(name);
      response->success = true;
      response->message = "Deleted " + name;
      if (debug_) {
        RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
      }
    } else {
      response->success = false;
      response->message = "Failed to remove object from planning scene";
      RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
    }
  }

  std::string base_frame_;
  bool debug_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;
  std::set<std::string> spawned_objects_;
  rclcpp::Service<so101_interfaces::srv::SpawnCube>::SharedPtr spawn_service_;
  rclcpp::Service<so101_interfaces::srv::DeleteCube>::SharedPtr delete_service_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SpawnCubeServer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
