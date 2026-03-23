cmake_minimum_required(VERSION 3.8)
project(so101_moveit_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(so101_interfaces REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(shape_msgs REQUIRED)

add_executable(move_to_pose_server src/move_to_pose_server.cpp)
target_include_directories(move_to_pose_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(move_to_pose_server PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  move_to_pose_server
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
  so101_interfaces
)

add_executable(move_gripper_server src/move_gripper_server.cpp)
target_include_directories(move_gripper_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(move_gripper_server PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  move_gripper_server
  "rclcpp"
  "moveit_ros_planning_interface"
  "so101_interfaces"
)

add_executable(spawn_cube_server src/spawn_cube_server.cpp)
target_include_directories(spawn_cube_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(spawn_cube_server PUBLIC c_std_99 cxx_std_17)
ament_target_dependencies(
  spawn_cube_server
  rclcpp
  moveit_ros_planning_interface
  moveit_msgs
  shape_msgs
  geometry_msgs
  so101_interfaces
)

install(TARGETS move_to_pose_server move_gripper_server spawn_cube_server
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
