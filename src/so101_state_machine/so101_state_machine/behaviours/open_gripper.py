"""Open gripper behaviour: calls /move_gripper with position 0.2618 rad (~15 deg)."""

import py_trees
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from so101_interfaces.srv import MoveGripper


class OpenGripper(py_trees.behaviour.Behaviour):
    """Calls /move_gripper to open gripper before pick and after place."""

    def __init__(self, name: str, node: Node):
        super().__init__(name)
        self.node = node
        self.callback_group = ReentrantCallbackGroup()
        self._gripper_client = self.node.create_client(
            MoveGripper, '/move_gripper', callback_group=self.callback_group
        )
        self._future = None
        self._wait_logged = False

    def initialise(self):
        self._future = None
        self._wait_logged = False
        self.node.get_logger().info(f"[{self.name}] Opening gripper...")

    def update(self) -> py_trees.common.Status:
        if not self._gripper_client.wait_for_service(timeout_sec=0.1):
            if not self._wait_logged:
                self._wait_logged = True
                self.node.get_logger().warn(f"[{self.name}] Waiting for move_gripper service...")
            return py_trees.common.Status.RUNNING

        if self._future is None:
            request = MoveGripper.Request()
            request.position = 0.2618  # rad (~15 deg) open
            self._future = self._gripper_client.call_async(request)
            return py_trees.common.Status.RUNNING

        if self._future.done():
            try:
                result = self._future.result()
                if result.success:
                    self.node.get_logger().info(f"[{self.name}] Gripper opened")
                    return py_trees.common.Status.SUCCESS
                self.node.get_logger().error(f"[{self.name}] Gripper failed: {result.message}")
                return py_trees.common.Status.FAILURE
            except Exception as e:
                self.node.get_logger().error(f"[{self.name}] Gripper exception: {e}")
                return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING
