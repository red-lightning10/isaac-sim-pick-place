#!/usr/bin/env python3
"""
BT Node - ROS 2 node that runs the pick-and-place behaviour tree.

Ticks the py_trees BehaviourTree at 10 Hz. Requires move_group, detection,
grasp, and moveit services to be running.
"""
import rclpy
import py_trees
from rclpy.node import Node

from .tree import create_tree


class BTNode(Node):
    """ROS 2 node hosting the pick-and-place behaviour tree."""

    def __init__(self):
        super().__init__("bt_interview_template_node")
        self.declare_parameter('grasp_debug', False)
        self.declare_parameter('detection_debug', False)
        self.declare_parameter('state_machine_debug', False)

        self.tree = py_trees.trees.BehaviourTree(create_tree(self))
        self.timer = self.create_timer(0.1, self._tick)
        self._completed_logged = False

        self.get_logger().warn("Execution Tree Started.")

    def _tick(self):
        """Tick the behaviour tree; log once when sequence completes."""
        self.tree.tick()
        if self.tree.root.status == py_trees.common.Status.SUCCESS and not self._completed_logged:
            self._completed_logged = True
            self.get_logger().warn("BT: Task sequence complete.")


def main():
    rclpy.init()
    node = BTNode()

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
