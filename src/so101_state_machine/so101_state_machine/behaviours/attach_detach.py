"""Attach/detach cube behaviour: publishes Bool to Isaac Sim attach topic after delay.

Isaac Sim subscribes to /isaac_attach_cube; true = attach, false = detach.
"""

import time

import py_trees
from rclpy.node import Node
from std_msgs.msg import Bool


class AttachDetachCube(py_trees.behaviour.Behaviour):
    """Waits delay_sec, then publishes attach/detach command to Isaac Sim."""

    def __init__(self, name: str, node: Node, topic_name: str, attach: bool, delay_sec: float = 1.0):
        super().__init__(name)
        self.node = node
        self.topic_name = topic_name
        self.attach = attach
        self.delay_sec = delay_sec
        self.pub = self.node.create_publisher(Bool, topic_name, 10)
        self._start_time = None
        self._done = False

    def initialise(self):
        self._start_time = time.monotonic()
        self._done = False

    def update(self) -> py_trees.common.Status:
        if not self._done and (time.monotonic() - self._start_time) >= self.delay_sec:
            msg = Bool()
            msg.data = self.attach
            self.pub.publish(msg)
            self.node.get_logger().info(f"BT: Isaac attach={self.attach} on {self.topic_name}")
            self._done = True
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING
