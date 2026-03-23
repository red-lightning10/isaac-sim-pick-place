"""Behaviour tree construction for pick-and-place pipeline.

Sequence: OpenGripper -> Grabbing -> AttachCube -> MoveToBoxPosition ->
          DetachCube -> OpenGripper. Each step wrapped in Retry decorator.
"""

import py_trees
from rclpy.node import Node

from .behaviours import OpenGripper, Grabbing, MoveToBoxPosition, AttachDetachCube


def create_tree(node: Node):
    """Build the pick-and-place behaviour tree with retries and OneShot wrapper."""
    STEP_RETRIES = 10
    ATTACH_TOPIC = "/isaac_attach_cube"  # Isaac Sim attach/detach topic
    ATTACH_DELAY = 0.5
    ATTACH_ONLY_TEST = False

    seq = py_trees.composites.Sequence(name="TaskSequence", memory=True)

    if ATTACH_ONLY_TEST:
        node.get_logger().info("BT: ATTACH_ONLY_TEST mode - detach only")
        seq.add_children([
            AttachDetachCube("DetachCube", node, ATTACH_TOPIC, attach=False, delay_sec=ATTACH_DELAY),
        ])
    else:
        seq.add_children([
            py_trees.decorators.Retry("RetryOpen1", OpenGripper("OpenGripper1", node), STEP_RETRIES),
            py_trees.decorators.Retry("RetryGrabbing", Grabbing("Grabbing", node), STEP_RETRIES),
            AttachDetachCube("AttachCube", node, ATTACH_TOPIC, attach=True, delay_sec=ATTACH_DELAY),
            py_trees.decorators.Retry("RetryMoveToBox", MoveToBoxPosition("MoveToBoxPosition", node), STEP_RETRIES),
            AttachDetachCube("DetachCube", node, ATTACH_TOPIC, attach=False, delay_sec=ATTACH_DELAY),
            py_trees.decorators.Retry("RetryOpen2", OpenGripper("OpenGripper2", node), STEP_RETRIES),
        ])

    root = py_trees.decorators.OneShot(
        name="RunOnce",
        child=seq,
        policy=py_trees.common.OneShotPolicy.ON_COMPLETION
    )
    return root
