"""
Minimal Demo: FrankaPanda Perception + Control

This script demonstrates the integrated use of:
1. Perception system - Getting point clouds from dual Azure Kinect cameras
2. Robot control - Moving the Franka Panda robot
3. Visualization - Viewing the point cloud with Open3D

Before running:
1. Make sure the perception pipeline is running:
   python -m frankapanda.perception.perception_pipeline --continuous

2. Make sure the robot is connected and powered on

Usage:
    python demo_perception_control.py
"""

import numpy as np
import time
from robo_utils.visualization.plotting import plot_pcd

# Import from frankapanda package
from frankapanda import FrankaPandaController
from frankapanda.perception import PerceptionPipeline


def main():

    # Initialize robot controller
    controller = FrankaPandaController()

    # Initialize perception pipeline client
    perception = PerceptionPipeline(publish_port=6556, timeout_ms=10000)

    # Get current robot state
    gripper_pose = controller.get_gripper_pose(as_transform=False, format='wxyz')

    # Capture point cloud
    print("4. Capturing point cloud from dual cameras...")
    try:
        pcd, rgb = perception.get_point_cloud()
    except TimeoutError as e:
        print("Make sure the perception pipeline is running!")
        return

    # Visualize point cloud

    plot_pcd(pcd, rgb, base_frame=True)

    # Demo robot movement
    controller.move_to_joints(controller.home_joints)
    controller.open_gripper()

    perception.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")