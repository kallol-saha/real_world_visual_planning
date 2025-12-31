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
import torch
import time
from scipy.spatial.transform import Rotation as R
from robo_utils.visualization.plotting import plot_pcd, make_gripper_visualization
from robo_utils.conversion_utils import pose_to_transformation, transform_pcd, rotate_pose_around_local_x

# Import from frankapanda package
from frankapanda import FrankaPandaController
from frankapanda.perception import PerceptionPipeline
from frankapanda.motionplanner import MotionPlanner


def main():

    # Initialize robot controller
    controller = FrankaPandaController()

    # Initialize perception pipeline client
    perception = PerceptionPipeline(publish_port=6556, timeout_ms=10000)

    current_joints = controller.get_robot_joints()
    current_joints = torch.tensor(current_joints, dtype=torch.float32, device="cuda:0")

    # Capture point cloud
    print("4. Capturing point cloud from dual cameras...")
    try:
        pcd, rgb = perception.get_point_cloud()
    except TimeoutError as e:
        print("Make sure the perception pipeline is running!")
        return

    # controller.move_to_joints(controller.home_joints)

    # T = np.eye(4)
    # T[:3, -1] = np.array([0.035, 0.05, 0.07])
    # theta_y = np.deg2rad(5) 
    # rot_y = R.from_euler('y', theta_y).as_matrix()
    # rot_x = R.from_euler('x', theta_x).as_matrix()
    # rot_combined = rot_x @ rot_y  # Apply Y first, then X
    # T[:3, :3] = rot_combined

    # new_pcd = transform_pcd(pcd, T)

    # Initialize motion planner
    motion_planner = MotionPlanner(pcd)

    # Get current robot state
    # gripper_pose = controller.get_gripper_pose(as_transform=False, format='wxyz')
    gripper_pose = motion_planner.fk(current_joints).cpu().numpy()
    # gripper_pose = rotate_pose_around_local_x(gripper_pose, np.deg2rad(10), format='wxyz')
    gripper_transform = pose_to_transformation(gripper_pose, format='wxyz')
    # rot_z = np.array([[0, -1, 0],
    #                   [1, 0, 0],
    #                   [0, 0, 1]]).T
    # # gripper_transform[:3, :3] = rot_z @ gripper_transform[:3, :3]
    # gripper_transform[:3, 2] = - gripper_transform[:3, 2]
    # gripper_transform[:3, 0] = - gripper_transform[:3, 0]

    gripper_pose = controller.get_gripper_pose(as_transform=False, format='wxyz')
    gripper_transform = pose_to_transformation(gripper_pose, format='wxyz')

    # gripper_transform = controller.get_gripper_pose(as_transform=True)

    gripper_points, gripper_colors = make_gripper_visualization(
        rotation=gripper_transform[:3, :3],
        translation=gripper_transform[:3, 3],
        length=0.05,
        density=50,
        color=(1, 0, 0)
    )

    combined_pcd = np.vstack([pcd, gripper_points])
    combined_rgb = np.vstack([rgb, gripper_colors])

    # Visualize point cloud

    # motion_planner.visualize_world_and_robot(current_joints)
    plot_pcd(combined_pcd, combined_rgb, base_frame=True)

    # Demo robot movement
    controller.move_to_joints(controller.home_joints)
    controller.open_gripper()

    perception.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")