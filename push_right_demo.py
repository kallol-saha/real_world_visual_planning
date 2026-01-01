"""
Push Left Demo: FrankaPanda Push Left Action

This script demonstrates a push left action:
1. Move to pre-push pose
2. Move to push start pose (along z-axis)
3. Push left along y-axis

Before running:
1. Make sure the perception pipeline is running:
   python -m frankapanda.perception.perception_pipeline --continuous

2. Make sure the robot is connected and powered on

Usage:
    python push_left_demo.py
"""

import numpy as np
import torch
from robo_utils.conversion_utils import move_pose_along_local_y, pose_to_transformation
from robo_utils.visualization.plotting import plot_pcd, make_gripper_visualization

# Import from frankapanda package
from frankapanda import FrankaPandaController
from frankapanda.perception import PerceptionPipeline
from frankapanda.motionplanner import MotionPlanner

# Pre-push offset along world y-axis
PRE_PUSH_Y_OFFSET = -0.1  # Move back along y before pushing


def visualize_poses_in_pointcloud(pcd, poses, rgb=None, colors=None):
    """Visualize multiple gripper poses in a point cloud."""
    if colors is None:
        colors = [(1, 0, 0)] * len(poses)

    combined_pcd = pcd.copy()
    combined_rgb = rgb.copy() if rgb is not None else None

    for i, pose in enumerate(poses):
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        gripper_transform = pose_to_transformation(pose, format='wxyz')
        gripper_points, gripper_colors = make_gripper_visualization(
            rotation=gripper_transform[:3, :3],
            translation=gripper_transform[:3, 3],
            length=0.05,
            density=50,
            color=colors[i % len(colors)]
        )
        combined_pcd = np.vstack([combined_pcd, gripper_points])
        if combined_rgb is not None:
            combined_rgb = np.vstack([combined_rgb, gripper_colors])

    if combined_rgb is not None:
        plot_pcd(combined_pcd, combined_rgb, base_frame=True)
    else:
        plot_pcd(combined_pcd, base_frame=True)


def main():

    # Initialize robot controller
    controller = FrankaPandaController()

    # Initialize perception pipeline client
    perception = PerceptionPipeline(publish_port=1235, timeout_ms=10000)

    # Capture point cloud
    print("Capturing point cloud from dual cameras...")
    try:
        pcd, rgb = perception.get_point_cloud()
    except TimeoutError as e:
        print("Make sure the perception pipeline is running!")
        return

    # Get current joints (robot is already at push start pose)
    current_joints = controller.get_robot_joints()
    current_joints = torch.tensor(current_joints, dtype=torch.float32, device="cuda:0")

    # Initialize motion planner
    motion_planner = MotionPlanner(pcd)

    # Get current pose via FK (this is push_start_pose)
    push_start_pose = motion_planner.fk(current_joints)

    # Push end pose: push_start moved along local y-axis
    push_end_pose = move_pose_along_local_y(push_start_pose, 0.2)
    push_end_pose = torch.tensor(push_end_pose, dtype=torch.float32, device="cuda:0")

    # Visualize poses: red=push_start, green=push_end
    print("Visualizing poses: red=push_start, green=push_end")
    visualize_poses_in_pointcloud(
        pcd,
        [push_start_pose, push_end_pose],
        rgb=rgb,
        colors=[(1, 0, 0), (0, 1, 0)]
    )

    # Plan push motion (along y-axis, disable all links)
    push_trajectories, push_success = motion_planner.plan_to_goal_poses(
        current_joints=current_joints.unsqueeze(0),
        goal_poses=push_end_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[:],
        plan_config=motion_planner.along_y_axis_plan_config
    )
    print(f"Push planning success: {push_success.item()}")

    if push_success.item():
        controller.move_along_trajectory(push_trajectories[0].cpu().numpy(), controller.close_gripper_action)

    controller.move_to_joints(controller.home_joints, controller.close_gripper_action)

    perception.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
