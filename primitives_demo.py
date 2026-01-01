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
from robo_utils.conversion_utils import (
    pose_to_transformation, 
    transform_pcd, 
    rotate_pose_around_local_x,
    move_pose_along_local_z
)

# Import from frankapanda package
from frankapanda import FrankaPandaController
from frankapanda.perception import PerceptionPipeline
from frankapanda.motionplanner import MotionPlanner

EE_LINK_CENTER_TO_GRIPPER_TIP = 0.105

def visualize_gripper_in_pointcloud(pcd, gripper_pose, rgb=None, base_frame=True):

    gripper_transform = pose_to_transformation(gripper_pose, format='wxyz')
    gripper_points, gripper_colors = make_gripper_visualization(
        rotation=gripper_transform[:3, :3],
        translation=gripper_transform[:3, 3],
        length=0.05,
        density=50,
        color=(1, 0, 0)
    )

    combined_pcd = np.vstack([pcd, gripper_points])
    if rgb is not None:
        combined_rgb = np.vstack([rgb, gripper_colors])
        plot_pcd(combined_pcd, combined_rgb, base_frame=base_frame)
    else:
        plot_pcd(combined_pcd, base_frame=base_frame)


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
    print("4. Capturing point cloud from dual cameras...")
    try:
        pcd, rgb = perception.get_point_cloud()
    except TimeoutError as e:
        print("Make sure the perception pipeline is running!")
        return

    controller.move_to_joints(controller.home_joints, controller.open_gripper_action)

    current_joints = controller.get_robot_joints()
    current_joints = torch.tensor(current_joints, dtype=torch.float32, device="cuda:0")

    # Initialize motion planner
    motion_planner = MotionPlanner(pcd)

    # Get current gripper pose
    gripper_pose = motion_planner.fk(current_joints).cpu().numpy()
    visualize_gripper_in_pointcloud(pcd, gripper_pose, rgb=rgb, base_frame=True)

    # Grasp Pose
    grasp_pose = torch.tensor(
        [0.43239057, -0.3163708, 0.22059757, 0.00328029, -0.9574287, 0.28860268, -0.00530971],
        dtype=torch.float32,
        device="cuda:0"
    )

    # motion_planner.visualize_world_and_robot(current_joints)
    # Pre Grasp Pose
    pre_grasp_pose = move_pose_along_local_z(grasp_pose, -0.12)
    pre_grasp_pose = torch.tensor(pre_grasp_pose, dtype=torch.float32, device="cuda:0")

    # Target Shelf Pose
    # TODO: Remember to change this to edge of the shelf, not inside it
    target_shelf_pose = torch.tensor(
        [0.519, -0.06, 0.375, 0.7071, -0.7071, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda:0"
    )

    # Visualize grasp and target poses: red=grasp, green=target
    print("Visualizing poses: red=grasp, green=target")
    visualize_poses_in_pointcloud(
        pcd,
        [grasp_pose, target_shelf_pose],
        rgb=rgb,
        colors=[(1, 0, 0), (0, 1, 0)]
    )

    # Lift Pose
    lift_pose = grasp_pose.clone()
    lift_pose[2] = target_shelf_pose[2]

    # Intermediate Pose 2 to rotate in place only
    inter_pose = target_shelf_pose.clone()
    inter_pose[0] = target_shelf_pose[0]   # X value same as target
    inter_pose[1] = -0.25   # Hard-coded retraction Y-value before inserting into shelf
    inter_pose[2] = target_shelf_pose[2]   # Z value (height) same as target

    pre_push_pose = torch.tensor(
        [0.519, -0.25, 0.399, -0.5, 0.5, 0.5, 0.5],
        dtype=torch.float32,
        device="cuda:0"
    )
    push_pose = move_pose_along_local_z(pre_push_pose, 0.33)
    push_pose = torch.tensor(push_pose, dtype=torch.float32, device="cuda:0")

    all_poses = [pre_grasp_pose, grasp_pose, lift_pose, inter_pose, target_shelf_pose, pre_push_pose, push_pose]
    # Visualize all poses in pointcloud
    # for pose in all_poses[-3:]:
    #     visualize_gripper_in_pointcloud(pcd, pose.cpu().numpy(), rgb=rgb, base_frame=True)

    # TODO: Plan between these
    # TODO: Reverse the plan after placing to come back to inter_pose2
    # TODO: Close and rotate the gripper, then plan forward across z-axis

    # TODO: 2 more demos: Pushing left, and Pushing right as skills.
    
    controller.open_gripper()
    
    pre_grasp_trajectories, pre_grasp_success = motion_planner.plan_to_goal_poses(
        current_joints=current_joints.unsqueeze(0),
        goal_poses=pre_grasp_pose.unsqueeze(0),
    )
    print(pre_grasp_success.item())

    grasp_trajectories, grasp_success = motion_planner.plan_to_goal_poses(
        current_joints=pre_grasp_trajectories[0, -1].unsqueeze(0),
        goal_poses=grasp_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-5:],   # Disable collision with gripper and fingers
        plan_config=motion_planner.along_z_axis_plan_config   # Plan along z-axis only
    )
    print(grasp_success.item())

    lift_trajectories, lift_success = motion_planner.plan_to_goal_poses(
        current_joints=grasp_trajectories[0, -1].unsqueeze(0),
        goal_poses=lift_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-5:],   # Disable collision with gripper and fingers
        plan_config=motion_planner.lift_plan_config   # Plan along world frame z-axis only
    )
    print(lift_success.item())

    # Enable blockers to force intermediate pose path
    # motion_planner.enable_intermediate_pose_blockers(True)

    inter_pose_trajectories, inter_pose_success = motion_planner.plan_to_goal_poses(
        current_joints=lift_trajectories[0, -1].unsqueeze(0),
        goal_poses=inter_pose.unsqueeze(0),
        plan_config=motion_planner.only_xy_translation_plan_config   # Plan along world frame x, y only
    )
    print(inter_pose_success.item())

    # Disable blockers for shelf insertion
    # motion_planner.enable_intermediate_pose_blockers(False)

    target_trajectories, target_success = motion_planner.plan_to_goal_poses(
        current_joints=inter_pose_trajectories[0, -1].unsqueeze(0),
        goal_poses=target_shelf_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-5:],   # Disable collision with gripper and fingers
        plan_config=motion_planner.along_z_axis_plan_config   # Plan along z-axis only
    )
    print(target_success.item())

    pre_push_trajectories, pre_push_success = motion_planner.plan_to_goal_poses(
        current_joints=target_trajectories[0, 0].unsqueeze(0),  # Will be used after reversing the target trajectory, so first joint state is the start
        goal_poses=pre_push_pose.unsqueeze(0),
        # plan_config=motion_planner.only_z_rot_and_xy_translation_plan_config   # Plan along world frame x, y and z rotation only
    )
    print(pre_push_success.item())

    push_trajectories, push_success = motion_planner.plan_to_goal_poses(
        current_joints=pre_push_trajectories[0, -1].unsqueeze(0),
        goal_poses=push_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-7:],   # Disable collision with gripper and fingers
        plan_config=motion_planner.along_z_axis_plan_config   # Push along z-axis
    )
    print(push_success.item())

    success = pre_grasp_success.item() & grasp_success.item() & lift_success.item() & inter_pose_success.item() & target_success.item() & pre_push_success.item() & push_success.item()

    if success:
        controller.move_along_trajectory(pre_grasp_trajectories[0].cpu().numpy(), controller.open_gripper_action)
        controller.move_along_trajectory(grasp_trajectories[0].cpu().numpy(), controller.open_gripper_action)
        controller.close_gripper(num_steps=80)
        controller.move_along_trajectory(lift_trajectories[0].cpu().numpy(), controller.close_gripper_action)
        controller.move_along_trajectory(inter_pose_trajectories[0].cpu().numpy(), controller.close_gripper_action)
        controller.move_along_trajectory(target_trajectories[0].cpu().numpy(), controller.close_gripper_action)
        controller.open_gripper(num_steps=80)
        controller.move_along_trajectory(target_trajectories[0].flip(0).cpu().numpy(), controller.open_gripper_action)  # Move back along reverse trajectory
        controller.move_along_trajectory(pre_push_trajectories[0].cpu().numpy(), controller.open_gripper_action)
        controller.close_gripper(num_steps=80)
        controller.move_along_trajectory(push_trajectories[0].cpu().numpy(), controller.close_gripper_action)

    controller.move_to_joints(controller.home_joints, controller.close_gripper_action)
    controller.open_gripper()

    perception.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")