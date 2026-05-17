"""
Debug rig: load saved placements for env_<ENV_ID>, sweep them through the
execute_plan placement plan/execute pipeline using start_joints derived from
a single (OBJECT_ID, GRASP_ID) grasp trajectory. No perception; static
collision world only.
"""

from pathlib import Path

import numpy as np
import torch

from frankapanda.motionplanner import MotionPlanner
from frankapanda import FrankaPandaController

from robo_utils.visualization.plotting import plot_pcd
from robo_utils.visualization.point_cloud_structures import make_gripper_visualization
from robo_utils.conversion_utils import pose_to_transformation


def visualize_poses_on_pcd(pcd, poses, colors, length=0.05):
    combined = pcd.copy()
    for pose, color in zip(poses, colors):
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        transform = pose_to_transformation(pose, format="wxyz")
        pts, _ = make_gripper_visualization(
            rotation=transform[:3, :3],
            translation=transform[:3, 3],
            length=length,
            density=50,
            color=color,
        )
        combined = np.vstack([combined, pts])
    plot_pcd(combined, base_frame=True)


ENV_ID = 1
OBJECT_ID = 0
GRASP_ID = 1

SAVE_DIR = Path("data/shelf_packing_scenes")

def execute_placement(motion_planner: MotionPlanner, controller: FrankaPandaController, start_joints: torch.tensor, placement_pose: torch.tensor, pcd_np: np.ndarray):

    # Offsets match execute_plan.plan_and_execute_placement exactly.
    Z_OFFSET = 0.
    Y_OFFSET = -0.3
    X_OFFSET = -0.1

    # Clone so caller's tensor is not mutated across debug iterations.
    placement_pose = placement_pose.clone()
    placement_pose[2] = placement_pose[2] + Z_OFFSET
    placement_pose[1] = placement_pose[1] - 0.2
    inter_pose = placement_pose.clone()
    inter_pose[0] = placement_pose[0] + X_OFFSET
    inter_pose[1] = Y_OFFSET
    inter_pose[2] = 0.3

    # Viz: pre-placement (cyan), placement (green). Close window to continue.
    print("  Visualizing inter_pose (cyan) + placement_pose (green).")
    visualize_poses_on_pcd(
        pcd_np,
        [inter_pose, placement_pose],
        colors=[(0.0, 1.0, 1.0), (0.0, 1.0, 0.0)],
    )

    # 1. Move to start (end-of-grasp config, object held).
    controller.move_to_joints(start_joints.cpu().detach().numpy(), controller.close_gripper_action)

    # 2. Plan pre-placement: start_joints -> inter_pose (unconstrained).
    pre_place_traj, pre_place_success = motion_planner.plan_to_goal_poses(
        current_joints=start_joints.unsqueeze(0),
        goal_poses=inter_pose.unsqueeze(0),
    )
    print(f"  pre-placement plan success: {bool(pre_place_success[0].item())}")

    # 3. Plan placement: end of pre-place traj -> placement_pose (Y/Z translation only).
    placement_pose_t = placement_pose.to("cuda:0").float().unsqueeze(0)    # (1, 7)
    if bool(pre_place_success[0].item()):
        plan_start = pre_place_traj[0, -1].unsqueeze(0)
    else:
        plan_start = start_joints.unsqueeze(0)
    place_traj, place_success = motion_planner.plan_to_goal_poses(
        current_joints=plan_start,
        goal_poses=placement_pose_t,
        plan_config=motion_planner.only_yz_translation_plan_config,
    )
    print(f"  placement plan success: {bool(place_success[0].item())}")

    # 4. Execute only if both plans succeeded.
    planning_success = bool(pre_place_success[0].item()) and bool(place_success[0].item())
    if not planning_success:
        print("  ABORT: one or more plans failed; not executing.")
        return

    pre_place_np = pre_place_traj[0].cpu().detach().numpy()
    place_np = place_traj[0].cpu().detach().numpy()
    controller.move_along_trajectory(pre_place_np, controller.close_gripper_action)
    controller.move_along_trajectory(place_np, controller.close_gripper_action)

    # Release object at placement pose.
    controller.open_gripper()

    # 5. Retract: reverse placement, then reverse pre-placement.
    controller.move_along_trajectory(place_np[::-1], controller.open_gripper_action)
    controller.move_along_trajectory(pre_place_np[::-1], controller.open_gripper_action)

    print("")


def main():
    npz_path = SAVE_DIR / f"env_{ENV_ID}" / "placement_poses.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing saved placements: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    placements_np = np.asarray(data["placements"], dtype=np.float32)        # (M, 7) wxyz
    # Match execute_plan: override every placement quaternion with a fixed
    # orientation. gripper x -> world +x, gripper z -> world +y
    # (so gripper y = g_z x g_x = world -z). 90 deg about world -x axis.
    # wxyz = (sqrt(2)/2, -sqrt(2)/2, 0, 0).
    FIXED_QUAT_WXYZ = np.array([0.70710677, -0.70710677, 0.0, 0.0], dtype=np.float32)
    placements_np[:, 3:7] = FIXED_QUAT_WXYZ
    placements = torch.from_numpy(placements_np).to("cuda:0")              # (M, 7) wxyz
    target_label = str(data["target_label"])
    pcd_np = np.asarray(data["pcd"], dtype=np.float32) if "pcd" in data.files else np.zeros((0, 3), np.float32)
    print(f"Loaded {npz_path}")
    print(f"  placements: {placements.shape}  target_label: '{target_label}'  pcd: {pcd_np.shape}")

    # Load grasp trajectory and derive start_joints = last config.
    grasps_npz_path = SAVE_DIR / f"env_{ENV_ID}" / f"grasps_{OBJECT_ID}.npz"
    if not grasps_npz_path.exists():
        raise FileNotFoundError(f"Missing saved grasps: {grasps_npz_path}")
    grasps_data = np.load(grasps_npz_path, allow_pickle=True)
    trajectories = grasps_data["trajectories"]
    if not (0 <= GRASP_ID < len(trajectories)):
        raise IndexError(
            f"GRASP_ID={GRASP_ID} out of range; npz has {len(trajectories)} trajectories."
        )
    grasp_traj = np.asarray(trajectories[GRASP_ID], dtype=np.float32)
    if grasp_traj.shape[0] == 0:
        raise RuntimeError(f"Grasp trajectory GRASP_ID={GRASP_ID} is empty.")
    start_joints = torch.from_numpy(grasp_traj[-1]).float().to("cuda:0")
    print(f"Loaded grasp traj from {grasps_npz_path}")
    print(f"  OBJECT_ID={OBJECT_ID} GRASP_ID={GRASP_ID} traj {grasp_traj.shape}  "
          f"start_joints={start_joints.tolist()}")

    controller = FrankaPandaController()
    motion_planner = MotionPlanner()                                       # static obstacles only

    for i in range(placements.shape[0]):
        print(f"\n=== Placement {i+1}/{placements.shape[0]} ===")
        execute_placement(motion_planner, controller, start_joints, placements[i], pcd_np)


if __name__ == "__main__":
    main()
