"""
Load saved placements for env_0 produced by sample_placements.py.
Also captures a fresh point cloud from the perception pipeline.
"""

from pathlib import Path

import numpy as np
import torch

from robo_utils.visualization.plotting import plot_pcd
from robo_utils.visualization.point_cloud_structures import make_gripper_visualization
from robo_utils.conversion_utils import pose_to_transformation
from frankapanda.perception import PerceptionPipeline
from frankapanda.motionplanner import MotionPlanner
from frankapanda import FrankaPandaController


def visualize_pose_on_pcd(pcd, pose, color=(0.0, 1.0, 1.0), length=0.05):
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    transform = pose_to_transformation(pose, format="wxyz")
    pts, cols = make_gripper_visualization(
        rotation=transform[:3, :3],
        translation=transform[:3, 3],
        length=length,
        density=50,
        color=color,
    )
    combined_pcd = np.vstack([pcd, pts])
    plot_pcd(combined_pcd, base_frame=True)



ENV_ID = 0
OBJECT_ID = 1
GRASP_ID = 0
SAVE_DIR = Path("data/shelf_packing_scenes")
PUBLISH_PORT = 1235

def execute_placement(motion_planner: MotionPlanner, controller: FrankaPandaController, start_joints: torch.tensor, placement_pose: torch.tensor, pcd_np: np.ndarray):

    Z_OFFSET = -0.1
    Y_OFFSET = -0.3
    X_OFFSET = -0.1

    placement_pose[2] = placement_pose[2] + Z_OFFSET
    placement_pose[1] = placement_pose[1] - 0.3
    inter_pose = placement_pose.clone()
    inter_pose[0] = placement_pose[0] + X_OFFSET
    inter_pose[1] = Y_OFFSET
    inter_pose[2] = 0.3

    # print("  Visualizing inter_pose (cyan). Close window to continue.")
    # visualize_pose_on_pcd(pcd_np, inter_pose, color=(0.0, 1.0, 1.0))

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

    # 5. Retract: reverse pre-placement trajectory back to start joint config.
    controller.move_along_trajectory(place_np[::-1], controller.close_gripper_action)
    controller.move_along_trajectory(pre_place_np[::-1], controller.close_gripper_action)

    print("")


def main():
    npz_path = SAVE_DIR / f"env_{ENV_ID}" / "placement_poses.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing saved placements: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    placements = torch.from_numpy(
        np.asarray(data["placements"], dtype=np.float32)
    ).to("cuda:0")                                                         # (M, 7) wxyz
    grasp_pose = data["grasp_pose"]                                        # (7,)
    target_label = str(data["target_label"])
    object_id = int(data["object_id"])
    grasp_id = int(data["grasp_id"])
    saved_pcd = data["pcd"]                                                # (N, 3) capture-time

    print(f"Loaded {npz_path}")
    print(f"  placements: {placements.shape}  grasp_pose: {grasp_pose.tolist()}")
    print(f"  target_label: '{target_label}'  object_id: {object_id}  grasp_id: {grasp_id}")
    print(f"  saved_pcd: {saved_pcd.shape}")

    print("Connecting to perception pipeline")
    perception = PerceptionPipeline(publish_port=PUBLISH_PORT, timeout_ms=10000)
    print("Capturing point cloud")
    try:
        pdata = perception.get_point_cloud_dict()
    except TimeoutError:
        print("Timeout waiting for PCD. Is `python -m frankapanda.perception."
              "perception_pipeline --continuous` running?")
        perception.close()
        return
    pcd_np = pdata["pcd"]
    print(f"  live pcd: {pcd_np.shape}")

    # Load grasp trajectory and derive start_joints = last config.
    grasps_npz_path = SAVE_DIR / f"env_{ENV_ID}" / f"grasps_{OBJECT_ID}.npz"
    if not grasps_npz_path.exists():
        raise FileNotFoundError(f"Missing saved grasps: {grasps_npz_path}")
    grasps_data = np.load(grasps_npz_path, allow_pickle=True)
    trajectories = grasps_data["trajectories"]                             # object array of (T_i, 7)
    if not (0 <= GRASP_ID < len(trajectories)):
        raise IndexError(
            f"GRASP_ID={GRASP_ID} out of range; npz has {len(trajectories)} trajectories."
        )
    grasp_traj = np.asarray(trajectories[GRASP_ID], dtype=np.float32)      # (T, 7)
    if grasp_traj.shape[0] == 0:
        raise RuntimeError(f"Grasp trajectory GRASP_ID={GRASP_ID} is empty (planning failed).")
    start_joints = torch.from_numpy(grasp_traj[-1]).float().to("cuda:0")   # (7,)
    print(f"Loaded grasp traj from {grasps_npz_path}")
    print(f"  GRASP_ID={GRASP_ID} traj: {grasp_traj.shape}  start_joints: {start_joints.tolist()}")

    controller = FrankaPandaController()
    motion_planner = MotionPlanner(pcd_np)

    for i in range(placements.shape[0]):
        print(f"\n=== Executing placement {i+1}/{placements.shape[0]} ===")
        execute_placement(motion_planner, controller, start_joints, placements[i], pcd_np)

    perception.close()


if __name__ == "__main__":
    main()
