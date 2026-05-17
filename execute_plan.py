"""
End-to-end plan execution: for a user-specified object list (e.g. [0, 1]),
pick a random object order, pick a random successful grasp per object,
replay the grasp trajectory, then sample a (previously-unused) placement
pose, plan + execute the placement, and retract back to home. Between every
(grasp, placement) the live pcd/rgb is verified by the user. On completion,
KeyboardInterrupt, or exception, prompt the user to mark the run success /
failure and dump everything to env_<ENV_ID>/<success|failure>/run_<ts>.npz.
"""

import os
import re
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import zmq

from robo_utils.visualization.plotting import plot_pcd
from robo_utils.visualization.point_cloud_structures import make_gripper_visualization
from robo_utils.conversion_utils import pose_to_transformation

from frankapanda.perception import PerceptionPipeline
from frankapanda.motionplanner import MotionPlanner
from frankapanda import FrankaPandaController


# --------------------------------------------------------------------------- #
#                                                                             #
#   >>>  USER EDIT POINT  <<<                                                 #
#                                                                             #
#   OBJECT_IDS: which object grasps_<id>.npz files to use (any subset).       #
#   ENV_ID: matches env_<ENV_ID>/ folder under SAVE_DIR.                      #
# --------------------------------------------------------------------------- #
ENV_ID = 2
OBJECT_IDS = [1, 2, 3]
RANDOMIZE_OBJECT_ORDER = False   # False -> use OBJECT_IDS as given

# Optional per-object grasp restriction. None -> all successful grasps are
# eligible (current default). dict -> only listed indices are eligible
# (bypasses the saved successes mask). Format: {object_id: [grasp_indices]}.
GRASP_DICT = {
    2: [2], #[1, 2],     # blue and purple lego brick
    1: [1], #[0, 1],     # green lego brick
    3: [4], #[0, 2, 3],  # red and green lego
    # 3: [0, 3, 4],  # green and blue lego
}

SAVE_DIR = Path("data/shelf_packing_scenes")
PUBLISH_PORT = 1235


# Placement offsets — keep same as execute_placements.execute_placement.
Z_OFFSET = -0.1
Y_OFFSET = -0.3
X_OFFSET = -0.1


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _as_float_traj(t):
    if t is None:
        return None
    a = np.asarray(t, dtype=np.float32)
    return a if a.ndim == 2 else None


def visualize_pose_on_pcd(pcd, pose, color=(0.0, 1.0, 1.0), length=0.05):
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
    combined = np.vstack([pcd, pts])
    plot_pcd(combined, base_frame=True)


_POSE_COLORS = [
    (1.0, 0.0, 0.0),  # red
    (0.0, 1.0, 0.0),  # green
    (0.0, 0.0, 1.0),  # blue
    (1.0, 1.0, 0.0),  # yellow
    (1.0, 0.0, 1.0),  # magenta
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 0.5, 0.0),  # orange
    (0.5, 0.0, 1.0),  # purple
    (1.0, 0.75, 0.8), # pink
]


def visualize_poses_on_pcd(pcd, poses, length=0.05):
    """Overlay all gripper widgets on a single pcd plot. poses: iterable of (7,) wxyz."""
    combined = pcd.copy()
    for i, pose in enumerate(poses):
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        transform = pose_to_transformation(pose, format="wxyz")
        pts, _ = make_gripper_visualization(
            rotation=transform[:3, :3],
            translation=transform[:3, 3],
            length=length,
            density=50,
            color=_POSE_COLORS[i % len(_POSE_COLORS)],
        )
        combined = np.vstack([combined, pts])
    plot_pcd(combined, base_frame=True)


def _drain_perception_socket(perception: PerceptionPipeline):
    """ZMQ SUB buffers all received msgs FIFO. Discard everything currently
    queued so the next blocking recv returns a fresh frame, not a stale one."""
    dropped = 0
    while True:
        try:
            perception.socket.recv(flags=zmq.DONTWAIT)
            dropped += 1
        except zmq.Again:
            break
    if dropped:
        print(f"  drained {dropped} stale frames from perception socket.")


def capture_until_user_ok(perception: PerceptionPipeline):
    """
    Capture pcd+rgb, plot both, prompt user. Loop until user accepts.
    Returns (pcd_np, rgb_np).
    """
    while True:
        _drain_perception_socket(perception)
        try:
            pdata = perception.get_point_cloud_dict()
        except TimeoutError:
            print("  perception timeout — retrying.")
            continue
        pcd_np = pdata["pcd"]
        rgb_np = pdata["rgb"]
        print(f"  captured pcd {pcd_np.shape}, rgb {rgb_np.shape}. Plotting (close window to continue).")
        plot_pcd(pcd_np, rgb_np, base_frame=True)
        ans = input("  Is this the right point cloud? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            return pcd_np, rgb_np
        print("  Re-capturing...")


def prompt_success() -> bool:
    while True:
        ans = input("\nWas this run a successful plan? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("", "n", "no"):
            return False
        print("  Please answer y or n.")


def plan_and_execute_placement(
    motion_planner: MotionPlanner,
    controller: FrankaPandaController,
    start_joints: torch.Tensor,
    placement_pose: torch.Tensor,
):
    """
    Mirrors execute_placements.execute_placement but returns the executed
    pre-placement + placement trajectories (np.ndarray) so they can be
    recorded. Returns (pre_place_np, place_np, success_bool).
    """
    placement_pose = placement_pose.clone()
    placement_pose[2] = placement_pose[2] + Z_OFFSET
    placement_pose[1] = placement_pose[1] - 0.15
    inter_pose = placement_pose.clone()
    inter_pose[0] = placement_pose[0] + X_OFFSET
    inter_pose[1] = Y_OFFSET
    inter_pose[2] = 0.3

    pre_place_traj, pre_place_success = motion_planner.plan_to_goal_poses(
        current_joints=start_joints.unsqueeze(0),
        goal_poses=inter_pose.unsqueeze(0),
    )
    print(f"    pre-placement plan success: {bool(pre_place_success[0].item())}")

    placement_pose_t = placement_pose.to("cuda:0").float().unsqueeze(0)
    if bool(pre_place_success[0].item()):
        plan_start = pre_place_traj[0, -1].unsqueeze(0)
    else:
        plan_start = start_joints.unsqueeze(0)
    place_traj, place_success = motion_planner.plan_to_goal_poses(
        current_joints=plan_start,
        goal_poses=placement_pose_t,
        plan_config=motion_planner.only_yz_translation_plan_config,
    )
    print(f"    placement plan success: {bool(place_success[0].item())}")

    planning_success = bool(pre_place_success[0].item()) and bool(place_success[0].item())
    if not planning_success:
        print("    ABORT: one or more placement plans failed.")
        return None, None, False

    pre_place_np = pre_place_traj[0].cpu().detach().numpy()
    place_np = place_traj[0].cpu().detach().numpy()

    controller.move_along_trajectory(pre_place_np, controller.close_gripper_action)
    controller.move_along_trajectory(place_np, controller.close_gripper_action)

    # Open gripper to release object.
    controller.open_gripper()

    # Retract: reverse placement, then reverse pre-placement.
    controller.move_along_trajectory(place_np[::-1], controller.open_gripper_action)
    controller.move_along_trajectory(pre_place_np[::-1], controller.open_gripper_action)

    return pre_place_np, place_np, True


def save_run(env_dir: Path, success: bool, record: dict):
    sub = "success" if success else "failure"
    out_dir = env_dir / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"run_{ts}.npz"
    np.savez(
        out_path,
        object_order=np.asarray(record["object_order"]),
        pcds=np.array(record["pcds"], dtype=object),
        rgbs=np.array(record["rgbs"], dtype=object),
        grasp_trajs=np.array(record["grasp_trajs"], dtype=object),
        place_pre_trajs=np.array(record["place_pre_trajs"], dtype=object),
        place_trajs=np.array(record["place_trajs"], dtype=object),
        grasp_poses=np.array(record["grasp_poses"], dtype=object),
        placement_poses=np.array(record["placement_poses"], dtype=object),
        target_labels=np.array(record["target_labels"], dtype=object),
        used_placement_indices=np.asarray(record["used_placement_indices"], dtype=np.int64),
    )
    print(f"Saved run -> {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    env_dir = SAVE_DIR / f"env_{ENV_ID}"
    if not env_dir.is_dir():
        raise FileNotFoundError(f"Env dir not found: {env_dir}")

    # Load per-object grasps.
    object_grasps = {}
    for oid in OBJECT_IDS:
        path = env_dir / f"grasps_{oid}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        data = np.load(path, allow_pickle=True)
        object_grasps[oid] = {
            "grasps": data["grasps"],                      # (G, 7)
            "trajectories": data["trajectories"],          # object array
            "successes": data["successes"].astype(bool),
            "target_label": str(data["target_label"]) if "target_label" in data.files else "?",
        }
        print(f"Loaded object {oid}: '{object_grasps[oid]['target_label']}' "
              f"({int(object_grasps[oid]['successes'].sum())}/{len(object_grasps[oid]['grasps'])} successful)")

    # Load placement poses (single per-env file).
    place_path = env_dir / "placement_poses.npz"
    if not place_path.exists():
        raise FileNotFoundError(f"Missing {place_path}")
    place_data = np.load(place_path, allow_pickle=True)
    all_placements = np.asarray(place_data["placements"], dtype=np.float32)  # (M, 7)
    print(f"Loaded {len(all_placements)} placement poses from {place_path}")

    # Object order first; we sample N=len(object_order) placements next.
    object_order = list(OBJECT_IDS)
    if RANDOMIZE_OBJECT_ORDER:
        np.random.shuffle(object_order)
        print(f"\nObject execution order (randomized): {object_order}\n")
    else:
        print(f"\nObject execution order (as given): {object_order}\n")

    # Override every placement quaternion with a fixed orientation:
    #   gripper z-axis = world +y, gripper x-axis = world +z
    # (so gripper y-axis = gripper_z x gripper_x = +y x +z = +x in world.)
    # Resulting wxyz quat = (0.5, -0.5, -0.5, -0.5) (norm 1, verified by
    # reconstruction). Position is taken from each saved placement.
    FIXED_QUAT_WXYZ = np.array([0.70710677, -0.70710677, 0.0, 0.0], dtype=np.float32)
    all_placements = all_placements.copy()
    all_placements[:, 3:7] = FIXED_QUAT_WXYZ

    # Stratified pick by descending y: sort all placements, split into n_obj
    # buckets, pick one random per bucket. Execute bucket 0 (highest y) first.
    n_pick = len(object_order)
    if n_pick > len(all_placements):
        raise RuntimeError(
            f"Need {n_pick} placements but only {len(all_placements)} available."
        )
    y_sorted_idx = np.argsort(-all_placements[:, 1])                       # descending y
    buckets = np.array_split(y_sorted_idx, n_pick)                         # n_pick chunks
    placement_queue = [int(np.random.choice(b)) for b in buckets]
    print(f"Sampled {n_pick} placements via y-stratified buckets "
          f"(bucket sizes {[len(b) for b in buckets]}); "
          f"queue {placement_queue} "
          f"(y={all_placements[placement_queue, 1].round(3).tolist()})")

    # Visualize placement_queue on saved pcd (color-coded in queue order).
    # if "pcd" in place_data.files:
    #     viz_pcd = np.asarray(place_data["pcd"], dtype=np.float32)
    #     queued_poses = [all_placements[i] for i in placement_queue]
    #     print("Visualizing placement_queue on saved pcd. Close window to continue.")
    #     visualize_poses_on_pcd(viz_pcd, queued_poses)
    # else:
    #     print("No pcd in placement_poses.npz; skipping queue visualization.")

    # Robot + perception setup.
    print("Connecting to perception pipeline")
    perception = PerceptionPipeline(publish_port=PUBLISH_PORT, timeout_ms=10000)
    controller = FrankaPandaController()

    used_placements: set[int] = set()
    record = {
        "object_order": object_order,
        "pcds": [],
        "rgbs": [],
        "grasp_trajs": [],
        "place_pre_trajs": [],
        "place_trajs": [],
        "grasp_poses": [],
        "placement_poses": [],
        "target_labels": [],
        "used_placement_indices": [],
    }

    success_overall = False
    try:
        for step_i, oid in enumerate(object_order):
            entry = object_grasps[oid]
            if GRASP_DICT is not None:
                if oid not in GRASP_DICT:
                    print(f"\n[{step_i+1}/{len(object_order)}] object {oid}: not in GRASP_DICT; skipping.")
                    continue
                requested = list(GRASP_DICT[oid])
                bad = [i for i in requested if not (0 <= i < len(entry["grasps"]))]
                if bad:
                    raise IndexError(
                        f"object_id={oid}: GRASP_DICT indices {bad} out of "
                        f"range; npz has {len(entry['grasps'])} grasps."
                    )
                success_idx = np.asarray(requested, dtype=np.int64)
            else:
                success_idx = np.where(entry["successes"])[0]
            if len(success_idx) == 0:
                print(f"\n[{step_i+1}/{len(object_order)}] object {oid}: no eligible grasps; skipping.")
                continue
            chosen = int(np.random.choice(success_idx))
            grasp_traj = _as_float_traj(entry["trajectories"][chosen])
            grasp_pose = np.asarray(entry["grasps"][chosen], dtype=np.float32)
            label = entry["target_label"]
            print(f"\n=== Step {step_i+1}/{len(object_order)}: object {oid} ('{label}') grasp idx {chosen} ===")
            if grasp_traj is None or grasp_traj.shape[0] == 0:
                print(f"  empty trajectory; skipping.")
                continue

            # Verified pcd capture before this object's pick+place.
            print("  Verifying live point cloud (pre-grasp)")
            pcd_np, rgb_np = capture_until_user_ok(perception)

            # Build motion planner with latest pcd.
            motion_planner = MotionPlanner(pcd_np)

            # Pop next placement from pre-ranked queue (top-20 by gripper-z . world_y,
            # ordered by descending y-coord). Sequential, not random.
            if not placement_queue:
                print("  Placement queue exhausted; aborting plan.")
                break
            place_idx = int(placement_queue.pop(0))
            used_placements.add(place_idx)
            placement_pose = torch.from_numpy(all_placements[place_idx]).float().to("cuda:0")
            print(f"  selected placement idx {place_idx}/{len(all_placements)} "
                  f"(y={float(all_placements[place_idx, 1]):.3f}); "
                  f"queue remaining {len(placement_queue)}")

            # 1. Replay saved grasp trajectory (pre_grasp + grasp + lift,
            # each of length T/3 = traj_tsteps from curobo). Gripper must
            # close at the boundary between grasp and lift, otherwise lift
            # leaves the object behind.
            T = grasp_traj.shape[0]
            if T % 3 != 0:
                raise RuntimeError(
                    f"Grasp trajectory length {T} not divisible by 3; cannot infer "
                    f"pre_grasp/grasp/lift split. Check sample_grasps save format."
                )
            seg = T // 3
            close_at = 2 * seg                                            # end of grasp segment
            print(f" grasp trajectory T={T} (segments of {seg}), close at idx {close_at}")
            controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
            controller.move_along_trajectory(grasp_traj[:close_at], controller.open_gripper_action)
            controller.close_gripper()
            controller.move_along_trajectory(grasp_traj[close_at:], controller.close_gripper_action)

            # 2. start_joints = last config of grasp trajectory.
            start_joints = torch.from_numpy(grasp_traj[-1]).float().to("cuda:0")

            # 3. Plan + execute placement (returns trajectories for record).
            pre_place_np, place_np, place_ok = plan_and_execute_placement(
                motion_planner, controller, start_joints, placement_pose,
            )
            if not place_ok:
                print("  Placement plan failed; aborting run.")
                break

            # 4. Reverse grasp trajectory back to home_joints.
            print(f"  reversing grasp trajectory back to home")
            # controller.move_along_trajectory(grasp_traj[::-1], controller.open_gripper_action)
            controller.move_to_joints(controller.home_joints, controller.open_gripper_action)

            # Record.
            record["pcds"].append(pcd_np.astype(np.float32))
            record["rgbs"].append(rgb_np)
            record["grasp_trajs"].append(grasp_traj)
            record["place_pre_trajs"].append(pre_place_np)
            record["place_trajs"].append(place_np)
            record["grasp_poses"].append(grasp_pose)
            record["placement_poses"].append(placement_pose.cpu().numpy())
            record["target_labels"].append(label)
            record["used_placement_indices"].append(place_idx)

        success_overall = prompt_success()

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — prompting for run outcome.")
        success_overall = prompt_success()
    except Exception:
        print("\nException during run:")
        traceback.print_exc()
        success_overall = prompt_success()
    finally:
        if record["pcds"]:
            save_run(env_dir, success_overall, record)
        else:
            print("Nothing executed; not saving.")
        perception.close()


if __name__ == "__main__":
    main()
