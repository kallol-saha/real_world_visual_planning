"""
For each (selected) saved grasp in env_<ENV_SRC>:
  1. Copy env_<ENV_SRC>/ -> env_<ENV_DST>/ (only if env_<ENV_DST>/ does not exist).
  2. Replay the saved pre_grasp + grasp segments (open gripper, then close at
     the grasp/lift boundary).
  3. Re-plan a NEW lift trajectory from the grasp-end joint config to a higher
     z (LIFT_Z). Executes the new lift, then opens the gripper, moves home.
  4. Replace traj[64:96] with the new lift trajectory and atomically write the
     updated npz back under env_<ENV_DST>/.

Each saved trajectory is pre_grasp + grasp + lift, length T=96 (3 * 32-step
curobo segments). The new lift segment has the same length, so the saved
trajectories stay shape-compatible with execute_saved_grasps / execute_plan.

GRASP_DICT mirrors execute_plan: {object_id: [grasp_indices]} -> only those
get a new lift. None -> every grasp marked successful.
"""

import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from frankapanda import FrankaPandaController
from frankapanda.motionplanner import MotionPlanner


# --------------------------------------------------------------------------- #
#                                                                             #
#   >>>  USER EDIT POINT  <<<                                                 #
#                                                                             #
# --------------------------------------------------------------------------- #
ENV_SRC = 0
ENV_DST = 1
SAVE_DIR = Path("data/shelf_packing_scenes")

LIFT_Z = 0.45                    # new lift end-effector z (world frame)

# Optional per-object grasp restriction. None -> all grasps marked successful.
# Format: {object_id: [grasp_indices]}. Indices bypass the successes mask.
GRASP_DICT = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _object_id_from_path(p: Path):
    m = re.match(r"grasps_(\d+)\.npz$", p.name)
    return int(m.group(1)) if m else None


def _as_float_traj(t):
    if t is None:
        return None
    a = np.asarray(t, dtype=np.float32)
    return a if a.ndim == 2 else None


def copy_env_if_needed(src_dir: Path, dst_dir: Path):
    if dst_dir.exists():
        print(f"Destination {dst_dir} already exists; not copying. Operating in place.")
        return
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source env dir not found: {src_dir}")
    print(f"Copying {src_dir} -> {dst_dir}")
    shutil.copytree(src_dir, dst_dir)


def save_npz_atomic(path: Path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    src_dir = SAVE_DIR / f"env_{ENV_SRC}"
    dst_dir = SAVE_DIR / f"env_{ENV_DST}"
    copy_env_if_needed(src_dir, dst_dir)

    object_files = sorted(
        (p for p in dst_dir.glob("grasps_*.npz") if _object_id_from_path(p) is not None),
        key=lambda p: _object_id_from_path(p),
    )
    if not object_files:
        print(f"No grasps_<id>.npz files in {dst_dir}")
        return

    controller = FrankaPandaController()
    motion_planner = MotionPlanner()                                       # static obstacles only

    total_done = 0
    total_failed = 0

    for p in object_files:
        object_id = _object_id_from_path(p)
        data = np.load(p, allow_pickle=True)
        grasps = data["grasps"]
        trajectories = data["trajectories"]
        successes = data["successes"].astype(bool)
        target_label = str(data["target_label"]) if "target_label" in data.files else "?"
        saved_pcd = data["pcd"] if "pcd" in data.files else None

        # Select indices.
        if GRASP_DICT is not None:
            if object_id not in GRASP_DICT:
                print(f"\n=== object {object_id} ('{target_label}'): not in GRASP_DICT; skipping ===")
                continue
            requested = list(GRASP_DICT[object_id])
            bad = [i for i in requested if not (0 <= i < len(grasps))]
            if bad:
                raise IndexError(
                    f"object {object_id}: GRASP_DICT indices {bad} out of range "
                    f"(npz has {len(grasps)} grasps)."
                )
            indices = np.asarray(requested, dtype=np.int64)
        else:
            indices = np.where(successes)[0]

        if len(indices) == 0:
            print(f"\n=== object {object_id} ('{target_label}'): no grasps selected; skipping ===")
            continue
        print(f"\n=== object {object_id} ('{target_label}'): re-lifting {len(indices)} grasp(s) ===")

        # Materialize as a fresh trajectories list (object array we will mutate).
        new_trajectories = list(trajectories)

        for n, i in enumerate(indices):
            traj = _as_float_traj(trajectories[i])
            if traj is None or traj.shape[0] == 0:
                print(f"  [{n+1}/{len(indices)}] grasp {i}: empty trajectory; skipping.")
                continue
            T = traj.shape[0]
            if T % 3 != 0:
                print(f"  [{n+1}/{len(indices)}] grasp {i}: T={T} not divisible by 3; skipping.")
                continue
            seg = T // 3
            close_at = 2 * seg                                             # end of grasp segment

            grasp_pose = np.asarray(grasps[i], dtype=np.float32)           # (7,) wxyz
            lift_pose_np = grasp_pose.copy()
            lift_pose_np[2] = LIFT_Z
            lift_pose = torch.from_numpy(lift_pose_np).float().to("cuda:0").unsqueeze(0)  # (1, 7)

            # End-of-grasp joint config (last row of grasp segment).
            grasp_end_joints = torch.from_numpy(traj[close_at - 1]).float().to("cuda:0").unsqueeze(0)

            print(f"  [{n+1}/{len(indices)}] grasp {i}: planning new lift to z={LIFT_Z:.3f}")
            new_lift_traj, lift_success = motion_planner.plan_to_goal_poses(
                current_joints=grasp_end_joints,
                goal_poses=lift_pose,
                disable_collision_links=motion_planner.links[-5:],
                plan_config=motion_planner.lift_plan_config,
            )
            ok = bool(lift_success[0].item())
            print(f"    lift plan success: {ok}")
            if not ok:
                print(f"    skipping execution + save for grasp {i}.")
                total_failed += 1
                continue

            new_lift_np = new_lift_traj[0].cpu().detach().numpy().astype(np.float32)
            if new_lift_np.shape[0] != seg:
                print(f"    WARNING: new lift length {new_lift_np.shape[0]} != original seg {seg}; "
                      f"saved trajectory will change shape.")

            # 1. Replay pre_grasp + grasp (open then close at boundary).
            controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
            controller.move_along_trajectory(traj[:close_at], controller.open_gripper_action)
            controller.close_gripper()

            # 2. Execute new lift with gripper closed.
            controller.move_along_trajectory(new_lift_np, controller.close_gripper_action)

            # 3. Release + home.
            controller.open_gripper()
            controller.move_to_joints(controller.home_joints, controller.open_gripper_action)

            # 4. Replace lift segment in saved traj and persist.
            new_full = np.concatenate([traj[:close_at], new_lift_np], axis=0).astype(np.float32)
            new_trajectories[i] = new_full
            total_done += 1
            print(f"    updated traj for grasp {i}; T_new={new_full.shape[0]}")

        # Save back this object's npz.
        save_kwargs = dict(
            grasps=np.asarray(grasps, dtype=np.float32),
            trajectories=np.array(new_trajectories, dtype=object),
            successes=successes,
            target_label=np.array(target_label),
            object_id=np.array(object_id),
            env_id=np.array(ENV_DST),
            pcd=(np.asarray(saved_pcd, dtype=np.float32)
                 if saved_pcd is not None else np.zeros((0, 3), dtype=np.float32)),
        )
        save_npz_atomic(p, **save_kwargs)
        print(f"  wrote {p}")

    print(f"\nDone. Re-lifted {total_done} grasp(s); {total_failed} lift plan failure(s).")


if __name__ == "__main__":
    main()
