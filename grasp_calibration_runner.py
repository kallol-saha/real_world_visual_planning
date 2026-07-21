"""
Grasp-calibration helper.

For every graspable object whose name matches `CLASSES_TO_COLLECT`, run the
grasp policy + CALIB_XY_OFFSET, plan pre-grasp / grasp / lift, drive the
robot to the PRE-GRASP pose so the user can visually inspect alignment,
prompt y/n/s:

    y  -> save the calibrated grasp pose under that object's name, home,
          continue to next object.
    n  -> re-infer and try again (loop on same object).
    s  -> skip this object (do not save), home, continue.

At end of run (or KeyboardInterrupt) save the accumulated dict to
`data/grasp_calibration/grasps_<ts>.json` as {name: [x, y, z, qw, qx, qy, qz]}.

The saved JSON is intended to be pasted into HARDCODED_GRASPS in
`table_bussing_runner_ordered.py` (manually, later).

Default CLASSES_TO_COLLECT excludes plates because their grasps are
already hardcoded in the ordered runner.

Prereq:
    python -m frankapanda.perception.perception_pipeline --continuous --segment \
        --seg_labels "white cup. green bowl. red plate. gray tray." \
        --num_points 1000000
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from robo_utils.conversion_utils import move_pose_along_local_z

from frankapanda.perception import PerceptionPipeline
from frankapanda.motionplanner import MotionPlanner
from frankapanda import FrankaPandaController

from policy_value_inference import PolicyValueInference

from execute_table_bussing_plan import (
    PUBLISH_PORT, DEVICE,
    N_VALUE_POINTS,
    LIFT_Z, PRE_GRASP_BACKOFF,
    GRASP_CKPT, PLACE_CKPT, VALUE_CKPT,
    fps_equal_parts,
)

from table_bussing_runner import (
    capture_until_user_ok,
    partition_scene,
    apply_calibration_offset,
)

from table_bussing_runner_ordered import (
    CALIB_XY_OFFSET,
    reclassify_inside_tray,
    infer_grasps_only,
)


# --------------------------------------------------------------------------- #
# USER EDIT POINT
# --------------------------------------------------------------------------- #
SAVE_DIR = Path("data/grasp_calibration")

# Substring match (case-insensitive). Objects whose name contains any of
# these will be calibrated. Plates intentionally omitted (already hardcoded
# in table_bussing_runner_ordered.HARDCODED_GRASPS).
CLASSES_TO_COLLECT = ["bowl", "cup"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _name_in_classes(name: str) -> bool:
    lname = name.lower()
    return any(c in lname for c in CLASSES_TO_COLLECT)


def _infer_single_calibrated_grasp(inference, k, name, seg_pcd, seg_rgb, tray_pcd):
    """Run grasp model on a single object spec, apply CALIB_XY_OFFSET, return
    the first grasp pose (7d) along with the scene_pcd used for FPS."""
    per_obj_list = infer_grasps_only(
        inference, [(k, name, seg_pcd, seg_rgb)], tray_pcd,
    )
    apply_calibration_offset(
        per_obj_list, CALIB_XY_OFFSET[0], CALIB_XY_OFFSET[1],
    )
    obj = per_obj_list[0]
    grasp = obj["grasps"][0]
    return grasp, obj["scene_pcd"]


def _plan_and_move_pregrasp(motion_planner, controller, current_joints,
                            grasp_pose, device):
    """Plan pre-grasp / grasp / lift; execute pre-grasp move only.
    Returns True on success, False if any of the 3 plans failed."""
    if isinstance(grasp_pose, torch.Tensor):
        grasp_pose = grasp_pose.to(device=device, dtype=torch.float32)
    else:
        grasp_pose = torch.tensor(grasp_pose, dtype=torch.float32, device=device)

    pre_grasp = move_pose_along_local_z(grasp_pose, -PRE_GRASP_BACKOFF)
    pre_grasp = torch.tensor(pre_grasp, dtype=torch.float32, device=device)
    lift_pose = grasp_pose.clone()
    lift_pose[2] = LIFT_Z

    print("  Plan pre_grasp")
    pre_traj, pre_ok = motion_planner.plan_to_goal_poses(
        current_joints=current_joints.unsqueeze(0),
        goal_poses=pre_grasp.unsqueeze(0),
    )
    print(f"    pre_grasp success: {bool(pre_ok[0].item())}")
    if not bool(pre_ok[0].item()):
        return False

    print("  Plan grasp (along local z)")
    grasp_traj, grasp_ok = motion_planner.plan_to_goal_poses(
        current_joints=pre_traj[0, -1].unsqueeze(0),
        goal_poses=grasp_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-5:],
        plan_config=motion_planner.along_z_axis_plan_config,
    )
    print(f"    grasp success: {bool(grasp_ok[0].item())}")
    if not bool(grasp_ok[0].item()):
        return False

    print("  Plan lift (world z)")
    lift_traj, lift_ok = motion_planner.plan_to_goal_poses(
        current_joints=grasp_traj[0, -1].unsqueeze(0),
        goal_poses=lift_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-5:],
        plan_config=motion_planner.lift_plan_config,
    )
    print(f"    lift success: {bool(lift_ok[0].item())}")
    if not bool(lift_ok[0].item()):
        return False

    pre_np = pre_traj[0].cpu().numpy()
    print("  EXEC: open gripper")
    controller.open_gripper()
    print(f"  EXEC: move along pre_grasp trajectory (T={len(pre_np)})")
    controller.move_along_trajectory(pre_np, controller.open_gripper_action)
    return True


def _prompt_yns(prompt: str) -> str:
    """Loop until user answers. Returns 'y' / 'n' / 's'."""
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("y", "yes"):
            return "y"
        if ans in ("", "n", "no"):
            return "n"
        if ans in ("s", "skip"):
            return "s"
        print("    Please answer y / n / s.")


def _to_list7(pose) -> list:
    if isinstance(pose, torch.Tensor):
        return [float(v) for v in pose.detach().cpu().numpy().tolist()]
    return [float(v) for v in np.asarray(pose, dtype=np.float32).tolist()]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def calibration_loop():
    device = torch.device(DEVICE)
    print("Building grasp model (no value model used)")
    inference = PolicyValueInference(
        grasp_ckpt=GRASP_CKPT,
        place_ckpt=PLACE_CKPT,
        value_ckpt=VALUE_CKPT,
        device=device,
        use_value=False,
    )

    print(f"\n>>> Connecting to perception pipeline on port {PUBLISH_PORT}")
    perception = PerceptionPipeline(publish_port=PUBLISH_PORT, timeout_ms=10000)
    controller = FrankaPandaController()

    saved: dict = {}
    out_path = None
    try:
        print("\n>>> Capturing live pcd (loop until user OK)")
        pcd_np, rgb_np, seg_labels, seg_label_names = capture_until_user_ok(perception)

        print("\n>>> Partitioning scene")
        object_specs, tray_pcd, tray_rgb, in_tray_objects = partition_scene(
            pcd_np, rgb_np, seg_labels, seg_label_names,
        )
        if object_specs is None:
            print("partition aborted by user; nothing to do.")
            return
        object_specs, in_tray_objects = reclassify_inside_tray(
            object_specs, in_tray_objects, tray_pcd,
        )
        if not object_specs:
            print("No graspable objects after reclassify; nothing to do.")
            return

        targets = [s for s in object_specs if _name_in_classes(s[1])]
        skipped = [s[1] for s in object_specs if not _name_in_classes(s[1])]
        print(f"\n##### CALIBRATION TARGETS ({len(targets)}): "
              f"{[s[1] for s in targets]} #####")
        if skipped:
            print(f"  ignoring (not in CLASSES_TO_COLLECT): {skipped}")
        if not targets:
            print("No target objects in scene; nothing to calibrate.")
            return

        # Whole-scene pcd for motion planner context.
        parts = [tray_pcd] + [s[2] for s in object_specs]
        print(f"\n>>> Whole-scene FPS for motion planner "
              f"({len(parts)} parts, total={N_VALUE_POINTS})")
        whole_pcd, _ = fps_equal_parts(parts, total_n=N_VALUE_POINTS)
        motion_planner = MotionPlanner(whole_pcd)

        for k, name, seg_pcd, seg_rgb in targets:
            print(f"\n========================================================")
            print(f"  >>> CALIBRATING '{name}' <<<")
            print(f"========================================================")
            while True:
                print(">>> Homing")
                controller.move_to_joints(
                    controller.home_joints, controller.open_gripper_action,
                )
                current_joints = torch.tensor(
                    controller.get_robot_joints(),
                    dtype=torch.float32, device=device,
                )

                grasp, _ = _infer_single_calibrated_grasp(
                    inference, k, name, seg_pcd, seg_rgb, tray_pcd,
                )
                print(f"  calibrated grasp7d = {_to_list7(grasp)}")

                ok = _plan_and_move_pregrasp(
                    motion_planner, controller, current_joints, grasp, device,
                )
                if not ok:
                    ans = _prompt_yns(
                        "  Plan failed. Retry inference? [y/N/s=skip]: "
                    )
                    if ans == "y":
                        continue
                    print(f"  SKIPPED '{name}' (plan failed).")
                    controller.move_to_joints(
                        controller.home_joints, controller.open_gripper_action,
                    )
                    break

                ans = _prompt_yns(
                    "  Grasp pose look good at pre-grasp? "
                    "[y=save / N=retry / s=skip]: "
                )
                if ans == "y":
                    saved[name] = _to_list7(grasp)
                    print(f"  SAVED '{name}' grasp = {saved[name]}")
                    controller.move_to_joints(
                        controller.home_joints, controller.open_gripper_action,
                    )
                    break
                if ans == "s":
                    print(f"  SKIPPED '{name}' (user).")
                    controller.move_to_joints(
                        controller.home_joints, controller.open_gripper_action,
                    )
                    break
                # 'n' -> retry
                print("  Retrying inference for this object...")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — ending calibration loop early.")
    except Exception:
        print("\nException during calibration:")
        traceback.print_exc()
    finally:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = SAVE_DIR / f"grasps_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(saved, f, indent=2)
        print(f"\n##### Calibration complete #####")
        print(f"Saved {len(saved)} grasp(s) -> {out_path}")
        for name, pose in saved.items():
            print(f"  '{name}': {pose}")
        try:
            perception.close()
        except Exception:
            pass


if __name__ == "__main__":
    calibration_loop()
