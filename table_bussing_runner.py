"""
Clean table-bussing runner. Wraps execute_table_bussing_plan helpers in a
loop:

  capture pcd -> user confirms pcd is right (re-capture loop)
  -> partition scene (tray + objects, merge in-tray, user confirms)
  -> sample grasps + placements per object, re-center, value-score
  -> argmax -> viz selected pair -> execute pick / carry / place / retract
  -> record trajectories
  -> ask user: task done?
        yes -> save run + exit
        no  -> next iteration

Saves to data/table_bussing/env_<ENV_ID>/run_<ts>.npz with one record per
executed step (object name, poses, full joint trajectories). The format is
replayable by a separate replay script: each step has pre_grasp_traj,
grasp_traj, lift_traj, carry_traj, place_traj (down); carry and place are
mirrored on retract.

Prereq:
    python -m frankapanda.perception.perception_pipeline --continuous --segment \
        --seg_labels "white cup. green bowl. red plate. gray tray." \
        --num_points 1000000
"""

import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from robo_utils.visualization.plotting import plot_pcd

from frankapanda.perception import PerceptionPipeline
from frankapanda.motionplanner import MotionPlanner
from frankapanda import FrankaPandaController

from policy_value_inference import PolicyValueInference

# Reuse all helpers / constants from the orchestrator.
from execute_table_bussing_plan import (
    PUBLISH_PORT, DEVICE, TRAY_LABEL,
    N_GRASPS, GRASP_CHUNK, PLACE_CHUNK, VALUE_CHUNK,
    N_GRASP_PLACE_POINTS, N_VALUE_POINTS,
    LIFT_Z, TRAY_CENTER_XY, INSIDE_TRAY_CLEARANCE,
    Z_BY_LABEL, PLACE_Z_ABOVE_GRASP,
    GRASP_CKPT, PLACE_CKPT, VALUE_CKPT,
    fps_equal_parts, _drain, enforce_z_down, _z_for_label,
    visualize_poses_on_pcd, _to_t,
    pick_vertical, carry_horizontal, place_vertical,
)


# --------------------------------------------------------------------------- #
# USER EDIT POINT
# --------------------------------------------------------------------------- #
SAVE_ROOT = Path("data/table_bussing")
ENV_ID = 0


# --------------------------------------------------------------------------- #
# Per-iteration helpers (composed from execute_table_bussing_plan parts)
# --------------------------------------------------------------------------- #

def capture_until_user_ok(perception: PerceptionPipeline):
    """Drain socket, recv, plot, prompt. Loop until user accepts."""
    while True:
        _drain(perception)
        try:
            pdata = perception.get_point_cloud_dict()
        except TimeoutError:
            print("  perception timeout — retrying.")
            continue
        pcd = np.asarray(pdata["pcd"], dtype=np.float32)
        rgb = np.asarray(pdata["rgb"], dtype=np.float32) if pdata.get("rgb") is not None else None
        seg_labels = pdata.get("seg_labels")
        seg_label_names = pdata.get("seg_label_names")
        print(f"  captured pcd {pcd.shape}"
              + (f", rgb {rgb.shape}" if rgb is not None else "")
              + (f", seg labels {len(seg_label_names)}" if seg_label_names else "")
              + ". Plotting (close window to continue).")
        if rgb is not None:
            plot_pcd(pcd, rgb, base_frame=True)
        else:
            plot_pcd(pcd, base_frame=True)
        while True:
            ans = input("  Is this the right point cloud? [y/N]: ").strip().lower()
            if ans in ("y", "yes"):
                return pcd, rgb, np.asarray(seg_labels, dtype=np.int64), list(seg_label_names)
            if ans in ("", "n", "no"):
                print("  Re-capturing...")
                break
            print("    Please answer y or n.")


def partition_scene(pcd_np, rgb_np, seg_labels, seg_label_names):
    """
    Identify tray, merge in-tray objects into tray, list graspable objects.
    Prompts user to confirm tray contents. Returns (object_specs, tray_pcd,
    tray_rgb) or (None, None, None) if user aborts.
    """
    if TRAY_LABEL not in seg_label_names:
        raise RuntimeError(f"TRAY_LABEL '{TRAY_LABEL}' not in {seg_label_names}.")
    tray_idx = seg_label_names.index(TRAY_LABEL)
    tray_mask = (seg_labels == tray_idx)
    print(f"  tray '{TRAY_LABEL}' idx={tray_idx} pts={int(tray_mask.sum())}")

    object_specs = []
    for k, name in enumerate(seg_label_names):
        if k == tray_idx:
            continue
        mask = (seg_labels == k)
        n = int(mask.sum())
        if n == 0:
            print(f"  skip empty segment '{name}'")
            continue
        object_specs.append((k, name, pcd_np[mask], rgb_np[mask] if rgb_np is not None else None))
        print(f"  object '{name}' idx={k} pts={n}")
    if not object_specs:
        print("  No non-tray object segments found.")
        return None, None, None

    tray_pcd = pcd_np[tray_mask]
    tray_rgb = rgb_np[tray_mask] if rgb_np is not None else None

    # In-tray merge.
    tx_min = float(tray_pcd[:, 0].min()) + INSIDE_TRAY_CLEARANCE
    tx_max = float(tray_pcd[:, 0].max()) - INSIDE_TRAY_CLEARANCE
    ty_min = float(tray_pcd[:, 1].min()) + INSIDE_TRAY_CLEARANCE
    ty_max = float(tray_pcd[:, 1].max()) - INSIDE_TRAY_CLEARANCE
    print(f"\n>>> Merging in-tray objects into tray pcd (clearance={INSIDE_TRAY_CLEARANCE} m)")
    print(f"  tray bbox shrunk: x=[{tx_min:.4f}, {tx_max:.4f}]  y=[{ty_min:.4f}, {ty_max:.4f}]")
    kept = []
    merged_pcds = [tray_pcd]
    merged_rgbs = [tray_rgb] if tray_rgb is not None else None
    in_tray_names = []
    graspable_names = []
    for k, name, seg_pcd, seg_rgb in object_specs:
        cx = float(seg_pcd[:, 0].mean())
        cy = float(seg_pcd[:, 1].mean())
        inside = (tx_min <= cx <= tx_max) and (ty_min <= cy <= ty_max)
        if inside:
            print(f"  MERGE -> tray '{name}' centroid=({cx:.4f}, {cy:.4f}) ({len(seg_pcd)} pts).")
            merged_pcds.append(seg_pcd)
            if merged_rgbs is not None and seg_rgb is not None:
                merged_rgbs.append(seg_rgb)
            in_tray_names.append(name)
        else:
            print(f"  KEEP graspable '{name}' centroid=({cx:.4f}, {cy:.4f}) ({len(seg_pcd)} pts).")
            kept.append((k, name, seg_pcd, seg_rgb))
            graspable_names.append(name)
    tray_pcd = np.concatenate(merged_pcds, axis=0).astype(np.float32)
    if merged_rgbs is not None and len(merged_rgbs) == len(merged_pcds):
        tray_rgb = np.concatenate(merged_rgbs, axis=0).astype(np.float32)
    print(f"  augmented tray pcd: {tray_pcd.shape}")
    if not kept:
        print("  All object segments are inside the tray. Nothing to bus.")
        return None, None, None

    print("\n##### Tray contents check #####")
    print(f"  Objects ALREADY IN TRAY (skipped): {in_tray_names if in_tray_names else '(none)'}")
    print(f"  Objects TO BUS (graspable)       : {graspable_names if graspable_names else '(none)'}")
    while True:
        ans = input("  Continue with policy inference + execution? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            return kept, tray_pcd, tray_rgb
        if ans in ("", "n", "no"):
            print("  Aborted by user.")
            return None, None, None
        print("    Please answer y or n.")


def infer_per_object(inference, object_specs, tray_pcd):
    """Sample grasps + placements for every kept object; z-down + z-override."""
    per_object = []
    for k, name, seg_pcd, seg_rgb in object_specs:
        print(f"\n=== Sampling grasps + placements for object '{name}' ({len(seg_pcd)} raw pts) ===")
        scene_pcd, part_labels = fps_equal_parts(
            [seg_pcd, tray_pcd], total_n=N_GRASP_PLACE_POINTS,
        )
        mask = (part_labels == 0).astype(np.float32)
        print(f"  scene_pcd {scene_pcd.shape}  mask_pos={int(mask.sum())}  mask_neg={int((1-mask).sum())}")

        print(f"  >>> GRASP policy (N={N_GRASPS}, chunk={GRASP_CHUNK})")
        grasps = inference.infer_grasps(scene_pcd, mask, num_grasps=N_GRASPS)
        grasps = [enforce_z_down(g) for g in grasps]
        matched_key, fixed_z = _z_for_label(name)
        if matched_key is not None:
            fz = float(fixed_z)
            for g in grasps:
                g[2] = fz
            print(f"  grasp z -> Z_BY_LABEL['{matched_key}']={fz} (seg '{name}')")
        else:
            print(f"  WARNING: no Z_BY_LABEL match for '{name}'; predicted grasp z kept.")
        for gi, g in enumerate(grasps):
            print(f"    grasp[{gi}] = {g.tolist()}")

        print(f"  >>> PLACE policy 1-per-grasp (total={N_GRASPS}, chunk={PLACE_CHUNK})")
        placements = []
        for g in grasps:
            ps = inference.infer_placements(
                pcd_np=scene_pcd, mask_np=mask, grasp_pose=g,
                num_placements=1, chunk_size=1,
            )
            placements.append(enforce_z_down(ps[0]))
        if matched_key is not None:
            place_z = float(fixed_z) + float(PLACE_Z_ABOVE_GRASP)
            for p in placements:
                p[2] = place_z
            print(f"  place z -> Z_BY_LABEL['{matched_key}']+{PLACE_Z_ABOVE_GRASP}={place_z}")
        else:
            print(f"  WARNING: no Z_BY_LABEL match for '{name}'; predicted place z kept.")
        for pi, p in enumerate(placements):
            print(f"    place[{pi}] = {p.tolist()}")

        per_object.append({
            "name": name,
            "label_idx": k,
            "scene_pcd": scene_pcd,
            "mask": mask,
            "grasps": grasps,
            "placements": placements,
        })
    return per_object


def recenter_placements(per_object):
    """Shift every placement xy so cluster mean lands on TRAY_CENTER_XY."""
    all_xy = []
    for obj in per_object:
        for p in obj["placements"]:
            arr = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
            all_xy.append(arr[:2])
    if not all_xy:
        return
    mean_xy = np.mean(np.stack(all_xy, axis=0), axis=0)
    dx = float(TRAY_CENTER_XY[0] - mean_xy[0])
    dy = float(TRAY_CENTER_XY[1] - mean_xy[1])
    print(f"\n>>> Re-centering placements onto tray center  shift=(dx={dx:+.4f}, dy={dy:+.4f})")
    for obj in per_object:
        for pi, p in enumerate(obj["placements"]):
            if isinstance(p, torch.Tensor):
                p[0] = p[0] + dx
                p[1] = p[1] + dy
            else:
                arr = np.asarray(p, dtype=np.float32).copy()
                arr[0] += dx
                arr[1] += dy
                obj["placements"][pi] = arr


def score_and_select(inference, whole_pcd, per_object):
    """Value-score all (object, pair) triples; return winner dict."""
    flat = []
    flat_grasps = []
    flat_places = []
    for obj_i, obj in enumerate(per_object):
        for pair_i, (g, p) in enumerate(zip(obj["grasps"], obj["placements"])):
            flat.append((obj_i, pair_i))
            flat_grasps.append(g)
            flat_places.append(p)
    print(f"\n>>> VALUE model on {len(flat)} triples (chunk={VALUE_CHUNK})")
    scores = inference.score_value_batched(
        pcd_np=whole_pcd, grasp_poses=flat_grasps, place_poses=flat_places,
        chunk_size=VALUE_CHUNK,
    )
    sorted_idx = torch.argsort(scores, descending=True).tolist()
    print("\nPer-triple value scores (sorted high -> low):")
    for rank, idx in enumerate(sorted_idx):
        obj_i, pair_i = flat[idx]
        print(f"  rank {rank+1:>2d}: obj={per_object[obj_i]['name']:>20s}  "
              f"pair={pair_i}  score={float(scores[idx]):.4f}")
    best = sorted_idx[0]
    best_obj_i, best_pair_i = flat[best]
    best_obj = per_object[best_obj_i]
    return {
        "obj": best_obj,
        "pair_i": best_pair_i,
        "score": float(scores[best]),
        "grasp": best_obj["grasps"][best_pair_i],
        "place": best_obj["placements"][best_pair_i],
    }


def run_one_step(perception, controller, inference, device, step_i: int):
    """
    One full step: capture -> partition -> sample -> score -> exec.
    Returns a record dict on success or None on user abort / nothing to do /
    failure. The accepted pcd is saved to disk before any execution starts.
    """
    print("\n========================================================")
    print(f"                  >>> STEP {step_i} <<<")
    print("========================================================")
    print(">>> Capturing live pcd (loop until user OK)")
    pcd_np, rgb_np, seg_labels, seg_label_names = capture_until_user_ok(perception)
    save_step_pcd(pcd_np, rgb_np, seg_labels, seg_label_names, step_i)

    print("\n>>> Partitioning scene")
    object_specs, tray_pcd, tray_rgb = partition_scene(
        pcd_np, rgb_np, seg_labels, seg_label_names,
    )
    if object_specs is None:
        return None

    per_object = infer_per_object(inference, object_specs, tray_pcd)
    recenter_placements(per_object)

    # Whole-scene FPS for value model.
    parts_for_value = [tray_pcd] + [s[2] for s in object_specs]
    print(f"\n>>> Whole-scene FPS for value scoring "
          f"({len(parts_for_value)} parts), total={N_VALUE_POINTS}")
    whole_pcd, _ = fps_equal_parts(parts_for_value, total_n=N_VALUE_POINTS)
    print(f"  whole_pcd {whole_pcd.shape}")

    winner = score_and_select(inference, whole_pcd, per_object)
    print(f"\n##### ARGMAX SELECTED #####")
    print(f"  object: '{winner['obj']['name']}'  pair={winner['pair_i']}  "
          f"score={winner['score']:.4f}")
    print(f"  grasp_pose: {winner['grasp'].tolist()}")
    print(f"  place_pose: {winner['place'].tolist()}")

    # Build motion planner with whole-scene pcd; home + capture current_joints.
    motion_planner = MotionPlanner(whole_pcd)
    print("\n>>> Homing before execution")
    controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
    current_joints = controller.get_robot_joints()
    current_joints = torch.tensor(current_joints, dtype=torch.float32, device=device)

    # Pre-place pose at LIFT_Z (same orientation as place).
    pre_place_pose = winner["place"].clone() if isinstance(winner["place"], torch.Tensor) \
        else torch.from_numpy(np.asarray(winner["place"], dtype=np.float32))
    pre_place_pose = pre_place_pose.clone()
    pre_place_pose[2] = LIFT_Z

    # Single pre-execution viz.
    nonbg = (seg_labels != -1)
    viz_pcd = pcd_np[nonbg]
    viz_rgb = rgb_np[nonbg] if rgb_np is not None else None
    print(f"\n>>> Visualizing selected grasp (green) + place (red). Close to start.")
    visualize_poses_on_pcd(
        viz_pcd, [winner["grasp"], winner["place"]],
        rgb=viz_rgb, colors=[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
    )

    # Execute.
    print("\n--- Pick vertical ---")
    lift_end_j, pick_record, pick_ok = pick_vertical(
        motion_planner, controller, current_joints, winner["grasp"], device,
    )
    if not pick_ok:
        print("ABORT: pick planning failed.")
        controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
        return None

    # Confirm grasp physically succeeded before committing to carry+place.
    while True:
        ans = input("\nGrasp successful? [y/N] (n -> release + home + retry step): ").strip().lower()
        if ans in ("y", "yes"):
            grasp_ok_user = True
            break
        if ans in ("", "n", "no"):
            grasp_ok_user = False
            break
        print("  Please answer y or n.")
    if not grasp_ok_user:
        print("Grasp marked failed; releasing + homing for retry.")
        controller.open_gripper()
        controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
        return None

    print("\n--- Carry horizontal ---")
    carry_end_j, carry_np, carry_ok = carry_horizontal(
        motion_planner, controller, lift_end_j, pre_place_pose, device,
    )
    if not carry_ok:
        print("ABORT: carry planning failed; releasing in place.")
        controller.open_gripper()
        controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
        return None

    print("\n--- Place vertical ---")
    place_end_j, place_np, place_ok = place_vertical(
        motion_planner, controller, carry_end_j,
        _to_t(winner["place"]).to(device), device,
    )
    if not place_ok:
        print("ABORT: place planning failed; releasing in place.")
        controller.open_gripper()
        controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
        return None

    print("\n--- Reverse carry (lift -> pre-place) ---")
    controller.move_along_trajectory(carry_np[::-1], controller.open_gripper_action)

    print("\n--- Home ---")
    controller.move_to_joints(controller.home_joints, controller.open_gripper_action)

    # Record everything replayable for this step.
    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    record = {
        "object_name": winner["obj"]["name"],
        "value_score": winner["score"],
        "grasp_pose": _np(winner["grasp"]),
        "place_pose": _np(winner["place"]),
        "pre_place_pose": _np(pre_place_pose),
        "pcd_at_step": pcd_np.astype(np.float32),
        "rgb_at_step": rgb_np.astype(np.float32) if rgb_np is not None else np.zeros((0, 3), np.float32),
        "seg_labels_at_step": seg_labels.astype(np.int64),
        "seg_label_names_at_step": np.array(seg_label_names),
        "pre_grasp_traj": pick_record["pre"].astype(np.float32),
        "grasp_traj":     pick_record["grasp"].astype(np.float32),
        "lift_traj":      pick_record["lift"].astype(np.float32),
        "carry_traj":     carry_np.astype(np.float32),
        "place_traj":     place_np.astype(np.float32),
    }
    return record


def prompt_task_done() -> bool:
    while True:
        ans = input("\nTask complete? [y/N] (y -> end loop, n -> next step): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("", "n", "no"):
            return False
        print("  Please answer y or n.")


def prompt_success() -> bool:
    """Mirrors execute_plan.prompt_success — decides success vs failure subdir."""
    while True:
        ans = input("\nWas this run a successful plan? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("", "n", "no"):
            return False
        print("  Please answer y or n.")


def save_step_pcd(pcd, rgb, seg_labels, seg_label_names, step_i: int):
    """Incremental per-step dump right after user confirms the pcd is right.
    Persists even if the rest of the step crashes."""
    inter_dir = SAVE_ROOT / f"env_{ENV_ID}" / "intermediate"
    inter_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = inter_dir / f"step_{step_i:03d}_{ts}.npz"
    np.savez(
        out_path,
        pcd=pcd.astype(np.float32),
        rgb=rgb.astype(np.float32) if rgb is not None else np.zeros((0, 3), np.float32),
        seg_labels=seg_labels.astype(np.int64),
        seg_label_names=np.array(seg_label_names),
        step_i=np.array(step_i),
    )
    print(f"  saved intermediate pcd -> {out_path}")


def save_run(records, success: bool):
    sub = "success" if success else "failure"
    out_dir = SAVE_ROOT / f"env_{ENV_ID}" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"run_{ts}.npz"
    np.savez(
        out_path,
        n_steps=np.array(len(records)),
        object_names=np.array([r["object_name"] for r in records]),
        value_scores=np.asarray([r["value_score"] for r in records], dtype=np.float32),
        grasp_poses=np.stack([r["grasp_pose"] for r in records], axis=0),
        place_poses=np.stack([r["place_pose"] for r in records], axis=0),
        pre_place_poses=np.stack([r["pre_place_pose"] for r in records], axis=0),
        pcds=np.array([r["pcd_at_step"] for r in records], dtype=object),
        rgbs=np.array([r["rgb_at_step"] for r in records], dtype=object),
        seg_labels=np.array([r["seg_labels_at_step"] for r in records], dtype=object),
        seg_label_names=np.array([r["seg_label_names_at_step"] for r in records], dtype=object),
        pre_grasp_trajs=np.array([r["pre_grasp_traj"] for r in records], dtype=object),
        grasp_trajs=np.array([r["grasp_traj"] for r in records], dtype=object),
        lift_trajs=np.array([r["lift_traj"] for r in records], dtype=object),
        carry_trajs=np.array([r["carry_traj"] for r in records], dtype=object),
        place_trajs=np.array([r["place_traj"] for r in records], dtype=object),
    )
    print(f"\nSaved run with {len(records)} step(s) -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    device = torch.device(DEVICE)

    print("Building policy + value models")
    inference = PolicyValueInference(
        grasp_ckpt=GRASP_CKPT,
        place_ckpt=PLACE_CKPT,
        value_ckpt=VALUE_CKPT,
        device=device,
        use_value=True,
    )

    print(f"\n>>> Connecting to perception pipeline on port {PUBLISH_PORT}")
    perception = PerceptionPipeline(publish_port=PUBLISH_PORT, timeout_ms=10000)
    controller = FrankaPandaController()

    records = []
    step_i = 0
    try:
        while True:
            step_i += 1
            rec = run_one_step(perception, controller, inference, device, step_i)
            if rec is not None:
                records.append(rec)
                print(f"\n>>> Step {step_i} complete. Total recorded: {len(records)}.")
                if prompt_task_done():
                    break
            else:
                # Step produced no record: grasp failure / plan failure / user
                # abort. Auto-retry without task_done prompt.
                print(f"\n>>> Step {step_i} produced no record. Auto-retrying.")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — ending loop early.")
    except Exception:
        print("\nException during run:")
        traceback.print_exc()
    finally:
        # Always prompt success/failure (mirrors execute_plan). Save to
        # env_<ENV_ID>/<success|failure>/run_<ts>.npz if any records exist.
        if records:
            success = prompt_success()
            save_run(records, success)
        else:
            print("No records this run; nothing to save (intermediate pcds may still be on disk).")
        perception.close()


if __name__ == "__main__":
    main()
