"""
Replay a saved table_bussing_runner run. No perception inference, no policy,
no value, no motion planner — just plays back the saved joint trajectories
on the robot.

Flow:
  1. Locate the saved npz (auto-picks the most recent run under
     data/table_bussing/env_<ENV_ID>/<STATUS>/ unless RUN_FILE is set).
  2. For every saved step k = 0..n-1: pop an Open3D window showing the
     step's saved pcd + green gripper widget at grasp_pose + red gripper
     widget at place_pose. Object name + step index printed in console.
  3. Prompt user for execution order, e.g. "2,0,1,3".
  4. Capture a live pcd + user-OK loop (sanity check that the live scene
     matches the saved scene).
  5. Replay each step in the requested order using the saved trajectories.

Replay sequence per step (matches the original runner execution):
    move_to_joints(home, open)
    move_along_trajectory(pre_grasp_traj, open)
    move_along_trajectory(grasp_traj,     open)
    close_gripper()
    move_along_trajectory(lift_traj,      close)
    move_along_trajectory(carry_traj,     close)
    move_along_trajectory(place_traj,     close)
    open_gripper()
    move_along_trajectory(place_traj[::-1], open)
    move_along_trajectory(carry_traj[::-1], open)
    move_to_joints(home, open)
"""

import re
import traceback
from pathlib import Path

import numpy as np
import torch
import zmq

from robo_utils.visualization.plotting import plot_pcd
from robo_utils.visualization.point_cloud_structures import make_gripper_visualization
from robo_utils.conversion_utils import pose_to_transformation

from frankapanda.perception import PerceptionPipeline
from frankapanda import FrankaPandaController

# Reuse runner's persistence helpers so replay saves are bit-identical in
# format to fresh runs. Override runner's globals at the bottom of this
# file so the save paths follow this script's ENV_ID / SAVE_ROOT.
import table_bussing_runner as _tbr


# --------------------------------------------------------------------------- #
# USER EDIT POINT
# --------------------------------------------------------------------------- #
ENV_ID = 0
STATUS = "failure"             # "success" or "failure"
RUN_FILE = None                # full Path to a specific run_*.npz, or None -> latest
SAVE_ROOT = Path("data/table_bussing")
PUBLISH_PORT = 1235
TIMEOUT_MS = 10000
SKIP_LIVE_PCD = False          # set True to skip the live capture sanity check
DRY_RUN_NO_GRASP = True       # True -> never close the gripper; replay motion only

# Per-step xy offset added to grasp_pose / place_pose after loading.
# Visualization-only correction — trajectories are NOT modified (they still
# replay to the originally planned configs). Format: {step_idx: (dx, dy)}.
# Independent dicts so the grasp can be corrected differently from the place.
GRASP_POSE_OFFSETS = {
    2: (0.3, 0.15),
}
PLACE_POSE_OFFSETS = {
    2: (0.0, 0.0),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def find_run_file() -> Path:
    if RUN_FILE is not None:
        p = Path(RUN_FILE)
        if not p.exists():
            raise FileNotFoundError(f"RUN_FILE not found: {p}")
        return p
    folder = SAVE_ROOT / f"env_{ENV_ID}" / STATUS
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    candidates = sorted(folder.glob("run_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No run_*.npz in {folder}")
    return candidates[-1]


def load_run(path: Path):
    print(f"Loading run from {path}")
    d = np.load(path, allow_pickle=True)
    n = int(d["n_steps"])
    steps = []
    for k in range(n):
        steps.append({
            "object_name":     str(d["object_names"][k]),
            "value_score":     float(d["value_scores"][k]),
            "grasp_pose":      np.asarray(d["grasp_poses"][k], dtype=np.float32),
            "place_pose":      np.asarray(d["place_poses"][k], dtype=np.float32),
            "pre_place_pose":  np.asarray(d["pre_place_poses"][k], dtype=np.float32),
            "pcd":             np.asarray(d["pcds"][k], dtype=np.float32),
            "rgb":             np.asarray(d["rgbs"][k], dtype=np.float32),
            "pre_grasp_traj":  np.asarray(d["pre_grasp_trajs"][k], dtype=np.float32),
            "grasp_traj":      np.asarray(d["grasp_trajs"][k], dtype=np.float32),
            "lift_traj":       np.asarray(d["lift_trajs"][k], dtype=np.float32),
            "carry_traj":      np.asarray(d["carry_trajs"][k], dtype=np.float32),
            "place_traj":      np.asarray(d["place_trajs"][k], dtype=np.float32),
        })
    print(f"  loaded {n} step(s).")
    # Visualization-only pose offsets (does not modify trajectories).
    for k_off, (dx, dy) in GRASP_POSE_OFFSETS.items():
        if not (0 <= k_off < n):
            print(f"  WARNING: GRASP_POSE_OFFSETS step {k_off} out of range [0, {n-1}]; skipping.")
            continue
        steps[k_off]["grasp_pose"][0] += dx
        steps[k_off]["grasp_pose"][1] += dy
        print(f"  GRASP_POSE_OFFSETS applied to step {k_off}: dx={dx:+.4f}, dy={dy:+.4f} "
              f"(viz only; trajectories unchanged).")
    for k_off, (dx, dy) in PLACE_POSE_OFFSETS.items():
        if not (0 <= k_off < n):
            print(f"  WARNING: PLACE_POSE_OFFSETS step {k_off} out of range [0, {n-1}]; skipping.")
            continue
        steps[k_off]["place_pose"][0] += dx
        steps[k_off]["place_pose"][1] += dy
        print(f"  PLACE_POSE_OFFSETS applied to step {k_off}: dx={dx:+.4f}, dy={dy:+.4f} "
              f"(viz only; trajectories unchanged).")
    for k, s in enumerate(steps):
        print(f"  step {k}: object='{s['object_name']}'  score={s['value_score']:.4f}  "
              f"grasp=({s['grasp_pose'][0]:.3f}, {s['grasp_pose'][1]:.3f}, {s['grasp_pose'][2]:.3f})  "
              f"place=({s['place_pose'][0]:.3f}, {s['place_pose'][1]:.3f}, {s['place_pose'][2]:.3f})")
    return steps


def visualize_step(step, k):
    """Plot saved pcd + grasp (green) + place (red) for step k."""
    pcd = step["pcd"]
    rgb = step["rgb"] if len(step["rgb"]) == len(pcd) else None
    combined = pcd.copy()
    combined_rgb = rgb.copy() if rgb is not None else None
    for pose, color in [(step["grasp_pose"], (0.0, 1.0, 0.0)),
                        (step["place_pose"], (1.0, 0.0, 0.0))]:
        transform = pose_to_transformation(pose, format="wxyz")
        pts, cols = make_gripper_visualization(
            rotation=transform[:3, :3],
            translation=transform[:3, 3],
            length=0.05, density=50, color=color,
        )
        combined = np.vstack([combined, pts])
        if combined_rgb is not None:
            combined_rgb = np.vstack([combined_rgb, cols])
    print(f"  >>> Visualizing step {k} ('{step['object_name']}'). "
          f"Green=grasp, Red=place. Close window to continue.")
    if combined_rgb is not None:
        plot_pcd(combined, combined_rgb, base_frame=True)
    else:
        plot_pcd(combined, base_frame=True)


def parse_order(raw: str, n_steps: int):
    """Parse '2,0,1,3' or '2 0 1 3' into a list[int]. Validates range +
    uniqueness; length may be < n_steps (partial replay)."""
    tokens = re.split(r"[,\s]+", raw.strip())
    tokens = [t for t in tokens if t]
    try:
        order = [int(t) for t in tokens]
    except ValueError:
        raise ValueError(f"Could not parse '{raw}' as a list of ints.")
    for idx in order:
        if not (0 <= idx < n_steps):
            raise ValueError(f"index {idx} out of range [0, {n_steps - 1}].")
    if len(set(order)) != len(order):
        raise ValueError(f"duplicate indices in {order}.")
    return order


def prompt_order(n_steps: int):
    while True:
        raw = input(f"\nEnter execution order as comma/space-separated step indices "
                    f"(0..{n_steps - 1}). Subset OK. Example: 2,0,1,3 : ").strip()
        try:
            order = parse_order(raw, n_steps)
            if not order:
                print("  empty order; please enter at least one index.")
                continue
            print(f"  parsed order: {order}")
            return order
        except ValueError as e:
            print(f"  {e}")


def _drain(perception: PerceptionPipeline):
    dropped = 0
    while True:
        try:
            perception.socket.recv(flags=zmq.DONTWAIT)
            dropped += 1
        except zmq.Again:
            break
    if dropped:
        print(f"  drained {dropped} stale frames.")


def capture_until_user_ok(perception: PerceptionPipeline):
    """Same pattern as runner: drain, recv, plot, prompt y/n; loop until y.
    Returns (pcd, rgb, seg_labels, seg_label_names) — last two may be None
    if perception was started without --segment."""
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
        if seg_labels is not None:
            seg_labels = np.asarray(seg_labels, dtype=np.int64)
        if seg_label_names is not None:
            seg_label_names = list(seg_label_names)
        print(f"  captured pcd {pcd.shape}"
              + (f", rgb {rgb.shape}" if rgb is not None else "")
              + (f", seg labels {len(seg_label_names)}" if seg_label_names else "")
              + ". Plotting (close to continue).")
        if rgb is not None:
            plot_pcd(pcd, rgb, base_frame=True)
        else:
            plot_pcd(pcd, base_frame=True)
        while True:
            ans = input("  Is this the right point cloud? [y/N]: ").strip().lower()
            if ans in ("y", "yes"):
                return pcd, rgb, seg_labels, seg_label_names
            if ans in ("", "n", "no"):
                print("  Re-capturing...")
                break
            print("    Please answer y or n.")


def prompt_continue(msg: str) -> bool:
    while True:
        ans = input(f"\n{msg} [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Please answer y or n.")


def replay_step(controller: FrankaPandaController, step):
    """Mirror the runner's execution sequence using saved trajectories.
    DRY_RUN_NO_GRASP=True keeps the gripper open throughout — useful as a
    motion rehearsal without actually picking the object."""
    pre = step["pre_grasp_traj"]
    grasp = step["grasp_traj"]
    lift = step["lift_traj"]
    carry = step["carry_traj"]
    place = step["place_traj"]
    print(f"  replay traj lengths: pre={len(pre)}, grasp={len(grasp)}, "
          f"lift={len(lift)}, carry={len(carry)}, place={len(place)}")
    if DRY_RUN_NO_GRASP:
        print(f"  DRY_RUN_NO_GRASP=True -> gripper stays open the whole time.")

    open_act = controller.open_gripper_action
    close_act = controller.open_gripper_action if DRY_RUN_NO_GRASP else controller.close_gripper_action

    controller.move_to_joints(controller.home_joints, open_act)

    print("  EXEC: pre_grasp (open)")
    controller.move_along_trajectory(pre, open_act)
    print("  EXEC: grasp (open)")
    controller.move_along_trajectory(grasp, open_act)
    if not DRY_RUN_NO_GRASP:
        print("  EXEC: close_gripper")
        controller.close_gripper()
    else:
        print("  EXEC: close_gripper SKIPPED (dry run)")
    print(f"  EXEC: lift ({'open' if DRY_RUN_NO_GRASP else 'close'})")
    controller.move_along_trajectory(lift, close_act)
    print(f"  EXEC: carry ({'open' if DRY_RUN_NO_GRASP else 'close'})")
    controller.move_along_trajectory(carry, close_act)
    print(f"  EXEC: place down ({'open' if DRY_RUN_NO_GRASP else 'close'})")
    controller.move_along_trajectory(place, close_act)
    if not DRY_RUN_NO_GRASP:
        print("  EXEC: open_gripper (release)")
        controller.open_gripper()
    else:
        print("  EXEC: open_gripper SKIPPED (dry run; never closed)")
    print("  EXEC: place reversed (open)")
    controller.move_along_trajectory(place[::-1], open_act)
    print("  EXEC: carry reversed (open)")
    controller.move_along_trajectory(carry[::-1], open_act)
    print("  EXEC: home (open)")
    controller.move_to_joints(controller.home_joints, open_act)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    # Make runner's save helpers write under THIS script's ENV_ID + SAVE_ROOT.
    _tbr.ENV_ID = ENV_ID
    _tbr.SAVE_ROOT = SAVE_ROOT

    run_path = find_run_file()
    steps = load_run(run_path)
    n = len(steps)
    if n == 0:
        print("Loaded run has 0 steps; nothing to do.")
        return

    # 1. Show every step's grasp + place on its saved pcd.
    print("\n>>> Visualizing every saved step (close each window to advance).")
    for k, s in enumerate(steps):
        visualize_step(s, k)

    # 2. Ask for order.
    order = prompt_order(n)

    perception = None
    controller = None
    records = []
    save_step_i = 0
    try:
        if SKIP_LIVE_PCD:
            print(">>> SKIP_LIVE_PCD=True — no per-step pcd capture; "
                  "saved replay run will contain empty pcds.")
        else:
            print(f"\n>>> Connecting to perception pipeline on port {PUBLISH_PORT}")
            perception = PerceptionPipeline(publish_port=PUBLISH_PORT, timeout_ms=TIMEOUT_MS)

        # 3. Replay in user-requested order. Per step: live pcd capture +
        #    user-OK loop -> intermediate save -> replay -> record.
        controller = FrankaPandaController()
        for slot, k in enumerate(order):
            s = steps[k]
            print(f"\n========================================================")
            print(f"  REPLAY {slot + 1}/{len(order)}  saved step k={k}  "
                  f"object='{s['object_name']}'")
            print(f"========================================================")
            if not prompt_continue("Execute this step?"):
                print("  Skipping this step.")
                continue

            # Live pcd capture (per-step, mirrors runner pattern).
            if perception is not None:
                print(">>> Capturing live pcd before replay (user-OK loop)")
                pcd_live, rgb_live, seg_labels_live, seg_label_names_live = \
                    capture_until_user_ok(perception)
                save_step_i += 1
                _tbr.save_step_pcd(
                    pcd_live, rgb_live,
                    seg_labels_live if seg_labels_live is not None
                        else np.zeros((len(pcd_live),), dtype=np.int64),
                    seg_label_names_live if seg_label_names_live is not None else [],
                    save_step_i,
                )
            else:
                pcd_live = np.zeros((0, 3), dtype=np.float32)
                rgb_live = np.zeros((0, 3), dtype=np.float32)
                seg_labels_live = np.zeros((0,), dtype=np.int64)
                seg_label_names_live = []

            replay_step(controller, s)
            print(f"  step k={k} replay complete.")

            # Record in runner format. trajectories come from the saved step.
            records.append({
                "object_name":      s["object_name"],
                "value_score":      s["value_score"],
                "grasp_pose":       s["grasp_pose"].astype(np.float32),
                "place_pose":       s["place_pose"].astype(np.float32),
                "pre_place_pose":   s["pre_place_pose"].astype(np.float32),
                "pcd_at_step":      pcd_live.astype(np.float32),
                "rgb_at_step":      rgb_live.astype(np.float32) if rgb_live is not None
                                    else np.zeros((0, 3), np.float32),
                "seg_labels_at_step":      seg_labels_live.astype(np.int64)
                                    if seg_labels_live is not None else np.zeros((0,), np.int64),
                "seg_label_names_at_step": np.array(seg_label_names_live)
                                    if seg_label_names_live is not None else np.array([]),
                "pre_grasp_traj":   s["pre_grasp_traj"].astype(np.float32),
                "grasp_traj":       s["grasp_traj"].astype(np.float32),
                "lift_traj":        s["lift_traj"].astype(np.float32),
                "carry_traj":       s["carry_traj"].astype(np.float32),
                "place_traj":       s["place_traj"].astype(np.float32),
            })
            print(f"  recorded replay step {len(records)} / {len(order)} requested.")
        print("\nAll requested replays done.")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — ending replay early.")
    except Exception:
        print("\nException during replay:")
        traceback.print_exc()
    finally:
        if records:
            success = _runner_prompt_success_or_local()
            _tbr.save_run(records, success)
        else:
            print("No replay records to save.")
        if perception is not None:
            perception.close()


def _runner_prompt_success_or_local() -> bool:
    """Thin wrapper so a future swap to a different prompt is one-liner."""
    return _tbr.prompt_success()


if __name__ == "__main__":
    main()
