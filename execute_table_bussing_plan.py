"""
Table bussing orchestrator.

1. Capture a live (large-N) pcd + per-point segmentation (--segment, no FPS).
2. For each segment except the gray tray: build (segment ∪ tray) pcd with the
   segment as the target mask, sample N grasps + N placements with the policy
   models, force every grasp + placement to be top-down (gripper z = -world z).
3. Visualize per-object grasps + placements with the masked pcd input.
4. Build whole-scene pcd (all segments + tray, no background) and score every
   (object, grasp_i, place_i) triple via the value model in chunks.
5. Argmax over scores: best (object, grasp, placement). Execute as
   pre-grasp -> grasp (vertical) -> lift -> XY carry -> pre-place ->
   placement (vertical) -> release -> retract.

Prereq:
    python -m frankapanda.perception.perception_pipeline --continuous --segment \
        --seg_labels "cup. bowl. plate. gray tray." --num_points 1000000
"""

import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import zmq
from scipy.spatial.transform import Rotation as R

from robo_utils.visualization.plotting import plot_pcd
from robo_utils.visualization.point_cloud_structures import make_gripper_visualization
from robo_utils.conversion_utils import (
    pose_to_transformation,
    move_pose_along_local_z,
)

from frankapanda.perception import PerceptionPipeline
from frankapanda.motionplanner import MotionPlanner
from frankapanda import FrankaPandaController

from policy_value_inference import PolicyValueInference


# --------------------------------------------------------------------------- #
#                                                                             #
#   >>>  USER EDIT POINT  <<<                                                 #
#                                                                             #
# --------------------------------------------------------------------------- #
PUBLISH_PORT = 1235
DEVICE = "cuda:0"

TRAY_LABEL = "gray tray"          # seg label name that the tray must match
N_GRASPS = 1                       # per object
N_PLACEMENTS = 1                   # per object (one placement per grasp)
GRASP_CHUNK = 4                    # batched forward in PolicyValueInference
PLACE_CHUNK = 4
VALUE_CHUNK = 16

# FPS targets. Models were trained on ~4096-point scenes; keep parity.
# Grasp/place input: equal points between target segment and tray.
# Value input:       equal points between every object segment and tray.
N_GRASP_PLACE_POINTS = 4096
N_VALUE_POINTS = 4096

LIFT_Z = 0.5                      # world-z height for lift + horizontal carry
PRE_GRASP_BACKOFF = 0.12           # meters along local gripper -z

# Measured tray midpoint (x, y) in robot-base frame. Used to re-center the
# placement cluster so its mean xy lands on the tray center (sim2real fix).
TRAY_CENTER_XY = (0.5654, 0.1745)

# Drop any object segment whose xy centroid falls inside the tray xy bbox
# shrunk by this clearance on every side. Such objects are presumed already
# bussed and should not be grasped or placed again.
INSIDE_TRAY_CLEARANCE = 0.03   # meters

# Per-object-class fixed end-effector z (world frame, meters). Used for BOTH
# the grasp pose z and the placement pose z. Predicted z is overwritten with
# this value before any execution. Label key must match the seg label name
# exactly. Missing keys -> predicted z is kept and a warning is printed.
Z_BY_LABEL = {
    "cup":   0.25,
    "bowl":  0.22,
    "plate": 0.19,
}

# Placement z is always this much higher than the grasp z (world frame).
PLACE_Z_ABOVE_GRASP = 0.03


def _z_for_label(name: str):
    """Substring match against Z_BY_LABEL keys (case-insensitive).
    Returns (matched_key, z) or (None, None). Handles seg labels like
    'white cup', 'blue bowl', etc."""
    lname = name.lower()
    for key, z in Z_BY_LABEL.items():
        if key.lower() in lname:
            return key, z
    return None, None

VISPLAN_WM = "/home/ksaha/Research/ModelBasedPlanning/visplanWM"
GRASP_CKPT = os.path.join(
    VISPLAN_WM,
    "models/flowmatch_actor/train_logs/Value_Function_Planning/sim2real_grasp_v3/best.pth",
)
PLACE_CKPT = os.path.join(
    VISPLAN_WM,
    "models/flowmatch_actor/train_logs/Value_Function_Planning/sim2real_place_v3/best.pth",
)
VALUE_CKPT = os.path.join(
    VISPLAN_WM,
    "models/flowmatch_actor/train_logs/Value_Function_Planning/sim2real_q_value_v3/best.pth",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def fps_equal_parts(parts, total_n):
    """
    FPS-downsample each input pcd part to a roughly equal share of total_n.
    If a part has fewer points than its allocation it is kept as-is (no
    upsampling). Returns:
      combined_pcd  : (M, 3) float32
      part_labels   : (M,) int64 — index into `parts` for every output point.
    """
    K = len(parts)
    if K == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    base = total_n // K
    rem = total_n - base * K
    per_part = [base + (1 if i < rem else 0) for i in range(K)]
    out_pcds = []
    out_labels = []
    for i, (pts, n_target) in enumerate(zip(parts, per_part)):
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) == 0:
            print(f"    fps part {i}: empty input; skipping.")
            continue
        if len(pts) <= n_target:
            sub = pts
            print(f"    fps part {i}: {len(pts)} -> {len(pts)} (input <= target, no FPS).")
        else:
            o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            sub = np.asarray(o3d_pcd.farthest_point_down_sample(n_target).points,
                             dtype=np.float32)
            print(f"    fps part {i}: {len(pts)} -> {len(sub)}.")
        out_pcds.append(sub)
        out_labels.append(np.full(len(sub), i, dtype=np.int64))
    if not out_pcds:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(out_pcds, axis=0), np.concatenate(out_labels, axis=0)


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


def enforce_z_down(pose7_wxyz):
    """
    Project pose onto top-down manifold: gripper local z-axis aligned with
    -world_z. Yaw kept from input gripper x-axis projection onto world xy.
    No random tilt (unlike sample_grasps.make_topdown). Returns same type as input.
    """
    is_tensor = isinstance(pose7_wxyz, torch.Tensor)
    p = pose7_wxyz.detach().cpu().numpy() if is_tensor else np.asarray(pose7_wxyz)
    pos = p[:3]
    qw, qx, qy, qz = p[3:7]
    mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
    x_world = mat[:, 0]
    yaw = float(np.arctan2(x_world[1], x_world[0]))
    R_target = R.from_euler("z", yaw) * R.from_euler("x", np.pi)
    qx_o, qy_o, qz_o, qw_o = R_target.as_quat()
    out = np.concatenate([pos, [qw_o, qx_o, qy_o, qz_o]]).astype(np.float32)
    return torch.from_numpy(out) if is_tensor else out


_POSE_COLORS = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
    (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (1.0, 0.75, 0.8),
]


def visualize_poses_on_pcd(pcd, poses, rgb=None, colors=None, length=0.05):
    combined = pcd.copy()
    combined_rgb = rgb.copy() if rgb is not None else None
    for i, pose in enumerate(poses):
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        transform = pose_to_transformation(pose, format="wxyz")
        c = (colors or _POSE_COLORS)[i % len(colors or _POSE_COLORS)]
        pts, cols = make_gripper_visualization(
            rotation=transform[:3, :3],
            translation=transform[:3, 3],
            length=length,
            density=50,
            color=c,
        )
        combined = np.vstack([combined, pts])
        if combined_rgb is not None:
            combined_rgb = np.vstack([combined_rgb, cols])
    if combined_rgb is not None:
        plot_pcd(combined, combined_rgb, base_frame=True)
    else:
        plot_pcd(combined, base_frame=True)


def _to_t(p):
    if isinstance(p, torch.Tensor):
        return p.detach().cpu().float()
    return torch.from_numpy(np.asarray(p, dtype=np.float32)).float()


# --------------------------------------------------------------------------- #
# Execution primitives (vertical pick / horizontal carry / vertical place)
# --------------------------------------------------------------------------- #

def pick_vertical(motion_planner: MotionPlanner, controller: FrankaPandaController,
                  current_joints: torch.Tensor, grasp_pose: torch.Tensor, device):
    """
    Pre-grasp (back off local -z) -> grasp (vertical down along gripper z) ->
    close gripper -> lift (world z to LIFT_Z).
    Returns (lift_end_joints, pcd_arr_dict_for_record, ok_bool).
    """
    grasp_pose = grasp_pose.to(device=device, dtype=torch.float32)
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
        return None, None, False

    print("  Plan grasp (along local z)")
    grasp_traj, grasp_ok = motion_planner.plan_to_goal_poses(
        current_joints=pre_traj[0, -1].unsqueeze(0),
        goal_poses=grasp_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-5:],
        plan_config=motion_planner.along_z_axis_plan_config,
    )
    print(f"    grasp success: {bool(grasp_ok[0].item())}")
    if not bool(grasp_ok[0].item()):
        return None, None, False

    print("  Plan lift (world z)")
    lift_traj, lift_ok = motion_planner.plan_to_goal_poses(
        current_joints=grasp_traj[0, -1].unsqueeze(0),
        goal_poses=lift_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-5:],
        plan_config=motion_planner.lift_plan_config,
    )
    print(f"    lift success: {bool(lift_ok[0].item())}")
    if not bool(lift_ok[0].item()):
        return None, None, False

    pre_np = pre_traj[0].cpu().numpy()
    grasp_np = grasp_traj[0].cpu().numpy()
    lift_np = lift_traj[0].cpu().numpy()
    print(f"  All 3 plans succeeded. pre_grasp T={len(pre_np)}, "
          f"grasp T={len(grasp_np)}, lift T={len(lift_np)}")

    print("  EXEC: open gripper")
    controller.open_gripper()
    print(f"  EXEC: move along pre_grasp trajectory (T={len(pre_np)})")
    controller.move_along_trajectory(pre_np, controller.open_gripper_action)
    print(f"  EXEC: move along grasp trajectory (T={len(grasp_np)})")
    controller.move_along_trajectory(grasp_np, controller.open_gripper_action)
    print("  EXEC: close gripper")
    controller.close_gripper()
    print(f"  EXEC: move along lift trajectory (T={len(lift_np)})")
    controller.move_along_trajectory(lift_np, controller.close_gripper_action)
    print("  Pick complete.")

    return lift_traj[0, -1], {"pre": pre_np, "grasp": grasp_np, "lift": lift_np}, True


def carry_horizontal(motion_planner: MotionPlanner, controller: FrankaPandaController,
                     current_joints: torch.Tensor, pre_place_pose: torch.Tensor, device):
    """
    Horizontal XY carry to pre_place_pose at LIFT_Z. Constrained to hold rotation,
    free xy (z held to keep object up). Uses only_xy_translation_plan_config.
    Returns (end_joints, traj_np, ok_bool).
    """
    pre_place_pose = pre_place_pose.to(device=device, dtype=torch.float32)
    pre_place_pose = pre_place_pose.clone()
    pre_place_pose[2] = LIFT_Z

    print("  Plan horizontal carry to pre_place (only xy)")
    carry_traj, carry_ok = motion_planner.plan_to_goal_poses(
        current_joints=current_joints.unsqueeze(0),
        goal_poses=pre_place_pose.unsqueeze(0),
        plan_config=motion_planner.only_xy_translation_plan_config,
    )
    print(f"    carry success: {bool(carry_ok[0].item())}")
    if not bool(carry_ok[0].item()):
        return None, None, False
    carry_np = carry_traj[0].cpu().numpy()
    print(f"  EXEC: move along carry trajectory (T={len(carry_np)})")
    controller.move_along_trajectory(carry_np, controller.close_gripper_action)
    print("  Carry complete.")
    return carry_traj[0, -1], carry_np, True


def place_vertical(motion_planner: MotionPlanner, controller: FrankaPandaController,
                   current_joints: torch.Tensor, place_pose: torch.Tensor, device):
    """
    Vertical descent (world z) from pre_place to place. Release + retract back
    along reverse(place) -> reverse(carry would be caller's job).
    Returns (end_joints, traj_np, ok_bool).
    """
    place_pose = place_pose.to(device=device, dtype=torch.float32)

    print("  Plan vertical place (world z)")
    place_traj, place_ok = motion_planner.plan_to_goal_poses(
        current_joints=current_joints.unsqueeze(0),
        goal_poses=place_pose.unsqueeze(0),
        disable_collision_links=motion_planner.links[-5:],
        plan_config=motion_planner.lift_plan_config,
    )
    print(f"    place success: {bool(place_ok[0].item())}")
    if not bool(place_ok[0].item()):
        return None, None, False

    place_np = place_traj[0].cpu().numpy()
    print(f"  EXEC: move along place trajectory down (T={len(place_np)})")
    controller.move_along_trajectory(place_np, controller.close_gripper_action)
    print("  EXEC: open gripper (release)")
    controller.open_gripper()
    # Vertical retract: reverse place to lift it back up.
    print(f"  EXEC: move along reversed place trajectory up (T={len(place_np)})")
    controller.move_along_trajectory(place_np[::-1], controller.open_gripper_action)
    print("  Place complete.")
    return place_traj[0, -1], place_np, True


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

    print(f"\n>>> Capturing live pcd from perception (port {PUBLISH_PORT})")
    perception = PerceptionPipeline(publish_port=PUBLISH_PORT, timeout_ms=10000)
    _drain(perception)
    pdata = perception.get_point_cloud_dict()
    perception.close()
    print("  Capture complete.")

    pcd_np = np.asarray(pdata["pcd"], dtype=np.float32)
    rgb_np = np.asarray(pdata["rgb"], dtype=np.float32)
    seg_labels = np.asarray(pdata["seg_labels"], dtype=np.int64)
    seg_label_names = list(pdata["seg_label_names"])
    print(f"  pcd {pcd_np.shape}  seg labels {seg_label_names}")

    if TRAY_LABEL not in seg_label_names:
        raise RuntimeError(f"TRAY_LABEL '{TRAY_LABEL}' not in {seg_label_names}.")
    tray_idx = seg_label_names.index(TRAY_LABEL)
    tray_mask = (seg_labels == tray_idx)
    print(f"  tray '{TRAY_LABEL}' idx={tray_idx} pts={int(tray_mask.sum())}")

    # Object segments = every non-tray, non-background label that has pts.
    object_specs = []  # list of (label_idx, label_name, segment_pcd, segment_rgb)
    for k, name in enumerate(seg_label_names):
        if k == tray_idx:
            continue
        mask = (seg_labels == k)
        n = int(mask.sum())
        if n == 0:
            print(f"  skip empty segment '{name}'")
            continue
        object_specs.append((k, name, pcd_np[mask], rgb_np[mask]))
        print(f"  object '{name}' idx={k} pts={n}")
    if not object_specs:
        raise RuntimeError("No non-tray object segments found.")

    tray_pcd = pcd_np[tray_mask]
    tray_rgb = rgb_np[tray_mask]

    # Objects already inside the tray xy bbox (shrunk by INSIDE_TRAY_CLEARANCE)
    # are presumed bussed. Do NOT consider them as graspable; instead MERGE
    # their points into the tray pcd so downstream stages (grasp/place input,
    # value scoring) treat them as part of the tray scene.
    if len(tray_pcd) > 0 and INSIDE_TRAY_CLEARANCE >= 0:
        tx_min = float(tray_pcd[:, 0].min()) + INSIDE_TRAY_CLEARANCE
        tx_max = float(tray_pcd[:, 0].max()) - INSIDE_TRAY_CLEARANCE
        ty_min = float(tray_pcd[:, 1].min()) + INSIDE_TRAY_CLEARANCE
        ty_max = float(tray_pcd[:, 1].max()) - INSIDE_TRAY_CLEARANCE
        print(f"\n>>> Merging in-tray objects into tray pcd "
              f"(clearance={INSIDE_TRAY_CLEARANCE} m)")
        print(f"  tray bbox shrunk: x=[{tx_min:.4f}, {tx_max:.4f}]  "
              f"y=[{ty_min:.4f}, {ty_max:.4f}]")
        kept = []
        merged_pcds = [tray_pcd]
        merged_rgbs = [tray_rgb]
        in_tray_names = []
        graspable_names = []
        for k, name, seg_pcd, seg_rgb in object_specs:
            cx = float(seg_pcd[:, 0].mean())
            cy = float(seg_pcd[:, 1].mean())
            inside = (tx_min <= cx <= tx_max) and (ty_min <= cy <= ty_max)
            if inside:
                print(f"  MERGE -> tray '{name}' centroid=({cx:.4f}, {cy:.4f}) "
                      f"({len(seg_pcd)} pts).")
                merged_pcds.append(seg_pcd)
                merged_rgbs.append(seg_rgb)
                in_tray_names.append(name)
            else:
                print(f"  KEEP graspable '{name}' centroid=({cx:.4f}, {cy:.4f}) "
                      f"({len(seg_pcd)} pts).")
                kept.append((k, name, seg_pcd, seg_rgb))
                graspable_names.append(name)
        tray_pcd = np.concatenate(merged_pcds, axis=0).astype(np.float32)
        tray_rgb = np.concatenate(merged_rgbs, axis=0).astype(np.float32) \
            if all(r is not None for r in merged_rgbs) else tray_rgb
        print(f"  augmented tray pcd: {tray_pcd.shape}")
        object_specs = kept
        if not object_specs:
            raise RuntimeError("All object segments are inside the tray. Nothing to bus.")

        # Summary + user confirmation before running policy inference.
        print(f"\n##### Tray contents check #####")
        print(f"  Objects ALREADY IN TRAY (skipped): {in_tray_names if in_tray_names else '(none)'}")
        print(f"  Objects TO BUS (graspable)       : {graspable_names if graspable_names else '(none)'}")
        while True:
            ans = input("  Continue with policy inference + execution? [y/N]: ").strip().lower()
            if ans in ("y", "yes"):
                break
            if ans in ("", "n", "no"):
                print("  Aborted by user.")
                return
            print("    Please answer y or n.")

    per_object = []  # list of dict per object with grasps, placements, scene_pcd, mask
    for k, name, seg_pcd, seg_rgb in object_specs:
        print(f"\n=== Sampling grasps + placements for object '{name}' ({len(seg_pcd)} raw pts) ===")
        print(f"  FPS equal between target segment + tray, target total={N_GRASP_PLACE_POINTS}")
        scene_pcd, part_labels = fps_equal_parts(
            [seg_pcd, tray_pcd], total_n=N_GRASP_PLACE_POINTS,
        )
        mask = (part_labels == 0).astype(np.float32)
        print(f"  scene_pcd {scene_pcd.shape}  mask_pos={int(mask.sum())}  mask_neg={int((1-mask).sum())}")
        print(f"  >>> Running GRASP policy (N={N_GRASPS}, chunk_size={GRASP_CHUNK})")
        grasps = inference.infer_grasps(scene_pcd, mask, num_grasps=N_GRASPS)
        grasps = [enforce_z_down(g) for g in grasps]
        # Override grasp z to per-class fixed value (substring match).
        matched_key, fixed_z = _z_for_label(name)
        if matched_key is not None:
            fixed_z = float(fixed_z)
            for g in grasps:
                g[2] = fixed_z
            print(f"  grasp z overridden to Z_BY_LABEL['{matched_key}']={fixed_z} "
                  f"(matched seg name '{name}')")
        else:
            print(f"  WARNING: no Z_BY_LABEL key matches seg name '{name}'; predicted grasp z kept.")
        print(f"  sampled {len(grasps)} grasps (z-down enforced):")
        for gi, g in enumerate(grasps):
            print(f"    grasp[{gi}] = {g.tolist()}")

        # One placement per grasp, conditioned on that grasp.
        print(f"  >>> Running PLACE policy 1-per-grasp (total={N_GRASPS}, chunk_size={PLACE_CHUNK})")
        placements = []
        for gi, g in enumerate(grasps):
            ps = inference.infer_placements(
                pcd_np=scene_pcd,
                mask_np=mask,
                grasp_pose=g,
                num_placements=1,
                chunk_size=1,
            )
            placements.append(enforce_z_down(ps[0]))
        # Override placement z to grasp z + PLACE_Z_ABOVE_GRASP.
        matched_key, fixed_z = _z_for_label(name)
        if matched_key is not None:
            place_z = float(fixed_z) + float(PLACE_Z_ABOVE_GRASP)
            for p in placements:
                p[2] = place_z
            print(f"  place z overridden to Z_BY_LABEL['{matched_key}']+{PLACE_Z_ABOVE_GRASP}"
                  f"={place_z} (matched seg name '{name}')")
        else:
            print(f"  WARNING: no Z_BY_LABEL key matches seg name '{name}'; predicted place z kept.")
        print(f"  sampled {len(placements)} placements (z-down enforced):")
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

    # 3b. Re-center placements: shift every placement by (tray_center_xy
    #     - mean(all_placements_xy)) so the placement cluster lands centered
    #     on the tray (sim2real correction for the placement policy).
    all_place_xy = []
    for obj in per_object:
        for p in obj["placements"]:
            arr = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
            all_place_xy.append(arr[:2])
    if all_place_xy:
        place_mean_xy = np.mean(np.stack(all_place_xy, axis=0), axis=0)
        dx = float(TRAY_CENTER_XY[0] - place_mean_xy[0])
        dy = float(TRAY_CENTER_XY[1] - place_mean_xy[1])
        print(f"\n>>> Re-centering placements onto tray center")
        print(f"  placements mean xy = ({place_mean_xy[0]:.4f}, {place_mean_xy[1]:.4f})")
        print(f"  tray center xy     = ({TRAY_CENTER_XY[0]:.4f}, {TRAY_CENTER_XY[1]:.4f})")
        print(f"  shift              = (dx={dx:+.4f}, dy={dy:+.4f})")
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
        # Verify post-shift mean.
        new_mean = np.mean(np.stack(
            [(p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p))[:2]
             for obj in per_object for p in obj["placements"]], axis=0,
        ), axis=0)
        print(f"  post-shift placements mean xy = ({new_mean[0]:.4f}, {new_mean[1]:.4f})")
        print(f"  per-object placements after shift:")
        for obj in per_object:
            for pi, p in enumerate(obj["placements"]):
                arr = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
                print(f"    {obj['name']:>20s} place[{pi}] = {arr.tolist()}")

    # 4. Build whole-scene pcd for value scoring. FPS equal between tray and
    # every object segment (no background) so the value model sees a balanced
    # mix and not a tray-dominated pcd.
    parts_for_value = [tray_pcd] + [s[2] for s in object_specs]            # tray first
    print(f"\n>>> Whole-scene FPS for value scoring "
          f"({len(parts_for_value)} parts: tray + {len(object_specs)} objects), "
          f"total target={N_VALUE_POINTS}")
    whole_pcd, _ = fps_equal_parts(parts_for_value, total_n=N_VALUE_POINTS)
    print(f"  whole_pcd {whole_pcd.shape}")

    # 5. Score every (object, i) triple. Flatten across objects so we batch
    #    through value model once.
    flat = []  # list of (obj_index, pair_index)
    flat_grasps = []
    flat_places = []
    for obj_i, obj in enumerate(per_object):
        for pair_i, (g, p) in enumerate(zip(obj["grasps"], obj["placements"])):
            flat.append((obj_i, pair_i))
            flat_grasps.append(g)
            flat_places.append(p)
    print(f"\n>>> Running VALUE model on {len(flat)} (object, grasp, place) triples "
          f"in chunks of {VALUE_CHUNK}")
    scores = inference.score_value_batched(
        pcd_np=whole_pcd,
        grasp_poses=flat_grasps,
        place_poses=flat_places,
        chunk_size=VALUE_CHUNK,
    )

    print("\nPer-triple value scores (unsorted):")
    for (obj_i, pair_i), s in zip(flat, scores.tolist()):
        print(f"  obj={per_object[obj_i]['name']:>20s}  pair={pair_i}  score={s:.4f}")

    sorted_idx = torch.argsort(scores, descending=True).tolist()
    print("\nPer-triple value scores (sorted high -> low):")
    for rank, idx in enumerate(sorted_idx):
        obj_i, pair_i = flat[idx]
        print(f"  rank {rank+1:>2d}: obj={per_object[obj_i]['name']:>20s}  "
              f"pair={pair_i}  score={float(scores[idx]):.4f}")

    best = sorted_idx[0]
    best_obj_i, best_pair_i = flat[best]
    best_obj = per_object[best_obj_i]
    best_grasp = best_obj["grasps"][best_pair_i]
    best_place = best_obj["placements"][best_pair_i]
    print(f"\n##### ARGMAX SELECTED for execution #####")
    print(f"  object      : '{best_obj['name']}'  (label_idx={best_obj['label_idx']})")
    print(f"  pair_idx    : {best_pair_i}")
    print(f"  value_score : {float(scores[best]):.4f}")
    print(f"  grasp_pose  : {best_grasp.tolist()}")
    print(f"  place_pose  : {best_place.tolist()}")

    # 6. Execute. Build MotionPlanner with whole-scene pcd for collision.
    print("\n>>> Setting up controller + motion planner")
    controller = FrankaPandaController()
    motion_planner = MotionPlanner(whole_pcd)

    print("  EXEC: home pose")
    controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
    current_joints = controller.get_robot_joints()
    current_joints = torch.tensor(current_joints, dtype=torch.float32, device=device)
    print(f"  current_joints (home) = {current_joints.tolist()}")

    # 6a. Pre-place pose = same xy as place, same orientation as place
    #     (both z-down), z = LIFT_Z.
    pre_place_pose = best_place.clone() if isinstance(best_place, torch.Tensor) \
        else torch.from_numpy(np.asarray(best_place, dtype=np.float32))
    pre_place_pose = pre_place_pose.clone()
    pre_place_pose[2] = LIFT_Z
    print(f"  pre_place_pose (at LIFT_Z={LIFT_Z}) = {pre_place_pose.tolist()}")

    # Pre-execution visualization: only the selected grasp + place pose on
    # the whole-scene pcd. Close window to start execution.
    nonbg = (seg_labels != -1)
    viz_pcd = pcd_np[nonbg]
    viz_rgb = rgb_np[nonbg] if rgb_np is not None else None
    _bg_arr = best_grasp.detach().cpu().numpy() if isinstance(best_grasp, torch.Tensor) else np.asarray(best_grasp)
    _bp_arr = best_place.detach().cpu().numpy() if isinstance(best_place, torch.Tensor) else np.asarray(best_place)
    _matched_key, expected_z = _z_for_label(best_obj["name"])
    print(f"  DEBUG viz pose check:")
    print(f"    best_grasp xyz = ({_bg_arr[0]:.4f}, {_bg_arr[1]:.4f}, {_bg_arr[2]:.4f})")
    print(f"    best_place xyz = ({_bp_arr[0]:.4f}, {_bp_arr[1]:.4f}, {_bp_arr[2]:.4f})")
    print(f"    Z_BY_LABEL match for '{best_obj['name']}' -> "
          f"key='{_matched_key}'  z={expected_z}")
    if expected_z is not None:
        if abs(_bg_arr[2] - expected_z) > 1e-5:
            print(f"    !! grasp z mismatch: {_bg_arr[2]} vs expected {expected_z}")
        if abs(_bp_arr[2] - expected_z) > 1e-5:
            print(f"    !! PLACE z mismatch: {_bp_arr[2]} vs expected {expected_z}")
    print(f"\n>>> Visualizing selected grasp (green) + place (red) for execution. "
          f"Close window to start.")
    visualize_poses_on_pcd(
        viz_pcd,
        [best_grasp, best_place],
        rgb=viz_rgb,
        colors=[(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
    )

    # 6b. Pick vertical.
    print("\n--- Pick vertical ---")
    lift_end_j, pick_record, pick_ok = pick_vertical(
        motion_planner, controller, current_joints, best_grasp, device,
    )
    if not pick_ok:
        print("ABORT: pick planning failed.")
        controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
        return

    # 6c. Carry horizontal.
    print("\n--- Carry horizontal ---")
    carry_end_j, carry_np, carry_ok = carry_horizontal(
        motion_planner, controller, lift_end_j, pre_place_pose, device,
    )
    if not carry_ok:
        print("ABORT: carry planning failed; releasing in place.")
        controller.open_gripper()
        controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
        return

    # 6d. Place vertical (down).
    print("\n--- Place vertical ---")
    place_end_j, place_np, place_ok = place_vertical(
        motion_planner, controller, carry_end_j,
        _to_t(best_place).to(device), device,
    )
    if not place_ok:
        print("ABORT: place planning failed; releasing in place.")
        controller.open_gripper()
        controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
        return

    # 6e. Retrace horizontal carry back to lift-end (above pre-grasp).
    print("\n--- Reverse carry (lift -> pre-place) ---")
    print(f"  EXEC: move along reversed carry trajectory (T={len(carry_np)})")
    controller.move_along_trajectory(carry_np[::-1], controller.open_gripper_action)
    print("  Reverse carry complete.")

    # 6f. Home.
    print("\n--- Home ---")
    controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
    print("Done.")


if __name__ == "__main__":
    main()
