"""
Ordered table-bussing runner. Class-ordered + GEOMETRIC placement (no
placement model). Per-step pipeline:

  1. Class ordering: cup -> plate -> bowl. Each step picks first class with a
     graspable object present; value model argmax-ranks objects within class.
  2. Geometric placement (drop-in replacement for placement model):
       - cup   : tray edge slot furthest from tray center (corners first,
                 mid-edges fallback). Blocks slots near in-tray cups.
       - plate : center of tray, or on top of existing plate stack.
                 z accumulates per in-tray plate.
       - bowl  : on top of in-tray bowl stack if any, else on plate stack,
                 else tray center.
     Formula: place_xy = target_xy + (grasp_xy - object_centroid_xy) so the
     OBJECT centroid lands on target_xy. Place orientation = z-down.
  3. Mid-pick grasp confirmation: pre-grasp / grasp / lift PLANNED, robot
     moved to pre-grasp, user prompted. No -> open + home + retry. Yes ->
     execute grasp + close + lift. No post-lift prompt.

Prereq:
    python -m frankapanda.perception.perception_pipeline --continuous --segment \
        --seg_labels "white cup. green bowl. red plate. gray tray." \
        --num_points 1000000
"""

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
    N_GRASPS, GRASP_CHUNK, PLACE_CHUNK, VALUE_CHUNK,
    N_GRASP_PLACE_POINTS, N_VALUE_POINTS,
    LIFT_Z, PRE_GRASP_BACKOFF,
    TRAY_CENTER_XY,
    Z_BY_LABEL, PLACE_Z_ABOVE_GRASP,
    GRASP_CKPT, PLACE_CKPT, VALUE_CKPT,
    fps_equal_parts, enforce_z_down, _z_for_label,
    visualize_poses_on_pcd, _to_t,
    carry_horizontal, place_vertical,
)

from table_bussing_runner import (
    capture_until_user_ok,
    partition_scene,
    apply_calibration_offset,
    prompt_task_done,
    prompt_success,
)

from robo_utils.visualization.plotting import plot_pcd


# --------------------------------------------------------------------------- #
# USER EDIT POINT
# --------------------------------------------------------------------------- #
SAVE_ROOT = Path("data/table_bussing_ordered")
ENV_ID = 0

# Required class processing order. Each step picks the first class in this
# list that still has at least one graspable object detected in the scene.
CLASS_ORDER = ["cup", "plate", "bowl"]

# Name-level priority override (case-insensitive substring match). When set,
# any object whose name contains one of these substrings is picked BEFORE the
# normal CLASS_ORDER traversal. Entries are checked in list order, so put
# higher-priority objects first. Set to [] to use pure class ordering.
NAME_PRIORITY_OVERRIDE = [
    "blue bowl",
    "green bowl",
    "orange cup",
    "pink plate",
    "white cup",
    "blue plate",
]

# Hand-eye calibration offset (meters). Subtracted from x and y of every
# predicted grasp pose right after inference. z is untouched. Placements
# are computed geometrically from real-world tray + calibrated grasps, so
# they do not need a separate calibration shift.
CALIB_XY_OFFSET = (0.0, 0.0)

# Geometric placement params.
PLATE_STACK_THICKNESS = 0.015     # m added to place-z per existing in-tray plate
BOWL_STACK_THICKNESS  = 0.06      # m added to place-z per existing in-tray bowl
EDGE_MARGIN           = 0.02      # extra m inside tray bbox past object radius
CORNER_OCCUPY_TOL_MULT = 2.0      # corner slot blocked if any in-tray cup
                                  # centroid lies within (mult * cup_radius)

# Cup slot logic uses a NOMINAL radius (not the FPS-pcd-measured one) because
# the measured radius is dominated by over-inclusive segmentation (the white
# cup's seg-pcd can include 17 cm of neighbouring points). A nominal radius
# gives sane edge margins + clipping. Tune to actual cup size + a bit.
CUP_NOMINAL_RADIUS = 0.04         # m; ~4 cm cup, used for clip margins

# Center-to-center spacing between successive cups along the row. Also acts
# as the blocker tolerance: a slot is considered blocked if any in-tray cup
# centroid is closer than CUP_SPACING_M.
CUP_SPACING_M = 0.13              # m; cup-to-cup pitch on the placement row

# Tray bbox computation uses this percentile to clip outlier tray points
# (perception sometimes labels stray points outside the rim as tray).
# Set to 0.0 to use raw min/max. 1.0 means we treat the 1st/99th
# percentile as the bbox edges -> ignores worst 1% of outliers each side.
TRAY_BBOX_PERCENTILE = 1.0

# Hardcoded grasp poses per object name (case-insensitive substring match).
# When set, the policy-inferred grasp for that object is REPLACED with this
# 7d pose [x, y, z, qw, qx, qy, qz] AFTER calibration is applied to the
# policy output — so these values are treated as final commanded grasp
# coords (no calibration shift applied to them). Placement is still
# computed geometrically from this grasp pose + the perceived object
# centroid, so the delta-correction puts the object centroid on target_xy.
HARDCODED_GRASPS = {
    # cups — from data/grasp_calibration/grasps_20260519_022210.json
    "white cup": [
        0.5705554175376892, -0.29167722880840302, 0.25,
        6.106410732755491e-17, 0.9972525238990784, -0.07407685369253159,
        -4.535899482854956e-18,
    ],
    "orange cup": [
        0.39264044165611267, -0.2650567293167114, 0.25,
        6.161204334583861e-18, 0.10062010586261749, -0.9949249029159546,
        -6.092158079928957e-17,
    ],
    # plates — kept from prior manual tuning
    "pink plate": [
        0.25916009306907654, -0.2791045731306076, 0.1899999976158142,
        6.055770073910504e-17, 0.9889822602272034, -0.1480339914560318,
        -9.064467663940854e-18,
    ],
    "blue plate": [
        0.50916009306907654, -0.2791045731306076, 0.1899999976158142,
        6.055770073910504e-17, 0.9889822602272034, -0.1480339914560318,
        -9.064467663940854e-18,
    ],
    # bowls — from data/grasp_calibration/grasps_20260519_022210.json
    "blue bowl": [
        0.3808218836784363, -0.4200177490711212, 0.2199999988079071,
        5.536427704169291e-17, 0.9041672348976135, 0.42717865109443665,
        2.6157148343543694e-17,
    ],
    "green bowl": [
        0.5110760927200317, -0.4114214777946472, 0.2199999988079071,
        2.851127127964511e-17, 0.4656243920326233, -0.8849824666976929,
        -5.4189544914247153e-17,
    ],
}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def _label_class(name: str):
    """Return the CLASS_ORDER key that matches `name`, or None."""
    lname = name.lower()
    for key in CLASS_ORDER:
        if key in lname:
            return key
    return None


def _object_metrics(obj):
    """Return (centroid_xy, radius_xy) from FPS scene_pcd[mask>0]. Both None
    if no points."""
    pcd = obj["scene_pcd"]
    m = obj["mask"]
    if isinstance(pcd, torch.Tensor):
        pcd = pcd.detach().cpu().numpy()
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    pts = pcd[m > 0.5]
    if len(pts) == 0:
        return None, 0.0
    centroid = pts[:, :2].mean(axis=0).astype(np.float32)
    radius = float(np.max(np.linalg.norm(pts[:, :2] - centroid[None, :], axis=1)))
    return centroid, radius


def _tray_bbox_xy(tray_pcd, percentile=None):
    """(tx_min, tx_max, ty_min, ty_max) of tray pcd in xy. By default uses
    TRAY_BBOX_PERCENTILE — clips that percent of outlier points on each side
    so noisy perception (a few stray tray-labeled points outside the rim)
    doesn't inflate the bbox. Pass percentile=0 to use raw min/max."""
    p = TRAY_BBOX_PERCENTILE if percentile is None else float(percentile)
    x = tray_pcd[:, 0]
    y = tray_pcd[:, 1]
    if p <= 0.0:
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())
    return (
        float(np.percentile(x, p)),
        float(np.percentile(x, 100.0 - p)),
        float(np.percentile(y, p)),
        float(np.percentile(y, 100.0 - p)),
    )


def reclassify_inside_tray(object_specs, in_tray_objects, tray_pcd):
    """Post-partition cleanup. `partition_scene` shrinks the tray bbox by
    INSIDE_TRAY_CLEARANCE to decide whether an object sits inside the tray;
    that can misclassify an object near the rim as graspable. Here we use
    the RAW tray bbox (no clearance) and move any graspable object whose
    xy centroid lies inside it into `in_tray_objects`. Returns
    (new_object_specs, new_in_tray_objects)."""
    if not object_specs:
        return object_specs, list(in_tray_objects or [])
    tx_min, tx_max, ty_min, ty_max = _tray_bbox_xy(tray_pcd)
    print(f"\n>>> Reclassify edge-of-tray objects: raw tray bbox "
          f"x=[{tx_min:.4f}, {tx_max:.4f}]  y=[{ty_min:.4f}, {ty_max:.4f}]")
    kept = []
    in_tray = list(in_tray_objects or [])
    for k, name, seg_pcd, seg_rgb in object_specs:
        cx = float(seg_pcd[:, 0].mean())
        cy = float(seg_pcd[:, 1].mean())
        if tx_min <= cx <= tx_max and ty_min <= cy <= ty_max:
            print(f"  RECLASSIFY '{name}' centroid=({cx:.4f}, {cy:.4f}) -> in-tray.")
            in_tray.append({
                "name": name,
                "centroid_xy": np.array([cx, cy], dtype=np.float32),
            })
        else:
            kept.append((k, name, seg_pcd, seg_rgb))
    return kept, in_tray


def _slot_blocked(slot, blockers_xy, tol):
    for c in blockers_xy:
        if float(np.linalg.norm(slot - c)) < tol:
            return True
    return False


def _pick_cup_slot(bbox, radius, in_tray_objects, tray_center_xy):
    """Cups packed along a single y-row.

    Row y is the y of the first in-tray object (or min-y edge if the tray is
    empty). Slots are ANCHORED to the first in-tray cup's x so that adjacent
    slots are EXACTLY CUP_SPACING_M from it (distance == tol -> `<` strict
    check leaves the slot free). Walks |k|=1, 2, 3, ... away from anchor,
    preferring +x first (deeper into tray), filtering anything outside the
    tray bbox. Falls back to tray center if every slot is blocked.

    Returns (slot_xy, is_fallback)."""
    tx_min, tx_max, ty_min, ty_max = bbox
    m = radius + EDGE_MARGIN
    blockers_xy = [
        np.asarray(ito["centroid_xy"], dtype=np.float32)
        for ito in (in_tray_objects or [])
    ]
    step = float(CUP_SPACING_M)
    # `<` is strict, but float32 precision in centroid + anchor + step can
    # produce distances that are `step - 1e-7` apart, falsely blocking the
    # adjacent slot. Use 1 mm safety so the next-step-away slot is reliably
    # free.
    tol = float(CUP_SPACING_M) - 1e-3

    if blockers_xy:
        anchor = blockers_xy[0]
        anchor_x = float(anchor[0])
        row_y = float(anchor[1])
        donor_name = (in_tray_objects or [{}])[0].get("name", "?")
        print(f"    [cup row] anchor='{donor_name}' centroid=({anchor_x:.4f}, {row_y:.4f}); "
              f"slots at anchor_x +/- k*step (step={step:.3f})")
    else:
        # Place the first cup at the FAR-RIGHT (max-y) edge of the tray.
        # Successive cups align to this y via the blockers_xy path above.
        anchor_x = float(tx_min + m)
        row_y = float(ty_max - m)
        print(f"    [cup row] no in-tray cup; row anchor at max-y edge "
              f"x={anchor_x:.4f}, y={row_y:.4f}")
    row_y = float(np.clip(row_y, ty_min + m, ty_max - m))

    # Build candidate slots in order of preference.
    candidates = []
    if not blockers_xy:
        candidates.append(np.array([anchor_x, row_y], dtype=np.float32))
    for k in range(1, 16):
        for sign in (+1, -1):  # prefer +x (deeper into tray) first
            x = anchor_x + sign * k * step
            candidates.append(np.array([x, row_y], dtype=np.float32))

    for slot in candidates:
        if not (tx_min + m <= slot[0] <= tx_max - m):
            print(f"    [cup slot] reject ({float(slot[0]):.4f}, {float(slot[1]):.4f}): "
                  f"out of bbox x=[{tx_min+m:.4f}, {tx_max-m:.4f}]")
            continue
        if _slot_blocked(slot, blockers_xy, tol):
            min_d = min(float(np.linalg.norm(slot - c)) for c in blockers_xy)
            print(f"    [cup slot] reject ({float(slot[0]):.4f}, {float(slot[1]):.4f}): "
                  f"blocked, min dist to in-tray={min_d:.4f} < tol={tol:.4f}")
            continue
        print(f"    [cup slot] accept ({float(slot[0]):.4f}, {float(slot[1]):.4f})")
        return slot, False
    return np.asarray(tray_center_xy, dtype=np.float32), True


def _clip_to_tray(target_xy, radius, bbox):
    """Clip target_xy into tray bbox shrunk by (radius + EDGE_MARGIN)."""
    tx_min, tx_max, ty_min, ty_max = bbox
    margin = radius + EDGE_MARGIN
    new_x = float(np.clip(target_xy[0], tx_min + margin, tx_max - margin))
    new_y = float(np.clip(target_xy[1], ty_min + margin, ty_max - margin))
    clipped = (abs(new_x - float(target_xy[0])) > 1e-6) or (abs(new_y - float(target_xy[1])) > 1e-6)
    return np.array([new_x, new_y], dtype=np.float32), clipped


def _make_place_pose(grasp_pose, target_xy, target_z, object_centroid_xy,
                     use_delta=True):
    """Build a place pose from a grasp.

    use_delta=True : place_xy = target_xy + (grasp_xy - object_centroid_xy).
                     Object centroid lands on target_xy. Used for plate/bowl
                     stacking where centroid alignment matters.
    use_delta=False: place_xy = target_xy directly. Gripper goes to target_xy
                     without compensating for grasp offset. Used for cups so
                     placed cup y exactly matches in-tray cup y.
    Quaternion copied from grasp.
    """
    is_tensor = isinstance(grasp_pose, torch.Tensor)
    g = grasp_pose.detach().cpu().numpy() if is_tensor else np.asarray(grasp_pose, dtype=np.float32)
    if use_delta:
        dx = float(g[0] - object_centroid_xy[0])
        dy = float(g[1] - object_centroid_xy[1])
    else:
        dx = 0.0
        dy = 0.0
    place = np.array([
        float(target_xy[0] + dx),
        float(target_xy[1] + dy),
        float(target_z),
        float(g[3]), float(g[4]), float(g[5]), float(g[6]),
    ], dtype=np.float32)
    return torch.from_numpy(place) if is_tensor else place


# --------------------------------------------------------------------------- #
# Grasp-only inference + geometric placements
# --------------------------------------------------------------------------- #

def _match_hardcoded_grasp(name: str):
    """Return (key, pose7d_np) for the first HARDCODED_GRASPS key whose
    lowercased form is a substring of `name`. Otherwise (None, None)."""
    lname = name.lower()
    for key, pose in HARDCODED_GRASPS.items():
        if key.lower() in lname:
            return key, np.asarray(pose, dtype=np.float32)
    return None, None


def apply_hardcoded_grasps(per_object):
    """Replace policy-inferred grasps with values from HARDCODED_GRASPS for any
    matching object name. Hardcoded values are treated as final commanded
    poses — NOT subject to CALIB_XY_OFFSET. Call AFTER apply_calibration_offset
    so the hardcoded values overwrite whatever the calibration produced."""
    for obj in per_object:
        key, pose_arr = _match_hardcoded_grasp(obj["name"])
        if pose_arr is None:
            continue
        new_grasps = []
        for g in obj["grasps"]:
            if isinstance(g, torch.Tensor):
                new_grasps.append(torch.from_numpy(pose_arr.copy()).to(g.device).to(g.dtype))
            else:
                new_grasps.append(pose_arr.copy())
        obj["grasps"] = new_grasps
        print(f"\n>>> HARDCODED grasp override for '{obj['name']}' (matched '{key}')")
        print(f"    pose7d = {pose_arr.tolist()}")


def infer_grasps_only(inference, object_specs, tray_pcd):
    """Build per_object entries with grasps only (skip placement model).
    If an object name matches HARDCODED_GRASPS, the grasp model is NOT
    called for it — the hardcoded pose is used directly. Otherwise the
    grasp policy runs as usual. `placements` is left empty; filled later
    by compute_placements_geometric."""
    per_object = []
    for k, name, seg_pcd, seg_rgb in object_specs:
        print(f"\n=== Processing object '{name}' ({len(seg_pcd)} raw pts) ===")
        scene_pcd, part_labels = fps_equal_parts(
            [seg_pcd, tray_pcd], total_n=N_GRASP_PLACE_POINTS,
        )
        mask = (part_labels == 0).astype(np.float32)
        print(f"  scene_pcd {scene_pcd.shape}  mask_pos={int(mask.sum())}  "
              f"mask_neg={int((1 - mask).sum())}")

        hard_key, hard_pose = _match_hardcoded_grasp(name)
        if hard_pose is not None:
            print(f"  HARDCODED grasp match -> '{hard_key}' (SKIP grasp model)")
            grasps = [enforce_z_down(torch.from_numpy(hard_pose.copy()))
                      for _ in range(N_GRASPS)]
        else:
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

        per_object.append({
            "name": name,
            "label_idx": k,
            "scene_pcd": scene_pcd,
            "mask": mask,
            "grasps": grasps,
            "placements": [],
        })
    return per_object


def compute_placements_geometric(per_object, tray_pcd, in_tray_objects):
    """Fill `placements` for every per_object entry from tray bbox + class rule.
    One placement per grasp. Mutates per_object in place."""
    bbox = _tray_bbox_xy(tray_pcd)
    tx_min, tx_max, ty_min, ty_max = bbox
    print(f"\n>>> Geometric placements: tray bbox "
          f"x=[{tx_min:.4f}, {tx_max:.4f}]  y=[{ty_min:.4f}, {ty_max:.4f}]")
    in_tray = in_tray_objects or []
    print(f"  ===== IN-TRAY DETECTED OBJECTS (placement references) =====")
    if not in_tray:
        print(f"    (none — every successive cup will fall back to min-y edge)")
    for ito in in_tray:
        cxy = ito["centroid_xy"]
        print(f"    '{ito['name']}'  centroid_xy=({float(cxy[0]):.4f}, {float(cxy[1]):.4f})")
    plate_centroids = [np.asarray(ito["centroid_xy"], dtype=np.float32)
                       for ito in in_tray if "plate" in ito["name"].lower()]
    bowl_centroids  = [np.asarray(ito["centroid_xy"], dtype=np.float32)
                       for ito in in_tray if "bowl"  in ito["name"].lower()]
    n_plates_in = len(plate_centroids)
    n_bowls_in  = len(bowl_centroids)
    plate_donor = plate_centroids[0] if plate_centroids else None
    bowl_donor  = bowl_centroids[0]  if bowl_centroids  else None
    print(f"  in-tray counts: plates={n_plates_in} bowls={n_bowls_in}")
    if plate_donor is not None:
        print(f"  plate stack donor xy=({plate_donor[0]:.4f}, {plate_donor[1]:.4f})")
    if bowl_donor is not None:
        print(f"  bowl stack donor  xy=({bowl_donor[0]:.4f}, {bowl_donor[1]:.4f})")

    for obj in per_object:
        cls = _label_class(obj["name"])
        centroid_xy, radius = _object_metrics(obj)
        if centroid_xy is None:
            print(f"  [geo] '{obj['name']}': no FPS points; placements left empty.")
            obj["placements"] = []
            continue
        print(f"  [geo] '{obj['name']}' class={cls} centroid_xy=({centroid_xy[0]:.4f}, {centroid_xy[1]:.4f}) radius={radius:.4f}")

        if cls == "cup":
            # Override measured radius for slot logic — segmentation noise
            # makes measured radius unreliable for cups.
            eff_radius = float(CUP_NOMINAL_RADIUS)
            print(f"    [cup] measured radius={radius:.4f} (noisy); using CUP_NOMINAL_RADIUS={eff_radius:.4f}")
            target_xy, fallback = _pick_cup_slot(bbox, eff_radius, in_tray, TRAY_CENTER_XY)
            if fallback:
                print(f"    [cup] all edge slots blocked; falling back to TRAY_CENTER_XY")
            target_z = float(Z_BY_LABEL["cup"]) + float(PLACE_Z_ABOVE_GRASP)
            print(f"    [cup] target_xy={target_xy.tolist()}  z={target_z:.4f}")
            # Use eff_radius for clipping too (further down).
            radius_for_clip = eff_radius
        elif cls == "plate":
            if plate_donor is not None:
                target_xy = plate_donor.copy()
                src = "in-tray plate donor"
            else:
                target_xy = np.asarray(TRAY_CENTER_XY, dtype=np.float32)
                src = "TRAY_CENTER_XY (no in-tray plate)"
            target_z = (float(Z_BY_LABEL["plate"]) + float(PLACE_Z_ABOVE_GRASP)
                        + n_plates_in * float(PLATE_STACK_THICKNESS))
            print(f"    [plate] target_xy={target_xy.tolist()} ({src})  z={target_z:.4f} (stack_level={n_plates_in})")
            radius_for_clip = radius
        elif cls == "bowl":
            if bowl_donor is not None:
                target_xy = bowl_donor.copy()
                target_z = (float(Z_BY_LABEL["bowl"]) + float(PLACE_Z_ABOVE_GRASP)
                            + n_bowls_in * float(BOWL_STACK_THICKNESS))
                src = f"in-tray bowl donor (level={n_bowls_in})"
            elif plate_donor is not None:
                target_xy = plate_donor.copy()
                target_z = (float(Z_BY_LABEL["bowl"]) + float(PLACE_Z_ABOVE_GRASP)
                            + n_plates_in * float(PLATE_STACK_THICKNESS))
                src = f"in-tray plate donor (above plate stack of {n_plates_in})"
            else:
                target_xy = np.asarray(TRAY_CENTER_XY, dtype=np.float32)
                target_z = float(Z_BY_LABEL["bowl"]) + float(PLACE_Z_ABOVE_GRASP)
                src = "TRAY_CENTER_XY (no donor)"
            print(f"    [bowl] target_xy={target_xy.tolist()} ({src})  z={target_z:.4f}")
            radius_for_clip = radius
        else:
            print(f"    [geo] unknown class for '{obj['name']}'; defaulting to TRAY_CENTER_XY.")
            target_xy = np.asarray(TRAY_CENTER_XY, dtype=np.float32)
            target_z = 0.25
            radius_for_clip = radius

        target_xy, clipped = _clip_to_tray(target_xy, radius_for_clip, bbox)
        if clipped:
            print(f"    [geo] target_xy clipped to fit tray -> {target_xy.tolist()}")

        # All classes use delta correction so the OBJECT CENTROID (not the
        # gripper target) lands on target_xy. For cups, this means the post-
        # placement white-cup centroid will equal the in-tray cup centroid
        # along the y-axis (target_y = in-tray cup y). The viz marker shows
        # the gripper target (offset by (grasp - centroid)); the print below
        # shows the predicted post-place centroid.
        placements = []
        for g in obj["grasps"]:
            p = _make_place_pose(g, target_xy, target_z, centroid_xy, use_delta=True)
            p = enforce_z_down(p)
            placements.append(p)
        obj["placements"] = placements
        # Predicted post-place centroid = place_xy + (centroid - grasp).
        for pi, (g, p) in enumerate(zip(obj["grasps"], placements)):
            g_arr = g.detach().cpu().numpy() if isinstance(g, torch.Tensor) else np.asarray(g)
            p_arr = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
            post_x = float(p_arr[0] + (centroid_xy[0] - g_arr[0]))
            post_y = float(p_arr[1] + (centroid_xy[1] - g_arr[1]))
            print(f"    [geo] pair {pi}: "
                  f"detected_centroid_xy=({centroid_xy[0]:.4f}, {centroid_xy[1]:.4f})  "
                  f"grasp_xy=({float(g_arr[0]):.4f}, {float(g_arr[1]):.4f})  "
                  f"place_xy=({float(p_arr[0]):.4f}, {float(p_arr[1]):.4f})  "
                  f"=> predicted post-place centroid_xy=({post_x:.4f}, {post_y:.4f})  "
                  f"target_xy=({float(target_xy[0]):.4f}, {float(target_xy[1]):.4f})")


_PREVIEW_COLORS = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
    (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (1.0, 0.75, 0.8),
]


def _select_target(object_specs):
    """Pick the next thing to bus.

    Returns (target_class, name_filter):
      - target_class : CLASS_ORDER key ("cup"/"plate"/"bowl") of the chosen
                       object(s).
      - name_filter  : a lowercased substring; only object names containing
                       it are eligible this step. When None, every object of
                       `target_class` is eligible.

    Priority:
      1. NAME_PRIORITY_OVERRIDE entries (in list order). First one whose
         substring matches at least one object name wins. name_filter set to
         that substring so only matching objects get planned.
      2. CLASS_ORDER traversal. First class with >=1 matching object wins.
         name_filter is None (every object of that class is eligible).
    """
    for override in NAME_PRIORITY_OVERRIDE:
        ov = override.lower()
        for spec in object_specs:
            name = spec[1].lower()
            if ov in name:
                cls = _label_class(spec[1])
                if cls is not None:
                    return cls, ov
    present = set()
    for spec in object_specs:
        c = _label_class(spec[1])
        if c is not None:
            present.add(c)
    for c in CLASS_ORDER:
        if c in present:
            return c, None
    return None, None


def preview_placements(per_object, in_tray_objects, tray_pcd, pcd_np, rgb_np, seg_labels):
    """Print + visualize all (grasp, place) pairs for the current target class.
    Each object gets a distinct color; its grasp and place share that color.
    Adds 4 WHITE markers at the tray bbox corners (uses the same percentile
    bbox the placement code is actually using) so the user can visually
    confirm the tray bounds. Prints predicted POST-PLACE centroid xy =
    place_xy + (centroid - grasp). That predicted centroid is what should
    align with the in-tray reference object's centroid. Prompts user to
    accept (y) or abort the step (n)."""
    if not per_object:
        print("  [preview] empty per_object; nothing to visualize.")
        return False
    print("\n##### PRE-EXECUTION PREVIEW (geometric placements) #####")
    bbox = _tray_bbox_xy(tray_pcd)
    raw_bbox = _tray_bbox_xy(tray_pcd, percentile=0.0)
    tx_min, tx_max, ty_min, ty_max = bbox
    rx_min, rx_max, ry_min, ry_max = raw_bbox
    print(f"  --- Tray bbox ---")
    print(f"    percentile={TRAY_BBOX_PERCENTILE:.1f}  -> x=[{tx_min:.4f}, {tx_max:.4f}]  y=[{ty_min:.4f}, {ty_max:.4f}]")
    print(f"    raw min/max         -> x=[{rx_min:.4f}, {rx_max:.4f}]  y=[{ry_min:.4f}, {ry_max:.4f}]")
    print("  --- In-tray reference objects ---")
    if not in_tray_objects:
        print("    (none)")
    for ito in (in_tray_objects or []):
        cxy = ito["centroid_xy"]
        print(f"    '{ito['name']}'  centroid_xy=({float(cxy[0]):.4f}, {float(cxy[1]):.4f})")
    print("  --- Per-object plan ---")
    poses = []
    colors = []
    for i, obj in enumerate(per_object):
        color = _PREVIEW_COLORS[i % len(_PREVIEW_COLORS)]
        centroid_xy, radius = _object_metrics(obj)
        if centroid_xy is None:
            print(f"  [{i}] '{obj['name']}' (no FPS points; skipping)")
            continue
        for pair_i, (g, p) in enumerate(zip(obj["grasps"], obj["placements"])):
            g_arr = g.detach().cpu().numpy() if isinstance(g, torch.Tensor) else np.asarray(g)
            p_arr = p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
            post_x = float(p_arr[0] + (centroid_xy[0] - g_arr[0]))
            post_y = float(p_arr[1] + (centroid_xy[1] - g_arr[1]))
            print(f"  [{i}] '{obj['name']}' pair={pair_i}  color=RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
            print(f"        detected_centroid_xy  = ({centroid_xy[0]:.4f}, {centroid_xy[1]:.4f})   radius={radius:.4f}")
            print(f"        calibrated_grasp_xy   = ({float(g_arr[0]):.4f}, {float(g_arr[1]):.4f})")
            print(f"        gripper_place_xy      = ({float(p_arr[0]):.4f}, {float(p_arr[1]):.4f})  (viz marker shows THIS)")
            print(f"        PREDICTED post-place  = ({post_x:.4f}, {post_y:.4f})  <-- should match in-tray ref")
            print(f"        grasp pose 7d  = {list(map(float, g_arr))}")
            print(f"        place pose 7d  = {list(map(float, p_arr))}")
            poses.append(g)
            colors.append(color)
            poses.append(p)
            colors.append(color)
    # Add the 4 tray bbox corners as WHITE markers so the user can verify
    # the tray bounds the placement code is using. z = mean tray z so corners
    # sit on the tray rim plane.
    corner_z = float(np.mean(tray_pcd[:, 2]))
    corner_quat = (0.0, 0.0, 1.0, 0.0)  # z-down quaternion (qw, qx, qy, qz)
    corner_xys = [
        (tx_min, ty_min), (tx_max, ty_min),
        (tx_max, ty_max), (tx_min, ty_max),
    ]
    print(f"  --- Tray bbox corners (WHITE markers in viz) ---")
    for (cx, cy) in corner_xys:
        corner_pose = np.array(
            [cx, cy, corner_z, *corner_quat], dtype=np.float32,
        )
        poses.append(corner_pose)
        colors.append((1.0, 1.0, 1.0))
        print(f"    corner xy=({cx:.4f}, {cy:.4f})  z={corner_z:.4f}")
    nonbg = (seg_labels != -1)
    viz_pcd = pcd_np[nonbg]
    viz_rgb = rgb_np[nonbg] if rgb_np is not None else None
    print(f"\n>>> Visualizing {len(per_object)} object(s); same color = same object's grasp + place.")
    print(f"    WHITE markers = tray bbox corners actually used by placement code.")
    print(f"    Note: viz markers show the GRIPPER place pose (offset from final object centroid by grasp offset).")
    print(f"    Close window to continue.")
    visualize_poses_on_pcd(viz_pcd, poses, rgb=viz_rgb, colors=colors)
    while True:
        ans = input("\nProceed with execution? [y/N] (n -> abort step, recapture): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("", "n", "no"):
            return False
        print("  Please answer y or n.")


def select_winner_ordered(inference, whole_pcd, per_object):
    """
    Pick the next object to bus per CLASS_ORDER. Within the chosen class,
    value-score (grasp, geometric-place) pairs and take argmax. Returns
    winner dict or None if no class in CLASS_ORDER matches any object.
    """
    by_class = {}
    for obj in per_object:
        c = _label_class(obj["name"])
        if c is None:
            print(f"  [select] skip object '{obj['name']}' — not in CLASS_ORDER {CLASS_ORDER}.")
            continue
        by_class.setdefault(c, []).append(obj)

    target_class = None
    for c in CLASS_ORDER:
        if by_class.get(c):
            target_class = c
            break
    if target_class is None:
        print("  [select] no remaining objects match CLASS_ORDER.")
        return None
    print(f"\n##### CLASS ORDERING: target class = '{target_class}' #####")
    for c in CLASS_ORDER:
        present = [o["name"] for o in by_class.get(c, [])]
        print(f"  class '{c}': {present if present else '(none)'}")

    class_objs = by_class[target_class]
    flat = []
    flat_grasps = []
    flat_places = []
    for obj_i, obj in enumerate(class_objs):
        for pair_i, (g, p) in enumerate(zip(obj["grasps"], obj["placements"])):
            flat.append((obj_i, pair_i))
            flat_grasps.append(g)
            flat_places.append(p)
    if not flat:
        print("  [select] no (grasp, place) pairs available.")
        return None
    if len(flat) == 1:
        # Single candidate -> no ranking needed, skip value model entirely.
        print(f"\n>>> SKIP value model: only 1 '{target_class}' triple.")
        best = 0
        best_score = 0.0
    else:
        print(f"\n>>> VALUE model on {len(flat)} '{target_class}' triples (chunk={VALUE_CHUNK})")
        scores = inference.score_value_batched(
            pcd_np=whole_pcd, grasp_poses=flat_grasps, place_poses=flat_places,
            chunk_size=VALUE_CHUNK,
        )
        sorted_idx = torch.argsort(scores, descending=True).tolist()
        print(f"Per-triple value scores within class '{target_class}' (sorted):")
        for rank, idx in enumerate(sorted_idx):
            obj_i, pair_i = flat[idx]
            print(f"  rank {rank+1:>2d}: obj={class_objs[obj_i]['name']:>20s}  "
                  f"pair={pair_i}  score={float(scores[idx]):.4f}")
        best = sorted_idx[0]
        best_score = float(scores[best])

    best_obj_i, best_pair_i = flat[best]
    best_obj = class_objs[best_obj_i]
    return {
        "obj": best_obj,
        "pair_i": best_pair_i,
        "score": best_score,
        "grasp": best_obj["grasps"][best_pair_i],
        "place": best_obj["placements"][best_pair_i],
        "target_class": target_class,
    }


# --------------------------------------------------------------------------- #
# Pick with mid-pick user confirmation
# --------------------------------------------------------------------------- #

def pick_with_confirm(motion_planner: MotionPlanner, controller: FrankaPandaController,
                      current_joints: torch.Tensor, grasp_pose: torch.Tensor, device):
    """
    Plan pre_grasp + grasp + lift. Execute pre_grasp move. Prompt user for
    grasp-pose approval. Yes -> execute grasp + close + lift. No -> open +
    home + return (None, None, False) so caller retries the whole step.
    Returns (lift_end_joints, pick_record, ok_bool).
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

    while True:
        ans = input("\nGrasp pose look good? [y/N] (n -> open + home + retry step): ").strip().lower()
        if ans in ("y", "yes"):
            grasp_ok_user = True
            break
        if ans in ("", "n", "no"):
            grasp_ok_user = False
            break
        print("  Please answer y or n.")
    if not grasp_ok_user:
        print("Grasp pose rejected; opening gripper and homing for retry.")
        controller.open_gripper()
        controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
        return None, None, False

    print(f"  EXEC: move along grasp trajectory (T={len(grasp_np)})")
    controller.move_along_trajectory(grasp_np, controller.open_gripper_action)
    print("  EXEC: close gripper")
    controller.close_gripper()
    print(f"  EXEC: move along lift trajectory (T={len(lift_np)})")
    controller.move_along_trajectory(lift_np, controller.close_gripper_action)
    print("  Pick complete (no post-lift prompt).")

    return lift_traj[0, -1], {"pre": pre_np, "grasp": grasp_np, "lift": lift_np}, True


# --------------------------------------------------------------------------- #
# Per-step driver
# --------------------------------------------------------------------------- #

def save_step_pcd_local(pcd, rgb, seg_labels, seg_label_names, step_i: int):
    """Local SAVE_ROOT wrapper so intermediates land under this script's dir."""
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


def save_run_local(records, success: bool):
    sub = "success" if success else "failure"
    out_dir = SAVE_ROOT / f"env_{ENV_ID}" / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"run_{ts}.npz"
    np.savez(
        out_path,
        n_steps=np.array(len(records)),
        object_names=np.array([r["object_name"] for r in records]),
        target_classes=np.array([r["target_class"] for r in records]),
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


def run_one_step(perception, controller, inference, device, step_i: int):
    """Capture -> partition -> sample -> ordered select -> exec with mid-pick confirm."""
    print("\n========================================================")
    print(f"                  >>> STEP {step_i} <<<")
    print("========================================================")
    print(">>> Capturing live pcd (loop until user OK)")
    pcd_np, rgb_np, seg_labels, seg_label_names = capture_until_user_ok(perception)
    save_step_pcd_local(pcd_np, rgb_np, seg_labels, seg_label_names, step_i)

    print("\n>>> Partitioning scene")
    object_specs, tray_pcd, tray_rgb, in_tray_objects = partition_scene(
        pcd_np, rgb_np, seg_labels, seg_label_names,
    )
    if object_specs is None:
        return None
    object_specs, in_tray_objects = reclassify_inside_tray(
        object_specs, in_tray_objects, tray_pcd,
    )
    if not object_specs:
        print("All objects inside tray after reclassify; nothing to bus.")
        return None
    if in_tray_objects:
        print(f"  in-tray donor candidates for stacking:")
        for ito in in_tray_objects:
            print(f"    '{ito['name']}' centroid_xy=({ito['centroid_xy'][0]:.4f}, {ito['centroid_xy'][1]:.4f})")

    # Selection short-circuit: NAME_PRIORITY_OVERRIDE first, then CLASS_ORDER.
    # Only sample grasps + placements for the chosen object(s); everything else
    # is deferred to later steps.
    target_class, name_filter = _select_target(object_specs)
    if target_class is None:
        print("No CLASS_ORDER / NAME_PRIORITY_OVERRIDE match in scene; nothing to do.")
        return None
    if name_filter is not None:
        target_specs = [s for s in object_specs if name_filter in s[1].lower()]
        skipped = [s[1] for s in object_specs if name_filter not in s[1].lower()]
        print(f"\n##### NAME OVERRIDE = '{name_filter}'  (class='{target_class}') #####")
    else:
        target_specs = [s for s in object_specs if _label_class(s[1]) == target_class]
        skipped = [s[1] for s in object_specs if _label_class(s[1]) != target_class]
        print(f"\n##### TARGET CLASS = '{target_class}' #####")
    print(f"  inferring grasps for: {[s[1] for s in target_specs]}")
    if skipped:
        print(f"  deferring: {skipped}")

    per_object = infer_grasps_only(inference, target_specs, tray_pcd)
    apply_calibration_offset(per_object, CALIB_XY_OFFSET[0], CALIB_XY_OFFSET[1])
    apply_hardcoded_grasps(per_object)
    compute_placements_geometric(per_object, tray_pcd, in_tray_objects)

    if not preview_placements(per_object, in_tray_objects, tray_pcd, pcd_np, rgb_np, seg_labels):
        print("User rejected preview; aborting step.")
        return None

    # Whole-scene FPS uses ALL graspable objects so the value model sees the
    # full scene context even though only target_class objects are scored.
    parts_for_value = [tray_pcd] + [s[2] for s in object_specs]
    print(f"\n>>> Whole-scene FPS for value scoring "
          f"({len(parts_for_value)} parts), total={N_VALUE_POINTS}")
    whole_pcd, _ = fps_equal_parts(parts_for_value, total_n=N_VALUE_POINTS)
    print(f"  whole_pcd {whole_pcd.shape}")

    winner = select_winner_ordered(inference, whole_pcd, per_object)
    if winner is None:
        print("No CLASS_ORDER match in scene; skipping step.")
        return None
    print(f"\n##### ORDERED SELECTION #####")
    print(f"  class : '{winner['target_class']}'")
    print(f"  object: '{winner['obj']['name']}'  pair={winner['pair_i']}  "
          f"score={winner['score']:.4f}")
    print(f"  grasp_pose: {winner['grasp'].tolist() if isinstance(winner['grasp'], torch.Tensor) else list(winner['grasp'])}")
    print(f"  place_pose: {winner['place'].tolist() if isinstance(winner['place'], torch.Tensor) else list(winner['place'])}")

    motion_planner = MotionPlanner(whole_pcd)
    print("\n>>> Homing before execution")
    controller.move_to_joints(controller.home_joints, controller.open_gripper_action)
    current_joints = controller.get_robot_joints()
    current_joints = torch.tensor(current_joints, dtype=torch.float32, device=device)

    pre_place_pose = winner["place"].clone() if isinstance(winner["place"], torch.Tensor) \
        else torch.from_numpy(np.asarray(winner["place"], dtype=np.float32))
    pre_place_pose = pre_place_pose.clone()
    pre_place_pose[2] = LIFT_Z

    print("\n--- Pick (with mid-pick grasp confirmation) ---")
    lift_end_j, pick_record, pick_ok = pick_with_confirm(
        motion_planner, controller, current_joints, winner["grasp"], device,
    )
    if not pick_ok:
        print("ABORT: pick failed or user rejected pre-grasp; retrying step.")
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

    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    record = {
        "object_name": winner["obj"]["name"],
        "target_class": winner["target_class"],
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
    pcd_np, rgb_np, seg_labels, seg_label_names = capture_until_user_ok(perception)
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
                print(f"\n>>> Step {step_i} produced no record. Auto-retrying.")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — ending loop early.")
    except Exception:
        print("\nException during run:")
        traceback.print_exc()
    finally:
        if records:
            success = prompt_success()
            save_run_local(records, success)
        else:
            print("No records this run; nothing to save (intermediate pcds may still be on disk).")
        perception.close()


if __name__ == "__main__":
    main()
