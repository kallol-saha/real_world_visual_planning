"""
Perception Pipeline Orchestrator

Pulls (H, W, 3) RGB + (H, W, 3) point cloud from each Azure Kinect via the
capture_single_camera.py subprocess, then runs:
  1. (optional) GSAM2 text-label segmentation -> (H, W) per-pixel label map
  2. Flatten + valid-pixel filter (drops zero-depth points)
  3. Combine cam0 + cam1 in lockstep (pcd, rgb, seg_labels, cluster_labels)
  4. Spatial bounds filter (x/y/z box; tight z = table -> roof)
  5. Per-pcd correction transform
  6. (optional) DBSCAN clustering with noise-reassignment
  7. FPS downsampling to args.num_points (default 4096)
  8. NN-mapping of seg / cluster labels onto the FPS subset
  9. Publishes {pcd, rgb, cluster_labels, seg_labels, seg_label_names, bounds}

Modes:
  default                       -> combine + bounds + FPS, no labels.
  --cluster                     -> + DBSCAN cluster labels.
  --segment                     -> + Grounded-SAM2 seg labels (text prompts).
  --segment + --cluster         -> both; consumer can pick.
  --no_downsample               -> alignment mode: skip combine/FPS, publish
                                   per-camera bounds-filtered pcds.

EDIT HERE TO CHANGE OBJECT LABELS FOR SEGMENTATION:
   DEFAULT_SEG_LABELS below. CLI `--seg_labels "..."` overrides.

Run:
   python -m frankapanda.perception.perception_pipeline --continuous \
       --segment --seg_labels "blue block. red cup. lego."
"""

import argparse
import subprocess
import numpy as np
import zmq
import pickle
import open3d as o3d
import time
import signal
import sys
from pathlib import Path
from robo_utils.conversion_utils import transform_pcd
from robo_utils.visualization.plotting import plot_pcd


# --------------------------------------------------------------------------- #
#                                                                             #
#   >>>  USER EDIT POINT  <<<                                                 #
#                                                                             #
#   Period-separated phrases. GroundingDINO + SAM2 will look for these in     #
#   each camera frame. Each label gets a stable integer index (its position   #
#   in this list, 0-based). Override at runtime with `--seg_labels "..."`.   #
# --------------------------------------------------------------------------- #
DEFAULT_SEG_LABELS = "duck. green lego."


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    print("\n\nShutdown requested... cleaning up")
    shutdown_requested = True


# --------------------------------------------------------------------------- #
# Per-camera unpack: (H,W,3) -> flat Nx3 + valid filter, lockstep with seg.
# --------------------------------------------------------------------------- #

def unpack_camera_data(data, seg_2d=None):
    """
    Args:
        data: dict from capture_single_camera with keys
              'pcd_hw3' (H,W,3 float), 'rgb_hw3' (H,W,3 uint8), 'valid_mask' (H,W bool).
        seg_2d: optional (H,W) int64 segmentation label map (-1 = bg).

    Returns:
        pcd (M, 3) float32 in robot-base frame,
        rgb (M, 3) float32 in [0, 1],
        seg (M,) int64 or None if seg_2d is None.
    """
    pcd_flat = data['pcd_hw3'].reshape(-1, 3)
    rgb_flat = data['rgb_hw3'].reshape(-1, 3).astype(np.float32) / 255.0
    valid_flat = data['valid_mask'].reshape(-1)
    seg_flat = seg_2d.reshape(-1) if seg_2d is not None else None

    pcd = pcd_flat[valid_flat]
    rgb = rgb_flat[valid_flat]
    seg = seg_flat[valid_flat] if seg_flat is not None else None
    return pcd, rgb, seg


# --------------------------------------------------------------------------- #
# Bounds: return mask so all parallel arrays slice in lockstep.
# --------------------------------------------------------------------------- #

def compute_bounds_mask(pcd, bounds):
    mask = np.ones(len(pcd), dtype=bool)
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        if axis_name in bounds:
            lo, hi = bounds[axis_name]
            mask &= (pcd[:, axis_idx] >= lo) & (pcd[:, axis_idx] <= hi)
    return mask


def apply_spatial_bounds(pcd, rgb, bounds):
    """Back-compat shim for no_downsample branch."""
    mask = compute_bounds_mask(pcd, bounds)
    return pcd[mask], rgb[mask]


# --------------------------------------------------------------------------- #
# DBSCAN clustering + noise reassignment.
# --------------------------------------------------------------------------- #

def cluster_points(pcd, eps=0.02, min_points=20):
    if len(pcd) == 0:
        return np.zeros(0, dtype=np.int64), 0

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    labels = np.asarray(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    num_clusters = int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0
    if num_clusters == 0:
        raise RuntimeError("DBSCAN found no clusters; loosen eps / lower min_points.")

    num_noise = int((labels == -1).sum())
    if num_noise > 0:
        non_noise = labels != -1
        kdtree = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd[non_noise]))
        )
        non_noise_labels = labels[non_noise]
        for ni in np.where(~non_noise)[0]:
            _, idx, _ = kdtree.search_knn_vector_3d(pcd[ni], 1)
            labels[ni] = non_noise_labels[idx[0]]
    return labels.astype(np.int64), num_clusters


# --------------------------------------------------------------------------- #
# FPS downsampling.
# --------------------------------------------------------------------------- #

def fps_downsample(pcd, rgb, num_points):
    if len(pcd) <= num_points:
        print(f"Point cloud has {len(pcd)} points, no downsampling needed")
        return pcd, rgb
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(rgb)
    print(f"Downsampling from {len(pcd)} to {num_points} points using FPS...")
    pcd_downsampled = pcd_o3d.farthest_point_down_sample(num_points)
    return np.asarray(pcd_downsampled.points), np.asarray(pcd_downsampled.colors)


def nn_carry_labels(pcd_source, labels_source, pcd_target):
    """Carry per-point labels from pcd_source onto pcd_target via 1-NN."""
    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_source))
    )
    out = np.empty(len(pcd_target), dtype=labels_source.dtype)
    for i, p in enumerate(pcd_target):
        _, idx, _ = tree.search_knn_vector_3d(p, 1)
        out[i] = labels_source[idx[0]]
    return out


# --------------------------------------------------------------------------- #
# Capture subprocess.
# --------------------------------------------------------------------------- #

def capture_camera(cam_id, zmq_port, no_align=False):
    print(f"\n{'='*60}")
    print(f"Starting capture for Camera {cam_id}")
    print(f"{'='*60}")
    script_path = Path(__file__).parent / "capture_single_camera.py"
    cmd = ['python', str(script_path), '--cam_id', str(cam_id), '--zmq_port', str(zmq_port)]
    if no_align:
        cmd.append('--no_align')
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        print(f"ERROR: Camera {cam_id} capture failed with return code {result.returncode}")
        return False
    return True


# --------------------------------------------------------------------------- #
# Pipeline iteration.
# --------------------------------------------------------------------------- #

def run_pipeline_iteration(
    receiver, publisher, bounds, num_points, receive_port,
    save=False, iteration=None,
    no_downsample=False,
    cluster=False, dbscan_eps=0.02, dbscan_min_points=20,
    segmenter=None, seg_labels_str=None,
    plot_cams=False, plot_crop=False, no_align=False,
):
    """One iteration of the perception pipeline. Returns success bool."""
    iter_prefix = f"[Iteration {iteration}] " if iteration is not None else ""

    # 1. Capture + receive (H,W,3) data from both cameras sequentially.
    camera_data = {}
    for cam_id in [0, 1]:
        if not capture_camera(cam_id, receive_port, no_align=no_align):
            print(f"{iter_prefix}ERROR: Failed to capture from camera {cam_id}")
            return False
        print(f"\n{iter_prefix}Waiting to receive data from Camera {cam_id}...")
        camera_data[cam_id] = pickle.loads(receiver.recv())
        H, W, _ = camera_data[cam_id]['pcd_hw3'].shape
        print(f"{iter_prefix}Received Camera {cam_id}: pcd {H}x{W}x3, "
              f"valid={int(camera_data[cam_id]['valid_mask'].sum())}")

    # 2. Optional segmentation per camera. seg_2d_per_cam[cid] = (H, W) int64.
    seg_2d_per_cam = {0: None, 1: None}
    seg_label_names = None
    if segmenter is not None:
        for cam_id in [0, 1]:
            print(f"\n{iter_prefix}Segmenting Camera {cam_id}...")
            seg_map, obj_list = segmenter.segment(
                camera_data[cam_id]['rgb_hw3'], seg_labels_str
            )
            seg_2d_per_cam[cam_id] = seg_map
            seg_label_names = obj_list

    # 3. Flatten + valid-filter each camera in lockstep with seg.
    per_cam_flat = {}
    for cam_id in [0, 1]:
        pcd_c, rgb_c, seg_c = unpack_camera_data(
            camera_data[cam_id], seg_2d_per_cam[cam_id]
        )
        per_cam_flat[cam_id] = dict(pcd=pcd_c, rgb=rgb_c, seg=seg_c)
        print(f"{iter_prefix}Camera {cam_id} valid pcd: {len(pcd_c)}")

    # Per-camera inspection: plot each camera's pcd separately (cam0 first,
    # then cam1). Blocks until window closed. Both in robot-base frame; with
    # --no_align cam1 is calibration-only (no cam1->cam0 alignment). Cropping
    # to workspace bounds is opt-in via plot_crop.
    if plot_cams:
        for cam_id in [0, 1]:
            pcd_c = per_cam_flat[cam_id]['pcd']
            rgb_c = per_cam_flat[cam_id]['rgb']
            if plot_crop:
                mask = compute_bounds_mask(pcd_c, bounds)
                pcd_c, rgb_c = pcd_c[mask], rgb_c[mask]
                crop_note = f" cropped to bounds ({len(pcd_c)}/{len(per_cam_flat[cam_id]['pcd'])} pts)"
            else:
                crop_note = f" ({len(pcd_c)} pts, uncropped)"
            print(f"{iter_prefix}Plotting Camera {cam_id} pcd{crop_note}. "
                  f"Close window to continue.")
            plot_pcd(pcd_c, rgb_c, base_frame=True)

    # ---------- no-downsample branch (alignment use case) ---------- #
    if no_downsample:
        print("\n" + "="*60)
        print(f"{iter_prefix}NO-DOWNSAMPLE MODE: bounds per camera, publish separately")
        print("="*60)
        print(f"{iter_prefix}Bounds: {bounds}")

        cam_filtered = {}
        for cam_id in [0, 1]:
            mask = compute_bounds_mask(per_cam_flat[cam_id]['pcd'], bounds)
            cam_filtered[cam_id] = dict(
                pcd=per_cam_flat[cam_id]['pcd'][mask],
                rgb=per_cam_flat[cam_id]['rgb'][mask],
            )
            print(f"{iter_prefix}Camera {cam_id} after bounds: "
                  f"{len(cam_filtered[cam_id]['pcd'])} points")

        final_data = {
            'no_downsample': True,
            'cam0_pcd': cam_filtered[0]['pcd'],
            'cam0_rgb': cam_filtered[0]['rgb'],
            'cam1_pcd': cam_filtered[1]['pcd'],
            'cam1_rgb': cam_filtered[1]['rgb'],
            'bounds': bounds,
        }
        publisher.send(pickle.dumps(final_data))
        print(f"{iter_prefix}Published per-camera point clouds")

        if save:
            output_dir = Path(__file__).parent / "data" / "perception_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            for cam_id in [0, 1]:
                np.save(output_dir / f"cam{cam_id}_pcd_raw.npy", cam_filtered[cam_id]['pcd'])
                np.save(output_dir / f"cam{cam_id}_rgb_raw.npy", cam_filtered[cam_id]['rgb'])
            print(f"{iter_prefix}Saved per-camera pcds to {output_dir}/")
        return True

    # ---------- normal branch: combine + bounds + (cluster) + FPS ---------- #
    print("\n" + "="*60)
    print(f"{iter_prefix}COMBINING POINT CLOUDS")
    print("="*60)
    pcd_combined = np.vstack([per_cam_flat[0]['pcd'], per_cam_flat[1]['pcd']])
    rgb_combined = np.vstack([per_cam_flat[0]['rgb'], per_cam_flat[1]['rgb']])
    seg_combined = None
    if segmenter is not None:
        seg_combined = np.concatenate([per_cam_flat[0]['seg'], per_cam_flat[1]['seg']])
    print(f"{iter_prefix}Combined point cloud: {len(pcd_combined)} points")

    # 4. Spatial bounds (lockstep across pcd, rgb, seg).
    print("\n" + "="*60)
    print(f"{iter_prefix}APPLYING SPATIAL BOUNDS")
    print("="*60)
    print(f"{iter_prefix}Bounds: {bounds}")
    bounds_mask = compute_bounds_mask(pcd_combined, bounds)
    pcd_filtered = pcd_combined[bounds_mask]
    rgb_filtered = rgb_combined[bounds_mask]
    seg_filtered = seg_combined[bounds_mask] if seg_combined is not None else None
    print(f"{iter_prefix}After bounds filtering: {len(pcd_filtered)} points")

    # 5. Per-pcd correction transform.
    T = np.eye(4)
    T[:3, -1] = np.array([0.035, 0.05, 0.07])
    pcd_filtered = transform_pcd(pcd_filtered, T)

    # 6. Optional DBSCAN clustering (pre-FPS so labels can be NN-mapped).
    full_cluster_labels = None
    if cluster:
        print("\n" + "="*60)
        print(f"{iter_prefix}DBSCAN CLUSTERING (eps={dbscan_eps}, min_points={dbscan_min_points})")
        print("="*60)
        full_cluster_labels, num_clusters = cluster_points(
            pcd_filtered, eps=dbscan_eps, min_points=dbscan_min_points
        )
        print(f"{iter_prefix}Found {num_clusters} clusters on {len(pcd_filtered)} pre-FPS points")

    # 7. FPS.
    print("\n" + "="*60)
    print(f"{iter_prefix}FPS DOWNSAMPLING")
    print("="*60)
    pcd_final, rgb_final = fps_downsample(pcd_filtered, rgb_filtered, num_points)
    print(f"{iter_prefix}Final point cloud: {len(pcd_final)} points")

    # 8. NN-carry of seg + cluster labels onto FPS subset.
    cluster_labels_final = None
    seg_labels_final = None
    if full_cluster_labels is not None or seg_filtered is not None:
        # Build the source-pcd KDTree once and reuse for both label sets.
        tree_src = o3d.geometry.KDTreeFlann(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_filtered))
        )
        nn_idx = np.empty(len(pcd_final), dtype=np.int64)
        for i, p in enumerate(pcd_final):
            _, idx, _ = tree_src.search_knn_vector_3d(p, 1)
            nn_idx[i] = idx[0]
        if full_cluster_labels is not None:
            cluster_labels_final = full_cluster_labels[nn_idx]
            print(f"{iter_prefix}Mapped cluster labels onto FPS pcd "
                  f"({int(cluster_labels_final.max() + 1)} clusters)")
        if seg_filtered is not None:
            seg_labels_final = seg_filtered[nn_idx]
            for k, name in enumerate(seg_label_names or []):
                n = int((seg_labels_final == k).sum())
                print(f"{iter_prefix}  seg label {k} ('{name}'): {n} pts on FPS")
            n_bg = int((seg_labels_final == -1).sum())
            print(f"{iter_prefix}  seg label -1 (background): {n_bg} pts on FPS")

    # 9. Publish.
    final_data = {
        'pcd': pcd_final,
        'rgb': rgb_final,
        'num_points': len(pcd_final),
        'bounds': bounds,
        'cluster_labels': cluster_labels_final,   # None unless --cluster
        'seg_labels': seg_labels_final,           # None unless --segment
        'seg_label_names': seg_label_names,       # None unless --segment
    }
    print("\n" + "="*60)
    print(f"{iter_prefix}PUBLISHING FINAL POINT CLOUD")
    print("="*60)
    publisher.send(pickle.dumps(final_data))
    print(f"{iter_prefix}Published final point cloud")

    if save:
        output_dir = Path(__file__).parent / "data" / "perception_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "final_pcd.npy", pcd_final)
        np.save(output_dir / "final_rgb.npy", rgb_final)
        if seg_labels_final is not None:
            np.save(output_dir / "final_seg_labels.npy", seg_labels_final)
        if cluster_labels_final is not None:
            np.save(output_dir / "final_cluster_labels.npy", cluster_labels_final)
        print(f"{iter_prefix}Saved to {output_dir}/")

    return True


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #

def main():
    global shutdown_requested
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Perception Pipeline')
    parser.add_argument('--receive_port', type=int, default=1234,
                        help='ZMQ port to receive camera data')
    parser.add_argument('--publish_port', type=int, default=1235,
                        help='ZMQ port to publish final point cloud')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Number of points for FPS downsampling')
    parser.add_argument('--save', action='store_true',
                        help='Save final point cloud to data/perception_output/')
    parser.add_argument('--no_downsample', action='store_true',
                        help='Skip FPS+combine; publish per-camera bounds-filtered pcds for alignment.')
    parser.add_argument('--cluster', action='store_true',
                        help='DBSCAN-cluster the bounds-filtered pcd before FPS; publish per-point cluster labels.')
    parser.add_argument('--dbscan_eps', type=float, default=0.02,
                        help='DBSCAN neighborhood radius in meters')
    parser.add_argument('--dbscan_min_points', type=int, default=20,
                        help='DBSCAN minimum cluster size')
    parser.add_argument('--plot_cams', action='store_true',
                        help='Plot each camera pcd separately per iteration (blocks until closed).')
    parser.add_argument('--plot_crop', action='store_true',
                        help='With --plot_cams: crop each camera pcd to workspace bounds (default: uncropped).')
    parser.add_argument('--no_align', action='store_true',
                        help='Skip cam1->cam0 alignment; cameras are calibrated-only (before alignment).')
    parser.add_argument('--segment', action='store_true',
                        help='Run Grounded-SAM2 text-label segmentation on each camera RGB.')
    parser.add_argument('--seg_labels', type=str, default=DEFAULT_SEG_LABELS,
                        help=('Period-separated label phrases for segmentation. '
                              f'Default: "{DEFAULT_SEG_LABELS}"'))
    parser.add_argument('--seg_device', type=str, default='cuda:0',
                        help='Torch device for the Segmenter (default: cuda:0).')
    parser.add_argument('--continuous', action='store_true',
                        help='Run continuously in a loop')
    parser.add_argument('--rate', type=float, default=1.0,
                        help='Loop rate in Hz for continuous mode')
    args = parser.parse_args()

    mode_str = "CONTINUOUS" if args.continuous else "SINGLE-SHOT"
    print("="*60)
    print(f"PERCEPTION PIPELINE STARTING ({mode_str} MODE)")
    print("="*60)
    if args.continuous:
        print(f"Loop rate: {args.rate} Hz (period: {1.0/args.rate:.2f}s)")
        print("Press Ctrl+C to stop\n")

    # Spatial bounds: tight z = table -> roof so FPS keeps a clean 4096.
    bounds = {
        # 'x': [0.3, 0.8],
        # 'y': [-0.5, 0.35],
        # 'z': [-0.03, 0.2],
        'x': [0.3, 0.6],
        'y': [-0.35, 0.35],
        'z': [-0.03, 0.2],
        }

    # ZMQ sockets.
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind(f"tcp://*:{args.receive_port}")
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://*:{args.publish_port}")
    print(f"ZMQ Configuration:")
    print(f"  Receiving camera data on port: {args.receive_port}")
    print(f"  Publishing final point cloud on port: {args.publish_port}")

    # Segmenter loaded once (heavy: ~2GB GPU).
    segmenter = None
    if args.segment:
        print(f"\nLoading Segmenter on {args.seg_device}")
        print(f"  Labels: \"{args.seg_labels}\"")
        from frankapanda.perception.segmenter import Segmenter
        segmenter = Segmenter(device=args.seg_device)

    try:
        if args.continuous:
            iteration = 1
            loop_period = 1.0 / args.rate
            while not shutdown_requested:
                start_time = time.time()
                print("\n" + "="*60)
                print(f"ITERATION {iteration} START")
                print("="*60 + "\n")
                success = run_pipeline_iteration(
                    receiver, publisher, bounds, args.num_points,
                    args.receive_port, args.save, iteration,
                    no_downsample=args.no_downsample,
                    cluster=args.cluster,
                    dbscan_eps=args.dbscan_eps,
                    dbscan_min_points=args.dbscan_min_points,
                    segmenter=segmenter,
                    seg_labels_str=args.seg_labels,
                    plot_cams=args.plot_cams,
                    plot_crop=args.plot_crop,
                    no_align=args.no_align,
                )
                if not success:
                    print(f"\nIteration {iteration} failed, stopping continuous mode")
                    break
                elapsed = time.time() - start_time
                print(f"\nIteration {iteration} completed in {elapsed:.2f}s")
                sleep_time = loop_period - elapsed
                if sleep_time > 0:
                    print(f"Sleeping for {sleep_time:.2f}s to maintain {args.rate} Hz rate...")
                    time.sleep(sleep_time)
                else:
                    print(f"WARNING: Iteration took longer ({elapsed:.2f}s) than loop period ({loop_period:.2f}s)")
                iteration += 1
            print("\n" + "="*60)
            print(f"CONTINUOUS MODE STOPPED (ran {iteration-1} iterations)")
            print("="*60)
        else:
            success = run_pipeline_iteration(
                receiver, publisher, bounds, args.num_points,
                args.receive_port, args.save,
                no_downsample=args.no_downsample,
                cluster=args.cluster,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_points=args.dbscan_min_points,
                segmenter=segmenter,
                seg_labels_str=args.seg_labels,
                plot_cams=args.plot_cams,
                plot_crop=args.plot_crop,
                no_align=args.no_align,
            )
            if success:
                print("\n" + "="*60)
                print("PERCEPTION PIPELINE COMPLETE")
                print("="*60)
    finally:
        print("\nCleaning up ZMQ sockets...")
        receiver.close()
        publisher.close()
        context.term()
        print("Shutdown complete")


if __name__ == '__main__':
    main()
