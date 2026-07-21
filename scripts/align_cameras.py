"""
Compute alignment transformation from camera 1 to camera 0 using ICP.

Subscribes to the perception pipeline publisher (run with --no_downsample)
to receive per-camera bounds-filtered point clouds. ICP gives a residual
correction relative to the current cam1_to_cam0; composes and saves the
updated cam1_to_cam0 and inverse cam0_to_cam1 transformations.

Usage:
    # Terminal 1
    python frankapanda/perception/perception_pipeline.py --no_downsample
    # Terminal 2
    python scripts/align_cameras.py
"""
import argparse
import os
import pickle
import numpy as np
import open3d as o3d
import zmq
from robo_utils.conversion_utils import invert_transformation
from robo_utils.visualization.plotting import plot_pcd


def compute_icp_alignment(source_pcd, target_pcd, threshold=0.01, visualize=False,
                          point_to_plane=False):
    """
    Compute ICP alignment from source to target point cloud.

    Args:
        source_pcd: Open3D point cloud (source)
        target_pcd: Open3D point cloud (target)
        threshold: Distance threshold for ICP
        visualize: Whether to visualize the alignment result

    Returns:
        4x4 transformation matrix that transforms source to align with target
    """
    print(f"Source point cloud: {len(source_pcd.points)} points")
    print(f"Target point cloud: {len(target_pcd.points)} points")

    trans_init = np.identity(4)

    if point_to_plane:
        # Point-to-plane: needs normals; faster but unstable on flat/sparse
        # scenes (ill-conditioned -> can diverge in one step).
        print("Estimating normals...")
        source_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        target_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        print(f"Running point-to-plane ICP with threshold={threshold}...")
    else:
        # Point-to-point: no normals, stable when clouds already overlap.
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        print(f"Running point-to-point ICP with threshold={threshold}...")

    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init, estimator
    )

    print("\nICP Registration Result:")
    print(reg_result)
    print(f"Fitness: {reg_result.fitness}")
    print(f"Inlier RMSE: {reg_result.inlier_rmse}")

    if visualize:
        src = np.asarray(source_pcd.points)   # cam1 (source)
        tgt = np.asarray(target_pcd.points)   # cam0 (target)
        red = np.tile([1.0, 0.0, 0.0], (len(src), 1))    # cam1
        blue = np.tile([0.0, 0.0, 1.0], (len(tgt), 1))   # cam0

        # BEFORE alignment: raw source (red) vs target (blue).
        print("Visualizing BEFORE alignment (red=cam1, blue=cam0). Close window to continue.")
        plot_pcd(np.vstack([src, tgt]), np.vstack([red, blue]), base_frame=True)

        # AFTER alignment: source transformed by the ICP result vs target.
        T = reg_result.transformation
        src_aligned = (np.hstack([src, np.ones((len(src), 1))]) @ T.T)[:, :3]
        print("Visualizing AFTER alignment (red=cam1 transformed, blue=cam0). Close window to continue.")
        plot_pcd(np.vstack([src_aligned, tgt]), np.vstack([red, blue]), base_frame=True)

    return reg_result.transformation


def receive_from_pipeline(subscribe_port, timeout_ms):
    """Subscribe to perception pipeline publisher, return one message payload."""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{subscribe_port}")
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

    print(f"Subscribed to perception pipeline on port {subscribe_port}")
    print(f"Waiting for per-camera point clouds (timeout: {timeout_ms}ms)...")
    try:
        data = pickle.loads(socket.recv())
    finally:
        socket.close()
        context.term()
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICP alignment from perception pipeline")
    parser.add_argument('--subscribe_port', type=int, default=1235,
                        help='ZMQ port to subscribe to (perception pipeline publish port)')
    parser.add_argument('--timeout_ms', type=int, default=120000,
                        help='Receive timeout in milliseconds')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='ICP correspondence distance threshold (meters)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize ICP result with Open3D')
    parser.add_argument('--point_to_plane', action='store_true',
                        help='Use point-to-plane ICP (needs normals; unstable on flat/sparse '
                             'scenes). Default: point-to-point (stable when clouds overlap).')
    args = parser.parse_args()

    data = receive_from_pipeline(args.subscribe_port, args.timeout_ms)

    if not data.get('no_downsample'):
        raise RuntimeError(
            "Received payload missing 'no_downsample' marker. "
            "Run perception_pipeline.py with --no_downsample."
        )

    cam0_pcd_array = data['cam0_pcd']
    cam1_pcd_array = data['cam1_pcd']
    print(f"Camera 0: {cam0_pcd_array.shape[0]} points")
    print(f"Camera 1: {cam1_pcd_array.shape[0]} points")

    cam0_pcd = o3d.geometry.PointCloud()
    cam0_pcd.points = o3d.utility.Vector3dVector(cam0_pcd_array)

    cam1_pcd = o3d.geometry.PointCloud()
    cam1_pcd.points = o3d.utility.Vector3dVector(cam1_pcd_array)

    # ICP: cam1 -> cam0, from identity init. Run the pipeline with --no_align so
    # cam1 is calibration-only; the ICP result is then the FULL cam1->cam0
    # alignment and is saved directly (no composition). This matches
    # align_multiple_cameras. Do NOT compose onto an existing file: under
    # --no_align the existing transform was not applied upstream, so composing
    # double-counts it and blows the result up.
    print("\n" + "="*60)
    print("Computing alignment: Camera 1 -> Camera 0")
    print("="*60)
    cam1_to_cam0 = compute_icp_alignment(
        cam1_pcd, cam0_pcd, threshold=args.threshold, visualize=args.visualize,
        point_to_plane=args.point_to_plane,
    )
    print(f"\ncam1_to_cam0 transformation:\n{cam1_to_cam0}")

    alignment_dir = os.path.join("data", "camera_alignments")
    os.makedirs(alignment_dir, exist_ok=True)
    cam1_to_cam0_file = os.path.join(alignment_dir, "cam1_to_cam0.npy")
    cam0_to_cam1_file = os.path.join(alignment_dir, "cam0_to_cam1.npy")

    cam0_to_cam1 = invert_transformation(cam1_to_cam0)

    print(f"\nFinal cam1_to_cam0:\n{cam1_to_cam0}")
    print(f"\nFinal cam0_to_cam1:\n{cam0_to_cam1}")

    np.save(cam1_to_cam0_file, cam1_to_cam0)
    np.save(cam0_to_cam1_file, cam0_to_cam1)

    print("\n" + "="*60)
    print("Saved transformations:")
    print(f"  cam1_to_cam0: {cam1_to_cam0_file}")
    print(f"  cam0_to_cam1: {cam0_to_cam1_file}")
    print("="*60)
