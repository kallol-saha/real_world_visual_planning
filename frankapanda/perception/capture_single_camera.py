"""
Single camera capture script that:
1. Opens one Azure Kinect camera
2. Captures RGBD frame and point cloud
3. Applies calibration transformation
4. Applies alignment transformation
5. Sends point cloud + RGB via ZMQ
6. Exits (allowing next camera to be opened)
"""

import argparse
import numpy as np
from pyk4a import PyK4A
from pyk4a.pyk4a import PyK4A as PyK4A_device
import zmq
import pickle
from pathlib import Path


def get_kinect_rgbd_frame(device: PyK4A_device):
    """
    Get a valid RGBD frame from the Kinect device.
    Attempts up to 20 times to get a valid capture.

    Returns:
        ir_frame: IR image
        rgb_frame: RGB image (color)
        pcd: Point cloud in depth camera coordinates
        depth_frame: Depth image
    """
    for _ in range(20):
        capture = device.get_capture()
        if capture.transformed_depth_point_cloud is not None:
            ir_frame = capture.ir
            rgb_frame = capture.color
            pcd = capture.transformed_depth_point_cloud
            depth_frame = capture.transformed_depth
            return ir_frame, rgb_frame, pcd, depth_frame

    raise RuntimeError("Failed to get valid capture after 20 attempts")


def transform_pcd(pcd, transform):
    """Apply 4x4 transformation to Nx3 point cloud."""
    # Convert to homogeneous coordinates
    pcd_homogeneous = np.hstack([pcd, np.ones((pcd.shape[0], 1))])
    # Apply transformation
    pcd_transformed = pcd_homogeneous @ transform.T
    # Convert back to 3D
    return pcd_transformed[:, :3]


def main():
    parser = argparse.ArgumentParser(description='Capture from single Azure Kinect camera')
    parser.add_argument('--cam_id', type=int, choices=[0, 1], default=0,
                        help='Camera ID (0 or 1)')
    parser.add_argument('--zmq_port', type=int, default=6555,
                        help='ZMQ port to send data to')
    parser.add_argument('--no_align', action='store_true',
                        help='Skip cam1->cam0 alignment (calibrated-only pcd).')
    args = parser.parse_args()

    print(f"[Camera {args.cam_id}] Initializing...")

    # Initialize ZMQ socket (PUSH mode)
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://localhost:{args.zmq_port}")

    # Initialize camera
    k4a = PyK4A(device_id=args.cam_id)
    k4a.start()

    try:
        # Capture frame
        print(f"[Camera {args.cam_id}] Capturing frame...")
        ir_frame, rgb_frame, pcd_raw, depth_frame = get_kinect_rgbd_frame(k4a)

        # Preserve (H, W, 3) shape so downstream segmentation can run on the
        # image grid and index the pcd by per-pixel masks.
        H, W = pcd_raw.shape[:2]

        # mm -> meters; keep image shape.
        pcd_hw3 = pcd_raw.astype(np.float32) / 1000.0   # (H, W, 3) meters in depth-cam frame

        # RGB: drop alpha + BGR->RGB; keep uint8 for SAM/DINO.
        rgb_hw3 = rgb_frame[..., :-1][..., ::-1].copy().astype(np.uint8)  # (H, W, 3) uint8

        # Invalid mask (depth == 0 -> zero-norm xyz).
        distances = np.linalg.norm(pcd_hw3, axis=-1)
        valid_mask = distances > 0.0  # (H, W) bool
        print(f"[Camera {args.cam_id}] Valid pixels: {int(valid_mask.sum())}/{valid_mask.size}")

        # Load and apply calibration transform (camera -> robot base).
        project_root = Path(__file__).parent.parent.parent
        calib_path = project_root / "data" / "calibration_results" / f"cam{args.cam_id}_calibration.npz"
        calib_data = np.load(calib_path)
        calibration_transform = calib_data['T']

        print(f"[Camera {args.cam_id}] Applying calibration transform...")
        pcd_hw3 = transform_pcd(pcd_hw3.reshape(-1, 3), calibration_transform).reshape(H, W, 3)

        # Cam1 -> cam0 alignment.
        if args.cam_id == 0:
            print(f"[Camera {args.cam_id}] No alignment needed (reference frame)")
        elif args.no_align:
            print(f"[Camera {args.cam_id}] --no_align set: skipping cam1->cam0 alignment (calibrated-only)")
        else:
            alignment_path = project_root / "data" / "camera_alignments" / "cam1_to_cam0.npy"
            alignment_transform = np.load(alignment_path)
            print(f"[Camera {args.cam_id}] Applying alignment transform...")
            pcd_hw3 = transform_pcd(pcd_hw3.reshape(-1, 3), alignment_transform).reshape(H, W, 3)

        # Send the un-flattened arrays so the pipeline can run segmentation on
        # rgb_hw3 and index pcd_hw3 by the resulting (H, W) mask.
        data = {
            'cam_id': args.cam_id,
            'pcd_hw3': pcd_hw3,         # (H, W, 3) float32, robot-base frame (cam0 frame for cam1)
            'rgb_hw3': rgb_hw3,         # (H, W, 3) uint8 RGB
            'valid_mask': valid_mask,   # (H, W) bool
        }

        print(f"[Camera {args.cam_id}] Sending data via ZMQ...")
        socket.send(pickle.dumps(data))

        print(f"[Camera {args.cam_id}] Complete! Sent ({H}, {W}, 3) rgb+pcd")

    finally:
        # Clean up
        k4a.stop()
        socket.close()
        context.term()
        print(f"[Camera {args.cam_id}] Shutdown complete")


if __name__ == '__main__':
    main()
