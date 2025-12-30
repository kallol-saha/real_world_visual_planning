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
    parser.add_argument('--zmq_port', type=int, default=5555,
                        help='ZMQ port to send data to')
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

        # Reshape point cloud from image format to Nx3
        pcd = pcd_raw.reshape(-1, 3)

        # Convert from millimeters to meters
        pcd = pcd / 1000.0

        # Process RGB
        rgb = rgb_frame[..., :-1].reshape(-1, 3)
        # Convert BGR to RGB
        rgb = rgb[:, [2, 1, 0]]
        # Normalize to [0, 1]
        rgb = rgb / 255.0

        # Filter invalid points (distance == 0)
        distances = np.linalg.norm(pcd, axis=1)
        mask = distances > 0.0
        pcd = pcd[mask]
        rgb = rgb[mask]

        print(f"[Camera {args.cam_id}] Valid points after filtering: {len(pcd)}")

        # Load and apply calibration transform (camera -> robot base)
        calib_path = f"/home/ksaha/Research/ModelBasedPlanning/real_world_visual_planning/data/calibration_results/cam{args.cam_id}_calibration.npz"
        calib_data = np.load(calib_path)
        calibration_transform = calib_data['T']

        print(f"[Camera {args.cam_id}] Applying calibration transform...")
        pcd = transform_pcd(pcd, calibration_transform)

        # Load and apply alignment transform
        if args.cam_id == 0:
            # Camera 0 is the reference frame, no alignment needed
            print(f"[Camera {args.cam_id}] No alignment needed (reference frame)")
        else:
            # Camera 1: apply alignment to camera 0 frame
            alignment_path = "/home/ksaha/Research/ModelBasedPlanning/real_world_visual_planning/data/camera_alignments/cam1_to_cam0.npy"
            alignment_transform = np.load(alignment_path)
            print(f"[Camera {args.cam_id}] Applying alignment transform...")
            pcd = transform_pcd(pcd, alignment_transform)

        # Prepare data to send
        data = {
            'cam_id': args.cam_id,
            'pcd': pcd,
            'rgb': rgb
        }

        print(f"[Camera {args.cam_id}] Sending data via ZMQ...")
        socket.send(pickle.dumps(data))

        print(f"[Camera {args.cam_id}] Complete! Sent {len(pcd)} points")

    finally:
        # Clean up
        k4a.stop()
        socket.close()
        context.term()
        print(f"[Camera {args.cam_id}] Shutdown complete")


if __name__ == '__main__':
    main()
