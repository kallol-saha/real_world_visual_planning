"""
Capture and save point cloud and RGB data from camera 0.
Applies extrinsics transformation and saves the results.
"""
from pyk4a import PyK4A, connected_device_count
from pyk4a.pyk4a import PyK4A as PyK4A_device
import cv2
import numpy as np
import time
import os
from robo_utils.conversion_utils import transform_pcd

def get_kinect_rgbd_frame(device: PyK4A_device, visualize=False):
    """
    Capture an IR frame from the Kinect camera.
    """
    rgb_frame = None
    capture = None
    for i in range(20):
        try:
            device.get_capture()
            capture = device.get_capture()
            if capture is not None:
                ir_frame = capture.ir
                depth_frame = capture.transformed_depth
                rgb_frame = capture.color
                pcd_frame = capture.transformed_depth_point_cloud
                return ir_frame, rgb_frame, pcd_frame, depth_frame
        except:
            time.sleep(0.1)
    else:
        print("Failed to capture frame after 20 attempts.")
        return None

if __name__ == "__main__":
    cam_id = 0
    
    # Load extrinsics calibration
    calibration_file = f"data/calibration_results/cam{cam_id}_calibration.npz"
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    
    content = np.load(calibration_file)
    T = content['T']
    print(f"Loaded extrinsics transformation for camera {cam_id}")
    print(f"Transformation matrix:\n{T}")
    
    # Initialize camera
    k4a = PyK4A(device_id=cam_id)
    k4a.start()
    print(f"Camera {cam_id} started")
    
    # Capture frame
    result = get_kinect_rgbd_frame(k4a)
    if result is None:
        raise RuntimeError("Failed to capture frame from camera")
    
    ir_frame, rgb_frame, pcd_frame, depth_frame = result
    
    # Process point cloud
    pcd = pcd_frame.reshape(-1, 3)
    rgb = rgb_frame[..., [2, 1, 0]].reshape(-1, 3) / 255.0  # Convert BGR to RGB and normalize
    pcd = pcd / 1000.0  # Convert from mm to meters
    
    # Remove points with distance = 0.0 (invalid points)
    distances = np.linalg.norm(pcd, axis=1)
    mask = distances > 0.0
    pcd = pcd[mask]
    rgb = rgb[mask]
    
    print(f"Point cloud shape before extrinsics: {pcd.shape}")
    
    # Apply extrinsics transformation
    pcd_transformed = transform_pcd(pcd, T)
    
    print(f"Point cloud shape after extrinsics: {pcd_transformed.shape}")
    
    # Save results
    output_dir = "data/camera_captures"
    os.makedirs(output_dir, exist_ok=True)
    
    pcd_file = os.path.join(output_dir, f"camera_{cam_id}_pcd.npy")
    rgb_file = os.path.join(output_dir, f"camera_{cam_id}_rgb.npy")
    
    np.save(pcd_file, pcd_transformed)
    np.save(rgb_file, rgb)
    
    print(f"\nSaved point cloud to: {pcd_file}")
    print(f"Saved RGB colors to: {rgb_file}")
    print(f"Point cloud shape: {pcd_transformed.shape}")
    print(f"RGB shape: {rgb.shape}")
    
    k4a.stop()
    print("Camera stopped")

