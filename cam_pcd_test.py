from pyk4a import PyK4A, connected_device_count
from pyk4a.pyk4a import PyK4A as PyK4A_device
import cv2
import numpy as np
import time
import json
from robo_utils.visualization.plotting import plot_pcd
from robo_utils.conversion_utils import transform_pcd

def estimate_intrinsics(points_3d, depth):
    """
    Estimate camera intrinsics matrix from 3D point cloud and depth.

    Args:
        points_3d: (H, W, 3) 3D points in camera frame
        depth: (H, W) depth map

    Returns:
        K_est: (3, 3) estimated camera intrinsics matrix
    """
    H, W = depth.shape
    X = points_3d[..., 0].ravel()
    Y = points_3d[..., 1].ravel()
    Z = points_3d[..., 2].ravel()

    # Pixel coordinates
    u = np.tile(np.arange(W), H)
    v = np.repeat(np.arange(H), W)

    # Remove points with zero depth to avoid division by zero
    valid = Z > 1e-6
    X, Y, Z = X[valid], Y[valid], Z[valid]
    u, v = u[valid], v[valid]

    # Solve fx from u = fx * X / Z + cx
    A_u = np.stack([X / Z, np.ones_like(X)], axis=1)
    fx_cx, _, _, _ = np.linalg.lstsq(A_u, u, rcond=None)
    fx, cx = fx_cx

    # Solve fy from v = fy * Y / Z + cy
    A_v = np.stack([Y / Z, np.ones_like(Y)], axis=1)
    fy_cy, _, _, _ = np.linalg.lstsq(A_v, v, rcond=None)
    fy, cy = fy_cy

    # Assemble intrinsic matrix
    K_est = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return K_est

def get_camera_intrinsics(calibration_file: str, camera_type: str = "depth") -> np.ndarray:
    """
    Extract the 3x3 camera intrinsics matrix from Azure Kinect calibration file.
    
    Args:
        calibration_file: Path to the calibration JSON file
        camera_type: Either "depth" or "color" to specify which camera's intrinsics to extract
    
    Returns:
        np.ndarray: 3x3 camera intrinsics matrix
    """
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    
    # Select the appropriate camera data
    camera_data = None
    for camera in calib_data["CalibrationInformation"]["Cameras"]:
        if (camera_type == "depth" and camera["Purpose"] == "CALIBRATION_CameraPurposeDepth") or \
           (camera_type == "color" and camera["Purpose"] == "CALIBRATION_CameraPurposePhotoVideo"):
            camera_data = camera
            break
    
    if camera_data is None:
        raise ValueError(f"Camera type {camera_type} not found in calibration data")
    
    # Get the parameters
    params = camera_data["Intrinsics"]["ModelParameters"]
    width = camera_data["SensorWidth"]
    height = camera_data["SensorHeight"]
    
    # The first four parameters are fx, fy, cx, cy in normalized coordinates
    fx = params[0] * width  # Denormalize by multiplying with sensor width
    fy = params[1] * height # Denormalize by multiplying with sensor height
    cx = params[2] * width  # Denormalize by multiplying with sensor width
    cy = params[3] * height # Denormalize by multiplying with sensor height
    
    # Create the 3x3 intrinsics matrix
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return intrinsics

def get_kinect_rgbd_frame(device: PyK4A_device, visualize=False):
    """
    Capture an IR frame from the Kinect camera.
    """
    # Capture an IR frame
    rgb_frame = None
    capture = None
    for i in range(20):
        try:
            device.get_capture()
            capture = device.get_capture()
            if capture is not None:
                ir_frame = capture.ir

                # depth_frame = capture.depth
                # ---
                depth_frame = capture.transformed_depth
                # ---
                
                # cv2.imshow('IR', ir_frame)
                rgb_frame = capture.color
                gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                gray_frame = np.clip(gray_frame, 0, 5e3) / 5e3  # Clip and normalize
                # cv2.imshow('color', rgb_frame)
                
                ir_frame_norm = np.clip(ir_frame, 0, 5e3) / 5e3  # Clip and normalize
                pcd_frame = capture.transformed_depth_point_cloud
                # print(pcd_frame.shape, ir_frame.shape)
                # print("successful capture")
                return ir_frame, rgb_frame, ir_frame_norm, pcd_frame, depth_frame
        except:
            time.sleep(0.1)
            # print("Failed to capture IR frame.")
    else:
        # print("Failed to capture IR frame after 20 attempts.")
        return None

connected_devices = connected_device_count()

cam_id = 1
calibration_file = f"data/calibration_results/cam{cam_id}_calibration.npz"
content = np.load(calibration_file)
T = content['T']


k4a = PyK4A(device_id=cam_id)
k4a.start()

ir_frame, rgb_frame, ir_frame_norm, pcd_frame, depth_frame = get_kinect_rgbd_frame(k4a)

distance_threshold = 1.

pcd = pcd_frame.reshape(-1, 3)
rgb = rgb_frame[..., [2, 1, 0]].reshape(-1, 3) / 255.0
pcd = pcd / 1000.0      # convert to meters
distances = np.linalg.norm(pcd, axis=1)

mask = (distances < distance_threshold) & (distances > 0.)
pcd = pcd[mask]
rgb = rgb[mask]
plot_pcd(pcd, rgb, base_frame=True) #, frame_size=20.0)

pcd = transform_pcd(pcd, T)
plot_pcd(pcd, rgb, base_frame=True) #, frame_size=20.0)

print("")


# k4a.save_calibration_json("calibration_0.json")

# Get intrinsics matrices
# K = get_camera_intrinsics("calibration_0.json", "color")

# color_intrinsics = get_camera_intrinsics("calibration_0.json", "color")


# print("Depth camera intrinsics:")
# print(depth_intrinsics)
# print("\nColor camera intrinsics:")
# print(color_intrinsics)

# ir_frame, rgb_frame, ir_frame_norm, pcd_frame, depth_frame = get_kinect_rgbd_frame(k4a)

# K_est = estimate_intrinsics(pcd_frame, depth_frame)
# K_inv = np.linalg.inv(K_est)

# # Backprojecting depth:
# H, W = depth_frame.shape
# u = np.arange(W)
# v = np.arange(H)
# u, v = np.meshgrid(u, v)

# # Backprojecting depth:
# x = (u - K[0, 2]) * depth_frame / K[0, 0]
# y = (v - K[1, 2]) * depth_frame / K[1, 1]
# z = depth_frame

# # Normalize the depth frame according to the max depth
# depth_frame = depth_frame / np.max(depth_frame) * 255
# depth_frame = depth_frame.astype(np.uint8)

# Normalize the ir frame according to the max ir
ir_frame = ir_frame / np.max(ir_frame) * 255
ir_frame = ir_frame.astype(np.uint8)

cv2.imwrite("ir_frame.png", ir_frame)
cv2.imwrite("rgb_frame.png", rgb_frame[..., :-1])
# cv2.imwrite("ir_frame_norm.png", ir_frame_norm)
# cv2.imwrite("pcd_frame.png", pcd_frame)
# cv2.imwrite("depth_frame.png", depth_frame)

print("")