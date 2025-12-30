"""
Load point clouds from both cameras, apply alignment transformations,
and visualize them together.
Filters points based on X, Y, Z bounds.
"""
import numpy as np
import os
from robo_utils.conversion_utils import transform_pcd
from robo_utils.visualization.plotting import plot_pcd

if __name__ == "__main__":
    # Define bounds for X, Y, Z axes [min, max] for each axis
    # Set to None to disable filtering for that axis
    bounds = {
        'x': [0.2, 0.8],  # X-axis bounds in meters
        'y': [-0.5, 0.5],  # Y-axis bounds in meters
        'z': [-0.1, 0.7],  # Z-axis bounds in meters
    }
    
    # You can also set individual bounds to None to disable filtering:
    # bounds = {
    #     'x': [-0.5, 0.5],
    #     'y': None,  # No filtering on Y axis
    #     'z': [0.0, 1.0],
    # }
    
    output_dir = "data/camera_captures"
    alignment_dir = os.path.join("data", "camera_alignments")
    
    # Load original point clouds (before cropping)
    cam0_pcd_file = os.path.join(output_dir, "camera_0_pcd.npy")
    cam0_rgb_file = os.path.join(output_dir, "camera_0_rgb.npy")
    cam1_pcd_file = os.path.join(output_dir, "camera_1_pcd.npy")
    cam1_rgb_file = os.path.join(output_dir, "camera_1_rgb.npy")
    
    # Load transformation matrices
    cam0_to_cam1_file = os.path.join(alignment_dir, "cam0_to_cam1.npy")
    cam1_to_cam0_file = os.path.join(alignment_dir, "cam1_to_cam0.npy")
    
    # Check if files exist
    if not os.path.exists(cam0_pcd_file):
        raise FileNotFoundError(f"Camera 0 point cloud not found: {cam0_pcd_file}")
    if not os.path.exists(cam0_rgb_file):
        raise FileNotFoundError(f"Camera 0 RGB not found: {cam0_rgb_file}")
    if not os.path.exists(cam1_pcd_file):
        raise FileNotFoundError(f"Camera 1 point cloud not found: {cam1_pcd_file}")
    if not os.path.exists(cam1_rgb_file):
        raise FileNotFoundError(f"Camera 1 RGB not found: {cam1_rgb_file}")
    if not os.path.exists(cam0_to_cam1_file):
        raise FileNotFoundError(f"Transformation cam0_to_cam1 not found: {cam0_to_cam1_file}")
    if not os.path.exists(cam1_to_cam0_file):
        raise FileNotFoundError(f"Transformation cam1_to_cam0 not found: {cam1_to_cam0_file}")
    
    # Load data
    print("Loading camera 0 point cloud and RGB...")
    cam0_pcd = np.load(cam0_pcd_file)
    cam0_rgb = np.load(cam0_rgb_file)
    print(f"Camera 0: {cam0_pcd.shape[0]} points, RGB shape: {cam0_rgb.shape}")
    
    print("Loading camera 1 point cloud and RGB...")
    cam1_pcd = np.load(cam1_pcd_file)
    cam1_rgb = np.load(cam1_rgb_file)
    print(f"Camera 1: {cam1_pcd.shape[0]} points, RGB shape: {cam1_rgb.shape}")
    
    # Ensure RGB values are in [0, 1] range (they should already be normalized)
    if cam0_rgb.max() > 1.0:
        cam0_rgb = cam0_rgb / 255.0
    if cam1_rgb.max() > 1.0:
        cam1_rgb = cam1_rgb / 255.0
    
    # Load transformations
    print("\nLoading transformations...")
    cam0_to_cam1 = np.load(cam0_to_cam1_file)
    cam1_to_cam0 = np.load(cam1_to_cam0_file)
    print(f"cam0_to_cam1 shape: {cam0_to_cam1.shape}")
    print(f"cam1_to_cam0 shape: {cam1_to_cam0.shape}")
    
    # cam1_to_cam0 transforms cam1's PCD to cam0's coordinate frame
    
    print("  Applying cam1_to_cam0 to camera 1 point cloud...")
    cam1_aligned = transform_pcd(cam1_pcd, cam1_to_cam0)
    
    # RGB colors stay with their respective points (no transformation needed)
    # Combine aligned point clouds and RGB
    combined_pcd = np.vstack([cam0_pcd, cam1_aligned])
    combined_rgb = np.vstack([cam0_rgb, cam1_rgb])
    
    print(f"\nAligned point clouds:")
    print(f"  Camera 0 (in its own frame): {cam0_pcd.shape[0]} points")
    print(f"  Camera 1 (transformed to cam0 frame): {cam1_aligned.shape[0]} points")
    print(f"  Combined: {combined_pcd.shape[0]} points")
    
    # Apply bounds filtering
    print(f"\nApplying bounds filtering:")
    print(f"  X bounds: {bounds['x']}")
    print(f"  Y bounds: {bounds['y']}")
    print(f"  Z bounds: {bounds['z']}")
    
    mask = np.ones(combined_pcd.shape[0], dtype=bool)
    
    if bounds['x'] is not None:
        x_mask = (combined_pcd[:, 0] >= bounds['x'][0]) & (combined_pcd[:, 0] <= bounds['x'][1])
        mask = mask & x_mask
        print(f"  X filtering: {np.sum(x_mask)}/{len(x_mask)} points remain")
    
    if bounds['y'] is not None:
        y_mask = (combined_pcd[:, 1] >= bounds['y'][0]) & (combined_pcd[:, 1] <= bounds['y'][1])
        mask = mask & y_mask
        print(f"  Y filtering: {np.sum(y_mask)}/{len(y_mask)} points remain")
    
    if bounds['z'] is not None:
        z_mask = (combined_pcd[:, 2] >= bounds['z'][0]) & (combined_pcd[:, 2] <= bounds['z'][1])
        mask = mask & z_mask
        print(f"  Z filtering: {np.sum(z_mask)}/{len(z_mask)} points remain")
    
    # Apply mask
    filtered_pcd = combined_pcd[mask]
    filtered_rgb = combined_rgb[mask]
    
    print(f"\nFiltered point cloud: {filtered_pcd.shape[0]} points (removed {combined_pcd.shape[0] - filtered_pcd.shape[0]} points)")
    
    # Visualize filtered aligned point clouds with RGB colors
    print("\nVisualizing filtered aligned point clouds...")
    print("  Using RGB colors from captured images")
    print("  Camera 0: in its own frame")
    print("  Camera 1: transformed to cam0 frame")
    plot_pcd(filtered_pcd, colors=filtered_rgb, base_frame=True)
    
    print("Visualization complete!")

