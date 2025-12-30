"""
Load and visualize point clouds from both cameras together.
Filters points based on X, Y, Z bounds.
"""
import numpy as np
import os
from robo_utils.visualization.plotting import plot_pcd

if __name__ == "__main__":
    # Define bounds for X, Y, Z axes [min, max] for each axis
    # Set to None to disable filtering for that axis
    # bounds = {
    #     'x': [0., 0.8],  # X-axis bounds in meters
    #     'y': [-0.7, 0.7],  # Y-axis bounds in meters
    #     'z': [-0.1, 1.0],  # Z-axis bounds in meters
    # }

    bounds = {
        'x': [0.2, 0.8],  # X-axis bounds in meters
        'y': [-0.3, 0.],  # Y-axis bounds in meters
        'z': [-0.1, 0.25],  # Z-axis bounds in meters
    }
    
    
    # You can also set individual bounds to None to disable filtering:
    # bounds = {
    #     'x': [-0.5, 0.5],
    #     'y': None,  # No filtering on Y axis
    #     'z': [0.0, 1.0],
    # }
    
    output_dir = "data/camera_captures"
    
    # Load camera 0 data
    cam0_pcd_file = os.path.join(output_dir, "camera_0_pcd.npy")
    
    # Load camera 1 data
    cam1_pcd_file = os.path.join(output_dir, "camera_1_pcd.npy")
    
    # Check if files exist
    if not os.path.exists(cam0_pcd_file):
        raise FileNotFoundError(f"Camera 0 point cloud not found: {cam0_pcd_file}")
    if not os.path.exists(cam1_pcd_file):
        raise FileNotFoundError(f"Camera 1 point cloud not found: {cam1_pcd_file}")
    
    # Load data
    print("Loading camera 0 data...")
    cam0_pcd = np.load(cam0_pcd_file)
    print(f"Camera 0: {cam0_pcd.shape[0]} points")
    
    print("Loading camera 1 data...")
    cam1_pcd = np.load(cam1_pcd_file)
    print(f"Camera 1: {cam1_pcd.shape[0]} points")
    
    # Create colors: camera_0 in red, camera_1 in blue
    cam0_color = np.array([1.0, 0.0, 0.0])  # Red
    cam1_color = np.array([0.0, 0.0, 1.0])  # Blue
    
    cam0_colors = np.tile(cam0_color, (cam0_pcd.shape[0], 1))
    cam1_colors = np.tile(cam1_color, (cam1_pcd.shape[0], 1))
    
    # Combine point clouds
    combined_pcd = np.vstack([cam0_pcd, cam1_pcd])
    combined_colors = np.vstack([cam0_colors, cam1_colors])
    
    print(f"\nCombined point cloud: {combined_pcd.shape[0]} points")
    print(f"Camera 0 (red): {cam0_pcd.shape[0]} points")
    print(f"Camera 1 (blue): {cam1_pcd.shape[0]} points")
    
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
    filtered_colors = combined_colors[mask]
    
    print(f"\nFiltered point cloud: {filtered_pcd.shape[0]} points (removed {combined_pcd.shape[0] - filtered_pcd.shape[0]} points)")
    
    # Save filtered/cropped PCD files for alignment
    filtered_cam0_mask = mask[:cam0_pcd.shape[0]]  # Mask for camera 0 points
    filtered_cam1_mask = mask[cam0_pcd.shape[0]:]  # Mask for camera 1 points
    
    filtered_cam0_pcd = cam0_pcd[filtered_cam0_mask]
    filtered_cam1_pcd = cam1_pcd[filtered_cam1_mask]
    
    cropped_output_dir = os.path.join(output_dir, "cropped")
    os.makedirs(cropped_output_dir, exist_ok=True)
    
    cam0_cropped_file = os.path.join(cropped_output_dir, "camera_0_cropped_pcd.npy")
    cam1_cropped_file = os.path.join(cropped_output_dir, "camera_1_cropped_pcd.npy")
    
    np.save(cam0_cropped_file, filtered_cam0_pcd)
    np.save(cam1_cropped_file, filtered_cam1_pcd)
    
    print(f"\nSaved cropped point clouds:")
    print(f"  Camera 0: {cam0_cropped_file} ({filtered_cam0_pcd.shape[0]} points)")
    print(f"  Camera 1: {cam1_cropped_file} ({filtered_cam1_pcd.shape[0]} points)")
    
    # Visualize filtered point cloud
    print("\nVisualizing filtered combined point clouds from both cameras...")
    print("  Camera 0: Red")
    print("  Camera 1: Blue")
    plot_pcd(filtered_pcd, colors=filtered_colors, base_frame=True)
    
    print("Visualization complete!")

