"""
Compute alignment transformation from camera 1 to camera 0 using ICP.
Saves both cam1_to_cam0 and cam0_to_cam1 (inverse) transformations.
"""
import numpy as np
import os
import open3d as o3d
from robo_utils.conversion_utils import invert_transformation

def compute_icp_alignment(source_pcd, target_pcd, threshold=0.01, visualize=False):
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
    
    # Estimate normals for point-to-plane ICP
    print("Estimating normals...")
    source_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Initial transformation (identity)
    trans_init = np.identity(4)
    
    # Run ICP
    print(f"Running ICP with threshold={threshold}...")
    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    print("\nICP Registration Result:")
    print(reg_result)
    print(f"\nTransformation matrix (cam1_to_cam0):")
    print(reg_result.transformation)
    print(f"Fitness: {reg_result.fitness}")
    print(f"Inlier RMSE: {reg_result.inlier_rmse}")
    
    if visualize:
        source_temp = source_pcd.copy()
        target_temp = target_pcd.copy()
        source_temp.paint_uniform_color([1, 0, 0])  # Red for source
        target_temp.paint_uniform_color([0, 0, 1])  # Blue for target
        source_temp.transform(reg_result.transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])
    
    return reg_result.transformation

if __name__ == "__main__":
    output_dir = "data/camera_captures"
    cropped_dir = os.path.join(output_dir, "cropped")
    
    # Load cropped point clouds
    cam0_cropped_file = os.path.join(cropped_dir, "camera_0_cropped_pcd.npy")
    cam1_cropped_file = os.path.join(cropped_dir, "camera_1_cropped_pcd.npy")
    
    if not os.path.exists(cam0_cropped_file):
        raise FileNotFoundError(f"Camera 0 cropped PCD not found: {cam0_cropped_file}")
    if not os.path.exists(cam1_cropped_file):
        raise FileNotFoundError(f"Camera 1 cropped PCD not found: {cam1_cropped_file}")
    
    print("Loading cropped point clouds...")
    cam0_pcd_array = np.load(cam0_cropped_file)
    cam1_pcd_array = np.load(cam1_cropped_file)
    
    print(f"Camera 0: {cam0_pcd_array.shape[0]} points")
    print(f"Camera 1: {cam1_pcd_array.shape[0]} points")
    
    # Convert to Open3D point clouds
    cam0_pcd = o3d.geometry.PointCloud()
    cam0_pcd.points = o3d.utility.Vector3dVector(cam0_pcd_array)
    
    cam1_pcd = o3d.geometry.PointCloud()
    cam1_pcd.points = o3d.utility.Vector3dVector(cam1_pcd_array)
    
    # Compute alignment: camera 1 to camera 0
    # This means: transform cam1_pcd to align with cam0_pcd
    print("\n" + "="*60)
    print("Computing alignment: Camera 1 -> Camera 0")
    print("="*60)
    cam1_to_cam0 = compute_icp_alignment(
        cam1_pcd, cam0_pcd, threshold=0.01, visualize=False
    )
    
    # Compute inverse: camera 0 to camera 1
    print("\n" + "="*60)
    print("Computing inverse transformation: Camera 0 -> Camera 1")
    print("="*60)
    cam0_to_cam1 = invert_transformation(cam1_to_cam0)
    print(f"\nTransformation matrix (cam0_to_cam1):")
    print(cam0_to_cam1)
    
    # Save transformations
    alignment_dir = os.path.join("data", "camera_alignments")
    os.makedirs(alignment_dir, exist_ok=True)
    
    cam1_to_cam0_file = os.path.join(alignment_dir, "cam1_to_cam0.npy")
    cam0_to_cam1_file = os.path.join(alignment_dir, "cam0_to_cam1.npy")
    
    np.save(cam1_to_cam0_file, cam1_to_cam0)
    np.save(cam0_to_cam1_file, cam0_to_cam1)
    
    print("\n" + "="*60)
    print("Saved transformations:")
    print(f"  cam1_to_cam0: {cam1_to_cam0_file}")
    print(f"  cam0_to_cam1: {cam0_to_cam1_file}")
    print("="*60)

