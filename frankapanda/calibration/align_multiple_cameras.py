import open3d as o3d
import numpy as np
import copy

from real_world.pcd_obs_env_dino import PCDObsEnv
camera_indices = [0, 1]
env1 = PCDObsEnv(
    # camera_alignments=False,
    dino_class="drawer",
    camera_indices=[0],
    voxel_size=0.002,
    use_sam=True)
env2 = PCDObsEnv(
    # camera_alignments=False,
    dino_class="drawer",
    camera_indices=[1],
    voxel_size=0.002,
    use_sam=True)

colors = [[0, 0, 1], [1, 0, 0]]

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def compute_align_to_target(target_pcd, other_pcds, threshold=0.01, visualize=False):
    """
    Compute alignments from other_pcds to base_pcd
    
    Input:
        target_pcd: Open3D point cloud
        other_pcds: dict of Open3D point clouds {cam_id: pcd}
    Return:
        dict of transforms {cam_id: transforms}
    """
    transforms = {}
    for cam_id, source in other_pcds.items():        
        print(f":: Aligning camera {cam_id} with target pcd")
        print(":: Apply point-to-point ICP")
        trans_init = np.identity(4)
        # reg_p2p = o3d.pipelines.registration.registration_icp(
        #     source, target, threshold, trans_init,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        if visualize:
            draw_registration_result(source, target_pcd, reg_p2p.transformation)

        transforms[cam_id] = reg_p2p.transformation.copy()
    
    # For base camer, set to identity
    transforms[base_cam_id] = np.identity(4)
    
    return transforms

def align_pcds(pcds, transforms):
    """
    Align point clouds using transforms
    
    Input:
        pcds: dict of Open3D point clouds {cam_id: pcd}
        transforms: dict of transforms {cam_id: transforms}.
    Return:
        Open3D point cloud
    """
    transformed_pcds = o3d.geometry.PointCloud()
    for cam_id in env.camera_indices:
        transformed_pcd = pcds[cam_id].transform(transforms[cam_id])
        # transformed_pcd.paint_uniform_color(colors[cam_id])
        transformed_pcds += transformed_pcd
    
    return transformed_pcds


base_cam_id = 1    # Set all other cameras to align with camera 1

# Obtain individual point clouds
pcds = {}
for idx, env in enumerate([env1, env2]):
    pcds[idx] = env.get_pcd(color=True, return_numpy=False)

# Compute alignments
transforms = compute_align_to_target(
    pcds[base_cam_id], pcds, threshold=0.01, visualize=False) # Set to True to visualize pairwise alignment


# before alignment
ori_pcd = o3d.geometry.PointCloud()
for cam_id in camera_indices:
    # pcds[cam_id].paint_uniform_color(colors[cam_id])
    ori_pcd += pcds[cam_id]
o3d.visualization.draw_geometries([ori_pcd])

# Transform all point clouds
camera_aligned_pcds = align_pcds(pcds, transforms)
o3d.visualization.draw_geometries([camera_aligned_pcds])

# Remove color
# camera_aligned_pcds_ = o3d.geometry.PointCloud(camera_aligned_pcds.points)
# camera_aligned_pcds_.normals = camera_aligned_pcds.normals 
# o3d.visualization.draw_geometries([camera_aligned_pcds_])


"""Save the results"""
import os, json
from scipy.spatial.transform import Rotation

this_file_path = os.path.abspath(__file__)
this_file_dir = os.path.dirname(this_file_path)
save_dir = os.path.join(this_file_dir, "alignment_results")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the transform matrices as npz
save_path = os.path.join(save_dir, "camera_alignments.npz")
save_content = {
    str(cam_id): transform for cam_id, transform in transforms.items()
} # Convert cam_id to string to save as npz
np.savez(save_path, **save_content)

# Also save the humann readable transforms as xyz, quat into json
save_content = {}
for cam_id, transform in transforms.items():
    quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
    save_content[cam_id] = {
        "xyz": transform[:3, 3].tolist(),
        "quaternion": quat.tolist()
    }
save_path = os.path.join(save_dir, "camera_alignments.json")
with open(save_path, "w") as f:
    json.dump(save_content, f, indent=2)