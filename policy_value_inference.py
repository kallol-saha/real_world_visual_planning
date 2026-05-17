"""
PolicyValueInference: load and run the sim2real cascaded grasp -> place
policies, with optional value-network scoring, on a real-world point cloud.

The class encapsulates checkpoint loading, sim2real center+scale
normalization, the 4-channel (xyz + target_mask) packing, and the
cascaded forward pass. Pass use_value=False to skip loading and running
the value network entirely.

Typical usage:
    inf = PolicyValueInference(
        grasp_ckpt, place_ckpt, value_ckpt=value_ckpt,
        device="cuda:0", use_value=True,
    )
    out = inf.infer(pcd_np, mask_np)  # pcd_np (N,3), mask_np (N,) in {0,1}
    print(out["grasp_pose"], out["place_pose"], out["value_score"])
"""

import os

import numpy as np
import torch
from tqdm import tqdm

# visplanWM model code, resolved via the .pth installed by
# setup_frankapanda_3dfa.sh into the conda env site-packages.
from models.flowmatch_actor.modeling.policy_grasp_place.denoise_actor_3d_packing import (
    DenoiseActor as GraspPlaceActor,
)
from models.flowmatch_actor.modeling.policy.value_network import ValueNetwork


# Policy ctor kwargs — must match the training config used for the
# sim2real_*_v3 checkpoints (see train_for_shelf_packing_grasp_place.py
# and visplan/action_sampling.py).
_POLICY_KWARGS = dict(
    embedding_dim=120,
    num_attn_heads=8,
    nhist=1,
    num_shared_attn_layers=4,
    relative=False,
    rotation_format="quat_wxyz",
    denoise_timesteps=10,
    denoise_model="rectified_flow",
    lv2_batch_size=1,
    pcd_input_channels=4,
)

_VALUE_KWARGS = dict(
    embedding_dim=120,
    num_attn_heads=8,
    num_shared_attn_layers=4,
)


def _load_checkpoint(checkpoint_path: str, model: torch.nn.Module) -> torch.nn.Module:
    """Mirrors visplan.shelf_packing_base.load_checkpoint_for_eval."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    try:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        if "numpy.core.multiarray.scalar" in str(e):
            blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        else:
            raise
    msn, unx = model.load_state_dict(blob["weight"], strict=False)
    name = os.path.basename(checkpoint_path)
    if msn:
        print(f"  [{name}] missing keys: {len(msn)}")
    if unx:
        print(f"  [{name}] unexpected keys: {len(unx)}")
    if not msn and not unx:
        print(f"  [{name}] all keys matched")
    del blob
    return model


def _center_and_scale(pcd_xyz: torch.Tensor, eps: float = 1e-6):
    """Sim2real per-sample preprocessing (matches sim2real_augment.center_and_scale)."""
    centroid = pcd_xyz.mean(dim=0)
    centered = pcd_xyz - centroid
    radius = centered.norm(dim=-1).max()
    scale = torch.clamp(radius, min=eps)
    return centered / scale, centroid, scale


def _unscale_xyz(xyz_norm: torch.Tensor, centroid: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return xyz_norm * scale + centroid


class PolicyValueInference:
    """
    Cascaded grasp -> place policy plus optional value scoring.

    Args:
        grasp_ckpt: path to grasp policy .pth (grasp_condition_dim=None).
            Pass None to skip loading the grasp model (e.g. when only
            placement sampling is needed); calls into infer() and
            infer_grasps() will then error.
        place_ckpt: path to place policy .pth (grasp_condition_dim=7).
            Pass None to skip loading the place model.
        value_ckpt: path to value network .pth. Required when use_value=True.
        device: target torch device (str or torch.device).
        use_value: if True, load + run the value network; output["value_score"]
            is a float in [0, 1]. If False, value_ckpt is ignored and
            output["value_score"] is None.
    """

    def __init__(
        self,
        grasp_ckpt: str = None,
        place_ckpt: str = None,
        value_ckpt: str = None,
        device="cuda:0",
        use_value: bool = False,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_value = use_value

        if grasp_ckpt is not None:
            print("Building grasp policy")
            grasp_kwargs = dict(_POLICY_KWARGS)
            grasp_kwargs["trajectory_length"] = 1
            grasp_kwargs["grasp_condition_dim"] = None
            self.grasp_model = (
                _load_checkpoint(grasp_ckpt, GraspPlaceActor(**grasp_kwargs))
                .to(self.device)
                .eval()
            )
        else:
            self.grasp_model = None

        if place_ckpt is not None:
            print("Building place policy")
            place_kwargs = dict(_POLICY_KWARGS)
            place_kwargs["trajectory_length"] = 1
            place_kwargs["grasp_condition_dim"] = 7
            self.place_model = (
                _load_checkpoint(place_ckpt, GraspPlaceActor(**place_kwargs))
                .to(self.device)
                .eval()
            )
        else:
            self.place_model = None

        if self.use_value:
            if value_ckpt is None:
                raise ValueError("use_value=True but value_ckpt is None")
            print("Building value network")
            self.value_model = (
                _load_checkpoint(value_ckpt, ValueNetwork(**_VALUE_KWARGS))
                .to(self.device)
                .eval()
            )
        else:
            self.value_model = None

    @torch.no_grad()
    def infer_grasps(self, pcd_np: np.ndarray, mask_np: np.ndarray, num_grasps: int = 5):
        """
        Run the grasp policy num_grasps times with independent denoising noise
        (batched in a single forward), return a list of (7,) grasp poses in
        robot-base frame. Place + value are NOT called.
        """
        pcd_t = torch.from_numpy(pcd_np).float()
        pcd_norm, centroid, scale = _center_and_scale(pcd_t)
        mask_t = (
            torch.from_numpy(mask_np.astype(np.float32))
            .to(pcd_norm.dtype)
            .unsqueeze(-1)
        )
        pcd_masked = torch.cat([pcd_norm, mask_t], dim=-1).unsqueeze(0)  # (1, N, 4)
        pcd_batched = pcd_masked.expand(num_grasps, -1, -1).contiguous().to(self.device)

        grasp_out = self.grasp_model(
            gt_trajectory=None,
            pcd=pcd_batched,
            proprioception=None,
            run_inference=True,
        )                                                              # (K, 1, 8)
        grasps = []
        for i in range(num_grasps):
            norm = grasp_out[i, 0].cpu()
            pos = _unscale_xyz(norm[:3], centroid, scale)
            grasps.append(torch.cat([pos, norm[3:7]]))                 # (7,) wxyz
        return grasps

    @torch.no_grad()
    def infer_placements(
        self,
        pcd_np: np.ndarray,
        mask_np: np.ndarray,
        grasp_pose: np.ndarray,
        num_placements: int = 32,
        chunk_size: int = 4,
    ):
        """
        Run the place policy num_placements times conditioned on a fixed
        grasp pose (xyz+quat_wxyz, robot-base frame). Calls are split into
        chunks of size chunk_size to bound peak attention memory; outputs
        are concatenated. Returns a list of (7,) place poses in robot-base
        frame.

        The grasp pose's xyz is normalized with the current pcd's centroid/
        scale before being passed as grasp_cond, matching the training-time
        convention (see infer() above).
        """
        pcd_t = torch.from_numpy(pcd_np).float()
        pcd_norm, centroid, scale = _center_and_scale(pcd_t)
        mask_t = (
            torch.from_numpy(mask_np.astype(np.float32))
            .to(pcd_norm.dtype)
            .unsqueeze(-1)
        )
        pcd_masked = torch.cat([pcd_norm, mask_t], dim=-1).unsqueeze(0).to(self.device)  # (1, N, 4)

        if isinstance(grasp_pose, np.ndarray):
            grasp_t = torch.from_numpy(grasp_pose).float()
        elif isinstance(grasp_pose, torch.Tensor):
            grasp_t = grasp_pose.detach().cpu().float()
        else:
            grasp_t = torch.tensor(grasp_pose, dtype=torch.float32)

        grasp_pos_norm = (grasp_t[:3] - centroid) / scale
        grasp_cond_one = torch.cat([grasp_pos_norm, grasp_t[3:7]]).unsqueeze(0).to(self.device)  # (1, 7)

        placements = []
        num_chunks = (num_placements + chunk_size - 1) // chunk_size
        pbar = tqdm(total=num_placements, desc="placement inference", unit="sample")
        remaining = num_placements
        for _ in range(num_chunks):
            k = min(chunk_size, remaining)
            pcd_chunk = pcd_masked.expand(k, -1, -1).contiguous()
            grasp_cond_chunk = grasp_cond_one.expand(k, -1).contiguous()
            place_out = self.place_model(
                gt_trajectory=None,
                pcd=pcd_chunk,
                proprioception=None,
                run_inference=True,
                grasp_cond=grasp_cond_chunk,
            )                                                              # (k, 1, 8)
            for i in range(k):
                norm = place_out[i, 0].cpu()
                pos = _unscale_xyz(norm[:3], centroid, scale)
                placements.append(torch.cat([pos, norm[3:7]]))             # (7,) wxyz
            del place_out, pcd_chunk, grasp_cond_chunk
            torch.cuda.empty_cache()
            remaining -= k
            pbar.update(k)
        pbar.close()
        return placements

    @torch.no_grad()
    def infer(self, pcd_np: np.ndarray, mask_np: np.ndarray) -> dict:
        """
        Run cascaded grasp -> place (and optional value scoring) on one scene.

        Args:
            pcd_np: (N, 3) float numpy array in robot-base frame.
            mask_np: (N,) numpy array in {0, 1} marking target-object points.

        Returns:
            dict with:
                "grasp_pose":      (7,) torch.Tensor — xyz + quat_wxyz, robot frame.
                "place_pose":      (7,) torch.Tensor — xyz + quat_wxyz, robot frame.
                "grasp_pose_norm": (8,) torch.Tensor — raw policy output (includes gripper).
                "place_pose_norm": (8,) torch.Tensor.
                "value_score":     float in [0, 1] if use_value else None.
                "centroid":        (3,) torch.Tensor — normalization centroid.
                "scale":           ()  torch.Tensor — normalization scale.
        """
        pcd_t = torch.from_numpy(pcd_np).float()                       # (N, 3) cpu
        pcd_norm, centroid, scale = _center_and_scale(pcd_t)

        mask_t = (
            torch.from_numpy(mask_np.astype(np.float32))
            .to(pcd_norm.dtype)
            .unsqueeze(-1)
        )                                                              # (N, 1) cpu
        pcd_masked_dev = (
            torch.cat([pcd_norm, mask_t], dim=-1)
            .unsqueeze(0)
            .to(self.device)
        )                                                              # (1, N, 4)

        grasp_out = self.grasp_model(
            gt_trajectory=None,
            pcd=pcd_masked_dev,
            proprioception=None,
            run_inference=True,
        )                                                              # (1, 1, 8)
        grasp_norm = grasp_out[0, 0].cpu()                             # (8,) pos+quat+gripper
        grasp_pos = _unscale_xyz(grasp_norm[:3], centroid, scale)
        grasp_pose = torch.cat([grasp_pos, grasp_norm[3:7]])           # (7,)

        grasp_cond_dev = (
            torch.cat([grasp_norm[:3], grasp_norm[3:7]])
            .unsqueeze(0)
            .to(self.device)
        )                                                              # (1, 7)
        place_out = self.place_model(
            gt_trajectory=None,
            pcd=pcd_masked_dev,
            proprioception=None,
            run_inference=True,
            grasp_cond=grasp_cond_dev,
        )                                                              # (1, 1, 8)
        place_norm = place_out[0, 0].cpu()
        place_pos = _unscale_xyz(place_norm[:3], centroid, scale)
        place_pose = torch.cat([place_pos, place_norm[3:7]])           # (7,)

        value_score = None
        if self.use_value:
            pcd_xyz_dev = pcd_norm.unsqueeze(0).to(self.device)        # (1, N, 3)
            actions_dev = (
                torch.stack([grasp_norm, place_norm], dim=0)
                .unsqueeze(0)
                .to(self.device)
            )                                                          # (1, 2, 8)
            value_score = float(
                self.value_model(pcd=pcd_xyz_dev, actions=actions_dev).cpu().item()
            )

        return {
            "grasp_pose": grasp_pose,
            "place_pose": place_pose,
            "grasp_pose_norm": grasp_norm,
            "place_pose_norm": place_norm,
            "value_score": value_score,
            "centroid": centroid,
            "scale": scale,
        }

    @torch.no_grad()
    def score_value_batched(
        self,
        pcd_np: np.ndarray,
        grasp_poses,
        place_poses,
        chunk_size: int = 16,
        gripper_open: float = 0.0,
        gripper_close: float = 1.0,
    ):
        """
        Batched value scoring over K (grasp, place) action pairs sharing one
        scene pcd. Each pose is a (7,) wxyz tensor/array in robot-base frame.
        Pcd is centered/scaled once; positions are normalized with that same
        centroid/scale, matching the training-time action convention.

        Args:
            pcd_np: (N, 3) scene point cloud in robot-base frame. NO target
                mask; the value model takes raw xyz.
            grasp_poses: list/array of K (7,) wxyz poses (robot frame).
            place_poses: list/array of K (7,) wxyz poses (robot frame).
            chunk_size: forward pass batch size.
            gripper_open / gripper_close: appended as the 8th channel to the
                grasp / place action vectors (matches the (8,) output of the
                grasp + place policies).

        Returns:
            scores: (K,) torch.Tensor on cpu, each in [0, 1].
        """
        if self.value_model is None:
            raise RuntimeError("score_value_batched called but value_model is None.")
        if len(grasp_poses) != len(place_poses):
            raise ValueError(
                f"grasp_poses ({len(grasp_poses)}) and place_poses "
                f"({len(place_poses)}) must have the same length."
            )

        K = len(grasp_poses)
        if K == 0:
            return torch.zeros((0,), dtype=torch.float32)

        # Center + scale pcd once. Value model takes raw xyz (no mask channel).
        pcd_t = torch.from_numpy(pcd_np).float()
        pcd_norm, centroid, scale = _center_and_scale(pcd_t)
        pcd_norm_dev = pcd_norm.unsqueeze(0).to(self.device)                # (1, N, 3)

        # Normalize positions; build (K, 2, 8) action stack on cpu.
        def _to_t(p):
            if isinstance(p, torch.Tensor):
                return p.detach().cpu().float()
            if isinstance(p, np.ndarray):
                return torch.from_numpy(p).float()
            return torch.tensor(p, dtype=torch.float32)

        grasp_acts = []
        place_acts = []
        for g, p in zip(grasp_poses, place_poses):
            gt = _to_t(g)
            pt = _to_t(p)
            g_pos = (gt[:3] - centroid) / scale
            p_pos = (pt[:3] - centroid) / scale
            g_act = torch.cat([g_pos, gt[3:7], torch.tensor([gripper_close], dtype=torch.float32)])  # (8,)
            p_act = torch.cat([p_pos, pt[3:7], torch.tensor([gripper_open], dtype=torch.float32)])   # (8,)
            grasp_acts.append(g_act)
            place_acts.append(p_act)
        actions_all = torch.stack(
            [torch.stack([g, p], dim=0) for g, p in zip(grasp_acts, place_acts)],
            dim=0,
        )                                                                    # (K, 2, 8)

        scores = torch.empty((K,), dtype=torch.float32)
        pbar = tqdm(total=K, desc="value scoring", unit="pair")
        for start in range(0, K, chunk_size):
            end = min(start + chunk_size, K)
            k = end - start
            pcd_chunk = pcd_norm_dev.expand(k, -1, -1).contiguous()         # (k, N, 3)
            actions_chunk = actions_all[start:end].to(self.device)          # (k, 2, 8)
            out = self.value_model(pcd=pcd_chunk, actions=actions_chunk)    # (k,) or (k, 1)
            out = out.detach().cpu().float().view(-1)
            scores[start:end] = out[:k]
            del pcd_chunk, actions_chunk, out
            torch.cuda.empty_cache()
            pbar.update(k)
        pbar.close()
        return scores
