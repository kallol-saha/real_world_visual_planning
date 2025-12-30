"""
Perception module for dual Azure Kinect cameras.

Provides tools for capturing, processing, and publishing point clouds from
dual Azure Kinect cameras with automatic calibration and alignment.
"""

from .perception_pipeline import (
    run_pipeline_iteration,
    apply_spatial_bounds,
    fps_downsample,
)
from .pipeline_wrapper import PerceptionPipeline

__all__ = [
    'PerceptionPipeline',
    'run_pipeline_iteration',
    'apply_spatial_bounds',
    'fps_downsample',
]
