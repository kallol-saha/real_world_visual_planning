# FrankaPanda Package Usage Guide

## Installation

To install the frankapanda package in development mode:

```bash
pip install -e .
```

This makes the package importable from anywhere in your Python environment.

## Package Structure

```
frankapanda/
├── __init__.py                    # Main package exports
├── perception/                    # Perception module
│   ├── __init__.py
│   ├── capture_single_camera.py   # Single camera capture script
│   ├── perception_pipeline.py     # Main pipeline orchestrator
│   ├── pipeline_wrapper.py        # PerceptionPipeline class
│   └── README.md                  # Perception documentation
├── controller.py                  # FrankaPandaController class
└── calibration/                   # Calibration tools
    └── ...
```

## Usage Examples

### 1. Import the Package

```python
# Import main components
from frankapanda import FrankaPandaController
from frankapanda.perception import PerceptionPipeline

# Or import submodules
from frankapanda import perception, calibration
```

### 2. Robot Control

```python
from frankapanda import FrankaPandaController

# Initialize controller
controller = FrankaPandaController()

# Get robot state
gripper_pose = controller.get_gripper_pose()
joints = controller.get_robot_joints()

# Move robot
controller.move_to_joints(controller.home_joints)
controller.open_gripper()
controller.close_gripper()

# Move with delta pose
delta = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]  # 5cm in X
controller.move_to_target_pose(delta, num_steps=50)
```

### 3. Perception System

#### Method A: Using the PerceptionPipeline Class (Recommended)

```python
from frankapanda.perception import PerceptionPipeline

# Create perception client
perception = PerceptionPipeline(publish_port=5556, timeout_ms=10000)

# Get single point cloud (blocking)
pcd, rgb = perception.get_point_cloud()

# Get point cloud with metadata
data = perception.get_point_cloud_dict()
# data contains: pcd, rgb, num_points, bounds

# Use context manager for automatic cleanup
with PerceptionPipeline() as perception:
    pcd, rgb = perception.get_point_cloud()
    # ... use point cloud ...
# Automatically cleaned up
```

#### Method B: Continuous Listening (Non-blocking)

```python
from frankapanda.perception import PerceptionPipeline

perception = PerceptionPipeline()

# Start background listener
perception.start_continuous_listener()

# In your control loop
while True:
    # Get latest point cloud (non-blocking, returns None if not ready)
    result = perception.get_latest()
    if result is not None:
        pcd, rgb = result
        # ... use point cloud ...

    # ... other operations ...

# Stop when done
perception.stop_continuous_listener()
perception.close()
```

### 4. Combined Perception + Control

```python
from frankapanda import FrankaPandaController
from frankapanda.perception import PerceptionPipeline

# Initialize both systems
controller = FrankaPandaController()
perception = PerceptionPipeline()

# Get point cloud
pcd, rgb = perception.get_point_cloud()

# Use point cloud to plan motion
# ... your planning code ...

# Execute motion
controller.move_to_target_pose(planned_delta, num_steps=100)

# Cleanup
perception.close()
```

## Running the Perception Pipeline

The perception pipeline must be running separately before you can use the `PerceptionPipeline` class.

### Single-shot mode (runs once and exits)
```bash
python -m frankapanda.perception.perception_pipeline
```

### Continuous mode (recommended for real-time applications)
```bash
# Run at 1 Hz (default)
python -m frankapanda.perception.perception_pipeline --continuous

# Run at 0.2 Hz (recommended for ~5 second pipeline time)
python -m frankapanda.perception.perception_pipeline --continuous --rate 0.2

# With custom settings
python -m frankapanda.perception.perception_pipeline \
    --continuous \
    --rate 0.2 \
    --num_points 4096 \
    --save
```

### Command-line shortcuts (after pip install -e .)
```bash
# These work from anywhere after installation
frankapanda-perception --continuous --rate 0.2
frankapanda-capture --cam_id 0 --zmq_port 5555
```

## Demo Script

A minimal demo script is provided that demonstrates both perception and control:

```bash
# Terminal 1: Start perception pipeline
python -m frankapanda.perception.perception_pipeline --continuous

# Terminal 2: Run demo
python demo_perception_control.py
```

The demo will:
1. Initialize robot controller
2. Connect to perception pipeline
3. Capture and visualize point cloud
4. Move robot to home position
5. Perform small movements
6. Capture another point cloud
7. Compare before/after

## Module-specific Documentation

- **Perception:** See `frankapanda/perception/README.md` for detailed perception pipeline documentation
- **Calibration:** See `frankapanda/calibration/` for camera calibration tools

## Common Patterns

### Pattern 1: Reactive Control Loop
```python
from frankapanda import FrankaPandaController
from frankapanda.perception import PerceptionPipeline

controller = FrankaPandaController()
perception = PerceptionPipeline()

perception.start_continuous_listener()

while True:
    # Get latest observation (non-blocking)
    result = perception.get_latest()
    if result is None:
        continue

    pcd, rgb = result

    # Process point cloud and make decision
    action = your_policy(pcd, rgb)

    # Execute action
    controller.move_to_target_pose(action, num_steps=10)

perception.stop_continuous_listener()
```

### Pattern 2: Sense-Plan-Act
```python
from frankapanda import FrankaPandaController
from frankapanda.perception import PerceptionPipeline

controller = FrankaPandaController()
perception = PerceptionPipeline()

# Sense
pcd, rgb = perception.get_point_cloud()

# Plan
trajectory = your_planner(pcd, rgb, controller.get_gripper_pose())

# Act
for waypoint in trajectory:
    controller.move_to_target_pose(waypoint, num_steps=50)

perception.close()
```

## Troubleshooting

### Import Error
```python
ImportError: No module named 'frankapanda'
```
**Solution:** Install the package with `pip install -e .` from the project root.

### Perception Timeout
```python
TimeoutError: No data received within 10000ms
```
**Solution:** Make sure the perception pipeline is running:
```bash
python -m frankapanda.perception.perception_pipeline --continuous
```

### Robot Not Responding
**Solution:**
1. Check robot is powered on
2. Verify `charmander.yml` config file exists
3. Check deoxys is properly installed

## Advanced: Custom Perception Processing

You can import and use the low-level perception functions:

```python
from frankapanda.perception import apply_spatial_bounds, fps_downsample
import numpy as np

# Apply custom bounds
pcd_filtered, rgb_filtered = apply_spatial_bounds(
    pcd, rgb,
    bounds={'x': [0.3, 0.7], 'y': [-0.2, 0.2], 'z': [0.0, 0.5]}
)

# Custom downsampling
pcd_small, rgb_small = fps_downsample(pcd_filtered, rgb_filtered, num_points=1024)
```
