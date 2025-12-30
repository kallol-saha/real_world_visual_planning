# Perception Pipeline Documentation

A complete perception pipeline for dual Azure Kinect cameras with ZMQ-based communication.

## Overview

The perception pipeline captures point clouds from two Azure Kinect cameras sequentially, combines them, applies filtering and downsampling, and publishes the result via ZMQ for downstream consumers.

### Why Sequential Capture?

The Azure Kinect SDK (PyK4A) requires that Python processes start and end around each camera's capture. You **cannot** open both cameras in the same Python workflow. The pipeline handles this by:
1. Starting a subprocess for camera 0
2. Capturing and sending data via ZMQ
3. Process exits
4. Starting a subprocess for camera 1
5. Capturing and sending data via ZMQ
6. Process exits
7. Main pipeline combines and processes the data

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  perception_pipeline.py                     │
│                    (Main Orchestrator)                      │
└───────────┬─────────────────────────────────┬───────────────┘
            │                                 │
            │ subprocess.run()                │ subprocess.run()
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│ capture_single_camera │         │ capture_single_camera │
│      --cam_id 0       │         │      --cam_id 1       │
└───────────┬───────────┘         └───────────┬───────────┘
            │                                 │
            │ ZMQ PUSH (port 5557)            │ ZMQ PUSH (port 5557)
            └─────────────┬───────────────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │  Combine Point Clouds   │
            │  Apply Spatial Bounds   │
            │  FPS Downsample (4096)  │
            └─────────────┬───────────┘
                          │
                          │ ZMQ PUB (port 5556)
                          ▼
            ┌─────────────────────────┐
            │ perception_client.py    │
            │  (Your Application)     │
            └─────────────────────────┘
```

## Components

### 1. `capture_single_camera.py`
**Purpose:** Capture from a single Azure Kinect camera

**Process:**
- Opens one camera (cam_id 0 or 1)
- Captures RGBD frame + point cloud
- Filters invalid points (distance = 0)
- Applies calibration transform (camera → robot base)
- Applies alignment transform (cam1 → cam0 frame, identity for cam0)
- Sends via ZMQ (PUSH socket)
- Exits

**Usage:**
```bash
python capture_single_camera.py --cam_id 0 --zmq_port 5557
```

**Arguments:**
- `--cam_id`: Camera ID (0 or 1)
- `--zmq_port`: ZMQ port to send data

### 2. `perception_pipeline.py`
**Purpose:** Main orchestrator that manages the complete pipeline

**Process:**
1. Binds ZMQ receiver (PULL socket) on port 5557
2. Binds ZMQ publisher (PUB socket) on port 5556
3. Launches subprocess for camera 0
4. Receives camera 0 data via ZMQ
5. Launches subprocess for camera 1
6. Receives camera 1 data via ZMQ
7. Combines point clouds
8. Applies spatial bounds: `x: [0.2, 0.8], y: [-0.5, 0.5], z: [-0.1, 0.7]`
9. FPS downsampling to 4096 points
10. Publishes final point cloud via ZMQ

**Modes:**
- **Single-shot mode (default):** Runs once, publishes one point cloud, then exits
- **Continuous mode:** Loops forever, continuously capturing and publishing at specified rate

**Usage:**
```bash
# Single-shot mode (default)
python perception_pipeline.py

# Continuous mode at 1 Hz (default rate)
python perception_pipeline.py --continuous

# Continuous mode at 2 Hz
python perception_pipeline.py --continuous --rate 2.0

# Continuous mode with custom settings
python perception_pipeline.py --continuous --rate 0.5 --num_points 2048

# Save output to file (each iteration in continuous mode)
python perception_pipeline.py --save
```

**Arguments:**
- `--receive_port`: Port to receive camera data (default: 5557)
- `--publish_port`: Port to publish final point cloud (default: 5556)
- `--num_points`: Number of points for FPS downsampling (default: 4096)
- `--save`: Save final point cloud to `data/perception_output/`
- `--continuous`: Run in continuous mode (loops forever)
- `--rate`: Loop rate in Hz for continuous mode (default: 1.0)

### 3. `perception_client_example.py`
**Purpose:** Example consumer that receives the final point cloud

**Usage:**
```bash
# Receive and print statistics
python perception_client_example.py

# Receive and visualize with Open3D
python perception_client_example.py --visualize

# Use custom port and timeout
python perception_client_example.py --port 5557 --timeout 120000
```

**Arguments:**
- `--port`: ZMQ port to receive data (default: 5556)
- `--visualize`: Visualize point cloud with Open3D
- `--timeout`: Timeout in milliseconds (default: 60000)

## Prerequisites

### Required Calibration Files

The pipeline expects the following calibration files to exist:

```
data/
├── calibration_results/
│   ├── cam0_calibration.npz  (4x4 transform: camera 0 → robot base)
│   └── cam1_calibration.npz  (4x4 transform: camera 1 → robot base)
└── camera_alignments/
    └── cam1_to_cam0.npy      (4x4 transform: camera 1 → camera 0)
```

These are generated by:
- `frankapanda/calibration/solve_calibration.py` → generates cam{0,1}_calibration.npz
- `align_cameras.py` → generates cam1_to_cam0.npy

### Python Dependencies

```bash
pip install numpy pyk4a zmq pickle-mixin open3d
```

## Running the Pipeline

### Method 1: Single-Shot Mode (Two Terminal Windows)

**Terminal 1:** Run the perception pipeline once
```bash
python perception_pipeline.py
```

**Terminal 2:** Run your client application
```bash
python perception_client_example.py --visualize
```

### Method 2: Continuous Mode (Recommended for Real-time Applications)

**Terminal 1:** Run the perception pipeline in continuous mode
```bash
# Run at 1 Hz (default)
python perception_pipeline.py --continuous

# Or at custom rate (e.g., 2 Hz)
python perception_pipeline.py --continuous --rate 2.0
```

**Terminal 2:** Run your client application
```bash
# Client will receive continuous updates
python perception_client_example.py --visualize
```

**Stop:** Press `Ctrl+C` in Terminal 1 to gracefully stop the pipeline

### Method 3: Background Pipeline

```bash
# Run continuous pipeline in background
python perception_pipeline.py --continuous &

# Run client
python perception_client_example.py --visualize

# When done, kill background process
pkill -f perception_pipeline.py
```

## Data Format

The final point cloud published via ZMQ is a Python dictionary:

```python
{
    'pcd': np.ndarray,      # Shape: (N, 3) - XYZ coordinates in meters
    'rgb': np.ndarray,      # Shape: (N, 3) - RGB colors in [0, 1]
    'num_points': int,      # Number of points (should be ≤ 4096)
    'bounds': dict          # Bounds used for filtering
}
```

## Integration with Your Application

To use the perception pipeline in your own script:

```python
import zmq
import pickle

# Setup ZMQ subscriber
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5556")
socket.setsockopt(zmq.SUBSCRIBE, b'')

# Receive point cloud
data = pickle.loads(socket.recv())
pcd = data['pcd']  # Nx3 point cloud
rgb = data['rgb']  # Nx3 RGB colors

# Use point cloud for your application
# - Motion planning
# - Grasp planning
# - Collision checking
# etc.

socket.close()
context.term()
```

## Customization

### Changing Spatial Bounds

Edit `perception_pipeline.py`, line ~250:

```python
bounds = {
    'x': [0.2, 0.8],    # Change X bounds (meters)
    'y': [-0.5, 0.5],   # Change Y bounds (meters)
    'z': [-0.1, 0.7]    # Change Z bounds (meters)
}
```

### Changing Number of Points

Use the `--num_points` argument:

```bash
python perception_pipeline.py --num_points 2048
```

Or modify the default in the script.

### Adding Additional Processing

Add your processing steps in `perception_pipeline.py` after FPS downsampling:

```python
# After line ~160 (FPS downsampling)
pcd_final, rgb_final = fps_downsample(pcd_filtered, rgb_filtered, args.num_points)

# Add your custom processing here
# Example: Remove statistical outliers
pcd_o3d = o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(pcd_final)
pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_final)
pcd_o3d, _ = pcd_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_final = np.asarray(pcd_o3d.points)
rgb_final = np.asarray(pcd_o3d.colors)
```

## Troubleshooting

### Camera Fails to Open
- **Error:** `Failed to get valid capture after 20 attempts`
- **Solution:** Check that cameras are plugged in and no other process is using them

### ZMQ Timeout
- **Error:** `Timeout after 60000ms`
- **Solution:** Make sure `perception_pipeline.py` is running before starting the client

### Port Already in Use
- **Error:** `Address already in use`
- **Solution:** Kill any existing processes using the ports:
  ```bash
  lsof -ti:5557 | xargs kill -9  # Camera data port
  lsof -ti:5556 | xargs kill -9  # Published point cloud port
  ```
- **Note:** Port 5555 is used by Deoxys robot control, so perception uses 5557

### Alignment Transform Not Found
- **Error:** `FileNotFoundError: cam1_to_cam0.npy`
- **Solution:** Run `align_cameras.py` first to generate alignment transforms

## Performance

Typical timing on a modern machine:
- Camera 0 capture: ~1-2 seconds
- Camera 1 capture: ~1-2 seconds
- Combine + filter + downsample: ~0.1-0.5 seconds
- **Total pipeline time: ~3-5 seconds per iteration**

ZMQ communication overhead: < 1ms per message

**Continuous Mode:**
- Maximum practical rate: ~0.2-0.3 Hz (one capture every 3-5 seconds)
- Recommended rate: 0.2 Hz or lower for reliable operation
- Setting `--rate` higher than the pipeline can handle will show warnings but continue operating

## Future Improvements

Potential enhancements:
1. Add support for continuous streaming (loop mode)
2. Add normal estimation before publishing
3. Add voxel grid downsampling as an alternative to FPS
4. Add support for saving intermediate results for debugging
5. Add multi-consumer support with ZMQ topics
6. Add compression for ZMQ messages (e.g., msgpack, protobuf)
