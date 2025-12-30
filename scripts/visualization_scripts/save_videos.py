import numpy as np
import cv2
import os

def save_video(frames: np.ndarray, output_path: str, fps: int = 30, is_rgb: bool = False):
    """
    Save a video from an array of frames.
    
    Args:
        frames: Array of frames with shape (N, H, W, C) where:
               N = number of frames
               H = height
               W = width
               C = channels (1 or 3)
        output_path: Path where to save the video
        fps: Frames per second (default: 30)
        is_rgb: Whether the input is RGB format (needs conversion to BGR for OpenCV)
    """
    if len(frames.shape) == 3:
        frames = frames[..., None]  # Add channel dimension if missing
        
    height, width = frames.shape[1:3]
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        if frame.shape[-1] == 1:
            # Convert single channel to 3 channels
            frame_norm = (frame.squeeze() / np.max(frame) * 255).astype(np.uint8)
            frame_3ch = np.stack([frame_norm] * 3, axis=-1)
            writer.write(frame_3ch)
        else:
            # For RGB/BGR frames
            if is_rgb:
                frame = frame[..., ::-1]  # Convert RGB to BGR
            writer.write(frame)
    
    writer.release()

def save_videos_from_npz(npz_path: str):
    """
    Load data from npz file and save rgb, depth, and ir videos.
    
    Args:
        npz_path: Path to the npz file containing rgb_data, depth_data, and ir_data
    """
    # Load the data
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    rgb_data = data['rgb_data']
    depth_data = data['depth_data']
    ir_data = data['ir_data']
    
    print(f"Found {len(rgb_data)} frames")
    
    # Save each video
    print("Saving RGB video...")
    save_video(rgb_data, 'rgb_video.mp4') #, is_rgb=True)
    
    print("Saving depth video...")
    save_video(depth_data, 'depth_video.mp4')
    
    print("Saving IR video...")
    save_video(ir_data, 'ir_video.mp4')
    
    print("Done! Videos saved as:")
    print("- rgb_video.mp4")
    print("- depth_video.mp4")
    print("- ir_video.mp4")

if __name__ == "__main__":
    if not os.path.exists("data.npz"):
        print("Error: data.npz not found!")
    else:
        save_videos_from_npz("data.npz")

    