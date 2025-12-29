import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import cv2
import argparse

def load_npz_data(npz_path):
    """Load data from npz file"""
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    
    # Get available keys
    keys = list(data.keys())
    print(f"Available data: {keys}")
    
    # Load data
    rgb_data = data.get('rgb_data', None)
    joint_data = data.get('joint_data', None)
    gripper_data = data.get('gripper_data', None)
    depth_data = data.get('depth_data', None)
    ir_data = data.get('ir_data', None)
    
    if rgb_data is not None:
        print(f"RGB data shape: {rgb_data.shape}")
    if joint_data is not None:
        print(f"Joint data shape: {joint_data.shape}")
    if gripper_data is not None:
        print(f"Gripper data shape: {gripper_data.shape}")
    
    return data, rgb_data, joint_data, gripper_data, depth_data, ir_data

def create_interactive_visualization(npz_path):
    """Create an interactive visualization with sliders"""
    data, rgb_data, joint_data, gripper_data, depth_data, ir_data = load_npz_data(npz_path)
    
    if rgb_data is None:
        print("No RGB data found!")
        return
    
    num_frames = len(rgb_data)
    print(f"Total frames: {num_frames}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Robot Data Visualization', fontsize=16)
    
    # RGB image subplot
    ax_rgb = axes[0, 0]
    ax_rgb.set_title('RGB Image')
    img_rgb = ax_rgb.imshow(rgb_data[0])
    ax_rgb.axis('off')
    
    # Joint states subplot
    ax_joints = axes[0, 1]
    ax_joints.set_title('Joint States')
    if joint_data is not None:
        joint_lines = ax_joints.plot(joint_data[0], 'o-', label='Current')
        ax_joints.set_xlabel('Joint Index')
        ax_joints.set_ylabel('Joint Angle (rad)')
        ax_joints.set_ylim([joint_data.min() - 0.1, joint_data.max() + 0.1])
        ax_joints.grid(True)
        ax_joints.legend()
    else:
        ax_joints.text(0.5, 0.5, 'No joint data available', 
                      ha='center', va='center', transform=ax_joints.transAxes)
    
    # Gripper state subplot
    ax_gripper = axes[1, 0]
    ax_gripper.set_title('Gripper State')
    if gripper_data is not None:
        gripper_line = ax_gripper.plot(gripper_data, 'b-', linewidth=2)
        ax_gripper.set_xlabel('Frame')
        ax_gripper.set_ylabel('Gripper State')
        ax_gripper.grid(True)
        ax_gripper.set_ylim([gripper_data.min() - 0.1, gripper_data.max() + 0.1])
    else:
        ax_gripper.text(0.5, 0.5, 'No gripper data available', 
                       ha='center', va='center', transform=ax_gripper.transAxes)
    
    # Depth/IR image subplot
    ax_depth = axes[1, 1]
    ax_depth.set_title('Depth/IR Image')
    if depth_data is not None:
        img_depth = ax_depth.imshow(depth_data[0], cmap='gray')
        ax_depth.axis('off')
    elif ir_data is not None:
        img_depth = ax_depth.imshow(ir_data[0], cmap='gray')
        ax_depth.axis('off')
    else:
        ax_depth.text(0.5, 0.5, 'No depth/IR data available', 
                     ha='center', va='center', transform=ax_depth.transAxes)
    
    # Add frame slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valstep=1)
    
    def update(val):
        frame_idx = int(slider.val)
        
        # Update RGB image
        img_rgb.set_array(rgb_data[frame_idx])
        
        # Update joint states
        if joint_data is not None:
            for i, line in enumerate(joint_lines):
                line.set_ydata(joint_data[frame_idx])
        
        # Update depth/IR image
        if depth_data is not None:
            img_depth.set_array(depth_data[frame_idx])
        elif ir_data is not None:
            img_depth.set_array(ir_data[frame_idx])
        
        # Update gripper indicator
        if gripper_data is not None:
            gripper_state = gripper_data[frame_idx]
            ax_gripper.axvline(x=frame_idx, color='r', alpha=0.5, linewidth=2)
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    plt.tight_layout()
    plt.show()

def create_animation(npz_path, output_path=None):
    """Create an animation of the data"""
    data, rgb_data, joint_data, gripper_data, depth_data, ir_data = load_npz_data(npz_path)
    
    if rgb_data is None:
        print("No RGB data found!")
        return
    
    num_frames = len(rgb_data)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Robot Data Animation', fontsize=16)
    
    # RGB image
    ax_rgb = axes[0]
    ax_rgb.set_title('RGB Image')
    img_rgb = ax_rgb.imshow(rgb_data[0])
    ax_rgb.axis('off')
    
    # Joint states
    ax_joints = axes[1]
    ax_joints.set_title('Joint States')
    if joint_data is not None:
        joint_lines = ax_joints.plot(joint_data[0], 'o-')
        ax_joints.set_xlabel('Joint Index')
        ax_joints.set_ylabel('Joint Angle (rad)')
        ax_joints.set_ylim([joint_data.min() - 0.1, joint_data.max() + 0.1])
        ax_joints.grid(True)
    else:
        ax_joints.text(0.5, 0.5, 'No joint data available', 
                      ha='center', va='center', transform=ax_joints.transAxes)
    
    def animate(frame):
        # Update RGB image
        img_rgb.set_array(rgb_data[frame])
        
        # Update joint states
        if joint_data is not None:
            for i, line in enumerate(joint_lines):
                line.set_ydata(joint_data[frame])
        
        return [img_rgb] + joint_lines if joint_data is not None else [img_rgb]
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                 interval=100, blit=True)
    
    if output_path:
        print(f"Saving animation to {output_path}...")
        anim.save(output_path, writer='pillow')
    
    plt.tight_layout()
    plt.show()

def plot_joint_trajectories(npz_path):
    """Plot joint trajectories over time"""
    data, rgb_data, joint_data, gripper_data, depth_data, ir_data = load_npz_data(npz_path)
    
    if joint_data is None:
        print("No joint data found!")
        return
    
    num_joints = joint_data.shape[1]
    num_frames = len(joint_data)
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Joint Trajectories Over Time', fontsize=16)
    
    axes = axes.flatten()
    
    for i in range(num_joints):
        if i < len(axes):
            ax = axes[i]
            ax.plot(joint_data[:, i], 'b-', linewidth=2)
            ax.set_title(f'Joint {i+1}')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Angle (rad)')
            ax.grid(True)
    
    # Hide unused subplots
    for i in range(num_joints, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def save_video_with_cv2(npz_path, output_path="data1.mp4", fps=10):
    """Save visualization as video using cv2"""
    data, rgb_data, joint_data, gripper_data, depth_data, ir_data = load_npz_data(npz_path)
    
    if rgb_data is None:
        print("No RGB data found!")
        return
    
    num_frames = len(rgb_data)
    print(f"Creating video with {num_frames} frames at {fps} fps...")
    
    # Get frame dimensions from RGB data
    height, width = rgb_data[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(num_frames):
        # Get current frame
        frame = rgb_data[frame_idx]
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add text overlay with frame info
        cv2.putText(frame_bgr, f'Frame: {frame_idx}/{num_frames-1}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add joint information if available
        if joint_data is not None:
            joint_text = f'Joint 1: {joint_data[frame_idx, 0]:.3f}'
            cv2.putText(frame_bgr, joint_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add gripper information if available
        if gripper_data is not None:
            gripper_state = "OPEN" if gripper_data[frame_idx] == 0.0 else "CLOSED"
            gripper_text = f'Gripper: {gripper_state}'
            cv2.putText(frame_bgr, gripper_text, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame_bgr)
        
        # Progress indicator
        if frame_idx % 10 == 0:
            print(f"Processing frame {frame_idx}/{num_frames-1}")
    
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Direct usage without arguments
    npz_path = "data5.npz"
    output_path = "data5.mp4"
    
    # Save as video using cv2
    save_video_with_cv2(npz_path, output_path, fps=10) 