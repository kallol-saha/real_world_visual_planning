#!/usr/bin/env python3
"""Debug script to test robot connection and state reception."""
import os
import sys
import time
import socket
from pathlib import Path

try:
    from deoxys import config_root
    from deoxys.franka_interface import FrankaInterface
    from deoxys.utils import YamlConfig
except ImportError as e:
    print(f"Error importing deoxys: {e}")
    print("Make sure deoxys is installed and PYTHONPATH includes ~/deoxys_control/deoxys")
    sys.exit(1)


def check_network_connectivity(config):
    """Check if we can reach the robot and NUC."""
    print("\n=== Network Connectivity Check ===")
    
    import subprocess
    
    robot_ip = config.get("ROBOT", {}).get("IP")
    nuc_ip = config.get("NUC", {}).get("IP")
    
    # Check ping connectivity
    if nuc_ip:
        print(f"Pinging NUC at {nuc_ip}...")
        try:
            result = subprocess.run(
                ["ping", "-c", "2", "-W", "2", nuc_ip],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"  ✓ NUC is reachable via ping")
            else:
                print(f"  ✗ Cannot ping NUC at {nuc_ip}")
                print(f"     This suggests a network connectivity issue!")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"  ? Could not test ping (ping command not available)")
    
    if robot_ip:
        print(f"Pinging robot at {robot_ip}...")
        try:
            result = subprocess.run(
                ["ping", "-c", "2", "-W", "2", robot_ip],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"  ✓ Robot is reachable via ping")
            else:
                print(f"  ✗ Cannot ping robot at {robot_ip}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"  ? Could not test ping")
    
    # Note: ZMQ uses UDP-like protocol, so TCP socket checks won't work
    print("\nNote: ZMQ uses its own protocol, so standard TCP port checks may not work.")
    print("      The important thing is that the state publisher is running on the NUC.")
    print()


def main():
    config_file = "charmander.yml"
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"ERROR: Config file not found: {config_file}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for: {os.path.abspath(config_file)}")
        sys.exit(1)
    
    print(f"Using config file: {os.path.abspath(config_file)}")
    
    # Load and display config
    config = YamlConfig(config_file).as_easydict()
    print("\n=== Config Summary ===")
    print(f"Robot IP: {config.get('ROBOT', {}).get('IP', 'N/A')}")
    print(f"NUC IP: {config.get('NUC', {}).get('IP', 'N/A')}")
    print(f"SUB_PORT: {config.get('NUC', {}).get('SUB_PORT', 'N/A')}")
    print(f"PUB_PORT: {config.get('NUC', {}).get('PUB_PORT', 'N/A')}")
    
    # Check network connectivity
    check_network_connectivity(config)
    
    # Try to initialize robot interface
    print("\n=== Initializing FrankaInterface ===")
    try:
        robot_interface = FrankaInterface(
            config_file,
            use_visualizer=False
        )
        print("✓ FrankaInterface initialized successfully")
        
        # Try to inspect ZMQ subscriber if available
        if hasattr(robot_interface, '_state_subscriber'):
            print("  ZMQ state subscriber found")
        if hasattr(robot_interface, '_gripper_subscriber'):
            print("  ZMQ gripper subscriber found")
        if hasattr(robot_interface, '_action_publisher'):
            print("  ZMQ action publisher found")
            
    except Exception as e:
        print(f"✗ Failed to initialize FrankaInterface: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Wait for state with detailed diagnostics
    print("\n=== Waiting for Robot State ===")
    max_wait = 15.0
    start_time = time.time()
    check_interval = 1.0
    last_check = start_time
    
    while time.time() - start_time < max_wait:
        current_time = time.time()
        
        # Check state buffer
        state_size = getattr(robot_interface, 'state_buffer_size', None)
        state_buffer_len = len(getattr(robot_interface, '_state_buffer', []))
        
        if current_time - last_check >= check_interval:
            elapsed = current_time - start_time
            print(f"[{elapsed:.1f}s] state_buffer_size={state_size}, _state_buffer length={state_buffer_len}")
            last_check = current_time
        
        # Check if we have state
        if (state_size and state_size > 0) or state_buffer_len > 0:
            print(f"\n✓ Robot state received after {time.time() - start_time:.1f} seconds!")
            
            # Try to get some state info
            if state_buffer_len > 0:
                state = robot_interface._state_buffer[-1]
                print(f"  Joint positions: {[f'{q:.3f}' for q in state.q[:3]]}...")
                print(f"  End-effector pose available: {hasattr(state, 'O_T_EE')}")
            
            robot_interface.close()
            print("\n✓ Connection test successful!")
            return
        
        time.sleep(0.1)
    
    print(f"\n✗ Timeout: No robot state received after {max_wait} seconds")
    print("\n" + "="*60)
    print("TROUBLESHOOTING GUIDE")
    print("="*60)
    print("\n1. CHECK STATE PUBLISHER ON NUC:")
    print(f"   SSH into the NUC (ssh user@{nuc_ip}) and run:")
    print("   - Check if deoxys state publisher is running:")
    print("     ps aux | grep deoxys")
    print("   - Start the state publisher if needed:")
    print("     cd ~/deoxys_control/deoxys")
    print("     python -m deoxys.examples.collect_data --interface-cfg charmander.yml")
    print("     (or whatever command starts your state publisher)")
    print("\n2. VERIFY NETWORK CONNECTIVITY:")
    print(f"   - Can you ping the NUC? Try: ping {nuc_ip}")
    print(f"   - Can you SSH to the NUC? Try: ssh user@{nuc_ip}")
    print("\n3. CHECK FIREWALL:")
    print("   - Make sure ports 5555 (SUB) and 5556 (PUB) are not blocked")
    print("   - On NUC: sudo ufw status")
    print("   - If firewall is active, allow ZMQ ports:")
    print("     sudo ufw allow 5555/udp")
    print("     sudo ufw allow 5556/udp")
    print("\n4. VERIFY CONFIG:")
    print(f"   - NUC IP in config: {nuc_ip}")
    print(f"   - SUB_PORT: {config.get('NUC', {}).get('SUB_PORT', 'N/A')}")
    print(f"   - Make sure these match what's running on the NUC")
    print("\n5. CHECK ROBOT STATUS:")
    print(f"   - Is the robot powered on? (IP: {config.get('ROBOT', {}).get('IP', 'N/A')})")
    print("   - Is the robot connected to the NUC?")
    print("\n6. TEST ZMQ CONNECTION MANUALLY:")
    print("   You can test ZMQ connectivity with:")
    print("   python -c \"import zmq; ctx=zmq.Context(); s=ctx.socket(zmq.SUB);")
    print(f"   s.connect('tcp://{nuc_ip}:{config.get('NUC', {}).get('SUB_PORT', 5555)}');")
    print("   s.setsockopt_string(zmq.SUBSCRIBE, ''); print('Connected!')\"")
    print("="*60)
    
    robot_interface.close()
    sys.exit(1)


if __name__ == "__main__":
    main()

