#!/usr/bin/env python3
"""Test ZMQ connection to robot state publisher."""
import sys
import time
import zmq
from deoxys.utils import YamlConfig

def test_zmq_connection(config_file="configs/charmander.yml"):
    """Test if we can connect to the ZMQ state publisher."""
    print("="*60)
    print("ZMQ Connection Test")
    print("="*60)
    
    # Load config
    config = YamlConfig(config_file).as_easydict()
    nuc_ip = config.get("NUC", {}).get("IP")
    sub_port = config.get("NUC", {}).get("SUB_PORT", 5555)
    pub_port = config.get("NUC", {}).get("PUB_PORT", 5556)
    
    print(f"\nConfig:")
    print(f"  NUC IP: {nuc_ip}")
    print(f"  SUB_PORT: {sub_port}")
    print(f"  PUB_PORT: {pub_port}")
    
    # Create ZMQ context and subscriber
    print(f"\nAttempting to connect to tcp://{nuc_ip}:{sub_port}...")
    
    try:
        ctx = zmq.Context()
        subscriber = ctx.socket(zmq.SUB)
        
        # Set socket options
        subscriber.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        subscriber.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all messages
        
        # Connect
        subscriber.connect(f"tcp://{nuc_ip}:{sub_port}")
        print(f"✓ Connected to ZMQ subscriber at tcp://{nuc_ip}:{sub_port}")
        
        # Try to receive a message
        print("\nWaiting for messages (timeout: 5 seconds)...")
        start_time = time.time()
        message_count = 0
        
        while time.time() - start_time < 5.0:
            try:
                # Try to receive a message (non-blocking with timeout)
                if subscriber.poll(timeout=100):  # 100ms poll
                    msg = subscriber.recv(zmq.NOBLOCK)
                    message_count += 1
                    if message_count == 1:
                        print(f"✓ Received first message! (size: {len(msg)} bytes)")
                        print(f"  Message type: {type(msg)}")
                        if message_count < 3:
                            print(f"  Waiting for more messages...")
                else:
                    time.sleep(0.1)
            except zmq.Again:
                # No message available yet
                time.sleep(0.1)
                continue
            except Exception as e:
                print(f"✗ Error receiving message: {e}")
                break
        
        if message_count > 0:
            print(f"\n✓ Success! Received {message_count} message(s) in 5 seconds")
            print("  The ZMQ state publisher is running and sending data!")
            return True
        else:
            print(f"\n✗ No messages received in 5 seconds")
            print("  Possible issues:")
            print("  1. State publisher is not running on NUC")
            print("  2. State publisher is running but not sending data")
            print("  3. Network/firewall blocking ZMQ messages")
            return False
            
    except zmq.error.ZMQError as e:
        print(f"✗ ZMQ Error: {e}")
        print("\nPossible causes:")
        print("  1. Cannot connect to NUC (network issue)")
        print("  2. State publisher not running on NUC")
        print("  3. Wrong IP address or port")
        print("  4. Firewall blocking connection")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            subscriber.close()
            ctx.term()
        except:
            pass

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "charmander.yml"
    success = test_zmq_connection(config_file)
    sys.exit(0 if success else 1)

