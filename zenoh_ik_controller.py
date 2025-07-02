import ikpy.chain
import numpy as np
import os
import time
import zenoh
import json
import logging
from ikpy.link import URDFLink, OriginLink, DHLink

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Robot and IK Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_file_path = os.path.join(script_dir, "x1/urdf/x1.urdf")

# Correctly define the list of link names (excluding joints)
left_arm_link_names = [
    "lumber_pitch",
    'left_shoulder_pitch_joint', 
    'left_shoulder_pitch', 
    'left_shoulder_roll_joint', 
    'left_shoulder_roll', 
    'left_shoulder_yaw_joint', 
    'left_shoulder_yaw', 
    'left_elbow_pitch_joint', 
    'left_elbow_pitch', 
    'left_elbow_yaw_joint', 
    'left_elbow_yaw', 
    'left_wrist_pitch_joint', 
    'left_wrist_pitch'
]

correct_mask = [False, True, True, True, True, True,False]

try:
    robot_chain = ikpy.chain.Chain.from_urdf_file(
        urdf_file_path,
        base_elements=left_arm_link_names,
        active_links_mask=correct_mask,
    )
except FileNotFoundError:
    log.error(f"Error: URDF file not found at {urdf_file_path}")
    exit()
except Exception as e:
    log.error(f"Error loading URDF: {e}")
    exit()

log.info("Successfully loaded URDF.")
log.info(f"Robot Name: {robot_chain.name}")

# --- 2. Inspect the loaded chain and links ---
print("\n--- Links in the chain ---")
for i, link in enumerate(robot_chain.links):
    is_active = False
    if i< robot_chain.active_links_mask.size and robot_chain.active_links_mask[i]:
        is_active = True
    link_type = "fixed"
    if isinstance(link,URDFLink):
        link_type = link.joint_type

    print(f"Link {i}: Name='{link.name}', Type='{link_type}', Active for IK: {is_active}")

# The active_links_mask indicates which links (joints) are part of the IK calculation
# True means it's an actuated joint, False means it's fixed or the base link.
print(f"\nActive links mask: {robot_chain.active_links_mask}")
num_active_joints = sum(robot_chain.active_links_mask)
print(f"Number of active (movable) joints for IK: {num_active_joints}")


# --- Zenoh Configuration ---
ZENOH_ROUTER_ADDRESS = "tcp/74.48.61.171:7447"
ZENOH_KEY = "fms/phone_server"

# --- Global State ---
# Initial joint configuration (e.g., all zeros)
initial_joint_positions = [0]+[0.01] * (len(robot_chain.links)-2)+[0]
current_angles_rad = np.array(initial_joint_positions)

# Calculate initial end-effector position from the initial joint angles
initial_fk = robot_chain.forward_kinematics(initial_joint_positions)
current_target_xyz = initial_fk[:3, 3]
log.info(f"Arm starting at position: {current_target_xyz}")

# --- Zenoh Subscriber Callback ---
def zenoh_callback(sample):
    global current_target_xyz, current_angles_rad
    try:
        data = json.loads(bytes(sample.payload).decode("utf-8"))
        
        # Extract data from the payload
        position_change = np.array(data.get("position_change", [0, 0, 0]))
        
        if np.allclose(position_change, [0.0, 0.0, 0.0]):
            log.info("Received zero position change. No motion.")
            return
        # --- Calculate New Target ---
        new_target_xyz = current_target_xyz + position_change
        
        log.info(f"Received position change: {position_change}, New Target: {new_target_xyz}")
        
        # --- IK Calculation ---
        # Use orientation_mode="Z" for stability. This keeps the end-effector pointing
        # along the Z-axis of its own frame, which is a common and stable setup.
        start_time = time.time()
        target_angles_rad = robot_chain.inverse_kinematics(
            target_position=new_target_xyz,
            initial_position=current_angles_rad
        )
        print(f"IK Time: {time.time() - start_time:.4f} seconds")
        # --- Validate and Update State ---
        # Check if the result contains any non-finite numbers (NaN, inf)
        print("IK Solution (Joint Angles): {np.rad2deg(target_angles_rad)}")
        if np.all(np.isfinite(target_angles_rad)):
            current_angles_rad = target_angles_rad
            current_fk = robot_chain.forward_kinematics(current_angles_rad)
            current_target_xyz = current_fk[:3, 3]
            
            log.info(f"IK Solution (Joint Angles): {np.rad2deg(current_angles_rad)}")
            log.info(f"New FK Position: {current_target_xyz}")
        else:
            log.warning("IK solution failed or returned non-finite values. Discarding.")
            # The state (current_angles_rad, current_target_xyz) remains unchanged.

        log.info("-" * 20)

    except json.JSONDecodeError:
        log.warning("Failed to decode JSON from Zenoh message.")
    except Exception as e:
        log.error(f"Error in IK calculation or callback: {e}")
        # No need to revert state here, as we now validate before updating.

# --- Main Execution ---
if __name__ == "__main__":
    log.info("Starting Zenoh IK Controller...")
    
    # --- Initialize Zenoh ---
    conf = zenoh.Config()
    if ZENOH_ROUTER_ADDRESS:
        connect_config = {"endpoints": [ZENOH_ROUTER_ADDRESS]}
        conf.insert_json5("connect", json.dumps(connect_config))
    
    try:
        session = zenoh.open(conf)
        log.info(f"Connected to Zenoh router at: {ZENOH_ROUTER_ADDRESS}")
        
        sub = session.declare_subscriber(ZENOH_KEY, zenoh_callback)
        log.info(f"Subscribed to Zenoh key: {ZENOH_KEY}")
        
        log.info("Controller is running. Waiting for data from Zenoh...")
        # Keep the script running to receive messages
        while True:
            time.sleep(1)
            
    except Exception as e:
        log.error(f"Failed to start Zenoh session or subscriber: {e}")
    finally:
        if 'session' in locals() and session.is_open():
            session.close()
            log.info("Zenoh session closed.")
