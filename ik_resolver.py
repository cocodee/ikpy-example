import ikpy.chain
import ikpy.utils.plot as plot_utils # For visualization (optional)
import numpy as np
import os
import time
from ikpy.link import URDFLink, OriginLink, DHLink

# Define the path to your URDF file
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_file_path = os.path.join(script_dir, "x1/urdf/x1.urdf")

# --- 1. Load the robot chain from the URDF file ---
# IKPy will automatically determine the active links (non-fixed joints)
# By default, the chain goes from the base to the last link defined in the URDF.
# If your URDF has a specific "end_effector" link you want to target that isn't the
# absolute last link, you might need to specify `last_link_vector` or manually
# construct the chain by listing links. For this example, we assume 'end_effector_link'
# is the one we want and it's the last in sequence for this simple URDF.

left_arm_link_names = [
    "left_shoulder_pitch",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch",
    "left_elbow_pitch_joint",
    "left_elbow_yaw",
    "left_elbow_yaw_joint",
    "left_wrist_pitch",
    "left_wrist_pitch_joint"
]

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
    robot_chain = ikpy.chain.Chain.from_urdf_file(urdf_file_path,
    base_elements=left_arm_link_names, # The first link in your desired chain
    active_links_mask=correct_mask,
    #last_link_vector=[0, 0.0, 0],
    #last_link_element=left_arm_link_names[-1], # The last link in your desired chain         
    )                                         
except FileNotFoundError:
    print(f"Error: URDF file not found at {urdf_file_path}")
    exit()
except Exception as e:
    print(f"Error loading URDF: {e}")
    exit()

print("Successfully loaded URDF.")
print(f"Robot Name: {robot_chain.name}")

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

# The target link for IK is implicitly the last link in the chain.
# For this URDF, it should be 'end_effector_link'.
target_link_name = robot_chain.links[-1].name
print(f"The IK will be computed for the end-effector: '{target_link_name}'")


# --- 3. Define a target position for the end-effector ---
# These coordinates are in the robot's base frame.
# Adjust these values based on your URDF's dimensions.
target_x = 0.2
target_y = 0.1
target_z = 0.3 # Height from the base_link origin
target_position = [target_x, target_y, target_z]
print(f"\nTarget End-Effector Position: {target_position}")

# Optional: Define target orientation (as a 3x3 rotation matrix)
# If not provided, IKPy will try to find a solution for the position only.
# For this example, we'll try with position first, then add orientation.
target_orientation = None #  Example: np.eye(3) for identity orientation
# target_orientation = np.array([ # Example: Pointing downwards along Z-axis
#     [1, 0, 0],
#     [0, -1, 0],
#     [0, 0, -1]
# ])
# orientation_mode = "all" # "X", "Y", "Z", "RX", "RY", "RZ", or "all"

# --- 4. Define an initial position for the joints (optional, but can help convergence) ---
# This should be a list/array of joint angles, one for each link in the chain (including fixed ones).
# For active links, these are the initial guess. For fixed links, the value is ignored.
# A common initial guess is all zeros.
initial_joint_positions = [0.0] * len(robot_chain.links)
# Example: if joint1 should start at 45 degrees, joint2 at 0, joint3 at -30 degrees
# initial_joint_positions = [0, np.deg2rad(45), np.deg2rad(0), np.deg2rad(-30), 0] # Base, J1, J2, J3, EEF
# Match the active_links_mask:
# [base_link (fixed), link1 (revolute), link2 (revolute), end_effector_link (revolute)]
# For our URDF, the active joints are roughly elements 1, 2, 3 of the links list.
# So, initial_joint_positions[1], initial_joint_positions[2], initial_joint_positions[3] will be used.
initial_joint_positions[1]=np.deg2rad(45)
initial_joint_positions[2]=np.deg2rad(45)
initial_joint_positions[3]=np.deg2rad(45)
initial_joint_positions[4]=np.deg2rad(45)
initial_joint_positions[5]=np.deg2rad(45)

print(f"Initial joint configuration (radians, full chain): {initial_joint_positions}")

initial_fk = robot_chain.forward_kinematics(initial_joint_positions)
current_target_xyz = initial_fk[:3, 3]
current_angles_rad= initial_joint_positions
print(f"Arm starting at position: {current_target_xyz}  {initial_fk}")

target_joint_positions = initial_joint_positions.copy()
target_joint_positions[3]=np.deg2rad(30)
target_joint_positions[4]=np.deg2rad(30)
other_fk = robot_chain.forward_kinematics(target_joint_positions)
other_xyz = other_fk[:3, 3]
print(f"target arm position: {other_xyz}")

active_joint_indices = [i for i, active in enumerate(robot_chain.active_links_mask) if active]

# 检查每个活动关节
for i, joint_idx in enumerate(active_joint_indices):
    link = robot_chain.links[joint_idx]
    lower_limit, upper_limit = link.bounds
    angle = initial_joint_positions[i]
    
    print(f"Active Joint {i} (Link Index {joint_idx}): Name='{link.name}'")
    print(f"  - Bounds: ({lower_limit:.4f}, {upper_limit:.4f})")
    print(f"  - Initial Angle: {angle:.4f}")
    
    if not (lower_limit <= angle <= upper_limit):
        print(f"  - !!! ERROR: Angle is OUTSIDE the defined limits!")

running = True
# --- 主循环 ---
while running:
    # 2. 计算新的目标位置
    #target_xyz = current_target_xyz + np.array([0, 0.02, -0.02])
    target_xyz = other_xyz
    try:
        print(f"current_angles_rad: {current_angles_rad}")
        print(f"target_xyz: {target_xyz}")
        # 3. IK求解
        # ikpy会找到最接近当前姿态的解
        # orientation_mode="Z" 是个非常有用的模式，它会尽力保持末端垂直向下，
        # 只关心位置(X,Y,Z)，大大简化了6-DOF手臂的控制。
        target_angles_rad = robot_chain.inverse_kinematics(
            target_position=target_xyz,
            initial_position=current_angles_rad,
            #orientation_mode="Z", # 可选 "X", "Y", "Z", "all", "None"
        )
        print(f"IK Solution (Joint Angles): {np.rad2deg(target_angles_rad)}")
        break
        # 5. 更新状态
        # 使用IK求解出的角度作为下一次计算的起点
        current_angles_rad = target_angles_rad
        # 重新用FK计算当前位置，而不是直接用target_xyz，这样可以防止误差累积
        current_fk = robot_chain.forward_kinematics(current_angles_rad)
        current_target_xyz = current_fk[:3, 3]

    except Exception as e:
        # 如果IK求解失败（例如目标点超出范围），则不移动并打印错误
        print(f"IK Error: {e}. Target unreachable.")
        # 回到上一个有效位置
        target_xyz = current_target_xyz


    time.sleep(5) # 控制循环频率，避免发送指令过快