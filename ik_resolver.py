import ikpy.chain
import ikpy.utils.plot as plot_utils # For visualization (optional)
import numpy as np
import os

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
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_elbow_yaw",
    "left_wrist_pitch"
]
try:
    robot_chain = ikpy.chain.Chain.from_urdf_file(urdf_file_path,
    base_elements=[left_arm_link_names[0]], # The first link in your desired chain
    last_link_element=left_arm_link_names[-1], # The last link in your desired chain         
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
    if i < len(robot_chain.active_links_mask) and robot_chain.active_links_mask[i]:
        is_active = True
    link_type = "Fixed/Base"
    if hasattr(link, 'joint') and hasattr(link.joint, 'type'):
        link_type = link.joint.type

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

print(f"Initial joint configuration (radians, full chain): {initial_joint_positions}")

# --- 5. Compute Inverse Kinematics ---
print("\n--- Computing IK ---")
try:
    # The ik_solution contains angles for ALL links in the chain (including fixed ones, which will be 0 or their fixed value)
    # in RADIANS.
    ik_solution_radians = robot_chain.inverse_kinematics(
        target_position=target_position,
        target_orientation=target_orientation, # Add this if you have a target orientation
        # orientation_mode=orientation_mode, # Add this if you have a target orientation
        initial_position=initial_joint_positions,
        # Optional parameters:
        # max_iter=10,
        # tolerance=1e-5
    )
    ik_solution_degrees = np.rad2deg(ik_solution_radians)

    print("\nIK Solution Found:")
    print(f"  Joint angles (radians, full chain): {ik_solution_radians}")
    print(f"  Joint angles (degrees, full chain): {ik_solution_degrees}")

    # Extracting active joint values
    active_joint_names = []
    active_joint_values_deg = []
    for i, link in enumerate(robot_chain.links):
        if robot_chain.active_links_mask[i]: # Check if the joint is active
            active_joint_names.append(link.name) # The link associated with the active joint
            active_joint_values_deg.append(ik_solution_degrees[i])

    print("\n  Active Joint Solution (degrees):")
    for name, val in zip(active_joint_names, active_joint_values_deg):
        print(f"    Joint for link '{name}': {val:.2f} degrees")

    # --- 6. Verify the solution with Forward Kinematics ---
    # This computes the end-effector pose given the found joint angles
    achieved_frame_matrix = robot_chain.forward_kinematics(ik_solution_radians)
    achieved_position = achieved_frame_matrix[:3, 3]
    achieved_orientation_matrix = achieved_frame_matrix[:3, :3]

    print("\n--- Verification with Forward Kinematics ---")
    print(f"  Achieved End-Effector Position: {achieved_position}")
    print(f"  Position Error (Euclidean distance): {np.linalg.norm(np.array(target_position) - achieved_position):.6f}")
    # print(f"  Achieved Orientation Matrix:\n{achieved_orientation_matrix}")

    # --- 7. (Optional) Plot the robot ---
    # Requires matplotlib: pip install matplotlib
    try:
        import matplotlib.pyplot as plt
        fig, ax = plot_utils.init_3d_figure()
        robot_chain.plot(ik_solution_radians, ax, target=target_position)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("IK Solution for x1_robot")
        # Set axis limits to better view the robot if needed
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([0, 0.7])
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping visualization. Install with 'pip install matplotlib'")
    except Exception as e:
        print(f"\nError during plotting: {e}")


except ikpy.exceptions.InverseKinematicsException as e:
    print(f"\nIK Error: Could not find a solution. {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
