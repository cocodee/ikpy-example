import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
import json
import logging
import os
import zenoh
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
from weighted_moving_filter import WeightedMovingFilter # 假设这个工具类存在
import traceback
# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ==============================================================================
# === STEP 1: NEW PINOCCHIO + CASADI IK SOLVER CLASS FOR X1's LEFT ARM ===
# ==============================================================================

class X1_LeftArmIK:
    """
    An optimization-based Inverse Kinematics solver for the X1 robot's left arm,
    using Pinocchio for kinematics and CasADi for optimization.
    """
    def __init__(self, urdf_path, visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.Visualization = visualization

        # --- 1. Load the full robot model ---
        try:
            # The URDF path needs a package directory for mesh files
            package_dir = os.path.dirname(urdf_path)
            self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dir)
        except Exception as e:
            log.error(f"Failed to load URDF: {e}")
            raise

        # --- 2. Define active joints and lock all others ---
        # These are the joints we want to control for the left arm.
        self.active_joints_names = [
            'left_shoulder_pitch_joint', 
            'left_shoulder_roll_joint', 
            'left_shoulder_yaw_joint', 
            'left_elbow_pitch_joint', 
            'left_elbow_yaw_joint', 
            'left_wrist_pitch_joint', 
        ]
        
        # Get all joint names from the model, excluding the 'universe' base.
        all_joint_names = [name for name in self.robot.model.names if name != 'universe']
        
        # Determine which joints to lock by finding the difference.
        joints_to_lock_names = [j for j in all_joint_names if j not in self.active_joints_names]
        log.info(f"Locking {len(joints_to_lock_names)} joints.")

        ee_parent_joint_name = 'left_wrist_pitch_joint' #'left_wrist_pitch_joint'
        ee_parent_joint_id = self.robot.model.getJointId(ee_parent_joint_name)
        print(f"Parent joint of end effector: {ee_parent_joint_id}")
        self.robot.model.addFrame(
            pin.Frame('L_ee', ee_parent_joint_id,
                      pin.SE3.Identity(), # No offset from the joint frame for now
                      pin.FrameType.OP_FRAME)
        )
        self.robot.rebuildData()
        # Build the reduced model containing only the active joints.
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=joints_to_lock_names,
            reference_configuration=pin.neutral(self.robot.model),
        )
        log.info(f"Reduced model created with {self.reduced_robot.model.nq} DoF.")
        log.info(f"Active joints in reduced model: {[name for name in self.reduced_robot.model.names if name != 'universe']}")
        


        # --- 3. Define the End-Effector (EE) Frame ---
        # We attach a frame to the last joint of the arm (the wrist).
        # This frame is what we will control to the target pose.

        self.ee_frame_id = self.reduced_robot.model.getFrameId('L_ee')
        
        #self.reduced_robot.data = self.reduced_robot.model.createData()
        # --- 4. Setup CasADi for Symbolic Optimization ---
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_target = casadi.SX.sym("tf_target", 4, 4) # Target pose (SE3)

        # Symbolic Forward Kinematics
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        
        # Error functions
        # Translational error: difference between current EE position and target position
        self.error_pos = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_target],  # 移除了 self.cTf_r
            [
                # 直接输出左手的误差向量，不再需要 vertcat
                self.cdata.oMf[self.ee_frame_id].translation - self.cTf_target[:3, 3]
            ],
        )       
        
        # Rotational error: difference between current EE orientation and target orientation
        self.error_rot = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_target],
            [cpin.log3(self.cdata.oMf[self.ee_frame_id].rotation @ self.cTf_target[:3, :3].T)],
        )        
        # --- 5. Define the Optimization Problem ---
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_target = self.opti.parameter(4, 4)

        # Cost function: a weighted sum of different objectives
        translational_cost = casadi.sumsqr(self.error_pos(self.var_q, self.param_tf_target))
        rotation_cost = casadi.sumsqr(self.error_rot(self.var_q, self.param_tf_target))
        regularization_cost = casadi.sumsqr(self.var_q)  # Keep joints close to zero
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last) # Ensure smooth motion

        # The weights (e.g., 50, 1) determine the priority of each objective.
        # Here, position tracking is much more important than orientation tracking.
        self.opti.minimize(
            50 * translational_cost + 1 * rotation_cost + 0.02 * regularization_cost + 0.1 * smooth_cost
        )

        # Constraints: ensure joint angles are within their physical limits
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit, 
                self.var_q,
                self.reduced_robot.model.upperPositionLimit
            )
        )
        
        # Solver configuration
        opts = {'ipopt': {'print_level': 0}, 'print_time': False}
        self.opti.solver("ipopt", opts)
        
        # --- 6. Initialize State and Filter ---
        self.q_current = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), self.reduced_robot.model.nq)
        
        # Optional Visualization Setup
        self.vis = None
        if self.Visualization:
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=False)
            self.vis.loadViewerModel("pinocchio")
            self.vis.display(self.q_current)
            self.vis.viewer['L_ee_target'].set_object(mg.Sphere(0.005), mg.MeshLambertMaterial(color=0xff0000, reflectivity=0.8))


    def solve(self, target_pose, q_guess=None):
        """Solves the IK problem for a given target pose."""
        if q_guess is None:
            q_guess = self.q_current
            
        # Set optimization parameters
        self.opti.set_initial(self.var_q, q_guess)
        self.opti.set_value(self.param_tf_target, target_pose)
        self.opti.set_value(self.var_q_last, self.q_current)

        if self.Visualization:
            self.vis.viewer['L_ee_target'].set_transform(target_pose)

        try:
            sol = self.opti.solve()
            sol_q = sol.value(self.var_q)

            # Apply smoothing filter for more stable output
            self.smooth_filter.add_data(sol_q)
            q_filtered = self.smooth_filter.filtered_data
            
            self.q_current = q_filtered # Update state

            if self.Visualization:
                self.vis.display(self.q_current)

            return self.q_current

        except Exception as e:
            log.warning(f"IK optimization failed: {e}. Returning last known configuration.")
            # On failure, return the last successful configuration to avoid jumps
            return self.q_current
            
    def forward_kinematics(self, q):
        """Computes forward kinematics for a given joint configuration."""
        pin.framesForwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        return self.reduced_robot.data.oMf[self.ee_frame_id].homogeneous


# ==============================================================================
# === STEP 2: REFACTORED MAIN SCRIPT USING THE NEW SOLVER ===
# ==============================================================================



# --- Zenoh Configuration ---
ZENOH_ROUTER_ADDRESS = "tcp/74.48.61.171:7447"
ZENOH_KEY = "fms/phone_server"




# --- Zenoh Subscriber Callback ---
def test(position_change):
    # --- Robot and IK Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
   # IMPORTANT: Make sure the path to the URDF is correct
    urdf_file_path = os.path.join(script_dir, "x1/urdf/x1.urdf") 



    log.info("Successfully initialized Pinocchio+CasADi IK solver.")
    # --- Initialize the new IK Solver ---
    try:
        ik_solver = X1_LeftArmIK(urdf_file_path, visualization=True) # Set to True to see a 3D view
    except Exception as e:
        log.error("Could not initialize IK solver. Exiting.")
        log.error(e)
        traceback.print_exc()
        exit()
# --- Global State ---
    # Initial joint configuration (all zeros for the reduced model)
    current_angles_rad = np.zeros(ik_solver.reduced_robot.model.nq)

    # Calculate initial end-effector pose from the initial joint angles
    initial_pose = ik_solver.forward_kinematics(current_angles_rad)
    current_target_position = initial_pose[:3, 3]

    # IMPORTANT: The new solver controls full pose (position + orientation).
    # Since the phone only sends position changes, we will keep the orientation fixed
    # to its initial state. This provides stability.
    initial_target_orientation = initial_pose[:3, :3]

    log.info(f"Arm starting at position: {current_target_position}")
    log.info(f"Arm orientation will be kept fixed to:\n{initial_target_orientation}")    
    try:
        if np.allclose(position_change, [0, 0, 0]):
            return
            
        # --- Calculate New Target Pose ---
        new_target_position = current_target_position + position_change
        
        print(f"Received position change: {position_change}, New Target Position: {new_target_position}")
        # Construct the full 4x4 target pose matrix
        target_pose = np.eye(4)
        target_pose[:3, :3] = initial_target_orientation # Keep orientation fixed
        target_pose[:3, 3] = new_target_position
        
        log.info(f"Received position change: {position_change}, New Target Position: {new_target_position}")
        
        # --- IK Calculation using the new solver ---
        start_time = time.time()
        # Use current_angles_rad as the initial guess for the optimization
        target_angles_rad = ik_solver.solve(target_pose, q_guess=current_angles_rad)
        log.info(f"IK Time: {time.time() - start_time:.4f} seconds")
        
        # --- Validate and Update State ---
        if np.all(np.isfinite(target_angles_rad)):
            log.info(f"IK solution result: {target_angles_rad}")
            pass
        else:
            log.warning("IK solution returned non-finite values. Discarding.")

        time.sleep(1000)
        log.info("-" * 20)

    except json.JSONDecodeError:
        log.warning("Failed to decode JSON from Zenoh message.")
    except Exception as e:
        log.error(f"Error in callback: {e}", exc_info=True)

def run_ik_test():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_file_path = os.path.join(script_dir, "x1", "urdf", "x1.urdf") 

    try:
        ik_solver = X1_LeftArmIK(urdf_file_path, visualization=True)
    except Exception as e:
        log.error("Could not initialize IK solver. Exiting.", exc_info=True)
        return

    log.info("Successfully initialized Pinocchio+CasADi IK solver.")

    # --- Persistent State Variables ---
    # These are now outside the loop/callback, so they retain their values.
    # We start with the solver's initial state (usually zeros).
    current_angles_rad = ik_solver.q_current
    
    # Calculate initial end-effector pose from the initial joint angles
    initial_pose = ik_solver.forward_kinematics(current_angles_rad)
    
    # The current target position starts at the arm's initial position
    current_target_position = initial_pose[:3, 3].copy() 
    
    # Orientation is kept fixed for stability
    target_orientation = initial_pose[:3, :3].copy()

    log.info(f"Arm starting at position: {current_target_position}")
    log.info(f"Arm orientation will be kept fixed to:\n{target_orientation}")    
    
    # This function would be your Zenoh callback
    def process_position_change(position_change):
        nonlocal current_target_position, current_angles_rad # Declare we are modifying the outer scope variables
        try:
            if np.allclose(position_change, [0, 0, 0]):
                return
                
            # --- Calculate New Target Pose ---
            # Update the target position incrementally
            current_target_position += np.array(position_change)
            
            log.info(f"Received change: {position_change}, New Target: {current_target_position}")
            
            target_pose = np.eye(4)
            target_pose[:3, :3] = target_orientation
            target_pose[:3, 3] = current_target_position
            
            # --- IK Calculation ---
            start_time = time.time()
            # Use the solver's last known configuration as the initial guess
            new_angles_rad = ik_solver.solve(target_pose)
            log.info(f"IK Time: {time.time() - start_time:.4f} seconds")
            
            # --- Validate and Update State ---
            if np.all(np.isfinite(new_angles_rad)):
                current_angles_rad = new_angles_rad # Update our state
                log.info(f"IK solution found: {current_angles_rad}")
            else:
                log.warning("IK solution returned non-finite values. Discarding.")

            # <<< FIX 3: Removed time.sleep(1000)
            log.info("-" * 20)

        except Exception as e:
            log.error(f"Error in callback: {e}", exc_info=True)

    # --- Simulating a few incoming messages ---
    log.info("--- STARTING SIMULATION ---")
    time.sleep(2) # Give viewer time to open
    
    while True:
        for i in range(5):
            process_position_change([0.0, 0.0, 0.01])
            time.sleep(1)
        
        for i in range(5):
            process_position_change([0.0, 0.01, 0.0])
            time.sleep(1)
        
        for i in range(5):
            process_position_change([0.01, 0.0, 0.0])
            time.sleep(1)
        
        for i in range(5):
            process_position_change([-0.01, -0.01, -0.01])
            time.sleep(1)

    time.sleep(1000)
    log.info("--- SIMULATION FINISHED ---")
# --- Main Execution ---
if __name__ == "__main__":
    #test([0.05, 0.15, 0.10])
    run_ik_test()
   