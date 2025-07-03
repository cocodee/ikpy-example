import pinocchio as pin
import numpy as np
from pinocchio.utils import *
import os

class PinocchioIKSolver:
    def __init__(self, urdf_path, end_effector_frame_name):
        """
        Initializes the Pinocchio-based Inverse Kinematics solver.

        :param urdf_path: Path to the URDF file of the robot.
        :param end_effector_frame_name: The name of the end-effector frame.
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at: {urdf_path}")

        # Load the robot model from the URDF file
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Get the frame ID for the end-effector
        if not self.model.existFrame(end_effector_frame_name):
            raise ValueError(f"End-effector frame '{end_effector_frame_name}' not found in the model. Available frames: {[f.name for f in self.model.frames]}")
        self.end_effector_frame_id = self.model.getFrameId(end_effector_frame_name)

        print(f"PinocchioIKSolver initialized for model: {self.model.name}")
        print(f"End-effector frame: '{end_effector_frame_name}' (ID: {self.end_effector_frame_id})")

    def solve(self, target_pose, initial_guess=None):
        """
        Computes the inverse kinematics for a given target pose.

        :param target_pose: A pinocchio.SE3 object representing the desired pose of the end-effector.
        :param initial_guess: An optional initial guess for the joint configuration. If None, uses random values.
        :return: The calculated joint angles (q) or None if the solution is not found.
        """
        # Constants for the IK algorithm
        DAMP = 1e-1
        IT_MAX = 1000
        EPS = 1e-3

        # Use provided initial guess or create a random one
        if initial_guess is not None:
            q = initial_guess
        else:
            q = pin.randomConfiguration(self.model)

        for i in range(IT_MAX):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # Get the current pose of the end-effector
            current_pose = self.data.oMf[self.end_effector_frame_id]
            
            # Calculate the error between current and target pose
            dMi = target_pose.actInv(current_pose)
            err = pin.log(dMi).vector
            
            if np.linalg.norm(err) < EPS:
                print(f"Convergence achieved in {i+1} iterations.")
                return q

            # Compute the Jacobian of the end-effector frame
            J = pin.computeFrameJacobian(self.model, self.data, q, self.end_effector_frame_id)
            
            # Moore-Penrose pseudo-inverse with damping
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + DAMP * np.eye(6), err))
            
            # Update joint configuration
            q = pin.integrate(self.model, q, v)

        print("Warning: IK failed to converge within the maximum number of iterations.")
        return None

if __name__ == '__main__':
    # --- Example Usage ---
    # This demonstrates how to use the PinocchioIKSolver class.
    
    # Get the absolute path to the URDF file
    # Note: This assumes the script is run from the project's root directory
    urdf_file = os.path.join(os.path.dirname(__file__), 'x1/urdf/x1.urdf')
    
    # IMPORTANT: You must specify the correct name of your robot's end-effector link.
    # This is a placeholder and will likely need to be changed.
    # You can find the link names in your URDF file.
    end_effector_name = "left_hand_v1.1_link" # <-- CHANGE THIS IF NEEDED

    try:
        # 1. Initialize the solver
        ik_solver = PinocchioIKSolver(urdf_file, end_effector_name)

        # 2. Define a target pose (position and orientation)
        # Let's target a position (x, y, z) and a standard orientation (identity matrix)
        target_position = np.array([0.5, 0.2, 0.5])
        target_orientation = np.eye(3) # Identity matrix for rotation
        target_pose = pin.SE3(target_orientation, target_position)

        print(f"\nAttempting to solve for target pose:\n{target_pose}")

        # 3. Provide an initial guess for the joint angles (optional, but recommended)
        # Using the neutral configuration (often all zeros) is a good starting point.
        initial_q = pin.neutral(ik_solver.model)

        # 4. Solve for the joint angles
        solution_q = ik_solver.solve(target_pose, initial_guess=initial_q)

        # 5. Print the results
        if solution_q is not None:
            print("\nIK Solution Found (joint angles):")
            print(solution_q)

            # Optional: Verify the solution with forward kinematics
            pin.forwardKinematics(ik_solver.model, ik_solver.data, solution_q)
            pin.updateFramePlacements(ik_solver.model, ik_solver.data)
            final_pose = ik_solver.data.oMf[ik_solver.end_effector_frame_id]
            print("\nResulting Pose from FK:")
            print(final_pose)
            
            final_error = pin.log(target_pose.actInv(final_pose)).vector
            print(f"\nFinal error norm: {np.linalg.norm(final_error)}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")