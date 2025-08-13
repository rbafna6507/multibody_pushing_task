# import gym
# import torch
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import time
# import pybullet as p # Ensure pybullet is imported

# # --- Import necessary components from project files ---
# try:
#     # Assuming cleaned files are in the same directory
#     from panda_pushing_env import PandaPushingEnv, BOX_SIZE, TARGET_POSE_MULTI
#     from learning_state_dynamics import ResidualDynamicsModel, PushingController, multi_body_cost_function
#     from visualizers import GIFVisualizer # Needed for GIF generation
#     print("Successfully imported components.")
# except ImportError as e:
#     print(f"Error importing necessary components: {e}")
#     print("Please ensure panda_pushing_env.py, learning_state_dynamics.py,")
#     print("visualizers.py, and mppi.py are accessible in the Python path.")
#     exit()

# # --- Configuration ---
# MODEL_FILENAME = 'contact_data_pushing_multi_body_multistep_residual_model.pt'
# GIF_FILENAME = 'pushing_demo.gif'
# ASSETS_DIR = 'assets' # Assumes 'assets' folder is in the same directory

# # Check if assets directory exists
# if not os.path.exists(ASSETS_DIR):
#     print(f"Error: Assets directory '{ASSETS_DIR}' not found.")
#     print("Please ensure the 'assets' folder containing URDF files is in the same directory as demo.py.")
#     exit()

# # Check if model file exists
# model_path = MODEL_FILENAME # Assumes model is in the same directory
# if not os.path.exists(model_path):
#     # Try searching in the assets directory as a fallback
#     model_path_alt = os.path.join(ASSETS_DIR, MODEL_FILENAME)
#     if os.path.exists(model_path_alt):
#         model_path = model_path_alt
#     else:
#         print(f"Error: Model file '{MODEL_FILENAME}' not found in the current directory or '{ASSETS_DIR}'.")
#         print("Please ensure the pre-trained model file is accessible.")
#         exit()
# print(f"Using model file: {model_path}")


# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # --- Environment and Model Setup ---
# print("Setting up environment and loading model...")

# # Environment setup (multi-object, for visualization)
# # Use a visualizer that saves frames for GIF
# visualizer = GIFVisualizer()

# env = PandaPushingEnv(
#     visualizer=visualizer,
#     render_non_push_motions=True, # Render all motions for the GIF
#     is_multi_object=True,
#     camera_heigh=480, # Adjust camera size for GIF
#     camera_width=640,
#     render_every_n_steps=1 # Render every step
# )

# # Model setup
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]

# dynamics_model = ResidualDynamicsModel(state_dim, action_dim)
# try:
#     dynamics_model.load_state_dict(torch.load(model_path, map_location=device))
#     dynamics_model.to(device)
#     dynamics_model.eval() # Set model to evaluation mode
#     print("Model loaded successfully.")
# except Exception as e:
#     print(f"Error loading model state_dict: {e}")
#     exit()

# # --- Controller Setup ---
# print("Setting up MPPI controller...")

# # MPPI parameters (tuned based on notebook)
# num_samples = 1000 # Number of samples per step
# horizon = 40       # Planning horizon

# try:
#     controller = PushingController(
#         env,
#         dynamics_model,
#         multi_body_cost_function, # Use the multi-body cost function
#         num_samples=num_samples,
#         horizon=horizon
#     )
#     print("Controller initialized.")
# except Exception as e:
#     print(f"Error initializing controller: {e}")
#     exit()

# # --- Run Control Loop ---
# print("Running MPPI control loop and generating GIF...")
# state = env.reset()
# print(f"Initial State: {state}")
# print(f"Target Goal State (Planar): {TARGET_POSE_MULTI}")

# num_steps_max = 80 # Maximum steps for the demo episode
# goal_reached = False
# frames = [] # Store frames for GIF

# try:
#     for i in tqdm(range(num_steps_max), desc="Demo Steps"):
#         action = controller.control(state)
#         state, reward, done, _ = env.step(action)

#         # Render frame using the environment's render_frame method
#         # which adds to the visualizer's frame list
#         env.render_frame()

#         if done:
#             # Check if done because the goal was reached
#             goal_distance = np.linalg.norm(state[3:5] - TARGET_POSE_MULTI[:2])
#             # Use a reasonable threshold for goal check in demo
#             goal_threshold = BOX_SIZE * 0.7
#             if goal_distance < goal_threshold:
#                 goal_reached = True
#                 print("\nGoal reached!")
#             else:
#                 print("\nEpisode finished (out of bounds or other reason).")
#             break
#         # time.sleep(0.05) # Optional delay

# except Exception as e:
#     print(f"\nAn error occurred during the control loop: {e}")
#     import traceback
#     traceback.print_exc()
# finally:
#     # --- Final Evaluation and GIF Saving ---
#     print(f"\nFinished control loop after {i+1} steps.")
#     print(f"GOAL REACHED: {goal_reached}")

#     # Save the collected frames as a GIF using the visualizer
#     print(f"Saving GIF to {GIF_FILENAME}...")
#     visualizer.get_gif()
#     print("GIF saved.")

#     # Clean up PyBullet connection
#     if p.isConnected():
#         p.disconnect()
#     print("PyBullet disconnected.")




import gym
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time # Import time for sleep
import pybullet as p # Ensure pybullet is imported

# --- Import necessary components from project files ---
try:
    # Assuming cleaned files are in the same directory
    from panda_pushing_env import PandaPushingEnv, BOX_SIZE, TARGET_POSE_MULTI
    from learning_state_dynamics import ResidualDynamicsModel, PushingController, multi_body_cost_function
    # visualizers.py is no longer needed for real-time rendering
    print("Successfully imported components.")
except ImportError as e:
    print(f"Error importing necessary components: {e}")
    print("Please ensure panda_pushing_env.py, learning_state_dynamics.py,")
    print("and mppi.py are accessible in the Python path.")
    exit()

# --- Configuration ---
MODEL_FILENAME = 'contact_data_pushing_multi_body_multistep_residual_model.pt'
# GIF_FILENAME = 'pushing_demo.gif' # No longer needed
ASSETS_DIR = 'assets' # Assumes 'assets' folder is in the same directory

# Check if assets directory exists
if not os.path.exists(ASSETS_DIR):
    print(f"Error: Assets directory '{ASSETS_DIR}' not found.")
    print("Please ensure the 'assets' folder containing URDF files is in the same directory as demo.py.")
    exit()

# Check if model file exists
model_path = MODEL_FILENAME # Assumes model is in the same directory
if not os.path.exists(model_path):
    model_path_alt = os.path.join(ASSETS_DIR, MODEL_FILENAME)
    if os.path.exists(model_path_alt):
        model_path = model_path_alt
    else:
        print(f"Error: Model file '{MODEL_FILENAME}' not found in the current directory or '{ASSETS_DIR}'.")
        print("Please ensure the pre-trained model file is accessible.")
        exit()
print(f"Using model file: {model_path}")


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Environment and Model Setup ---
print("Setting up environment (GUI mode) and loading model...")

# Environment setup for real-time rendering
# Use debug=True to connect in GUI mode
env = PandaPushingEnv(
    debug=True, # <<< Enable GUI mode
    visualizer=None, # <<< No separate visualizer needed
    render_non_push_motions=True, # Render all motions
    is_multi_object=True,
    camera_heigh=600, # Adjust camera size if needed
    camera_width=800,
    render_every_n_steps=1 # Render every step (handled by GUI)
)

# Model setup
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

dynamics_model = ResidualDynamicsModel(state_dim, action_dim)
try:
    dynamics_model.load_state_dict(torch.load(model_path, map_location=device))
    dynamics_model.to(device)
    dynamics_model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    exit()

# --- Controller Setup ---
print("Setting up MPPI controller...")

# MPPI parameters (tuned based on notebook)
num_samples = 1000 # Number of samples per step
horizon = 40       # Planning horizon

try:
    controller = PushingController(
        env,
        dynamics_model,
        multi_body_cost_function, # Use the multi-body cost function
        num_samples=num_samples,
        horizon=horizon
    )
    print("Controller initialized.")
except Exception as e:
    print(f"Error initializing controller: {e}")
    exit()

# --- Run Control Loop ---
print("Running MPPI control loop with real-time rendering...")
state = env.reset()
print(f"Initial State: {state}")
print(f"Target Goal State (Planar): {TARGET_POSE_MULTI}")

num_steps_max = 80 # Maximum steps for the demo episode
goal_reached = False

try:
    for i in tqdm(range(num_steps_max), desc="Demo Steps"):
        action = controller.control(state)
        state, reward, done, _ = env.step(action)

        # In GUI mode, rendering happens automatically with p.stepSimulation()
        # Add a small delay to make it watchable
        time.sleep(0.05) # Adjust delay as needed

        if done:
            goal_distance = np.linalg.norm(state[3:5] - TARGET_POSE_MULTI[:2])
            goal_threshold = BOX_SIZE * 0.7
            if goal_distance < goal_threshold:
                goal_reached = True
                print("\nGoal reached!")
            else:
                print("\nEpisode finished (out of bounds or other reason).")
            break

except Exception as e:
    print(f"\nAn error occurred during the control loop: {e}")
    import traceback
    traceback.print_exc()
finally:
    # --- Final Evaluation ---
    print(f"\nFinished control loop after {i+1} steps.")
    print(f"GOAL REACHED: {goal_reached}")

    # Keep the GUI open for a bit after finishing
    print("Simulation finished. Closing GUI in 5 seconds...")
    time.sleep(5)

    # Clean up PyBullet connection
    if p.isConnected():
        p.disconnect()
    print("PyBullet disconnected.")

