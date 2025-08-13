import gym
from gym import spaces
import os
import pybullet as p
import pybullet_data as pd
import math
import numpy as np
from tqdm import tqdm
import argparse
import time

print("imported panda_pushing_env from /again")

# Asset paths
hw_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(hw_dir, 'assets')
if not os.path.exists(assets_dir):
     assets_dir = os.path.join(os.getcwd(), 'assets')
     if not os.path.exists(assets_dir):
          print("Warning: Assets directory not found. Please ensure 'assets' folder is accessible.")
          assets_dir = "."

BOX_SIZE = 0.1

# Multi-Object Poses
INTERMEDIATE_START_POSE_PLANAR = np.array([0.35, 0.05, 0.0])
TARGET_START_POSE_PLANAR = np.array([0.5, 0.0, 0.0])
TARGET_POSE_MULTI = np.array([0.75, 0.0, 0.0]) # Adjusted goal

# Single-Object Poses (Fallback)
TARGET_POSE_FREE = np.array([0.8, 0., 0.])
TARGET_POSE_OBSTACLES = np.array([0.8, -0.1, 0.])
OBSTACLE_CENTRE = np.array([0.6, 0.2, 0.])
OBSTACLE_HALFDIMS = np.array([0.05, 0.25, 0.05])

class PandaPushingEnv(gym.Env):
    def __init__(self, debug=False, visualizer=None, render_non_push_motions=True,
                 render_every_n_steps=1, camera_heigh=84, camera_width=84,
                 is_multi_object=True):
        self.debug = debug
        self.visualizer = visualizer
        self.render_every_n_steps = render_every_n_steps
        self.is_multi_object = is_multi_object
        self.include_obstacle = False if is_multi_object else False

        if p.getConnectionInfo()['isConnected'] == 0:
            connection_mode = p.GUI if debug else p.DIRECT
            options = "--opengl2" if connection_mode == p.DIRECT else ""
            p.connect(connection_mode, options=options)
        p.setAdditionalSearchPath(pd.getDataPath())

        self.episode_step_counter = 0
        self.episode_counter = 0
        self.frames = []

        self.pandaUid = None
        self.tableUid = None
        self.intermediateUid = None
        self.targetUid_actual = None
        self.targetUid_viz = None
        self.obstacleUid = None

        self.object_file_path = os.path.join(assets_dir, "objects/cube/cube.urdf")
        self.target_file_path = os.path.join(assets_dir, "objects/cube/cube.urdf")
        self.obstacle_file_path = os.path.join(assets_dir, "objects/obstacle/obstacle.urdf")

        self.init_panda_joint_state = np.array([0., 0., 0., -np.pi * 0.5, 0., np.pi * 0.5, 0.])
        self.left_finger_idx = 9
        self.right_finger_idx = 10
        self.end_effector_idx = 11
        self.ik_precision_treshold = 1e-4
        self.max_ik_repeat = 50
        self.fixed_orientation = p.getQuaternionFromEuler([0., -math.pi, 0.])

        self.lower_z = 0.02
        self.raise_z = 0.3
        self.push_length = 0.1

        self.render_non_push_motions = render_non_push_motions
        self.is_render_on = True
        self.camera_height = camera_heigh
        self.camera_width = camera_width
        if self.debug:
            p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=0, cameraPitch=-45,
                                         cameraTargetPosition=[0.5, 0.0, 0.1])

        self.block_size = BOX_SIZE
        self.space_limits = [np.array([0.05, -0.35]), np.array([0.95, 0.35])]

        if self.is_multi_object:
            low_bounds = np.array([self.space_limits[0][0], self.space_limits[0][1], -np.pi] * 2, dtype=np.float32)
            high_bounds = np.array([self.space_limits[1][0], self.space_limits[1][1], np.pi] * 2, dtype=np.float32)
            self.state_dim = 6
            self.target_goal_pose_planar = TARGET_POSE_MULTI
        else:
            print("Warning: Initializing PandaPushingEnv in single-object mode.")
            low_bounds = np.array([self.space_limits[0][0], self.space_limits[0][1], -np.pi], dtype=np.float32)
            high_bounds = np.array([self.space_limits[1][0], self.space_limits[1][1], np.pi], dtype=np.float32)
            self.state_dim = 3
            self.target_goal_pose_planar = TARGET_POSE_FREE

        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1, -np.pi * 0.5, 0], dtype=np.float32),
                                       high=np.array([1, np.pi * 0.5, 1], dtype=np.float32))
        self.action_dim = 3

        self.intermediate_start_pose = None
        self.target_start_pose = None
        self.target_goal_pose = None

    def _set_object_positions(self):
        if self.is_multi_object:
            self.intermediate_start_pose = self._planar_pose_to_world_pose(INTERMEDIATE_START_POSE_PLANAR)
            self.target_start_pose = self._planar_pose_to_world_pose(TARGET_START_POSE_PLANAR)
            self.target_goal_pose = self._planar_pose_to_world_pose(self.target_goal_pose_planar)
        else:
            start_pose_planar = np.array([0.4, 0., 0.0])
            self.target_start_pose = self._planar_pose_to_world_pose(start_pose_planar)
            self.target_goal_pose = self._planar_pose_to_world_pose(self.target_goal_pose_planar)

    def reset(self):
        self._set_object_positions()
        self.episode_counter += 1
        self.episode_step_counter = 0
        self.is_render_on = True
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        self.pandaUid = p.loadURDF(os.path.join(assets_dir, "franka_panda/panda.urdf"), useFixedBase=True)
        for i in range(len(self.init_panda_joint_state)):
            p.resetJointState(self.pandaUid, i, self.init_panda_joint_state[i])
        self.tableUid = p.loadURDF(os.path.join(assets_dir, "objects/table/table.urdf"), basePosition=[0.5, 0, -0.65])

        if self.is_multi_object:
            self.intermediateUid = p.loadURDF(self.object_file_path,
                                              basePosition=self.intermediate_start_pose[:3],
                                              baseOrientation=self.intermediate_start_pose[3:], globalScaling=1.0)
            p.changeVisualShape(self.intermediateUid, -1, rgbaColor=[0.9, 0.9, 0.1, 1]) # Yellow

            self.targetUid_actual = p.loadURDF(self.object_file_path,
                                               basePosition=self.target_start_pose[:3],
                                               baseOrientation=self.target_start_pose[3:], globalScaling=1.0)
            p.changeVisualShape(self.targetUid_actual, -1, rgbaColor=[0.9, 0.1, 0.1, 1]) # Red

            self.targetUid_viz = p.loadURDF(self.target_file_path,
                                            basePosition=self.target_goal_pose[:3],
                                            baseOrientation=self.target_goal_pose[3:],
                                            globalScaling=1., useFixedBase=True)
            p.setCollisionFilterGroupMask(self.targetUid_viz, -1, 0, 0)
            p.changeVisualShape(self.targetUid_viz, -1, rgbaColor=[0.05, 0.95, 0.05, .5]) # Green ghost

            for i in range(-1, p.getNumJoints(self.pandaUid)):
                 p.setCollisionFilterPair(self.pandaUid, self.targetUid_actual, i, -1, 0)
                 p.setCollisionFilterPair(self.pandaUid, self.targetUid_viz, i, -1, 0)
        else:
            self.targetUid_actual = p.loadURDF(self.object_file_path, basePosition=self.target_start_pose[:3], baseOrientation=self.target_start_pose[3:], globalScaling=1.)
            self.targetUid_viz = p.loadURDF(self.target_file_path, basePosition=self.target_goal_pose[:3], baseOrientation=self.target_goal_pose[3:], globalScaling=1., useFixedBase=True)
            p.setCollisionFilterGroupMask(self.targetUid_viz, -1, 0, 0)
            p.changeVisualShape(self.targetUid_viz, -1, rgbaColor=[0.05, 0.95, 0.05, .1])

        if self.include_obstacle:
            self.obstacleUid = p.loadURDF(self.obstacle_file_path, basePosition=OBSTACLE_CENTRE, useFixedBase=True)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        state = self.get_state()
        return state.astype(np.float32)

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        action = np.clip(action, self.action_space.low, self.action_space.high)

        if not self.action_space.contains(action):
             print(f"Warning: Action {action} is still not valid after clipping. Check bounds.")
             action = np.clip(action, self.action_space.low, self.action_space.high)

        self.episode_step_counter += 1
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        push_location_fraction, push_angle, push_length_fraction = action[0], action[1], action[2]
        push_location = push_location_fraction * self.block_size * 0.5 * 0.95
        push_length = push_length_fraction * self.push_length

        object_to_push_id = self.intermediateUid if self.is_multi_object else self.targetUid_actual
        if object_to_push_id is None:
             raise RuntimeError("Object to push is None. Check environment mode and reset.")

        self.push(object_to_push_id, push_location, push_angle, push_length=push_length)

        state = self.get_state()
        reward = 0.
        done = self._is_done(state)
        info = {}
        return state.astype(np.float32), reward, done, info

    def _is_done(self, state):
        done_bounds = False
        at_goal = False
        goal_distance = float('inf')

        if not hasattr(self, 'target_goal_pose_planar') or self.target_goal_pose_planar is None:
             print("ERROR in _is_done: self.target_goal_pose_planar not set! Check env.reset().")
             return True

        target_goal_pose = self.target_goal_pose_planar
        low_bounds_xy = self.space_limits[0]
        high_bounds_xy = self.space_limits[1]

        if self.is_multi_object:
            if state.shape[0] != 6:
                print(f"ERROR in _is_done (multi): Expected 6D state, got {state.shape}")
                return True

            state_intermediate_xy = state[:2]
            state_target_xy = state[3:5]
            state_target = state[3:]

            out_of_bounds_intermediate = not (np.all(state_intermediate_xy >= low_bounds_xy) and np.all(state_intermediate_xy <= high_bounds_xy))
            out_of_bounds_target = not (np.all(state_target_xy >= low_bounds_xy) and np.all(state_target_xy <= high_bounds_xy))
            done_bounds = out_of_bounds_intermediate or out_of_bounds_target

            goal_distance = np.linalg.norm(state_target[:2] - target_goal_pose[:2])
            goal_threshold = BOX_SIZE * 0.7 # Adjusted threshold for demo
            at_goal = goal_distance < goal_threshold

        else:
            if state.shape[0] != 3:
                 print(f"ERROR in _is_done (single): Expected 3D state, got {state.shape}")
                 return True

            state_xy = state[:2]
            out_of_bounds_single = not (np.all(state_xy >= low_bounds_xy) and np.all(state_xy <= high_bounds_xy))
            done_bounds = out_of_bounds_single
            goal_distance = np.linalg.norm(state[:2] - target_goal_pose[:2])
            at_goal = goal_distance < (BOX_SIZE * 0.5)

        done = done_bounds or at_goal
        return done

    def lower_down(self, step_size=0.05):
        current_pos = self.get_end_effector_pos()
        target_pos = current_pos.copy()
        target_pos[-1] = self.lower_z
        self._move_ee_trajectory(target_pos, step_size=step_size)

    def raise_up(self, step_size=0.05):
        current_pos = self.get_end_effector_pos()
        target_pos = current_pos.copy()
        target_pos[-1] = self.raise_z
        self._move_ee_trajectory(target_pos, step_size=step_size)

    def planar_push(self, push_angle, push_length=None, step_size=0.001):
        if push_length is None: push_length = self.push_length
        if abs(push_length) < 1e-6: return
        current_pos = self.get_end_effector_pos()
        target_pos = current_pos + push_length * np.array([np.cos(push_angle), np.sin(push_angle), 0])
        self._move_ee_trajectory(target_pos, step_size=step_size)

    def set_planar_xy(self, xy, theta=0., step_size=0.05):
        current_z = self.get_end_effector_pos()[-1]
        target_pos = np.array([xy[0], xy[1], current_z])
        self._move_ee_trajectory(target_pos, step_size=step_size)

    def push(self, object_id_to_push, push_location, push_angle, push_length=None):
        current_block_pose = self.get_object_pos_planar(object_id_to_push)
        theta = current_block_pose[-1]

        original_render_state = self.is_render_on
        if not self.render_non_push_motions: self.is_render_on = False
        self.raise_up()

        start_gap = 0.1
        start_xy_bf = np.array([- (BOX_SIZE / 2.0 + start_gap), push_location])
        w_R_bf = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        start_xy_wf = w_R_bf @ start_xy_bf + current_block_pose[:2]

        self.set_planar_xy(start_xy_wf, theta=theta)
        self.lower_down()

        approach_dist = start_gap - 0.01
        if approach_dist > 0:
            self.planar_push(theta, push_length=approach_dist, step_size=0.005)

        self.is_render_on = original_render_state
        self.planar_push(push_angle + theta, push_length=push_length, step_size=0.005)

    def _move_ee_trajectory(self, target_ee_pos, step_size=0.001):
        start_ee_pos = self.get_end_effector_pos()
        goal_error = target_ee_pos - start_ee_pos
        goal_length = np.linalg.norm(goal_error)
        if goal_length < 1e-6: return
        goal_dir = goal_error / goal_length
        num_steps = max(1, int(goal_length // step_size))

        for step_i in range(num_steps):
            frac = (step_i + 1) / num_steps
            target_ee_pos_i = start_ee_pos + frac * goal_length * goal_dir
            render_step_i = step_i % self.render_every_n_steps == 0
            self._move_robot_ee(target_ee_pos_i, render=render_step_i)
        self._move_robot_ee(target_ee_pos, render=True)

    def _move_robot_ee(self, target_ee_pos, render=True):
        distance = float('inf')
        repeat_counter = 0
        current_joint_poses = self.get_all_joint_pos()
        lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        joint_ranges = [u - l for u, l in zip(upper_limits, lower_limits)]
        rest_poses = self.init_panda_joint_state[:7]

        while distance > self.ik_precision_treshold and repeat_counter < self.max_ik_repeat:
            try:
                computed_ik_joint_pos = p.calculateInverseKinematics(
                    self.pandaUid, self.end_effector_idx, target_ee_pos,
                    self.fixed_orientation, maxNumIterations=100,
                    residualThreshold=self.ik_precision_treshold,
                    lowerLimits=lower_limits, upperLimits=upper_limits,
                    jointRanges=joint_ranges, restPoses=rest_poses
                    )

                computed_ik_joint_pos_clipped = [np.clip(p, l, u) for p, l, u in zip(computed_ik_joint_pos[:7], lower_limits, upper_limits)]

                p.setJointMotorControlArray(self.pandaUid, list(range(7)), p.POSITION_CONTROL,
                                            targetPositions=computed_ik_joint_pos_clipped, forces=[500.0] * 7)
                p.setJointMotorControl2(self.pandaUid, self.right_finger_idx, p.POSITION_CONTROL, 0., force=100)
                p.setJointMotorControl2(self.pandaUid, self.left_finger_idx, p.POSITION_CONTROL, 0., force=100)

                p.stepSimulation()

                actual_ee_pos = self.get_end_effector_pos()
                distance = np.linalg.norm(target_ee_pos - actual_ee_pos)
                repeat_counter += 1
            except Exception as e:
                print(f"Error in IK or simulation step: {e}")
                break

        if self.debug:
            self._debug_step()
        elif render and self.is_render_on:
            self.render_frame()

    def get_state(self):
        if self.is_multi_object:
            if self.intermediateUid is None or self.targetUid_actual is None:
                 raise RuntimeError("Objects not loaded correctly in multi-object mode during get_state()")
            intermediate_planar = self.get_object_pos_planar(self.intermediateUid)
            target_planar = self.get_object_pos_planar(self.targetUid_actual)
            state = np.concatenate([intermediate_planar, target_planar])
        else:
            if self.targetUid_actual is None: raise RuntimeError("Target object not loaded in single-object mode during get_state()")
            state = self.get_object_pos_planar(self.targetUid_actual)
        return state.astype(np.float32)

    def get_object_pose(self, object_id):
        try:
            pos, quat = p.getBasePositionAndOrientation(object_id)
            return np.concatenate([np.asarray(pos), np.asarray(quat)])
        except Exception as e:
            print(f"Error getting pose for object ID {object_id}: {e}")
            raise e

    def get_object_pos_planar(self, object_id):
        object_pos_wf = self.get_object_pose(object_id)
        object_pos_planar = self._world_pose_to_planar_pose(object_pos_wf)
        return object_pos_planar

    def get_end_effector_pos(self):
        try:
             link_state = p.getLinkState(self.pandaUid, self.end_effector_idx)
             if link_state:
                 return np.asarray(link_state[0])
             else:
                 joint_states = [p.getJointState(self.pandaUid, i)[0] for i in range(7)]
                 fk_result = p.getLinkState(self.pandaUid, self.end_effector_idx, computeForwardKinematics=1)
                 if fk_result: return np.asarray(fk_result[0])
                 return np.array([0.5, 0.0, 0.1])
        except Exception as e:
             print(f"Error getting end effector position: {e}")
             return np.array([0.5, 0.0, 0.1])

    def get_all_joint_pos(self):
        try:
            joint_states = p.getJointStates(self.pandaUid, list(range(7)))
            return np.array([state[0] for state in joint_states])
        except Exception as e:
            print(f"Error getting joint positions: {e}")
            return np.zeros(7)

    def _planar_pose_to_world_pose(self, planar_pose):
        theta = planar_pose[-1]
        plane_z = BOX_SIZE / 2.0
        world_pos = np.array([planar_pose[0], planar_pose[1], plane_z])
        quat = p.getQuaternionFromEuler([0, 0, theta])
        return np.concatenate([world_pos, np.asarray(quat)])

    def _world_pose_to_planar_pose(self, world_pose):
        pos = world_pose[:3]
        quat = world_pose[3:]
        euler = p.getEulerFromQuaternion(quat)
        theta = euler[2]
        return np.array([pos[0], pos[1], theta])

    def render_image(self, camera_pos, camera_orn_pyr, camera_width, camera_height, nearVal=0.01, distance=0.7):
        yaw, pitch, roll = camera_orn_pyr[0], camera_orn_pyr[1], camera_orn_pyr[2]
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_pos, distance=distance,
                                                          yaw=yaw, pitch=pitch, roll=roll,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(camera_width) / camera_height,
                                                   nearVal=nearVal, farVal=100.0)
        try:
            renderer = p.ER_BULLET_HARDWARE_OPENGL if self.debug else p.ER_TINY_RENDERER
            flags = p.ER_NO_SEGMENTATION_MASK
            connection_info = p.getConnectionInfo()
            if connection_info['connectionMethod'] == p.DIRECT:
                 renderer = p.ER_TINY_RENDERER

            width, height, rgba, _, _ = p.getCameraImage(
                width=camera_width, height=camera_height, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix, renderer=renderer, flags=flags
            )
            rgb_array = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
            return rgb_array
        except Exception as e:
            print(f"Error getting camera image: {e}")
            return np.zeros((camera_height, camera_width, 3), dtype=np.uint8)

    def _debug_step(self):
        pass

    def render_frame(self):
        if self.debug:
            time.sleep(0.01)
            pass
        elif self.visualizer is not None:
            if self.is_render_on:
                rgb_img = self.render_image(camera_pos=[0.5, 0.0, 0.2],
                                            camera_orn_pyr=[0, -45, 0],
                                            camera_width=self.camera_width,
                                            camera_height=self.camera_height,
                                            distance=1.6)
                self.frames.append(rgb_img)
                if self.visualizer is not None:
                    self.visualizer.set_data(rgb_img)
        else:
            pass

_EPS = np.finfo(float).eps * 4.0

def quaternion_matrix(quaternion):
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS: return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0 - q[1, 1] - q[2, 2], q[0, 1] - q[2, 3], q[0, 2] + q[1, 3], 0.0),
        (q[0, 1] + q[2, 3], 1.0 - q[0, 0] - q[2, 2], q[1, 2] - q[0, 3], 0.0),
        (q[0, 2] - q[1, 3], q[1, 2] + q[0, 3], 1.0 - q[0, 0] - q[1, 1], 0.0),
        (0.0, 0.0, 0.0, 1.0)
    ), dtype=np.float64)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help="Run in PyBullet GUI mode.")
    script_args, _ = parser.parse_known_args()

    print("Testing Multi-Object Environment...")
    env = PandaPushingEnv(debug=script_args.debug, is_multi_object=True)
    state = env.reset()
    print(f"Initial State (Multi-Object): {state}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    for i in tqdm(range(15), desc="Multi-Object Steps"):
        action_i = env.action_space.sample()
        try:
             state, reward, done, info = env.step(action_i)
             if done:
                 print("Episode finished.")
                 state = env.reset()
                 print("Environment reset.")
        except Exception as e:
             print(f"Error during step {i}: {e}")
             import traceback
             traceback.print_exc()
             break
    print("Multi-Object Test Finished.")
    p.disconnect()
