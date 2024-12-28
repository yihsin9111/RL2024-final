import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import os
import time
import json
import collections
import h5py

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import SIM_TASK_CONFIGS

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from pyquaternion import Quaternion
import argparse

import IPython
e = IPython.embed

def read_text(text_path, offset=0, half=0):
    with open(text_path, 'r') as txt_file:
        data = txt_file.readline().split(" ")
        data = list(filter(lambda x: x != "", data))

    if half:
        data_list = np.array(data)[offset:half].tolist(
        ) + np.array(data)[half+offset:].tolist()
        return np.array(data_list).reshape((-1, 3)).astype(np.float32)
    else:
        return np.array(data)[offset:].reshape((-1, 3)).astype(np.float32)

def rotation_matrix_from_vectors(a, b):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def rotation_axis_angle(a, b):
    # Normalize the vectors
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    
    # Calculate the rotation axis (cross product)
    rotation_axis = np.cross(a, b)
    
    # Calculate the rotation angle (dot product)
    dot_product = np.dot(a, b)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Normalize the rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    
    return rotation_axis, angle_deg

def axis_angle_to_quaternion(axis, degrees):
    # Normalize the axis
    axis = axis / np.linalg.norm(axis)
    # Convert degrees to radians
    radians = np.deg2rad(degrees)
    # Create the quaternion
    quaternion = R.from_rotvec(axis * radians)
    return quaternion

def combine_rotations(axis1, degrees1, axis2, degrees2):
    # Convert the first rotation to a quaternion
    quat1 = axis_angle_to_quaternion(axis1, degrees1)
    # Convert the second rotation to a quaternion
    quat2 = axis_angle_to_quaternion(axis2, degrees2)
    # Combine the two quaternions by multiplying them
    combined_quat = R.from_quat(quat1) * R.from_quat(quat2)
    return combined_quat

def if_close(vec, a, b, c):
    # a - x, b - y, c - z
    if vec[0]**2/(a**2) + vec[1]**2/(b**2) + vec[2]**2/(c**2) < 1:
        return True
    else:
        return False

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None, cube_pose=None, cube_quat=None):
        super().__init__(random=random)
        self.cube_pose = np.array(cube_pose)
        self.cube_quat = np.array(cube_quat)

    def before_step(self, action, physics):

        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        print("initialize_robots")
        physics.named.data.qpos[:16] = START_ARM_POSE

        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084]) # initialize gripper position
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0]) # initialize gripper orientation
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        # print(np.append(self.cube_pose, self.cube_quat))
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7],np.append(self.cube_pose, self.cube_quat))
        # randomize box position
        # cube_pose = sample_box_pose()
        # box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        # np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs
        # print("touch_table: ",physics.data.qpos[box_start_idx+2])
        reward = 0
        if touch_right_gripper or touch_left_gripper:
            reward = 1
        if (touch_right_gripper or touch_left_gripper) and not touch_table: # lifted
            reward = 2
        return reward

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError

class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None, cube_pose=None, cube_quat=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.cube_pose = np.array(cube_pose)
        self.cube_quat = np.array(cube_quat)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            # print("datactrl: ", physics.data.ctrl)
            np.copyto(physics.data.ctrl, START_ARM_POSE)
        #     # assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = np.append(self.cube_pose, self.cube_quat)
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        # print("all_contact_pairs: ", all_contact_pairs)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        touch_table = ("red_box", "table") in all_contact_pairs
        # print("touch_table: ",physics.data.xpos[physics.model.name2id('red_box', 'geom')])
        reward = 0
        if touch_right_gripper or touch_left_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper and not touch_table: # lifted
            reward = 2
        if touch_right_gripper and touch_table:
            reward = 1
        if touch_left_gripper and touch_table:
            reward = 1
        return reward
    
class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        # print(curr_waypoint, next_waypoint, t)
        if next_waypoint["t"] - curr_waypoint["t"] < 1e-6:
            t_frac = 0
        else:
            t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        while self.left_trajectory[0]['t'] <= self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        while self.right_trajectory[0]['t'] <= self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)
        # print("left_gripper: ", [left_gripper])
        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])
        # print("action_left: ", action_left)
        self.step_count += 1
        return np.concatenate([action_left, action_right])

class ArmPolicy(BasePolicy):
    def __init__(self, inject_noise=False, left_trajectory=None, right_trajectory=None):
        super().__init__(inject_noise=inject_noise)
        self.left_trajectory = right_trajectory
        self.right_trajectory = left_trajectory
            
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        print("generate_trajectory")
        # print("init_mocap_pose_right: ", init_mocap_pose_right)
        # print("init_mocap_pose_left: ", init_mocap_pose_left)
        for idx, t in enumerate(self.left_trajectory):
            if idx == 0:
                new_left_ts = {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}
            else:
                new_left_ts = {
                    "t": t["t"],
                    "xyz": np.array(t["xyz"]),
                    "quat": np.array(t["quat"]),
                    "gripper": t["gripper"]
                }
            self.left_trajectory[idx] = new_left_ts
        
        for idx, t in enumerate(self.right_trajectory):
            if idx == 0:
                new_right_ts = {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}
            else:
                new_right_ts = {
                    "t": t["t"],
                    "xyz": np.array(t["xyz"]),
                    "quat": np.array(t["quat"]),
                    "gripper": t["gripper"]
                }
            self.right_trajectory[idx] = new_right_ts
        # print("left: ", len(left))
        # print("right: ", len(right))
        # self.right_trajectory = right
        # self.left_trajectory = left
        # for idx, t in trajectory:
        #     new_left_ts = {"t": idx, "xyz": t[:3], "quat": t[3:7], "gripper": t[7]}, # (gripper) xyz: position, quat: orientation, gripper: gripper control
        #     self.left_trajectory.append(new_left_ts)
        #     new_right_ts = {"t": idx, "xyz": t[8:11], "quat": t[11:15], "gripper": t[15]},
        #     self.right_trajectory.append(new_right_ts)

def rotation_matrix_z(degrees):
    # Convert degrees to radians
    radians = np.deg2rad(degrees)
    
    # Create the rotation matrix
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians), np.cos(radians), 0],
        [0, 0, 1]
    ])
    
    return rotation_matrix    
# initialize mujoco environment

hand_connection = {
        (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20),
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        (21+1, 21+2), (21+2, 21+3), (21+3, 21+4),
        (21+5, 21+6), (21+6, 21+7), (21+7, 21+8),
        (21+9, 21+10), (21+10, 21+11), (21+11, 21+12),
        (21+13, 21+14), (21+14, 21+15), (21+15, 21+16),
        (21+17, 21+18), (21+18, 21+19), (21+19, 21+20),
        (21+0, 21+1), (21+0, 21+5), (21+0, 21+9), (21+0, 21+13), (21+0, 21+17)
    }

obj_connection = {
    (1, 2), (2, 3), (3, 4), (4, 1),
    (1, 5), (2, 6), (3, 7), (4, 8),
    (5, 6), (6, 7), (7, 8), (8, 5)
}

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

colorclass = plt.cm.ScalarMappable(cmap='jet')
colors = colorclass.to_rgba(np.linspace(0,1,int(42)))
colormap = (colors[:, 0:3])


obj_colorclass = plt.cm.ScalarMappable(cmap='rainbow')
obj_colors = obj_colorclass.to_rgba(np.linspace(0,1,int(21)))
obj_colormap = (obj_colors[:, 0:3])

color_array = np.zeros((21, 3))

# Set the first element to red (RGB: [1, 0, 0])
color_array[6] = [1, 0, 0]
color_array[7] = [0, 1, 0]
color_array[2] = [0, 0, 1]
color_array[5] = [0, 0, 1]

hand_color = np.zeros((42, 3))
hand_color[0] = [1,0,0]
hand_color[21] = [1,0,0]
left_thumb = [1, 2, 3, 4]
left_first = [5, 6, 7, 8]
right_thumb = [22, 23, 24, 25]
right_first = [26, 27, 28, 29]
left_third_joint = [1, 5, 9, 13, 17]
right_third_joint = [22, 26, 30, 34, 38]
left_finger = [4, 8, 12, 16, 20]
right_finger = [25, 29, 33, 37, 41]
for i in left_thumb:
    hand_color[i] = [0, 1, 0]
for i in left_first:
    hand_color[i] = [0, 0, 1]
for i in right_thumb:
    hand_color[i] = [1, 1, 0]
for i in right_first:
    hand_color[i] = [1, 0, 1]

# data_dir = [ "./cocoa", "./milk", "chips","./cappuccino",  "./espresso", ]
episode_len = 600

left_trajectory = []
right_trajectory = []

# np.random.seed(int(time.time()))


def simulate_episode(ddir, odir):

        time_ = 0

        recenter_center = np.array([0 + np.random.uniform(-0.15, 0.15), 0.4 + np.random.uniform(-0.1, 0.2), 0])
        rand_rotation = np.random.uniform(0, 45)
        cube_pose = np.array([0, 0, 0.1])
        cube_quat = np.array([0, 0, 0, 1])

        left_hand_open = 1
        right_hand_open = 1
        set_obj = False
        hand_offset = 0
        left_hand = True

        hand_directory = ddir + "/hand_pose/"
        obj_directory = ddir + "/obj_pose/"

        hand_filenames = sorted(os.listdir(hand_directory))
        obj_filenames = sorted(os.listdir(obj_directory))
        print(hand_filenames)
        exist_rotations = False
        iteration_len = min(len(hand_filenames), len(obj_filenames))
        for i in range(min(len(hand_filenames), len(obj_filenames))):  
            # with open(action_directory+action_filenames[i], 'r') as txt_file:
            #     action_label = txt_file.readline().split(" ")
            #     # print(action_label)
            #     if not (int(action_label[0]) > 0 and int(action_label[0]) < 10):
            #         continue

            print(hand_points.shape)

            hand_points = read_text(hand_directory+hand_filenames[i], 1, 64)
            # print(hand_points.shape)

            # obj_dir = "./obj_pose/0000" + str(i) + ".txt"
            box_points = read_text(obj_directory+obj_filenames[i], 1)
            # print(box_points.shape)

            # if z_min > np.min(box_points[:, 2]):
            #     z_min = np.min(box_points[:, 2])
            
            if exist_rotations == False:
                # check orientation 

                rotation_z_matrix = rotation_matrix_z(rand_rotation)
                left_hand_pos = np.mean(hand_points[0:21], axis=0)
                right_hand_pos = np.mean(hand_points[21:], axis=0)
                plane_vector = [np.cross(left_hand_pos-box_points[6], right_hand_pos-box_points[6])/np.linalg.norm(np.cross(left_hand_pos-box_points[6], right_hand_pos-box_points[6])),
                                np.cross(left_hand_pos-box_points[7], right_hand_pos-box_points[7])/np.linalg.norm(np.cross(left_hand_pos-box_points[7], right_hand_pos-box_points[7])),
                                np.cross(left_hand_pos-box_points[0], right_hand_pos-box_points[0])/np.linalg.norm(np.cross(left_hand_pos-box_points[0], right_hand_pos-box_points[0]))]
                
                # axis end points: 2, 5, 7
                min_z_axis = np.cross(box_points[7]-box_points[6], box_points[2]-box_points[6])
                min_z_axis = min_z_axis / np.linalg.norm(min_z_axis)
                max_dot = np.min([abs(np.dot(plane_vector[0], min_z_axis)), abs(np.dot(plane_vector[1], min_z_axis)), abs(np.dot(plane_vector[2], min_z_axis))])
                box_x_axis = np.linalg.norm(box_points[2] - box_points[6])/2
                box_y_axis = np.linalg.norm(box_points[7] - box_points[6])/2
                box_z_axis = np.linalg.norm(box_points[5] - box_points[6])/2

                candidate_1 = np.cross(box_points[7]-box_points[6], box_points[5]-box_points[6])
                cand1_z_axis = candidate_1 / np.linalg.norm(candidate_1)
                cand1_dot = np.min([abs(np.dot(plane_vector[0], cand1_z_axis)), abs(np.dot(plane_vector[1], cand1_z_axis)), abs(np.dot(plane_vector[2], cand1_z_axis))])
                if cand1_dot > max_dot or np.linalg.norm(box_points[2] - box_points[6]) > 2*min(np.linalg.norm(box_points[7] - box_points[6]), np.linalg.norm(box_points[5] - box_points[6])):
                    max_dot = cand1_dot
                    min_z_axis = cand1_z_axis
                    box_x_axis = np.linalg.norm(box_points[5] - box_points[6])/2
                    box_y_axis = np.linalg.norm(box_points[7] - box_points[6])/2
                    box_z_axis = np.linalg.norm(box_points[2] - box_points[6])/2
                
                candidate_2 = np.cross(box_points[5]-box_points[6], box_points[2]-box_points[6])
                cand2_z_axis = candidate_2 / np.linalg.norm(candidate_2)
                cand2_dot = np.min([abs(np.dot(plane_vector[0], cand2_z_axis)), abs(np.dot(plane_vector[1], cand2_z_axis)), abs(np.dot(plane_vector[2], cand2_z_axis))])
                if cand2_dot > max_dot or np.linalg.norm(box_points[7] - box_points[6])> 2*min(np.linalg.norm(box_points[2] - box_points[6]), np.linalg.norm(box_points[5] - box_points[6])):
                    max_dot = cand2_dot
                    min_z_axis = cand2_z_axis
                    box_x_axis = np.linalg.norm(box_points[5] - box_points[6])/2
                    box_y_axis = np.linalg.norm(box_points[2] - box_points[6])/2
                    box_z_axis = np.linalg.norm(box_points[7] - box_points[6])/2

                if np.dot(min_z_axis, plane_vector[2]) < 0:
                    min_z_axis = -min_z_axis

                z_axis = min_z_axis
                z_axis = z_axis / np.linalg.norm(z_axis)

                


                rotation_matrix = rotation_matrix_from_vectors(z_axis, np.array([0, 0, 1]))

                hand_points = np.dot(hand_points, rotation_matrix.T)
                box_points = np.dot(box_points, rotation_matrix.T)

                xy_vector = (np.mean(hand_points[0:21], axis=0) - np.mean(hand_points[21:], axis=0))
                # print(xy_vector)
                xy_vector[2] = 0
                
                xy_vector = xy_vector / np.linalg.norm(xy_vector)
                # print(xy_vector)
                # if np.dot(xy_vector, np.array([1, 0, 0])) < 0:
                #     print("rotate")
                #     # rotation_matrix_x = rotation_matrix_from_vectors(xy_vector, np.array([0, 1, 0]))
                #     rotation_matrix_x = np.eye(3)
                #     # rotation_matrix = np.matmul(rotation_matrix_x, rotation_matrix)
                #     hand_points = np.dot(hand_points, rotation_matrix_x.T)
                #     box_points = np.dot(box_points, rotation_matrix_x.T)
                # else:
                #     rotation_matrix_x = np.eye(3)

                x_center = box_points[0,0]
                y_center = box_points[0,1]
                z_min = np.min(box_points[:, 2])
                # print(z_axis)
                # exist_rotations = True
            else:
                hand_points = np.dot(hand_points, rotation_matrix.T)
                box_points = np.dot(box_points, rotation_matrix.T)
                # hand_points = np.dot(hand_points, rotation_matrix_x.T)
                # box_points = np.dot(box_points, rotation_matrix_x.T)

            # if z_min > np.min(box_points[:, 2]):
            #     z_min = np.min(box_points[:, 2])
            #     print("z_min", z_min)
            # put the object on the table
            hand_points[:, 2] = hand_points[:, 2] - z_min
            box_points[:, 2] = box_points[:, 2] - z_min

            

            
            # recenter the points to the center of the object
            hand_points[:, 0] = hand_points[:, 0] - x_center
            hand_points[:, 1] = hand_points[:, 1] - y_center 
            box_points[:, 0] = box_points[:, 0] - x_center
            box_points[:, 1] = box_points[:, 1] - y_center 

            hand_points = np.dot(hand_points, rotation_z_matrix.T)
            box_points = np.dot(box_points, rotation_z_matrix.T)

            if exist_rotations == False:
                cube_pose = box_points[0]

            if exist_rotations == False:
                cube_pose = box_points[0]
                cube_axis, cube_angle = rotation_axis_angle(np.array([1, 0, 0]), box_points[7]-box_points[6])
                if cube_axis[2] <= 0:
                    cube_angle = -cube_angle
                cube_quant = Quaternion(axis=[0,0,1], degrees=cube_angle)
                exist_rotations = True

            # get the plane of the hand
            hand_plane_vec = np.cross(hand_points[4]-hand_points[2], hand_points[12]-hand_points[10])
            hand_plane_vec = hand_plane_vec * 0.1 / np.linalg.norm(hand_plane_vec) + hand_points[0]
            hand_plane_vec_r = np.cross(hand_points[33]-hand_points[31], hand_points[25]-hand_points[23])
            hand_plane_vec_r = hand_plane_vec_r * 0.1 / np.linalg.norm(hand_plane_vec_r) + hand_points[21+0]
            # print(np.min(box_points[:, 2]), np.max(box_points[:, 2]))

            # get the direction of the hand
            left_dir = np.array([np.mean(hand_points[left_third_joint, 0]), np.mean(hand_points[left_third_joint, 1]), np.mean(hand_points[left_third_joint, 2])]) - hand_points[0]
            normalize_factor = np.linalg.norm(left_dir)/0.065
            right_dir = np.array([np.mean(hand_points[right_third_joint, 0]), np.mean(hand_points[right_third_joint, 1]), np.mean(hand_points[right_third_joint, 2])]) - hand_points[21+0]
            left_dir = left_dir - np.dot(left_dir, np.array([0,0,1])) * np.array([0,0,1])
            left_dir = left_dir / np.linalg.norm(left_dir)
            right_dir = right_dir - np.dot(right_dir, np.array([0,0,1])) * np.array([0,0,1])
            right_dir = right_dir / np.linalg.norm(right_dir)
            # print(normalize_factor)

            # get quantization of the direction
            left_r_axis, left_r_angle = rotation_axis_angle(np.array([-1, 0, 0]), left_dir)
            right_r_axis, right_r_angle = rotation_axis_angle(np.array([1, 0, 0]), right_dir)
            left_p_axis, left_p_angle = rotation_axis_angle(np.array([0, 0, 1]), hand_plane_vec)
            right_p_axis, right_p_angle = rotation_axis_angle(np.array([0, 0, 1]), hand_plane_vec_r)
            # left_quat = combine_rotations(left_r_axis, left_r_angle, left_p_axis, left_p_angle)
            # right_quat = combine_rotations(right_r_axis, right_r_angle, right_p_axis, right_p_angle)
            # print(right_r_axis, right_r_angle)
            left_quat = Quaternion(axis=left_r_axis, degrees=left_r_angle) * Quaternion(axis=[0,-1,0], degrees=left_p_angle*0.5)
            right_quat = Quaternion(axis=right_r_axis, degrees=right_r_angle) * Quaternion(axis=[0,1,0], degrees=right_p_angle*0.5)
            # print(left_quat.as_quat())
            # append trajectory
            #{"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            # left_position = (hand_points[0]/normalize_factor - recenter_center).tolist()
            left_xyz = ([np.mean(hand_points[left_finger, 0]), np.mean(hand_points[left_finger, 1]), np.mean(hand_points[:, 2])]/ normalize_factor + recenter_center + np.array([0, 0, -0.01])).tolist()
            right_xyz = ([np.mean(hand_points[right_finger, 0]), np.mean(hand_points[right_finger, 1]), np.mean(hand_points[:, 2])]/normalize_factor + recenter_center+ np.array([0, 0, -0.01])).tolist()
            # check if left hand is closeed
            if if_close(np.mean(hand_points[left_finger,:], axis=0)-box_points[0], box_x_axis, box_y_axis, box_z_axis):
                left_hand_open = 0
                if not set_obj:
                    cube_pose = np.mean(hand_points[left_finger, :], axis=0)
                    cube_quant = Quaternion(axis=left_r_axis, degrees=left_r_angle)
                    hand_offset = left_xyz[2] - 0.01
                    left_xyz[2] = 0.01
                    left_trajectory[-1]["xyz"][2] = 0.01
                    left_trajectory[-2]["xyz"][2] = 0.01 + hand_offset*0.5
                    left_trajectory[-3]["xyz"][2] = 0.01 + hand_offset*0.7
                    set_obj = True
                    left_hand = True
                else:
                    left_xyz[2] -= hand_offset
                    left_hand = False
            else:
                left_hand_open = 1
            # check if right hand is closed
            if if_close(np.mean(hand_points[right_finger,:], axis=0)-box_points[0], box_x_axis, box_y_axis, box_z_axis):
                right_hand_open = 0
                if not set_obj:
                    cube_pose = np.mean(hand_points[right_finger, :], axis=0)
                    cube_quant = Quaternion(axis=right_r_axis, degrees=right_r_angle)
                    hand_offset = right_xyz[2] - 0.01
                    right_xyz[2] = 0.01
                    right_trajectory[-1]["xyz"][2] = 0.01
                    right_trajectory[-2]["xyz"][2] = 0.01 + hand_offset*0.5
                    right_trajectory[-3]["xyz"][2] = 0.01 + hand_offset*0.7
                    set_obj = True
                else:
                    right_xyz[2] -= hand_offset
            else:
                right_hand_open = 1
            # if np.linalg.norm(box_points[0]-cube_pose) > 0.01: # and left_hand_open == 1 and right_hand_open == 1:
            #     if np.linalg.norm(box_points[0]-cube_pose) < np.linalg.norm(box_points) and left_hand_open == 1:
            #         left_hand_open = 0
            #         # cube_pose = ([np.mean(hand_points[left_finger, 0]), np.mean(hand_points[left_finger, 1]), np.mean(hand_points[:, 2])])
            #         # left_xyz[2] = 0.01
            #     elif right_hand_open == 1:
            #         right_hand_open = 0
            #         cube_pose = ([np.mean(hand_points[right_finger, 0]), np.mean(hand_points[right_finger, 1]), np.mean(hand_points[:, 2])])
            #         # right_xyz[2] = 0.01
            left_trajectory.append({"t": time_, 
                                    "xyz": left_xyz, 
                                    "quat": left_quat.elements.tolist(), 
                                    "gripper": left_hand_open})
            right_trajectory.append({"t": time_, 
                                    "xyz": right_xyz, 
                                    "quat": right_quat.elements.tolist(), 
                                    "gripper": right_hand_open})
            time_ += int(episode_len/(iteration_len))
            print("time: ", time_)
            # ax1.clear ()
            # ax1.set_xlim3d(-0.3, 0.3)
            # ax1.set_ylim3d(-0.3, 0.3)
            # ax1.set_zlim3d(0, 0.6)
            # # print(landmarks)
            # # ax.set_xlim3d(-1, 1)
            # # ax. set_ylim3d(-1, 1)
            # # ax.set_zlim3d(-1, 1)
            # ax1.scatter(hand_points[ :, 0], hand_points[ :, 1], hand_points[ :, 2], s=50, color=hand_color)
            # ax1.scatter(box_points[ :, 0], box_points[ :, 1], box_points[ :, 2], s=50, color = color_array)
            # # print(np.min(box_points[:, 2]), np.max(box_points[:, 2]))
            # ax1.scatter(hand_plane_vec[0], hand_plane_vec[1], hand_plane_vec[2], s=50, color = [0, 0, 0])
            # ax1.scatter(hand_plane_vec_r[0], hand_plane_vec_r[1], hand_plane_vec_r[2], s=50, color = [0, 0, 0])
            # ax1.plot([hand_points[0, 0], hand_plane_vec[0]], [hand_points[0, 1], hand_plane_vec[1]], [hand_points[0, 2], hand_plane_vec[2]], 'k')
            # ax1.plot([hand_points[21, 0], hand_plane_vec_r[0]], [hand_points[21, 1], hand_plane_vec_r[1]], [hand_points[21, 2], hand_plane_vec_r[2]], 'k')
            # for _c in hand_connection:
            #     ax1.plot ( [hand_points[ _c[0], 0], hand_points[ _c[1], 0]],
            #     [hand_points[ _c[0], 1], hand_points[ _c[1], 1]],
            #     [hand_points[ _c[0], 2], hand_points[ _c[1], 2]], 'k')

            # for _c in obj_connection:
            #     ax1.plot ( [box_points[ _c[0], 0], box_points[ _c[1], 0]],
            #     [box_points[ _c[0], 1], box_points[ _c[1], 1]],
            #     [box_points[ _c[0], 2], box_points[ _c[1], 2]], 'k')
            # plt.plot(0.1)
        print("episode len: ", episode_len)
        print("iteration len: ", iteration_len)
        print("time: ", time_)
        if time_ < episode_len:
            left_trajectory.append({"t": episode_len+10, 
                                    "xyz": left_xyz, 
                                    "quat": left_quat.elements.tolist(), 
                                    "gripper": left_hand_open})
            right_trajectory.append({"t": episode_len+10, 
                                    "xyz": right_xyz, 
                                    "quat": right_quat.elements.tolist(), 
                                    "gripper": right_hand_open})
            time_ += int(episode_len/(iteration_len-1.2))

        plt.close()
        ax1.clear()

        data = {
            "left_trajectory": left_trajectory,
            "right_trajectory": right_trajectory,
            "cube_pose": (cube_pose/normalize_factor + recenter_center).tolist(),
            "cube_quat": cube_quant.elements.tolist(),
            "cube_size": [box_x_axis*2/normalize_factor, box_y_axis*2/normalize_factor, box_z_axis*2/normalize_factor]

        } 

        # print(data) 

        # initialize the task



        xml_path = './assets/bimanual_viperx_ee_transfer_cube.xml'
        physics = mujoco.Physics.from_xml_path(xml_path)


        onscreen_render = True
        num_episodes = 1

        camera_names = ['top', 'angle', 'vis']
        render_cam_name = 'angle' # top angle vis


        # print(data.get('right_trajectory', []))
        # print(data.get('left_trajectory', []))
        box_geom_id = physics.model.name2id('red_box', 'geom')
        new_size = np.array(data.get('cube_size', []))  # New half-dimensions
        # physics.model.geom_size[box_geom_id][2] = new_size[2]
        task = BimanualViperXEETask(cube_pose=(cube_pose/normalize_factor + recenter_center).tolist(), cube_quat=cube_quant.elements.tolist())
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                        n_sub_steps=None, flat_observation=False)

        
        policy = ArmPolicy(inject_noise=False, left_trajectory=left_trajectory, right_trajectory=right_trajectory)


        # policy.generate_trajectory(data.get('left_trajectory', []), data.get('right_trajectory', []))

        # clear the environment

        ts = env.reset() # timestep: step_type, reward, discount, observation
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for eps_idx in range(num_episodes):
            for step in range(episode_len):
                action = policy(ts)
                # print("action: ", ts.observation['qpos'])
                ts = env.step(action)
                episode.append(ts)
                if onscreen_render:
                    plt_img.set_data(ts.observation['images'][render_cam_name])
                    plt.pause(0.01)
        plt.close()

        subtask_info = episode[0].observation['env_state'].copy()

        joint_traj = [ts.observation['qpos'] for ts in episode]
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        max_reward = np.max([ts.reward for ts in episode[1:]])
        print("max_reward: ", np.max([ts.reward for ts in episode[1:]]))
        # print("joint_traj: ", joint_traj[0])
        # print("gripper_ctrl: ", gripper_ctrl_traj[0])
        # replace gripper pose with gripper control

        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl
            # print("joint: ", joint)

        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        xml_path = './assets/bimanual_viperx_transfer_cube.xml'
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(cube_pose=data.get('cube_pose', []), cube_quat=data.get('cube_quat', []))
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                        n_sub_steps=None, flat_observation=False)
        box_geom_id = physics.model.name2id('red_box', 'geom')
        new_size = np.array(data.get('cube_size', [])) 
        # BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()
        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            # print("action: ", action)
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.01)

        plt.close()

        replay_reward = np.sum([ts.reward == 2 for ts in episode_replay[1:]])
        print("replay_reward: ", replay_reward)

        if replay_reward >= 20:

            joint_traj = joint_traj[:-1]
            episode_replay = episode_replay[:-1]
            # print("joint_traj: ", len(joint_traj))
            # print("episode: ", len(episode_replay))
            max_timesteps = len(joint_traj)
            data_dict = {
                        '/observations/qpos': [],
                        '/observations/qvel': [],
                        '/action': [],
                    }
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'] = []

            while joint_traj:
                action = joint_traj.pop(0)
                ts = episode_replay.pop(0)
                data_dict['/observations/qpos'].append(ts.observation['qpos'])
                data_dict['/observations/qvel'].append(ts.observation['qvel'])
                data_dict['/action'].append(action)
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

            if left_hand:
                dic_path = odir + "/findnpick_left"
            else:
                dic_path = odir + "/findnpick_right"
            num_data = len(os.listdir(dic_path))
            dataset_path = os.path.join(dic_path, f'findnpick_{num_data}')
            t0 = time.time()
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                                chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (max_timesteps, 14))
                qvel = obs.create_dataset('qvel', (max_timesteps, 14))
                action = root.create_dataset('action', (max_timesteps, 14))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')
        del episode_replay
        del env
        del task
        del data
        del physics
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images to extract 3D skeletons and save results as JSON.")
    parser.add_argument("--input", required=True, help="Directory containing input images.")
    parser.add_argument("--output", required=True, help="Directory to save output JSON files.")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simulate_episode(input_dir, output_dir)
    print(f"Processed {input_dir} and saved to {output_dir}.")