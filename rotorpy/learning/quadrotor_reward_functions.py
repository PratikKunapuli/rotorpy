import numpy as np
from scipy.spatial.transform import Rotation

import gymnasium as gym
from gymnasium import spaces

import math

"""
Reward functions for quadrotor tasks. 
"""

def hover_reward(observation, action, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}):
    """
    Rewards hovering at (0, 0, 0). It is a combination of position error, velocity error, body rates, and 
    action reward.
    """

    # Compute the distance to goal
    dist_reward = -weights['x']*np.linalg.norm(observation[0:3])

    # Compute the velocity reward
    vel_reward = -weights['v']*np.linalg.norm(observation[3:6])

    # Compute the angular rate reward
    ang_rate_reward = -weights['w']*np.linalg.norm(observation[10:13])

    # Compute the action reward
    action_reward = -weights['u']*np.linalg.norm(action)

    return dist_reward + vel_reward + action_reward + ang_rate_reward

def datt_reward(state, reference, action, t):
    """
    Rewards for tracking a reference trajectory. Combination of position, velocity and yaw error. 
    """
    # Assumes the action is CTBR command
    # Assumes state contains pos, vel, quat minimum

    reference_flat_outputs = reference.update(t)

    yaw = Rotation.from_quat(state['q']).as_euler('ZYX')[0]
    yawcost = 0.5 * min(abs(reference_flat_outputs['yaw'] - yaw), abs(reference_flat_outputs['yaw'] - yaw))
    poscost = np.linalg.norm(state['x'] - reference_flat_outputs['x'])#min(np.linalg.norm(state.pos), 1.0)
    velcost = 0.1 * min(np.linalg.norm(state['v']), 1.0)

    # ucost = 0.2 * abs(action[0]) + 0.1 * np.linalg.norm(action[1:])

    cost = yawcost + poscost + velcost

    return -cost

def rpg_reward(state, reference, action, t):
    """
    Reward function for the RPG group, assumes the reference is given in flat outputs, 
    with the flatness diffeomorphism applied for the desired orientation and angular velocity
    """

    reference_flat_outputs = reference.update(t)

    # TODO: Implement the reward function for the RPG group

    return 0