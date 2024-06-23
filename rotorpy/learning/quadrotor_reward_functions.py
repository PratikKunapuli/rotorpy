import numpy as np
from scipy.spatial.transform import Rotation

import gymnasium as gym
from gymnasium import spaces

import math

from rotorpy.utils.flatness import Flatness

"""
Reward functions for quadrotor tasks. 
"""

def hover_reward(state, reference, action, t, **kwargs): 
    weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}
    """
    Rewards hovering at (0, 0, 0). It is a combination of position error, velocity error, body rates, and 
    action reward.
    """

    state = np.hstack([state['x'], state['v'], state['q'], state['w']])
    current_ref = reference.update(t)

    # Compute the distance to goal
    dist_reward = -weights['x']*np.linalg.norm(state[0:3] - current_ref['x'])

    # Compute the velocity reward
    vel_reward = -weights['v']*np.linalg.norm(state[3:6] - current_ref['x_dot'])

    # Compute the angular rate reward
    ang_rate_reward = -weights['w']*np.linalg.norm(state[10:13])

    # Compute the action reward
    action_reward = -weights['u']*np.linalg.norm(action)

    return dist_reward + vel_reward + action_reward + ang_rate_reward

def datt_reward(state, reference, action, t, **kwargs):
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

def rpg_reward(state, reference, action, t, **kwargs):
    """
    Reward function for the RPG group, assumes the reference is given in flat outputs, 
    with the flatness diffeomorphism applied for the desired orientation and angular velocity
    """

    reference_flat_outputs = reference.update(t)

    # TODO: Implement the reward function for the RPG group

    return 0

def DRL_sejong_reward(state, reference, action, t, **kwargs):
    """
    Reward function for the paper:
    "Deep Reinforcement Learning-based Quadcopter Controller: A Practical Approach and Experiments"
    from Sejong University.
    """
    flatness_obj = kwargs.get('flatness_obj')

    if flatness_obj is None:
        raise ValueError("Flatness object is required for this reward function")

    reference_flat_outputs = reference.update(t)

    R_des = flatness_obj.get_orientations_from_flat(state, reference_flat_outputs)

    pos_error = state['x'] - reference_flat_outputs['x']
    ori_error = R_des - Rotation.from_quat(state['q']).as_matrix()
    vel_error = state['v'] - reference_flat_outputs['x_dot']

    #nominal action depends on the chosen action modaility. Here we're assuming CTBR or TRPY (only thrust is penalized)
    action_error = action[0] - kwargs.get('hover_thrust') if 'hover_thrust' in kwargs.keys() else 0

    weights = {
        'stay_alive': 2,
        'pos_error': 2.5,
        'ori_error': 2.5,
        'vel_error': 0.05,
        'action_error': 0.05
    }

    reward = weights['stay_alive'] - weights['pos_error']*np.linalg.norm(pos_error) - \
            weights['ori_error']*np.linalg.norm(ori_error, ord='fro') - weights['vel_error']*np.linalg.norm(vel_error) - \
            weights['action_error']*np.linalg.norm(action_error)

    return reward
    
    