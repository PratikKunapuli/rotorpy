import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import torch

from scipy.spatial.transform import Rotation

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_trajectory_tracking import QuadrotorTrackingEnv

# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import hover_reward, datt_reward
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.hover_traj import HoverTraj

from custom_policies import *
from configurations import load_experiment

from stable_baselines3 import PPO

from joblib import Parallel, delayed

from argparse import ArgumentParser
from pathlib import Path

DEFAULT_LOG_DIR = Path('./logs/')
DEFAULT_DATA_DIR = Path('./data/')
SAVED_POLICY_DIR = Path('./saved_policies/')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config',
        required=True,
        help='Name of the experiment configuration. '    
    )
    parser.add_argument('-n', '--name', dest='name', 
        default=None,
        help='Name of the policy to train. If such policy already exists in ./saved_policies/, continues training it.'
    )
    parser.add_argument('-an', '--adapt-name', dest='adapt_name',
        default=None,
        help='Name of the adaptation network to eval.')
    parser.add_argument('--num-envs', dest='num_envs',
        type=int, default=10,
        help='Number of environments to run in parallel.'
    )
    parser.add_argument('--ref', dest='ref', 
        type=str, default='lissajous_ref')

    parser.add_argument('--action', type=str, default='cmd_ctbr', required=True)

    return parser.parse_args()

def remove_env_info(obs, env_dimension):
    base_obs = obs[:13]
    env_params = obs[-env_dimension:]
    return base_obs, env_params

def add_env_info(obs, env_params):
    obs = np.concatenate([obs, env_params])
    return obs

def run_rollout(model, args, experiment_dict, action_mode, seed, adaptation_network=None):
    """
    Run a rollout with the given model and reference trajectory. 
    """
    if args.ref == "lissajous_ref":
        traj = TwoDLissajous(A=1, B=1, a=1, b=2, delta=np.pi/2, height=0.5, yaw_bool=False, dt=0.01, seed = 2024, fixed_seed = False, env_diff_seed=True)
    elif args.ref == "hover_ref":
        traj = HoverTraj()
    else:
        raise NotImplementedError

    experiment_dict['integrator'] = 'RK45' # make the integrator more accurate

    env = QuadrotorTrackingEnv(quad_params, experiment_dict=experiment_dict, render_mode='None', control_mode=action_mode, reference = traj, reward_fn = datt_reward)
    obs, _  = env.reset(seed=seed, options={'pos_bound': 0.2,'vel_bound':0})
    if adaptation_network is not None:
        adaptation_history = np.zeros((100, 13+4))
        
    rollout_states = []
    rollout_ref_states = []

    rollout_positions = []
    rollout_rotations = []

    done = False
    total_reward = 0
    while not done:
        if adaptation_network is not None:
            latent = adaptation_network(adaptation_history)
            obs = add_env_info(obs, latent)

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, _, _ = env.step(action)

        if adaptation_network is not None:
            base_obs, env_params = remove_env_info(obs, 10)
            gt_latent = model.policy.encoder(torch.from_numpy(env_params).to(device)).detach().numpy()
            adaptation_history = np.roll(adaptation_history, -1, axis=0)
            adaptation_history[-1] = np.concatenate([base_obs, action])

        rollout_states.append(env._get_current_state())
        rollout_ref_states.append(env._get_current_reference()['x'])
        rollout_positions.append(env._get_current_state()[:3])
        rollout_rotations.append(Rotation.from_quat(env._get_current_state()[6:10]).as_matrix())
        
        total_reward += reward
    rollout_states = np.asarray(rollout_states)
    rollout_ref_states = np.asarray(rollout_ref_states)
    control_error = np.mean(np.linalg.norm(rollout_states[:, :3] - rollout_ref_states[:, :], axis=1))
    rollout_positions = np.asarray(rollout_positions)
    rollout_rotations = np.asarray(rollout_rotations)

    return control_error, total_reward, rollout_states, rollout_ref_states, rollout_positions, rollout_rotations

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = PPO.load(SAVED_POLICY_DIR / f'{args.name}.zip')

    if args.adapt_name is None:
        adaptation_network = None
    else:
        action_dims = 4
        if os.path.exists(SAVED_POLICY_DIR / f'{args.name}_adapt' / f'{args.adapt_name}'):
            adaptation_network_state_dict = torch.load(SAVED_POLICY_DIR / f'{args.name}_adapt' / f'{args.adapt_name}', map_location=torch.device('cpu'))
        elif os.path.exists(SAVED_POLICY_DIR / f'{args.adapt_name}'):
            adaptation_network_state_dict = torch.load(SAVED_POLICY_DIR / f'{args.adapt_name}', map_location=torch.device('cpu'))
        else:
            raise ValueError(f'Invalid adaptation network name: {args.adapt_name}')

        
        adaptation_network = AdaptationNetwork(input_dims=13 + action_dims, e_dims=5)
        adaptation_network.load_state_dict(adaptation_network_state_dict)

    experiment_dict = load_experiment(args.config)

    returns = Parallel(n_jobs=args.num_envs)(delayed(run_rollout)(model, args, experiment_dict, args.action, seed, adaptation_network) for seed in range(args.num_envs))
    # print(returns)
    #0th index is the control error, 1st index is the total reward, 2nd index is the rollout states, 3rd index is the rollout reference states, 4th index is the rollout positions, 5th index is the rollout rotations
    returns = np.asarray(returns, dtype=object)
    print("Mean control error: ", np.mean(returns[:, 0]))
    print("Std Dev control error: ", np.std(returns[:, 0]))

    print("\nMean total reward: ", np.mean(returns[:, 1]))
    print("Std Dev total reward: ", np.std(returns[:, 1]))

    # create animation
    all_positions = np.zeros((10*100 + 1, args.num_envs, 3))
    all_rotations = np.zeros((10*100 + 1, args.num_envs, 3, 3))
    reference_positions = np.zeros((10*100, 3))

    if args.ref == "lissajous_ref":
        ref = TwoDLissajous(A=1, B=1, a=1, b=2, delta=np.pi/2, height=0.5, yaw_bool=False, dt=0.01, seed = 2024, fixed_seed = False, env_diff_seed=True)
    elif args.ref == "hover_ref":
        ref = HoverTraj()
    else:
        raise NotImplementedError

    T = 10*100
    M = returns.shape[0]
    time_vec = np.arange(0, 10.01, 0.01)

    for i in np.arange(0, 10, 0.01):
        reference_positions[int(i*100)] = ref.update(i)['x']

    for env_number in range(returns.shape[0]):
        positions = returns[env_number, 4]
        rotations = returns[env_number, 5]
        time = positions.shape[0]
        all_positions[:time, env_number, :] = positions
        all_rotations[:time, env_number, :, :] = rotations

        # fill rest of the time with the last position and rotation
        all_positions[time:, env_number, :] = positions[-1]
        all_rotations[time:, env_number, :, :] = rotations[-1]

    from rotorpy.utils.animate import animate

    outputfile_name = "Eval_" + args.name + ".mp4"

    from rotorpy.world import World
    world = World.empty((-4, 4, -4, 
                                       4, -4, 4))
    animation = animate(time_vec, all_positions, all_rotations, wind=np.zeros((T,M,3)), animate_wind=False, world=world, filename=outputfile_name, blit=False, show_axes=True, close_on_finish=True, reference=reference_positions)
