import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_trajectory_tracking import QuadrotorTrackingEnv

# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import hover_reward, datt_reward
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.minsnap_datt import MinSnap
from rotorpy.utils.helper_functions import sample_waypoints

from argparse import ArgumentParser
from custom_policies import *
from configurations import load_experiment
from pathlib import Path

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3 import PPO

DEFAULT_LOG_DIR = Path('./logs/')
DEFAULT_DATA_DIR = Path('./data/')
SAVED_POLICY_DIR = Path('./saved_policies/')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config',
        required=True,
        help='Name of the experiment configuration. '    
    )
    parser.add_argument("-m", "--model", dest='model',
        required=True,
        help="Which model to train. Options are \'DATT\', \'RPG\', \'RMA\' etc.)"
    )
    parser.add_argument('--reward', dest='reward',
        default="datt", 
        help='Reward function to use. Default: datt')
    parser.add_argument('-n', '--name', dest='name', 
        default=None,
        help='Name of the policy to train. If such policy already exists in ./saved_policies/, continues training it.'
    )
    parser.add_argument('-d', '--log-dir', dest='log_dir',
        default=None,
        help='The directory to log training info to. Can run tensorboard from this directory to view.'   
    )
    parser.add_argument('-ts', '--timesteps', dest='timesteps',
        type=int, default=10e6,
        help='Number of timesteps to train for. Default: 10 million'    
    )
    parser.add_argument('-ch', '--checkpoint', dest='checkpoint',
        type=bool, default=False,
        help='Whether to save checkpoints.'
    )
    parser.add_argument('-de', '--device', dest='device',
        type=int, default=0,
        help='GPU ID to use.'
    )
    parser.add_argument('--rate', dest='rate',
                        type=int, default=100,
        help='Rate at which to run the environment in Hz. Default: 100 Hz'
    )

    parser.add_argument('--n-envs', type=int, help='How many "parallel" environments to run', default=10)
    parser.add_argument('-r', '--ref', dest='ref', type=str, default="lissajous_ref")
    parser.add_argument('--seed', dest='seed', type=int, default=None,
        help='Seed to use for randomizing reference trajectories during training.'
    )

    args = parser.parse_args()

    return args

def train():
    args = parse_args()

    if args.name is None:
        policy_name = f'model_mismatch_env_policy'

    if args.log_dir is None:
        log_dir = DEFAULT_LOG_DIR / f'{args.name}_logs'
        log_dir.mkdir(exist_ok=True, parents=True)
    if not log_dir.exists():
        raise FileNotFoundError(f'{log_dir} does not exist')
    
    # Set up Env
    reward_function = datt_reward if args.reward == 'datt' else hover_reward
    experiment_dict = load_experiment(args.config)
    # TODO add more configurability here
    if args.ref == "lissajous_ref":
        traj = TwoDLissajous(A=1, B=1, a=1, b=2, delta=0, height=0.5, yaw_bool=False, dt=1/args.rate, seed = 2024, fixed_seed = False, env_diff_seed=True)
    elif args.ref == "minsnap_ref":
        traj = MinSnap(points=sample_waypoints(seed=5), yaw_angles=np.zeros(4), v_avg=2, dt=1/args.rate, seed = 2024, fixed_seed = True, env_diff_seed=True)
    else:
        raise NotImplementedError

    

    
    # Set up Policy
    algo_class = PPO

    if not (SAVED_POLICY_DIR / f'{args.name}.zip').exists():
        print("TRAINING NEW POLICY!")
        if args.model.upper() == "DATT":
            policy_class = 'MlpPolicy'
            features_extractor_kwargs = {}
            features_extractor_class = FeedforwardFeaturesExtractor
            features_extractor_kwargs['extra_state_features'] = 3
            net_arch = dict(pi=[64, 64, 64], vf=[64, 64, 64])
            policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=net_arch,
                     features_extractor_class=features_extractor_class,
                     features_extractor_kwargs=features_extractor_kwargs,
                     share_features_extractor = True
            )
            print("Using policy: DATT")
        elif args.model.upper() == "RMA":
            encoder_net_arch = [64, 64]
            encoder_output_dim = 5
            encoder_input_dim = 10

            policy_net_arch = dict(pi=[64, 64, 64], vf=[64, 64, 64])
            policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=policy_net_arch,
                        encoder_input_dim = encoder_input_dim,
                        encoder_output_dim = encoder_output_dim, 
                        encoder_network_architecture= encoder_net_arch,
                        share_features_extractor = True
            )
            policy_class = RMAPolicy
            print("Using policy: RMA")
        elif args.model.upper() == "RPG":
            policy_net_arch = dict(pi=[128, 128, 128], vf=[128, 128, 128])
            policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=policy_net_arch,
                        share_features_extractor = True
            )
            policy_class = RPGPolicy
            print("Using policy: RPG")
        else:
            raise NotImplementedError

        env_class = QuadrotorTrackingEnv
        env_kwargs = {
            'quad_params': quad_params,
            'experiment_dict': experiment_dict,
            'render_mode': 'None',
            'reference': traj,
            'reward_fn': reward_function,
            'sim_rate': args.rate,
        }

        if issubclass(env_class, VecEnv):
            env = VecMonitor(env_class(args.n_envs))
        else:
            env = make_vec_env(env_class, n_envs=args.n_envs, env_kwargs=env_kwargs)
        
        kwargs = {}
        policy: BaseAlgorithm = algo_class(
            policy_class, 
            env, 
            tensorboard_log=log_dir, 
            policy_kwargs=policy_kwargs,
            device=f'cuda:{args.device}',
            verbose=0,
            **kwargs
        )
    else:
        env_class = QuadrotorTrackingEnv
        experiment_dict['reference_randomize_threshold'] = 0 # begin randomizing the reference trajectory immediately
        env_kwargs = {
            'quad_params': quad_params,
            'experiment_dict': experiment_dict,
            'render_mode': 'None',
            'reference': traj,
            'reward_fn': reward_function,
            'sim_rate': args.rate,
        }

        if issubclass(env_class, VecEnv):
            env = VecMonitor(env_class(args.n_envs))
        else:
            env = make_vec_env(env_class, n_envs=args.n_envs, env_kwargs=env_kwargs)
        policy: BaseAlgorithm = algo_class.load(SAVED_POLICY_DIR / f'{args.name}.zip', env, device=f'cuda:{args.device}')
        print('CONTINUING TRAINING!')

    if args.checkpoint:
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            # save_freq = 100,
            save_path=SAVED_POLICY_DIR,
            name_prefix=args.name
        )
    else:
        checkpoint_callback = None

    policy.learn(total_timesteps=args.timesteps, progress_bar=True, callback=checkpoint_callback)
    policy.save(SAVED_POLICY_DIR / args.name)


if __name__ == '__main__':
    train()