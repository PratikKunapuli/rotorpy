"""
Imports
"""
import numpy as np
from scipy.spatial.transform import Rotation  # This is a useful library for working with attitude.
from rotorpy.vehicles.crazyflie_params import quad_params as crazyflie_params

import torch
from stable_baselines3 import PPO
from pathlib import Path

import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming your_project is two levels up from rotorpy
project_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
# Construct the path to the rl directory
rl_dir = os.path.join(project_dir, 'rl')
# Add the rl directory to the system path
sys.path.append(rl_dir)

# Now you can import custom_policies
from custom_policies import *


class NNControl(object):
    """
    The controller is implemented as a class with two required methods: __init__() and update(). 
    The __init__() is used to instantiate the controller, and this is where any model parameters or 
    controller gains should be set. 
    In update(), the current time, state, and desired flat outputs are passed into the controller at 
    each simulation step. The output of the controller depends on the control abstraction for Multirotor...
        if cmd_motor_speeds: the output dict should contain the key 'cmd_motor_speeds'
        if cmd_motor_thrusts: the output dict should contain the key 'cmd_rotor_thrusts'
        if cmd_vel: the output dict should contain the key 'cmd_v'
        if cmd_ctatt: the output dict should contain the keys 'cmd_thrust' and 'cmd_q'
        if cmd_ctbr: the output dict should contain the keys 'cmd_thrust' and 'cmd_w'
        if cmd_ctbm: the output dict should contain the keys 'cmd_thrust' and 'cmd_moment'
    """
    def __init__(self, model_name, experiment_dict, control_mode, assumed_quad_params=None):
        """
        Use this constructor to save vehicle parameters, set controller gains, etc. 
        Parameters:
            vehicle_params, dict with keys specified in a python file under /rotorpy/vehicles/

        """
        self.control_mode = control_mode
        self.experiment_dict = experiment_dict
        self.time_horizon = experiment_dict.get('time_horizon', 0)
        self.feedback_horizon = experiment_dict.get('feedback_horizon', 0)

        self.true_quad_params = crazyflie_params
        if not assumed_quad_params:
            self.assumed_quad_params = crazyflie_params

        # Limits for action scaling
        self.rotor_speed_min = self.assumed_quad_params['rotor_speed_min']
        self.rotor_speed_max = self.assumed_quad_params['rotor_speed_max']
        self.max_thrust = self.assumed_quad_params['k_eta'] * self.assumed_quad_params['rotor_speed_max']**2
        self.min_thrust = self.assumed_quad_params['k_eta'] * self.assumed_quad_params['rotor_speed_min']**2
        self.num_rotors = 4

        # Max orintations for ct_att
        self.max_roll = np.pi/4
        self.max_pitch = np.pi/4
        self.max_yaw = np.pi

        # Max body rates for ctbr
        self.max_roll_br = 7.0
        self.max_pitch_br = 7.0 
        self.max_yaw_br = 3.0


        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # current path is actually where the TOP MOST entry point is (rotorpy/examples)
        self.model = PPO.load('../rl/saved_policies/'+ model_name, device=device, print_system_info=True)



    def update(self, t, state, flat_outputs):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys

                key, description, unit, (applicable control abstraction)                                                        
                cmd_motor_speeds, the commanded speed for each motor, rad/s, (cmd_motor_speeds)
                cmd_thrust, the collective thrust of all rotors, N, (cmd_ctatt, cmd_ctbr, cmd_ctbm)
                cmd_moment, the control moments on each boxy axis, N*m, (cmd_ctbm)
                cmd_q, desired attitude as a quaternion [i,j,k,w], , (cmd_ctatt)
                cmd_w, desired angular rates in body frame, rad/s, (cmd_ctbr)
                cmd_v, desired velocity vector in world frame, m/s (cmd_vel)
        """        

        obs = self._get_obs(state, flat_outputs)


        action, _ =  self.model.predict(obs, deterministic=True)

        control_dict = self.rescale_action(action)


        return control_dict

    def _get_obs(self, state, flat_output):
        self.current_state = np.hstack([state['x'], state['v'], state['q'], state['w']])
        if self.experiment_dict.get('fb_body_frame', True):
            pos = state['x']
            rot = Rotation.from_quat(state['q'])

            fb_term = pos - rot.inv().apply(flat_output ['x'])
            if self.time_horizon > 0:
                futures = np.hstack([pos - rot.inv().apply(flat_output['future_pos'][i]) for i in range(self.time_horizon)])
                obs = np.hstack([fb_term, self.current_state[3:], futures])
            else:
                obs = np.hstack([fb_term, self.current_state[3:]])
        else:
            fb_term = pos - flat_output ['x']
            if self.time_horizon > 0:
                futures = np.hstack([self.ref.update(self.t + i*self.t_step)['x'] for i in range(self.time_horizon)])
                obs = np.hstack([fb_term, self.current_state[3:], futures])
            else:
                obs = np.hstack([fb_term, self.current_state[3:]])

        if self.experiment_dict.get('l1_simulation', False):
            d_hat = self.l1_simulation()
            obs = np.hstack([obs, d_hat])
        
        
        if self.feedback_horizon > 0:
            self.feedback_buffer = np.roll(self.feedback_buffer, -1, axis=0)
            self.feedback_buffer[-1] = self.current_state
            obs = np.hstack([obs, self.feedback_buffer.flatten()])
        
        if self.experiment_dict.get('include_env_params', False):
            # Include the assumed and true parameters
            obs = np.hstack([obs, np.array([self.assumed_quad_params['mass'], self.assumed_quad_params['Ixx'], self.assumed_quad_params['Iyy'], self.assumed_quad_params['Izz'], self.assumed_quad_params['k_eta']])])
            obs = np.hstack([obs, np.array([self.true_quad_params['mass'], self.true_quad_params['Ixx'], self.true_quad_params['Iyy'], self.true_quad_params['Izz'], self.true_quad_params['k_eta']])])
        
        return obs
    
    def rescale_action(self, action):
            """
            Rescales the action to within the control limits and then assigns the appropriate dictionary. 
            """

            control_dict = {
                'cmd_motor_speeds': np.zeros((4,)),
                'cmd_motor_thrusts' : np.zeros((4,)),
                'cmd_thrust' : 0,
                'cmd_moment' : np.zeros((3,)),
                'cmd_q' : np.array([0,0,0,1]),
                'cmd_w' : np.zeros((3,)),
                'cmd_v' : np.zeros((3,)),
            }


            if self.control_mode == 'cmd_ctbm':
                # Scale action[0] to (0,1) and then scale to the max thrust
                cmd_thrust = np.interp(action[0], [-1,1], [self.num_rotors*self.min_thrust, self.num_rotors*self.max_thrust])

                # Scale the moments
                cmd_roll_moment = np.interp(action[1], [-1,1], [-self.max_roll_moment, self.max_roll_moment])
                cmd_pitch_moment = np.interp(action[2], [-1,1], [-self.max_pitch_moment, self.max_pitch_moment])
                cmd_yaw_moment = np.interp(action[3], [-1,1], [-self.max_yaw_moment, self.max_yaw_moment])

                control_dict['cmd_thrust'] = cmd_thrust
                control_dict['cmd_moment'] = np.array([cmd_roll_moment, cmd_pitch_moment, cmd_yaw_moment])
            
            elif self.control_mode == 'cmd_ctbr':
                # Scale action to min and max thrust.
                cmd_thrust = np.interp(action[0], [-1, 1], [self.num_rotors*self.min_thrust, self.num_rotors*self.max_thrust])

                # Scale the body rates. 
                cmd_roll_br = np.interp(action[1], [-1,1], [-self.max_roll_br, self.max_roll_br])
                cmd_pitch_br = np.interp(action[2], [-1,1], [-self.max_pitch_br, self.max_pitch_br])
                cmd_yaw_br = np.interp(action[3], [-1,1], [-self.max_yaw_br, self.max_yaw_br])

                control_dict['cmd_thrust'] = cmd_thrust
                control_dict['cmd_w'] = np.array([cmd_roll_br, cmd_pitch_br, cmd_yaw_br])

            elif self.control_mode == 'cmd_motor_speeds':
                # Scale the action to min and max motor speeds. 
                control_dict['cmd_motor_speeds'] = np.interp(action, [-1,1], [self.rotor_speed_min, self.rotor_speed_max])

            elif self.control_mode == 'cmd_motor_thrusts':
                # Scale the action to min and max rotor thrusts. 
                control_dict['cmd_motor_thrusts'] = np.interp(action, [-1,1], [self.min_thrust, self.max_thrust])
            
            elif self.control_mode == 'cmd_vel':
                # Scale the velcoity to min and max values. 
                control_dict['cmd_v'] = np.interp(action, [-1,1], [-self.max_vel, self.max_vel])
            
            elif self.control_mode == 'cmd_ctatt':
                cmd_thrust = np.interp(action[0], [-1, 1], [self.num_rotors*self.min_thrust, self.num_rotors*self.max_thrust])

                roll = np.interp(action[1], [-1,1], [-self.max_roll, self.max_roll])
                pitch = np.interp(action[2], [-1,1], [-self.max_pitch, self.max_pitch])
                yaw = np.interp(action[3], [-1,1], [-self.max_yaw, self.max_yaw])

                # print("Desired roll (rads): ", roll)
                # print("Desired pitch (rads): ", pitch)

                cmd_q = Rotation.from_euler('ZYX', [roll, pitch, yaw]).as_quat() # guessing the euler convention here

                control_dict['cmd_thrust'] = cmd_thrust
                control_dict['cmd_q'] = cmd_q

            return control_dict