import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from enum import Enum
from rotorpy.world import World
from rotorpy.vehicles.multirotor_model_mismatch import MultirotorModelMismatch
from rotorpy.vehicles.crazyflie_params import quad_params as crazyflie_params
from rotorpy.learning.quadrotor_reward_functions import hover_reward
from rotorpy.utils.shapes import Quadrotor
from rotorpy.trajectories.lissajous_traj import TwoDLissajous

import gymnasium as gym
from gymnasium import spaces

import math
from copy import deepcopy

class QuadrotorTrackingEnv(gym.Env):
    """

    A quadrotor environment for reinforcement learning using Gymnasium. 

    Inputs:
        initial_state: the initial state of the quadrotor. The default is hover. 
        control_mode: the appropriate control abstraction that is used by the controller, options are...
                                    'cmd_motor_speeds': the controller directly commands motor speeds. 
                                    'cmd_motor_thrusts': the controller commands forces for each rotor.
                                    'cmd_ctbr': the controller commands a collective thrsut and body rates. 
                                    'cmd_ctbm': the controller commands a collective thrust and moments on the x/y/z body axes
                                    'cmd_vel': the controller commands a velocity vector in the body frame. 
        reward_fn: the reward function, default to hover, but the user can pass in any function that is used as a reward. 
        quad_params: the parameters for the quadrotor. 
        max_time: the maximum time of the session. After this time, the session will exit. 
        world: the world for the quadrotor to operate within. 
        sim_rate: the simulation rate (in Hz), i.e. the timestep. 
        aero: boolean, determines whether or not aerodynamic wrenches are computed. 
        render_mode: render the quadrotor.
        render_fps: rendering frames per second, lower this for faster visualization. 
        ax: for plotting purposes, you can supply an axis object that the quadrotor will visualize on. 
        color: choose the color of the quadrotor. If none, it will randomly select a color.
    """

    metadata = {"render_modes": ["None", "3D", "console"], 
                "render_fps": 30,
                "control_modes": ['cmd_motor_speeds', 'cmd_motor_thrusts', 'cmd_ctbr', 'cmd_ctbm', 'cmd_vel']}

    def __init__(self, 
                 initial_state = {'x': np.array([0,0,0]),
                                  'v': np.zeros(3,),
                                  'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                                  'w': np.zeros(3,),
                                  'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                                  'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])},
                 control_mode = 'cmd_ctbr',
                 reward_fn = hover_reward,            
                 quad_params = crazyflie_params,                   
                 max_time = 10,                # Maximum time to run the simulation for in a single session.
                 wind_profile = None,         # wind profile object, if none is supplied it will choose no wind. 
                 world        = None,         # The world object
                 sim_rate = 100,              # The update frequency of the simulator in Hz
                 aero = False,                 # Whether or not aerodynamic wrenches are computed.
                 render_mode = "None",        # The rendering mode
                 render_fps = 30,             # The rendering frames per second. Lower this for faster visualization. 
                 fig = None,                  # Figure for rendering. Optional. 
                 ax = None,                   # Axis for rendering. Optional. 
                 color = None,                # The color of the quadrotor. 
                 reference = TwoDLissajous(),  # Reference trajectory for the quadrotor to track 
                 experiment_dict = None       # Dictionary detailing experiment parameters such as domain randomization etc. 
                ):
        super(QuadrotorTrackingEnv, self).__init__()

        self.metadata['render_fps'] = render_fps
        self.experiment_dict = experiment_dict

        self.quad_params = quad_params

        self.initial_state = initial_state

        self.vehicle_state = initial_state

        assert control_mode in self.metadata["control_modes"]  # Don't accept improper control modes
        self.control_mode = control_mode

        self.sim_rate = sim_rate
        self.t_step = 1/self.sim_rate
        self.reward_fn = reward_fn

        self.ref = reference
        self.aero = aero

        # Create quadrotor from quad params and control abstraction. 
        self.quadrotor = MultirotorModelMismatch(quad_params=quad_params, initial_state=initial_state, control_abstraction=control_mode, aero=aero)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.t = 0  # The current time of the instance
        self.max_time = max_time

        self.reset_count = 0

        ############ OBSERVATION SPACE

        # The observation state is the full state of the quadrotor.
        #     position, x, observation_state[0:3]
        #     velocity, v, observation_state[3:6]
        #     orientation, q, observation_state[6:10]
        #     body rates, w, observation_state[10:13]
        #     wind, wind, observation_state[13:16]
        #     motor_speeds, rotor_speeds, observation_state[16:20]
        # For simplicitly, we assume these observations can lie within -inf to inf. 

        # base number of observations: 3 for position, 3 for velocity, 4 for orientation, 3 for body rates
        self.num_observations = 13

        self.time_horizon = self.experiment_dict.get('time_horizon', 0)
        self.num_observations += 3*(self.time_horizon + 1)# only position for the time horizon future reference points
        
        if self.experiment_dict.get('l1_simulation', False):
            self.num_observations += 3

        self.feedback_horizon = 0
        if self.experiment_dict.get('feedback_horizon', 0) > 0:
            self.feedback_horizon = self.experiment_dict['feedback_horizon']
            self.feedback_buffer = np.zeros((self.feedback_horizon, 13)) # full state feedback
            self.num_observations += 13*self.feedback_horizon

        if self.experiment_dict.get('include_env_params', False):
            # 5 parameters: mass, ixx, iyy, izz, k_eta, k_m
            # One for the assumed parameters and one for the true parameters
            self.num_observations += 5 * 2


        self.observation_space = spaces.Box(low = -np.inf, high=np.inf, shape = (self.num_observations,), dtype=np.float32)
        
        ############ ACTION SPACE

        # For generalizability, we assume the controller outputs 4 numbers between -1 and 1. Depending on the control mode, we scale these appropriately. 
        
        if self.control_mode == 'cmd_vel':
            self.action_space = spaces.Box(low = -1, high = 1, shape = (3,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low = -1, high = 1, shape = (4,), dtype=np.float32)

        ######  Min/max values for scaling control outputs.

        self.rotor_speed_max = self.quadrotor.rotor_speed_max
        self.rotor_speed_min = self.quadrotor.rotor_speed_min

        # Compute the min/max thrust by assuming the rotor is spinning at min/max speed. (also generalizes to bidirectional rotors)
        self.max_thrust = self.quadrotor.k_eta * self.quadrotor.rotor_speed_max**2
        self.min_thrust = self.quadrotor.k_eta * self.quadrotor.rotor_speed_min**2

        # Find the maximum moment on each axis, N-m
        self.max_roll_moment = self.max_thrust * np.abs(self.quadrotor.rotor_pos['r1'][1])
        self.max_pitch_moment = self.max_thrust * np.abs(self.quadrotor.rotor_pos['r1'][0])
        self.max_yaw_moment = self.quadrotor.k_m * self.quadrotor.rotor_speed_max**2

        # Set the maximum body rate on each axis (this is hand selected), rad/s
        self.max_roll_br = 7.0
        self.max_pitch_br = 7.0 
        self.max_yaw_br = 3.0

        # Set the maximum speed command (this is hand selected), m/s
        self.max_vel = 3/math.sqrt(3)   # Selected so that at most the max speed is 3 m/s

        ###################################################################################################

        # Save the order of magnitude of the rotor speeds for later normalization
        self.rotor_speed_order_mag = math.floor(math.log(self.quadrotor.rotor_speed_max, 10))

        if world is None:
            # If no world is specified, assume that it means that the intended world is free space.
            wbound = 4 
            self.world = World.empty((-wbound, wbound, -wbound, 
                                       wbound, -wbound, wbound))
        else:
            self.world = world

        if wind_profile is None:
            # If wind is not specified, default to no wind. 
            from rotorpy.wind.default_winds import NoWind
            self.wind_profile = NoWind()
        else:
            self.wind_profile = wind_profile

        if self.render_mode == '3D':
            if fig is None and ax is None:
                self.fig = plt.figure('Visualization')
                self.ax = self.fig.add_subplot(projection='3d')
            else:
                self.fig = fig
                self.ax = ax
            if color is None:
                colors = list(mcolors.CSS4_COLORS)
            else:
                colors = [color]
            self.quad_obj = Quadrotor(self.ax, wind=True, color=np.random.choice(colors), wind_scale_factor=5)
            self.world_artists = None
            self.title_artist = self.ax.set_title('t = {}'.format(self.t))

        self.rendering = False   # Bool for tracking when the renderer is actually rendering a frame. 

        return 

    def render(self):
        if self.render_mode == '3D':
            self._plot_quad()
        elif self.render_mode == 'console':
            self._print_quad()

    def close(self):
        if self.fig is not None:
            # Close the plots
            plt.close('all')
    
    def reset(self, seed=None, initial_state='random', options={'pos_bound': 0.5, 'vel_bound': 0}):
        """
        Reset the environment
        Inputs:
            seed: the seed for any random number generation, mostly for reproducibility. 
            initial_state: determines how to set the quadrotor again. Options are...
                        'random': will randomly select the state of the quadrotor. 
                        'deterministic': will set the state to the initial state selected by the user when creating
                                         the quadrotor environment (usually hover). 
                        the user can also specify the state itself as a dictionary... e.g. 
                            reset(options={'initial_state': 
                                 {'x': np.array([0,0,0]),
                                  'v': np.zeros(3,),
                                  'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                                  'w': np.zeros(3,),
                                  'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                                  'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
                                  })
            options: dictionary for misc options for resetting the scene. 
                        'pos_bound': the min/max position region for random placement. 
                        'vel_bound': the min/max velocity region for random placement
                                
        """
        assert options['pos_bound'] >= 0 and options['vel_bound'] >= 0 , "Bounds must be greater than or equal to 0."

        super().reset(seed=seed)


        self.reset_count += 1 

        if self.experiment_dict.get('reference_randomize_threshold', -1) > 0:
            if self.reset_count > self.experiment_dict['reference_randomize_threshold']:
                self.ref.reset()

        if initial_state == 'random':
            # Randomly select an initial state for the quadrotor. At least assume it is level. 
            pos = np.random.uniform(low=-options['pos_bound'], high=options['pos_bound'], size=(3,))
            vel = np.random.uniform(low=-options['vel_bound'], high=options['vel_bound'], size=(3,))
            state = {'x': pos,
                     'v': vel,
                     'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                     'w': np.zeros(3,),
                     'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                     'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

        elif initial_state == 'deterministic':
            state = self.initial_state
        
        elif isinstance(initial_state, dict):
            # Ensure the correct keys are in dict.  
            if all(key in initial_state for key in ('x', 'v', 'q', 'w', 'wind', 'rotor_speeds')):
                state = initial_state
            else:
                raise KeyError("Missing state keys in your initial_state. You must specify values for ('x', 'v', 'q', 'w', 'wind', 'rotor_speeds')")

        else:
            raise ValueError("You must either specify 'random', 'deterministic', or provide a dict containing your desired initial state.")
        
        quad_params_copy = deepcopy(self.quad_params)
        assumed_quad_params = deepcopy(self.quad_params)
        
        domain_rand = self.experiment_dict.get('domain_randomization', 0.0) # amount of domain randomization in percent 

        # Randomize parameters
        if domain_rand > 0:
            # Randomize the mass
            quad_params_copy['mass'] = np.random.uniform(low=quad_params_copy['mass']*(1-domain_rand), high=quad_params_copy['mass']*(1+domain_rand))
            # Randomize the inertia
            quad_params_copy['Ixx'] = np.random.uniform(low=quad_params_copy['Ixx']*(1-domain_rand), high=quad_params_copy['Ixx']*(1+domain_rand))
            quad_params_copy['Iyy'] = np.random.uniform(low=quad_params_copy['Iyy']*(1-domain_rand), high=quad_params_copy['Iyy']*(1+domain_rand))
            quad_params_copy['Izz'] = np.random.uniform(low=quad_params_copy['Izz']*(1-domain_rand), high=quad_params_copy['Izz']*(1+domain_rand))
            # Randomize the thrust coefficient
            quad_params_copy['k_eta'] = np.random.uniform(low=quad_params_copy['k_eta']*(1-domain_rand), high=quad_params_copy['k_eta']*(1+domain_rand))
            # Randomize the moment coefficient
            quad_params_copy['k_m'] = np.random.uniform(low=quad_params_copy['k_m']*(1-domain_rand), high=quad_params_copy['k_m']*(1+domain_rand))
        
        if self.experiment_dict.get('model_mismatch', False):
            self.assumed_quad_params = assumed_quad_params
        else:
            self.assumed_quad_params = quad_params_copy
        self.true_quad_params = quad_params_copy
        
        # Update the quadrotor model with the new parameters
        self.quadrotor = MultirotorModelMismatch(quad_params=quad_params_copy, initial_state=state, control_abstraction=self.control_mode, aero=self.aero, assumed_quad_params=self.assumed_quad_params)
    
        # Set up scaling of the action space
        self.rotor_speed_max = self.assumed_quad_params['rotor_speed_max']
        self.rotor_speed_min = self.assumed_quad_params['rotor_speed_min']

        # Compute the min/max thrust by assuming the rotor is spinning at min/max speed. (also generalizes to bidirectional rotors)
        self.max_thrust = self.assumed_quad_params['k_eta'] * self.rotor_speed_max**2
        self.min_thrust = self.assumed_quad_params['k_eta'] * self.rotor_speed_min**2

        # Find the maximum moment on each axis, N-m
        self.max_roll_moment = self.max_thrust * np.abs(self.assumed_quad_params['rotor_pos']['r1'][1])
        self.max_pitch_moment = self.max_thrust * np.abs(self.assumed_quad_params['rotor_pos']['r1'][0])
        self.max_yaw_moment = self.assumed_quad_params['k_m'] * self.rotor_speed_max**2

        # Set the initial state. 
        self.vehicle_state = state

        # Reset the time
        self.t = 0.0

        # Reset the reward
        self.reward = 0.0

        if self.experiment_dict.get('l1_simulation', False):
            self.v_hat = np.zeros(3)
            self.d_hat = np.zeros(3)
            self.d_hat_t = np.zeros(3)
        
        # Now get observation and info using the new state
        if self.feedback_horizon > 0:
            self.feedback_buffer = np.zeros((self.feedback_horizon, 13)) # full state feedback
        observation = self._get_obs()
        info = self._get_info()

        self.render()

        return (observation, info)
    

    def step(self, action):

        """
        Step the quadrotor dynamics forward by one step based on the policy action. 
        Inputs:
            action: The action is a 4x1 vector which depends on the control abstraction: 

            if control_mode == 'cmd_vel':
                action[0] (-1,1) := commanded velocity in x direction (will be rescaled to m/s)
                action[1] (-1,1) := commanded velocity in y direction (will be rescaled to m/s)
                action[2] (-1,1) := commanded velocity in z direction (will be rescaled to m/s)
            if control_mode == 'cmd_ctbr':
                action[0] (-1,1) := the thrust command (will be rescaled to Newtons)
                action[1] (-1,1) := the roll body rate (will be rescaled to rad/s)
                action[2] (-1,1) := the pitch body rate (will be rescaled to rad/s)
                action[3] (-1,1) := the yaw body rate (will be rescaled to rad/s)
            if control_mode == 'cmd_ctbm':
                action[0] (-1,1) := the thrust command (will be rescaled to Newtons)
                action[1] (-1,1) := the roll moment (will be rescaled to Newton-meters)
                action[2] (-1,1) := the pitch moment (will be rescaled to Newton-meters)
                action[3] (-1,1) := the yaw moment (will be rescaled to Newton-meters)
            if control_mode == 'cmd_motor_speeds':
                action[0] (-1,1) := motor 1 speed (will be rescaled to rad/s)
                action[1] (-1,1) := motor 2 speed (will be rescaled to rad/s)
                action[2] (-1,1) := motor 3 speed (will be rescaled to rad/s)
                action[3] (-1,1) := motor 4 speed (will be rescaled to rad/s)
            if control_mode == 'cmd_motor_forces':
                action[0] (-1,1) := motor 1 force (will be rescaled to Newtons)
                action[1] (-1,1) := motor 2 force (will be rescaled to Newtons)
                action[2] (-1,1) := motor 3 force (will be rescaled to Newtons)
                action[3] (-1,1) := motor 4 force (will be rescaled to Newtons)

        """
        
        # First rescale the action and get the appropriate control dictionary given the control mode.
        self.control_dict = self.rescale_action(action)

        # Now update the wind state using the wind profile
        self.vehicle_state['wind'] = self.wind_profile.update(self.t, self.vehicle_state['x'])

        # Last perform forward integration using the commanded motor speed and the current state
        self.vehicle_state = self.quadrotor.step(self.vehicle_state, self.control_dict, self.t_step)
        observation = self._get_obs()
        
        # Update t by t_step
        self.t += self.t_step

        # Check for safety
        safe = self.safety_exit()

        # Determine whether or not the session should terminate. 
        terminated = (self.t >= self.max_time) or not safe

        # Now compute the reward based on the current state
        self.reward = self._get_reward(self.vehicle_state, self.ref, action)

        # Finally get info
        info = self._get_info()

        truncated = False

        self.render()

        return (observation, self.reward, terminated, truncated, info)
    
    def close(self):
        """
        Close the environment
        """
        return None
    
    def rescale_action(self, action):
            """
            Rescales the action to within the control limits and then assigns the appropriate dictionary. 
            """

            control_dict = {}

            if self.control_mode == 'cmd_ctbm':
                # Scale action[0] to (0,1) and then scale to the max thrust
                cmd_thrust = np.interp(action[0], [-1,1], [self.quadrotor.num_rotors*self.min_thrust, self.quadrotor.num_rotors*self.max_thrust])

                # Scale the moments
                cmd_roll_moment = np.interp(action[1], [-1,1], [-self.max_roll_moment, self.max_roll_moment])
                cmd_pitch_moment = np.interp(action[2], [-1,1], [-self.max_pitch_moment, self.max_pitch_moment])
                cmd_yaw_moment = np.interp(action[3], [-1,1], [-self.max_yaw_moment, self.max_yaw_moment])

                control_dict['cmd_thrust'] = cmd_thrust
                control_dict['cmd_moment'] = np.array([cmd_roll_moment, cmd_pitch_moment, cmd_yaw_moment])
            
            elif self.control_mode == 'cmd_ctbr':
                # Scale action to min and max thrust.
                cmd_thrust = np.interp(action[0], [-1, 1], [self.quadrotor.num_rotors*self.min_thrust, self.quadrotor.num_rotors*self.max_thrust])

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

            return control_dict
    
    def _get_reward(self, state, ref, action):
        """
        Compute the reward for the current state and goal.
        Inputs:
            observation: The current state of the quadrotor
            action: The current action of the controller
        Outputs:
            the current reward.
        """

        return self.reward_fn(state, ref, action, self.t)
    
    def safety_exit(self):
        """
        Return exit status if any safety condition is violated, otherwise None.
        """
        if np.any(np.abs(self.vehicle_state['v']) > 100):
            return False
        if np.any(np.abs(self.vehicle_state['w']) > 100):
            return False
        if self.vehicle_state['x'][0] < self.world.world['bounds']['extents'][0] or self.vehicle_state['x'][0] > self.world.world['bounds']['extents'][1]:
            return False
        if self.vehicle_state['x'][1] < self.world.world['bounds']['extents'][2] or self.vehicle_state['x'][1] > self.world.world['bounds']['extents'][3]:
            return False
        if self.vehicle_state['x'][2] < self.world.world['bounds']['extents'][4] or self.vehicle_state['x'][2] > self.world.world['bounds']['extents'][5]:
            return False

        if len(self.world.world.get('blocks', [])) > 0:
            # If a world has objects in it we need to check for collisions.  
            collision_pts = self.world.path_collisions(self.vehicle_state['x'], 0.25)
            no_collision = collision_pts.size == 0
            if not no_collision:
                return False
        return True
    
    def _get_obs(self):
        self.current_state = np.hstack([self.vehicle_state['x'], self.vehicle_state['v'], self.vehicle_state['q'], self.vehicle_state['w']])
        obs = self.current_state
        if self.experiment_dict.get('fb_body_frame', True):
            pos = self.vehicle_state['x']
            rot = Rotation.from_quat(self.vehicle_state['q'])

            fb_term = pos - rot.inv().apply(self.ref.update(self.t)['x'])
            if self.time_horizon > 0:
                futures = np.hstack([pos - rot.inv().apply(self.ref.update(self.t + i*self.t_step)['x']) for i in range(self.time_horizon)])
                obs = np.hstack([obs, fb_term, futures])
            else:
                obs = np.hstack([obs, fb_term])
        else:
            fb_term = pos - self.ref.update(self.t)['x']
            if self.time_horizon > 0:
                futures = np.hstack([self.ref.update(self.t + i*self.t_step)['x'] for i in range(self.time_horizon)])
                obs = np.hstack([obs, fb_term, futures])
            else:
                obs = np.hstack([obs, fb_term])

        if self.experiment_dict.get('l1_simulation', False):
            d_hat = self.l1_simulation()
            obs = np.hstack([obs, d_hat])
        
        
        if self.feedback_horizon > 0:
            self.feedback_buffer = np.roll(self.feedback_buffer, -1, axis=0)
            self.feedback_buffer[-1] = current_state
            obs = np.hstack([obs, self.feedback_buffer.flatten()])
        
        if self.experiment_dict.get('include_env_params', False):
            # Include the assumed and true parameters
            obs = np.hstack([obs, np.array([self.assumed_quad_params['mass'], self.assumed_quad_params['Ixx'], self.assumed_quad_params['Iyy'], self.assumed_quad_params['Izz'], self.assumed_quad_params['k_eta']])])
            obs = np.hstack([obs, np.array([self.true_quad_params['mass'], self.true_quad_params['Ixx'], self.true_quad_params['Iyy'], self.true_quad_params['Izz'], self.true_quad_params['k_eta']])])
        

        return obs

    def _get_current_state(self):
        return self.current_state
    
    def _get_current_reference(self):
        return self.ref.update(self.t)
    
    def l1_simulation(self):
        vel = self.vehicle_state['v']
        rot = Rotation.from_quat(self.vehicle_state['q'])
        f = rot.apply([0, 0, self.quadrotor.previous_thrust[-1]]) # body z force rotated into world frame
    
        self.A = -0.01
        self.lamb = 0.1

        g_vec = np.array([0, 0, -1]) * self.quadrotor.g
        alpha = 0.99
        phi = 1 / self.A * (np.exp(self.A * self.t_step) - 1)

        a_t_hat = g_vec + f / self.assumed_quad_params['mass'] - self.d_hat_t + self.A * (self.v_hat - vel)
        
        self.v_hat += a_t_hat * self.t_step
        v_tilde = self.v_hat - vel
        
        self.d_hat_t = 1 / phi * np.exp(self.A * self.t_step) * v_tilde
        self.d_hat = -(1 - alpha) * self.d_hat_t + alpha * self.d_hat

        return self.d_hat

    def _get_info(self):
        return {}

    def _plot_quad(self):

        if abs(self.t / (1/self.metadata['render_fps']) - round(self.t / (1/self.metadata['render_fps']))) > 5e-2:
            self.rendering = False  # Set rendering bool to false.
            return
        
        self.rendering = True # Set rendering bool to true. 

        plot_position = deepcopy(self.vehicle_state['x'])
        plot_rotation = Rotation.from_quat(self.vehicle_state['q']).as_matrix()
        plot_wind = deepcopy(self.vehicle_state['wind'])

        if self.world_artists is None and not ('x' in self.ax.get_xlabel()):
            self.world_artists = self.world.draw(self.ax)
            self.ax.plot(0, 0, 0, 'go')

        self.quad_obj.transform(position=plot_position, rotation=plot_rotation, wind=plot_wind)
        self.title_artist.set_text('t = {:.2f}'.format(self.t))

        plt.pause(1e-9)

        return 
    
    def _print_quad(self):

        print("Time: %3.2f \t Position: (%3.2f, %3.2f, %3.2f) \t Reward: %3.2f" % (self.t, self.vehicle_state['x'][0], self.vehicle_state['x'][1], self.vehicle_state['x'][2], self.reward))