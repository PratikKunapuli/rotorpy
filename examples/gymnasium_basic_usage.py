import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# For this demonstration, we'll just use the SE3 controller. 
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.vehicles.crazyflie_params import quad_params

controller = SE3Control(quad_params)

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_trajectory_tracking import QuadrotorTrackingEnv

# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import hover_reward, datt_reward

from rotorpy.trajectories.lissajous_traj import TwoDLissajous

# First, we need to make the gym environment. The inputs to the model are as follows...
"""
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
    max_time: the maximum time of the session. 
    world: the world for the quadrotor to operate within. 
    sim_rate: the simulation rate (in Hz), i.e. the timestep. 
    render_mode: render the quadrotor.
            'None': no rendering
            'console': output text describing the environment. 
            '3D': will render the quadrotor in 3D. WARNING: THIS IS SLOW. 

"""
SIM_RATE = 100
sim_dt = 1/SIM_RATE

traj = TwoDLissajous(A=1, B=1, a=1, b=1, delta=0, height=1, yaw_bool=False, dt=sim_dt)

experiment_dict = {
            'domain_randomization': 0.1,
            'model_mismatch': False,
            'time_horizon': 10,
            'include_env_params': False,
            'reference_randomize_threshold': 0,                  # Env resets before refernce is randomized
            'integrator': "RK45" # Euler or RK45
}

env = QuadrotorTrackingEnv(quad_params, experiment_dict=experiment_dict, render_mode='None', reference = traj, reward_fn=datt_reward, control_mode="cmd_ctbr", sim_rate=SIM_RATE) #control_mode = "cmd_motor_speeds" or "cmd_ctbr"


# Now reset the quadrotor.
# Setting initial_state to 'random' will randomly place the vehicle in the map near`` the origin.
# But you can also set the environment resetting to be deterministic. 
observation, info = env.reset(initial_state='random', options={'pos_bound': 0, 'vel_bound': 0})
input()

# Number of timesteps
T = 500
time = np.arange(T)*(1/100)      # Just for plotting purposes.
position = np.zeros((T, 3))      # Just for plotting purposes. 
velocity = np.zeros((T, 3))      # Just for plotting purposes.
reward_sum = np.zeros((T,))      # Just for plotting purposes.
actions = np.zeros((T, 4))       # Just for plotting purposes.
rotor_speeds = np.zeros((T, 4))

reference_pos = np.zeros((T, 3))

for i in range(T):
    ##### Below is just code for computing the action via the SE3 controller and converting it to an action [-1,1]

    # Unpack the observation from the environment
    state = env.vehicle_state
    flat = env._get_current_reference()
    control_dict = controller.update(i, state, flat)

    # SRT - Extract the commanded motor speeds.
    # cmd_motor_speeds = control_dict['cmd_motor_speeds']
    # action = np.interp(cmd_motor_speeds, [env.rotor_speed_min, env.rotor_speed_max], [-1,1])
    print("\nTime: ", time[i])

    # print("Max/Min thrust: ", env.max_thrust * 4, env.min_thrust * 4)
    # print("Max body rates: ", env.max_roll_br, env.max_pitch_br, env.max_yaw_br)

    # CTBR - Extact command and scale to [-1,1]
    thrust = control_dict['cmd_thrust']
    body_rates = control_dict['cmd_w']
    thrust_cmd = np.interp(thrust, [env.min_thrust * 4, env.max_thrust * 4], [-1,1])
    body_rates_cmd = body_rates
    body_rates_cmd[0] = np.interp(body_rates_cmd[0], [-env.max_roll_br, env.max_roll_br], [-1,1])
    body_rates_cmd[1] = np.interp(body_rates_cmd[1], [-env.max_pitch_br, env.max_pitch_br], [-1,1])
    body_rates_cmd[2] = np.interp(body_rates_cmd[2], [-env.max_yaw_br, env.max_yaw_br], [-1,1])
    action = np.hstack([thrust_cmd, body_rates_cmd])
    print("Action: ", action)

    # Step forward in the environment
    observation, reward, terminated, truncated, info = env.step(action)

    print("Obs: ", observation)

    # For plotting, save the relevant information
    position[i, :] = state['x']
    velocity[i, :] = state['v']
    reference_pos[i, :] = flat['x']
    rotor_speeds[i, :] = env.vehicle_state['rotor_speeds']
    if i == 0:
        reward_sum[i] = reward
    else:
        reward_sum[i] = reward_sum[i-1] + reward
    actions[i, :] = action

print("Final Reward: ", reward_sum[-1])
env.close()
print("Env closed.")

# Plotting
(fig, axes) = plt.subplots(nrows=2, ncols=1, num='Quadrotor State')
ax = axes[0]
ax.plot(time, position[:, 0], 'r', label='X')
ax.plot(time, position[:, 1], 'g', label='Y')
ax.plot(time, position[:, 2], 'b', label='Z')
ax.set_ylabel("Position, m")
ax.legend()

ax = axes[1]
ax.plot(time, velocity[:, 0], 'r', label='X')
ax.plot(time, velocity[:, 1], 'g', label='Y')
ax.plot(time, velocity[:, 2], 'b', label='Z')
ax.set_ylabel("Velocity, m/s")
ax.set_xlabel("Time, s")

(fig, axes) = plt.subplots(nrows=2, ncols=1, num="Action and Reward")
ax = axes[0]
ax.plot(time, actions[:, 0], 'r', label='action 1')
ax.plot(time, actions[:, 1], 'g', label='action 2')
ax.plot(time, actions[:, 2], 'b', label='action 3')
ax.plot(time, actions[:, 3], 'm', label='action 4')
ax.set_ylabel("Action")
ax.legend()

ax = axes[1]
ax.plot(time, reward_sum, 'k')
ax.set_xlabel("Time, s")
ax.set_ylabel("Reward Sum")
plt.savefig("gymnasium_basic_usage.png")

plt.figure()
plt.subplot(3,1,1)
plt.plot(time, position[:, 0], 'r', label='X')
plt.plot(time, reference_pos[:, 0], 'r--', label='X Ref')
plt.ylabel("X (m)")
plt.legend()
plt.subplot(3,1,2)
plt.plot(time, position[:, 1], 'g', label='Y')
plt.plot(time, reference_pos[:, 1], 'g--', label='Y Ref')
plt.ylabel("Y (m)")
plt.legend()
plt.subplot(3,1,3)
plt.plot(time, position[:, 2], 'b', label='Z')
plt.plot(time, reference_pos[:, 2], 'b--', label='Z Ref')
plt.ylabel("Z (m)")
plt.legend()
plt.xlabel("Time (s)")
plt.savefig("gymnasium_basic_usage_tracking.png")

plt.figure()
plt.plot(time, rotor_speeds[:, 0], 'r', label='Rotor 1')
plt.plot(time, rotor_speeds[:, 1], 'g', label='Rotor 2')
plt.plot(time, rotor_speeds[:, 2], 'b', label='Rotor 3')
plt.plot(time, rotor_speeds[:, 3], 'm', label='Rotor 4')
plt.xlabel("Time, s")
plt.ylabel("Rotor Speed, rad/s")
plt.legend()
plt.savefig("gymnasium_basic_usage_rotor_speeds.png")
plt.show()