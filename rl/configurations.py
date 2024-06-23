"""
This file contains a collection of dictionaries corresponding to different experiments that can be queried by name
"""

# Experiment dictionary for the trajectory tracking task
datt_model_mismatch_20 = {
    'domain_randomization': 0.2,
    'model_mismatch': True,
    'time_horizon': 10,
    'include_env_params': False,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': True,
    'integrator': "Euler"
}

# Experiment dictionary for the trajectory tracking task
datt_model_mismatch_40 = {
    'domain_randomization': 0.4,
    'model_mismatch': True,
    'time_horizon': 10,
    'include_env_params': False,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': True,
    'integrator': "Euler"
}

# Experiment dictionary for the trajectory tracking task
datt_model_mismatch_60 = {
    'domain_randomization': 0.6,
    'model_mismatch': True,
    'time_horizon': 10,
    'include_env_params': False,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': True,
    'integrator': "Euler"
}

# Experiment dictionary for the trajectory tracking task
datt_no_model_mismatch_20 = {
    'domain_randomization': 0.2,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': False,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': True,
    'integrator': "Euler"
}

# Experiment dictionary for the trajectory tracking task
datt_no_model_mismatch_40 = {
    'domain_randomization': 0.4,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': False,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': True,
    'integrator': "Euler"
}

# Experiment dictionary for the trajectory tracking task
datt_no_model_mismatch_60 = {
    'domain_randomization': 0.6,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': False,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': True,
    'integrator': "Euler"
}

rma_model_mismatch_20 = {
    'domain_randomization':0.2,
    'model_mismatch': True,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': False,
    'integrator': "Euler"
}

rma_model_mismatch_40 = {
    'domain_randomization':0.4,
    'model_mismatch': True,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': False,
    'integrator': "Euler"
}

rma_model_mismatch_60 = {
    'domain_randomization':0.6,
    'model_mismatch': True,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': False,
    'integrator': "Euler"
}

rma_no_model_mismatch_20 = {
    'domain_randomization':0.2,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': False,
    'integrator': "Euler"
}

rma_no_model_mismatch_40 = {
    'domain_randomization':0.4,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': False,
    'integrator': "Euler"
}

rma_no_model_mismatch_60 = {
    'domain_randomization':0.6,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': False,
    'integrator': "Euler"
}

rma_traj_tracking = {
    'domain_randomization': 0.05,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 100000,                  # Env resets before refernce is randomized
    'l1_simulation': False,
    'integrator': "Euler",
    'reward': 'sejong'
}

rma_traj_tracking_no_fb = {
    'domain_randomization': 0.05,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 100000,                  # Env resets before refernce is randomized
    'l1_simulation': False,
    'integrator': "Euler",
    'fb_body_frame': False,
    'reward': 'sejong'
}

known_experiments = {
    'datt_model_mismatch_20': datt_model_mismatch_20,
    'datt_model_mismatch_40': datt_model_mismatch_40,
    'datt_model_mismatch_60': datt_model_mismatch_60,
    'datt_no_model_mismatch_20': datt_no_model_mismatch_20,
    'datt_no_model_mismatch_40': datt_no_model_mismatch_40,
    'datt_no_model_mismatch_60': datt_no_model_mismatch_60,
    'rma_model_mismatch_20': rma_model_mismatch_20,
    'rma_model_mismatch_40': rma_model_mismatch_40,
    'rma_model_mismatch_60': rma_model_mismatch_60,
    'rma_no_model_mismatch_20': rma_no_model_mismatch_20,
    'rma_no_model_mismatch_40': rma_no_model_mismatch_40,
    'rma_no_model_mismatch_60': rma_no_model_mismatch_60,
    'rma_traj_tracking': rma_traj_tracking,
    'rma_traj_tracking_no_fb': rma_traj_tracking_no_fb
}

# Load experiment by name
def load_experiment(name):
    if name in known_experiments:
        return known_experiments[name]
    else:
        raise ValueError('Experiment {} not found'.format(name))