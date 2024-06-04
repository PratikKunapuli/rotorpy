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
    'l1_simulation': True
}

# Experiment dictionary for the trajectory tracking task
datt_model_mismatch_40 = {
    'domain_randomization': 0.4,
    'model_mismatch': True,
    'time_horizon': 10,
    'include_env_params': False,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': True
}

# Experiment dictionary for the trajectory tracking task
datt_model_mismatch_60 = {
    'domain_randomization': 0.6,
    'model_mismatch': True,
    'time_horizon': 10,
    'include_env_params': False,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': True
}

# Experiment dictionary for the trajectory tracking task
datt_no_model_mismatch_20 = {
    'domain_randomization': 0.2,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': False,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': True
}

# Experiment dictionary for the trajectory tracking task
datt_no_model_mismatch_40 = {
    'domain_randomization': 0.4,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': False,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': True
}

# Experiment dictionary for the trajectory tracking task
datt_no_model_mismatch_60 = {
    'domain_randomization': 0.6,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': False,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': True
}

rma_model_mismatch_20 = {
    'domain_randomization':0.2,
    'model_mismatch': True,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': False
}

rma_model_mismatch_40 = {
    'domain_randomization':0.4,
    'model_mismatch': True,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': False
}

rma_model_mismatch_60 = {
    'domain_randomization':0.6,
    'model_mismatch': True,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': False
}

rma_no_model_mismatch_60 = {
    'domain_randomization':0.6,
    'model_mismatch': False,
    'time_horizon': 10,
    'include_env_params': True,
    'reference_randomize_threshold': 5000,                  # Env resets before refernce is randomized
    'l1_simulation': False
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
    'rma_no_model_mismatch_60': rma_no_model_mismatch_60
}

# Load experiment by name
def load_experiment(name):
    if name in known_experiments:
        return known_experiments[name]
    else:
        raise ValueError('Experiment {} not found'.format(name))