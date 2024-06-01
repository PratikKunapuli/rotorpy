from gymnasium.envs.registration import register

register(
     id="Quadrotor-v0",
     entry_point="rotorpy.learning.quadrotor_environments:QuadrotorEnv",
)

register(
     id="Quadrotor-v1",
     entry_point="rotorpy.learning.quadrotor_environments:QuadrotorTrackingEnv",
)