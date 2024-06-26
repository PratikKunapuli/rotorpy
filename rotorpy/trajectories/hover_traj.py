import numpy as np

class HoverTraj(object):
    """
    This trajectory simply has the quadrotor hover at the origin indefinitely.
    By modifying the initial condition, you can create step response
    experiments.
    """
    def __init__(self):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission.
        """
        self.future_buffer = np.zeros((10,3))
        self.fill_buffer(t=0)
    
    def reset(self):
        """
        Reset the trajectory to the initial conditions.
        """
        self.fill_buffer(t=0)

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        current_ref = self.update_once(t=t)
        self.fill_buffer(t=t)
        flat_output = { 'x': current_ref['x'], 'x_dot': current_ref['x_dot'], 'x_ddot': current_ref['x_ddot'], 'x_dddot': current_ref['x_dddot'], 'x_ddddot': current_ref['x_ddddot'],
                        'yaw': current_ref['yaw'], 'yaw_dot': current_ref['yaw_dot'], 'yaw_ddot': current_ref['yaw_ddot'], 'future_pos': self.future_buffer}
    
        return flat_output
    
    def update_once(self, t=0):
        x    = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw    = 0
        yaw_dot = 0
        yaw_ddot = 0

        current_ref = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot, 'yaw_ddot':yaw_ddot}
        return current_ref

    def fill_buffer(self, t=0):
        """
        This function fills the buffer with future desired states. 
        """
        for i in range(10):
            self.future_buffer[i] = self.update_once(t + i*0.01)['x']
