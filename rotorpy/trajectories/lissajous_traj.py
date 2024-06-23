import numpy as np
import random
"""
Lissajous curves are defined by trigonometric functions parameterized in time. 
See https://en.wikipedia.org/wiki/Lissajous_curve

"""
class TwoDLissajous(object):
    """
    The standard Lissajous on the XY curve as defined by https://en.wikipedia.org/wiki/Lissajous_curve
    This is planar in the XY plane at a fixed height. 
    """
    def __init__(self, A=1, B=1, a=1, b=1, delta=0, height=0, yaw_bool=False, dt=0.01, seed=2024, fixed_seed = False, env_diff_seed=True):
        """
        This is the constructor for the Trajectory object. A fresh trajectory
        object will be constructed before each mission.

        Inputs:
            A := amplitude on the X axis
            B := amplitude on the Y axis
            a := frequency on the X axis
            b := frequency on the Y axis
            delta := phase offset between the x and y parameterization
            height := the z height that the lissajous occurs at
            yaw_bool := determines whether the vehicle should yaw
        """

        self.A, self.B = A, B
        self.a, self.b = a, b 
        self.delta = delta
        self.height = height

        self.yaw_bool = yaw_bool

        self.seed = seed
        self.fixed_seed = fixed_seed
        self.env_diff_seed = env_diff_seed
        self.reset_count = 0

        self.dt = dt
        self.future_buffer = np.zeros((10,3))
        self.fill_buffer(t=0)

    def gen_coefficients(self):
        # self.A = np.random.uniform(-2, 2)
        # self.B = np.random.uniform(-2, 2)
        self.A = 1
        self.B = 1
        self.a = np.random.uniform(0.5, 2)
        self.b = np.random.uniform(0.5, 2)
        self.delta = np.random.uniform(0, np.pi)

    def reset(self):
        """
        Reset the trajectory to the initial conditions and randomize the parameters
        """
        if self.fixed_seed:
            np.random.seed(self.seed)
        elif self.env_diff_seed and self.reset_count > 0:
            np.random.seed(random.randint(0, 1000000))
            
        self.reset_count += 1
        self.gen_coefficients()

        self.fill_buffer(t=0)
    

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.1

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
        x        = np.array([self.A*np.sin(self.a*t + self.delta),
                             self.B*np.sin(self.b*t),
                             self.height])
        x_dot    = np.array([self.a*self.A*np.cos(self.a*t + self.delta),
                             self.b*self.B*np.cos(self.b*t),
                             0])
        x_ddot   = np.array([-(self.a)**2*self.A*np.sin(self.a*t + self.delta),
                             -(self.b)**2*self.B*np.sin(self.b*t),
                             0])
        x_dddot  = np.array([-(self.a)**3*self.A*np.cos(self.a*t + self.delta),
                             -(self.b)**3*self.B*np.cos(self.b*t),
                             0])
        x_ddddot = np.array([(self.a)**4*self.A*np.sin(self.a*t + self.delta),
                             (self.b)**4*self.B*np.sin(self.b*t),
                             0])

        if self.yaw_bool:
            yaw = np.pi/4*np.sin(np.pi*t)
            yaw_dot = np.pi*np.pi/4*np.cos(np.pi*t)
            yaw_ddot = np.pi*np.pi*np.pi/4*np.cos(np.pi*t)
        else:
            yaw = 0
            yaw_dot = 0
            yaw_ddot = 0
        
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot, 'yaw_ddot':yaw_ddot}
        return flat_output


    def fill_buffer(self, t=0):
        """
        This function fills the buffer with future desired states. 
        """
        for i in range(10):
            self.future_buffer[i] = self.update_once(t + i*self.dt)['x']

if __name__ == "__main__":
    traj = TwoDLissajous(A=1, B=1, a = 2, b=0.5, delta=0, height=1, yaw_bool=False, dt=0.01)
    # traj.reset()
    pos = []
    for i in range(300):
        pos.append(traj.update(i*0.01)['x'])
    
    import matplotlib.pyplot as plt
    pos = np.array(pos)
    plt.plot(pos[:,0], pos[:,1])
    plt.savefig('lissajous.png')
