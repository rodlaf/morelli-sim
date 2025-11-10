# distutils: language = c
# cython: language_level=3

"""
Cython wrapper for f16_waypoint.h
Minimal binding - all logic is in C
"""

cimport numpy as np
import numpy as np
from libc.stdlib cimport srand, rand

cdef extern from "f16_waypoint.h":
    cdef int OBS_DIM_TOTAL
    
    ctypedef struct F16Waypoint:
        double state[16]
        double waypoint[3]
        double time
        double step_size
        double time_limit
        double u_ref[4]
        double Nz
        double ps
        double Ny_r
        int tick
        unsigned int seed
        double reward
    
    void f16_waypoint_reset(F16Waypoint* env, int keep_position)
    int f16_waypoint_step(F16Waypoint* env, const double u_ref[4])
    void f16_waypoint_render(F16Waypoint* env)
    int f16_waypoint_should_close()
    void f16_waypoint_close()
    void f16_waypoint_clear_trail()
    void get_observation(const F16Waypoint* env, double obs[28])



cdef class F16WaypointEnv:
    """Thin Cython wrapper for C F16Waypoint environment"""
    cdef F16Waypoint env
    cdef bint initialized
    
    def __init__(self, double step_size=1.0/30.0, double time_limit=100.0, 
                 unsigned int random_seed=0):
        """Initialize environment"""
        self.env.step_size = step_size
        self.env.time_limit = time_limit
        self.env.seed = random_seed if random_seed != 0 else <unsigned int>42
        
        # Seed C random
        srand(self.env.seed)
        
        self.initialized = True
    
    def reset(self, bint keep_position=False):
        """Reset environment"""
        f16_waypoint_reset(&self.env, 1 if keep_position else 0)
        
        # Return state and info
        cdef int i
        state = np.zeros(13, dtype=np.float64)
        for i in range(13):
            state[i] = self.env.state[i]
        
        info = {
            'time': self.env.time,
            'waypoint': np.array([self.env.waypoint[0], self.env.waypoint[1], 
                                 self.env.waypoint[2]], dtype=np.float64),
        }
        return state, info
    
    def step(self, np.ndarray[double, ndim=1] u_ref):
        """Step environment with control input"""
        cdef double u_ref_c[4]
        u_ref_c[0] = u_ref[0]
        u_ref_c[1] = u_ref[1]
        u_ref_c[2] = u_ref[2]
        u_ref_c[3] = u_ref[3]
        
        cdef int result = f16_waypoint_step(&self.env, u_ref_c)
        
        # Return state, reward, terminated, truncated, info
        cdef int i
        state = np.zeros(13, dtype=np.float64)
        for i in range(13):
            state[i] = self.env.state[i]
        
        cdef bint terminated = (result == 1) or (result == 2)
        cdef bint truncated = (result == 3)
        # Use the random reward generated in the C code
        cdef double reward = self.env.reward
        
        info = {
            'time': self.env.time,
            'waypoint': np.array([self.env.waypoint[0], self.env.waypoint[1], 
                                 self.env.waypoint[2]], dtype=np.float64),
            'Nz': self.env.Nz,
            'ps': self.env.ps,
            'Ny_r': self.env.Ny_r,
        }
        
        if result == 1:
            info['termination_reason'] = 'success'
        elif result == 2:
            info['termination_reason'] = 'physics'
        
        return state, reward, terminated, truncated, info
    
    def render(self):
        """Render current state"""
        f16_waypoint_render(&self.env)
    
    def should_close_window(self):
        """Check if window should close"""
        return f16_waypoint_should_close() != 0
    
    def close_window(self):
        """Close window"""
        f16_waypoint_close()
    
    def clear_trail(self):
        """Clear trail"""
        f16_waypoint_clear_trail()
    
    @property
    def state(self):
        """Get current state"""
        cdef int i
        state = np.zeros(13, dtype=np.float64)
        for i in range(13):
            state[i] = self.env.state[i]
        return state
    
    @property
    def waypoint(self):
        """Get current waypoint"""
        return np.array([self.env.waypoint[0], self.env.waypoint[1], 
                        self.env.waypoint[2]], dtype=np.float64)
    
    @property
    def time(self):
        """Get current time"""
        return self.env.time
    
    def get_observation(self):
        """Get RL-compatible observation vector (28 dimensions)
        
        Returns observation matching JAX implementation:
        - [0-18]: F16 state (velocity, sin/cos angles, rates, altitude, power, integrators)
        - [19-21]: Previous actions (Nz, ps, throttle)
        - [22-27]: Waypoint info (sin/cos azimuth, sin/cos elevation, symlog range, time)
        """
        cdef int i
        obs = np.zeros(28, dtype=np.float64)
        cdef double[::1] obs_view = obs
        get_observation(&self.env, &obs_view[0])
        return obs
