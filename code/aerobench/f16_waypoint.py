"""
F-16 Waypoint Environment

Single-file environment for F-16 waypoint following task.
Combines environment, agent, and simulation in Gymnasium-style API.
"""

import time
import numpy as np
from typing import Tuple, Dict, Any

from aerobench.highlevel import f16_model
from aerobench.util import get_state_names, Euler, StateIndex
from aerobench.visualize import raylib_renderer_cy as raylib_renderer


class F16Waypoint:
    """
    F-16 waypoint following environment with random waypoint generation.
    """
    
    def __init__(self, 
                 num_waypoints: int = 3,
                 initial_state: np.ndarray = None,
                 step_size: float = 1/30,
                 time_limit: float = 150.0,
                 extended_states: bool = True,
                 waypoint_radius: float = 10000.0,
                 altitude_range: tuple = (1000.0, 4000.0),
                 random_seed: int = None):
        """
        Initialize F-16 waypoint environment with random waypoint generation
        
        Args:
            num_waypoints: Number of waypoints to generate
            initial_state: Initial state vector (13+ elements), generated if None
            step_size: Simulation time step in seconds
            time_limit: Maximum simulation time
            extended_states: Whether to compute extended states
            waypoint_radius: Maximum distance from origin for waypoints (ft)
            altitude_range: (min_alt, max_alt) in feet
            random_seed: Random seed for reproducibility (None for random)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.num_waypoints = num_waypoints
        self.waypoint_radius = waypoint_radius
        self.altitude_range = altitude_range
        self.step_size = step_size
        self.time_limit = time_limit
        self.extended_states = extended_states
        
        # Generate random waypoints
        self.waypoints = self._generate_waypoints()
        
        # Generate or use provided initial state
        if initial_state is None:
            initial_state = self._generate_initial_state()
        
        # State management
        num_vars = len(get_state_names()) + 3  # num integrators
        if initial_state.size < num_vars:
            self.initial_state = np.zeros(num_vars)
            self.initial_state[:initial_state.shape[0]] = initial_state
        else:
            self.initial_state = initial_state.copy()
        
        self.state = None
        self.time = None
        self.integrator = None
        
        # History tracking
        self.times = []
        self.states = []
        
        if self.extended_states:
            self.xd_list = []
            self.u_list = []
            self.Nz_list = []
            self.ps_list = []
            self.Ny_r_list = []
        
        self.integrator_class = Euler
        self.wall_time_start = None
    
    def _generate_waypoints(self) -> list:
        """Generate random waypoints within specified constraints"""
        waypoints = []
        for _ in range(self.num_waypoints):
            # Random position within circular region
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, self.waypoint_radius)
            
            east = radius * np.cos(angle)
            north = radius * np.sin(angle)
            altitude = np.random.uniform(self.altitude_range[0], self.altitude_range[1])
            
            waypoints.append([east, north, altitude])
        
        return waypoints
    
    def _generate_initial_state(self) -> np.ndarray:
        """Generate a reasonable initial state for F-16"""
        from numpy import deg2rad
        
        # Standard initial conditions
        power = 9
        alpha = deg2rad(2.1215)
        beta = 0
        alt = 1500
        vt = 540
        phi = 0
        theta = deg2rad(2.1215)  # Match alpha for level flight
        psi = 0
        
        # [vt, alpha, beta, phi, theta, psi, p, q, r, pos_n, pos_e, alt, power]
        return np.array([vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power], dtype=float)
        
    def render(self):
        """Render current state using Raylib"""
        if self.state is None or self.time is None:
            return
        
        # Get current extended states (Nz, ps) - use most recent or compute
        if self.extended_states and len(self.Nz_list) > 0:
            nz_g = float(self.Nz_list[-1])
            ps_rad_s = float(self.ps_list[-1])
        else:
            # Compute on the fly if not tracking
            nz_g = 0.0
            ps_rad_s = 0.0
        
        # Build render state dict for C renderer
        render_state = {
            'time_sec': float(self.time),
            'speed_fps': float(self.state[StateIndex.VT]),
            'alpha_rad': float(self.state[StateIndex.ALPHA]),
            'beta_rad': float(self.state[StateIndex.BETA]),
            'phi_rad': float(self.state[StateIndex.PHI]),
            'theta_rad': float(self.state[StateIndex.THETA]),
            'psi_rad': float(self.state[StateIndex.PSI]),
            'position_ft': (
                float(self.state[StateIndex.POSE]),
                float(self.state[StateIndex.POSN]),
                float(self.state[StateIndex.ALT])
            ),
            'nz_g': nz_g,
            'ps_rad_s': ps_rad_s,
            'waypoints': self.waypoints
        }
        
        raylib_renderer.render(render_state)
    
    def should_close_window(self) -> bool:
        """Check if render window should close"""
        return raylib_renderer.window_should_close()
    
    def close_window(self):
        """Close render window"""
        raylib_renderer.close()
    
    def reset_rendering(self):
        """Reset rendering state (trail, etc.)"""
        raylib_renderer.reset()
        
    def reset(self, initial_time: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to initial state and generate new waypoints"""
        # Generate new random waypoints
        self.waypoints = self._generate_waypoints()
        
        self.state = self.initial_state.copy()
        self.time = initial_time
        
        self.times = [self.time]
        self.states = [self.state.copy()]
        
        if self.extended_states:
            self.xd_list = []
            self.u_list = []
            self.Nz_list = []
            self.ps_list = []
            self.Ny_r_list = []
        
        self.integrator = self.integrator_class(
            self._dynamics, self.time, self.state, np.inf, step=self.step_size)
        
        self.wall_time_start = time.perf_counter()
        
        return self.state.copy(), self._get_info()
    
    def step(self, u_ref: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one simulation step with control input u_ref"""
        self._current_u_ref = u_ref
        
        next_time = self.time + self.step_size
        
        while self.integrator.t < next_time - 1e-7 and self.integrator.status == 'running':
            self.integrator.step()
        
        if self.integrator.status != 'running':
            info = self._get_info()
            info['status'] = self.integrator.status
            return self.state.copy(), 0.0, True, False, info
        
        self.time = next_time
        self.state = self.integrator.dense_output()(self.time)
        
        self.times.append(self.time)
        self.states.append(self.state.copy())
        
        if self.extended_states:
            xd, u, Nz, ps, Ny_r = f16_model.controlled_f16_wrapper(self.state, u_ref)
            self.xd_list.append(xd)
            self.u_list.append(u)
            self.Nz_list.append(Nz)
            self.ps_list.append(ps)
            self.Ny_r_list.append(Ny_r)
        
        terminated = self._check_terminated()
        truncated = self.time >= self.time_limit
        
        return self.state.copy(), 0.0, terminated, truncated, self._get_info()
    
    def _dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """Dynamics function for integration"""
        alpha = state[StateIndex.ALPHA]
        if not -2 < alpha < 2:
            raise RuntimeError(f"alpha ({alpha}) out of bounds")
        
        vel = state[StateIndex.VEL]
        if not 200 <= vel <= 3000:
            raise RuntimeError(f"velocity ({vel}) out of bounds")
        
        alt = state[StateIndex.ALT]
        if not -10000 < alt < 100000:
            raise RuntimeError(f"altitude ({alt}) out of bounds")
        
        return f16_model.controlled_f16_wrapper(state, self._current_u_ref)[0]
    
    def _check_terminated(self) -> bool:
        """Check termination conditions"""
        try:
            alpha = self.state[StateIndex.ALPHA]
            if not -2 < alpha < 2:
                return True
            
            vel = self.state[StateIndex.VEL]
            if not 200 <= vel <= 3000:
                return True
            
            alt = self.state[StateIndex.ALT]
            if not -10000 < alt < 100000:
                return True
        except:
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Build info dictionary"""
        info = {
            'time': self.time,
            'wall_time': time.perf_counter() - self.wall_time_start if self.wall_time_start else 0.0,
        }
        
        if self.extended_states and len(self.xd_list) > 0:
            info['xd'] = self.xd_list[-1]
            info['u'] = self.u_list[-1]
            info['Nz'] = self.Nz_list[-1]
            info['ps'] = self.ps_list[-1]
            info['Ny_r'] = self.Ny_r_list[-1]
        
        return info
    
    def get_history(self) -> Dict[str, Any]:
        """Get complete simulation history"""
        history = {
            'times': self.times.copy(),
            'states': np.array(self.states, dtype=float),
            'wall_time': time.perf_counter() - self.wall_time_start if self.wall_time_start else 0.0,
        }
        
        if self.extended_states:
            history['xd_list'] = self.xd_list.copy()
            history['u_list'] = self.u_list.copy()
            history['Nz_list'] = self.Nz_list.copy()
            history['ps_list'] = self.ps_list.copy()
            history['Ny_r_list'] = self.Ny_r_list.copy()
        
        return history
