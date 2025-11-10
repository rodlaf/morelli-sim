"""
F-16 Waypoint Environment

Single-file environment for F-16 waypoint following task.
Combines environment, agent, and simulation in Gymnasium-style API.

The environment terminates when:
  - Alpha (angle of attack) exceeds [-2, 2] radians
  - Velocity falls outside [200, 3000] ft/s
  - Altitude falls outside [-10000, 100000] ft
  - Integration fails

The environment truncates when:
  - Time limit is exceeded (default 150 seconds)

The task is continuous waypoint reaching - when a waypoint is reached
(within WAYPOINT_CAPTURE_RADIUS), a new random waypoint is automatically
generated and the episode continues indefinitely until physics bounds are
violated or time limit is reached.
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
    
    # Physics bounds for termination
    ALPHA_MIN = -2.0  # radians
    ALPHA_MAX = 2.0   # radians
    VELOCITY_MIN = 200.0   # ft/s
    VELOCITY_MAX = 3000.0  # ft/s
    ALTITUDE_MIN = -10000.0  # ft
    ALTITUDE_MAX = 100000.0  # ft
    
    # Waypoint generation bounds (east, north, altitude)
    WAYPOINT_MIN = np.array([-3000.0, 15000.0, 1000.0])
    WAYPOINT_MAX = np.array([3000.0, 20000.0, 4000.0])
    
    # Waypoint capture radius
    WAYPOINT_CAPTURE_RADIUS = 500.0  # ft
    
    def __init__(self, 
                 initial_state: np.ndarray = None,
                 step_size: float = 1/30,
                 time_limit: float = 200.0,
                 extended_states: bool = True,
                 random_seed: int = None):
        """
        Initialize F-16 waypoint environment with random waypoint generation
        
        Args:
            initial_state: Initial state vector (13+ elements), generated if None
            step_size: Simulation time step in seconds
            time_limit: Maximum simulation time
            extended_states: Whether to compute extended states
            random_seed: Random seed for reproducibility (None for random)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.step_size = step_size
        self.time_limit = time_limit
        self.extended_states = extended_states
        
        # Generate single random waypoint
        self.waypoint = self._generate_waypoint()
        
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
    
    def _generate_waypoint(self) -> np.ndarray:
        """Generate a single random waypoint within specified bounds"""
        return np.random.uniform(self.WAYPOINT_MIN, self.WAYPOINT_MAX)
    
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
            'waypoints': [self.waypoint],  # Wrap single waypoint in list for renderer
            'waypoint_radius': float(self.WAYPOINT_CAPTURE_RADIUS)
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
        """Reset to initial state and generate new waypoint"""
        # Generate new random waypoint
        self.waypoint = self._generate_waypoint()
        
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
        if not self.ALPHA_MIN < alpha < self.ALPHA_MAX:
            raise RuntimeError(f"alpha ({alpha}) out of bounds [{self.ALPHA_MIN}, {self.ALPHA_MAX}]")
        
        vel = state[StateIndex.VEL]
        if not self.VELOCITY_MIN <= vel <= self.VELOCITY_MAX:
            raise RuntimeError(f"velocity ({vel}) out of bounds [{self.VELOCITY_MIN}, {self.VELOCITY_MAX}]")
        
        alt = state[StateIndex.ALT]
        if not self.ALTITUDE_MIN < alt < self.ALTITUDE_MAX:
            raise RuntimeError(f"altitude ({alt}) out of bounds [{self.ALTITUDE_MIN}, {self.ALTITUDE_MAX}]")
        
        return f16_model.controlled_f16_wrapper(state, self._current_u_ref)[0]
    
    def _check_terminated(self) -> bool:
        """Check termination conditions"""
        alpha = self.state[StateIndex.ALPHA]
        if not self.ALPHA_MIN < alpha < self.ALPHA_MAX:
            return True
        
        vel = self.state[StateIndex.VEL]
        if not self.VELOCITY_MIN <= vel <= self.VELOCITY_MAX:
            return True
        
        alt = self.state[StateIndex.ALT]
        if not self.ALTITUDE_MIN < alt < self.ALTITUDE_MAX:
            return True
        
        # Check if waypoint reached
        pos_e = self.state[StateIndex.POSE]
        pos_n = self.state[StateIndex.POSN]
        pos_alt = self.state[StateIndex.ALT]
        
        distance = np.linalg.norm([
            pos_e - self.waypoint[0],
            pos_n - self.waypoint[1],
            pos_alt - self.waypoint[2]
        ])
        
        if distance < self.WAYPOINT_CAPTURE_RADIUS:
            # Waypoint reached, generate new one and continue
            self.waypoint = self._generate_waypoint()
            return False  # Don't terminate, continue with new waypoint
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Build info dictionary"""
        info = {
            'time': self.time,
            'wall_time': time.perf_counter() - self.wall_time_start if self.wall_time_start else 0.0,
            'waypoint': self.waypoint.copy(),
            'waypoint_radius': self.WAYPOINT_CAPTURE_RADIUS,
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
