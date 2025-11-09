"""
F-16 Waypoint Environment

Single-file environment for F-16 waypoint following task.
Combines environment, agent, and simulation in Gymnasium-style API.
"""

import time
import numpy as np
from typing import Tuple, Dict, Any, Optional

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.util import get_state_names, Euler, StateIndex


class F16Waypoint:
    """
    F-16 waypoint following environment
    
    Fixed waypoints for U-turn maneuver demonstration.
    """
    
    # Waypoints for U-turn scenario (east, north, altitude)
    WAYPOINTS = [
        [-5000, -7500, 1500],
        [-15000, -7500, 1000],
        [-15000, 6000, 3500]
    ]
    
    def __init__(self, 
                 initial_state: np.ndarray,
                 step_size: float = 1/30,
                 time_limit: float = 150.0,
                 extended_states: bool = True):
        """
        Initialize F-16 waypoint environment
        
        Args:
            initial_state: Initial state vector (13+ elements)
            step_size: Simulation time step in seconds
            time_limit: Maximum simulation time
            extended_states: Whether to compute extended states
        """
        self.step_size = step_size
        self.time_limit = time_limit
        self.extended_states = extended_states
        
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
        
    def reset(self, initial_time: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to initial state"""
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
            xd, u, Nz, ps, Ny_r = controlled_f16(self.state, u_ref)
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
        
        return controlled_f16(state, self._current_u_ref)[0]
    
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
