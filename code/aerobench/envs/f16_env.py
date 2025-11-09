"""
Gymnasium-style environment for F-16 simulation

Decouples the autopilot/agent from the simulator
"""

import time
import numpy as np
from typing import Tuple, Dict, Any, Optional

from aerobench.highlevel.controlled_f16 import controlled_f16
from aerobench.util import get_state_names, Euler, StateIndex


class F16Env:
    """
    Gymnasium-style F-16 flight dynamics environment
    
    The environment manages the F-16 simulation, while agents/autopilots
    provide control inputs through the step() function.
    """
    
    def __init__(self, 
                 initial_state: np.ndarray,
                 step_size: float = 1/30,
                 integrator: str = 'euler',
                 time_limit: float = np.inf,
                 extended_states: bool = False):
        """
        Initialize F-16 environment
        
        Args:
            initial_state: Initial state vector (13+ elements)
            step_size: Simulation time step in seconds
            integrator: Integration method ('euler' supported)
            time_limit: Maximum simulation time
            extended_states: Whether to compute extended states (xd, u, Nz, ps, Ny_r)
        """
        self.step_size = step_size
        self.time_limit = time_limit
        self.extended_states = extended_states
        
        # State management
        num_vars = len(get_state_names()) + 3  # num integrators
        if initial_state.size < num_vars:
            # Append integral error states to state vector
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
        
        # Extended state tracking
        if self.extended_states:
            self.xd_list = []
            self.u_list = []
            self.Nz_list = []
            self.ps_list = []
            self.Ny_r_list = []
        
        # Integration setup
        assert integrator == 'euler', "Only euler integrator currently supported"
        self.integrator_class = Euler
        
        # Timing
        self.wall_time_start = None
        self.wall_time_total = 0.0
        
    def reset(self, 
              initial_state: Optional[np.ndarray] = None,
              initial_time: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state
        
        Args:
            initial_state: Optional new initial state (uses default if None)
            initial_time: Initial time value
            
        Returns:
            state: Initial state observation
            info: Dictionary with extended states if enabled
        """
        if initial_state is not None:
            num_vars = len(get_state_names()) + 3
            if initial_state.size < num_vars:
                self.initial_state = np.zeros(num_vars)
                self.initial_state[:initial_state.shape[0]] = initial_state
            else:
                self.initial_state = initial_state.copy()
        
        self.state = self.initial_state.copy()
        self.time = initial_time
        
        # Reset history
        self.times = [self.time]
        self.states = [self.state.copy()]
        
        if self.extended_states:
            self.xd_list = []
            self.u_list = []
            self.Nz_list = []
            self.ps_list = []
            self.Ny_r_list = []
        
        # Initialize integrator
        self.integrator = self.integrator_class(
            self._dynamics,
            self.time,
            self.state,
            np.inf,
            step=self.step_size
        )
        
        self.wall_time_start = time.perf_counter()
        self.wall_time_total = 0.0
        
        # Build info dict
        info = self._get_info()
        
        return self.state.copy(), info
    
    def step(self, 
             u_ref: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one simulation step with the given control input
        
        Args:
            u_ref: Control reference vector [Nz_ref, ps_ref, Ny_r_ref, throttle]
            
        Returns:
            state: New state observation
            reward: Reward (0.0 for base environment, override in subclasses)
            terminated: Whether episode is done due to terminal condition
            truncated: Whether episode is done due to time limit
            info: Dictionary with extended states and status
        """
        # Store u_ref for dynamics function
        self._current_u_ref = u_ref
        
        # Advance integrator
        next_time = self.time + self.step_size
        
        while self.integrator.t < next_time - 1e-7 and self.integrator.status == 'running':
            self.integrator.step()
        
        if self.integrator.status != 'running':
            # Integration failed
            terminated = True
            truncated = False
            info = self._get_info()
            info['status'] = self.integrator.status
            return self.state.copy(), 0.0, terminated, truncated, info
        
        # Update state and time
        self.time = next_time
        dense_output = self.integrator.dense_output()
        self.state = dense_output(self.time)
        
        # Record history
        self.times.append(self.time)
        self.states.append(self.state.copy())
        
        # Compute extended states if needed
        if self.extended_states:
            xd, u, Nz, ps, Ny_r = self._compute_extended_states(u_ref)
            self.xd_list.append(xd)
            self.u_list.append(u)
            self.Nz_list.append(Nz)
            self.ps_list.append(ps)
            self.Ny_r_list.append(Ny_r)
        
        # Check termination conditions
        terminated = self._check_terminated()
        truncated = self.time >= self.time_limit
        
        # Build info dict
        info = self._get_info()
        
        # Compute reward (default 0, override in subclasses)
        reward = 0.0
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def _dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Dynamics function for integration
        
        Uses self._current_u_ref which should be set before integration
        """
        # Validate state bounds
        alpha = state[StateIndex.ALPHA]
        if not -2 < alpha < 2:
            raise RuntimeError(f"alpha ({alpha}) out of bounds")
        
        vel = state[StateIndex.VEL]
        if not 200 <= vel <= 3000:
            raise RuntimeError(f"velocity ({vel}) out of bounds")
        
        alt = state[StateIndex.ALT]
        if not -10000 < alt < 100000:
            raise RuntimeError(f"altitude ({alt}) out of bounds")
        
        # Compute derivative
        xd = controlled_f16(state, self._current_u_ref)[0]
        return xd
    
    def _compute_extended_states(self, u_ref: np.ndarray) -> Tuple:
        """Compute extended states (xd, u, Nz, ps, Ny_r) at current state"""
        xd, u, Nz, ps, Ny_r = controlled_f16(self.state, u_ref)
        return xd, u, Nz, ps, Ny_r
    
    def _check_terminated(self) -> bool:
        """
        Check if episode should terminate due to failure conditions
        
        Override in subclasses for custom termination logic
        """
        # Check for state bounds violations
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
        """Build info dictionary with current state information"""
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
        """
        Get complete history of simulation
        
        Returns:
            Dictionary with times, states, and extended states if enabled
        """
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
    
    @property
    def observation(self) -> np.ndarray:
        """Current state observation"""
        return self.state.copy() if self.state is not None else None
