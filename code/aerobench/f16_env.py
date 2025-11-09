"""
F-16 Simulation Environment (Gymnasium-style API)

Decoupled environment/agent architecture for F-16 flight dynamics.
Combines environment, agent wrapper, and runner utilities in a single module.
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
        
        assert integrator == 'euler', "Only euler integrator currently supported"
        self.integrator_class = Euler
        
        self.wall_time_start = None
        
    def reset(self, 
              initial_state: Optional[np.ndarray] = None,
              initial_time: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to initial state"""
        if initial_state is not None:
            num_vars = len(get_state_names()) + 3
            if initial_state.size < num_vars:
                self.initial_state = np.zeros(num_vars)
                self.initial_state[:initial_state.shape[0]] = initial_state
            else:
                self.initial_state = initial_state.copy()
        
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
        """Execute one simulation step"""
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


class AutopilotAgent:
    """Wrapper for existing Autopilot classes"""
    
    def __init__(self, autopilot):
        self.autopilot = autopilot
        self.mode = autopilot.mode
    
    def get_action(self, state: np.ndarray, time: float) -> np.ndarray:
        """Get control action from autopilot"""
        self.autopilot.advance_discrete_mode(time, state)
        self.mode = self.autopilot.mode
        u_ref = self.autopilot.get_u_ref(time, state)
        return np.array(u_ref, dtype=float)
    
    def is_done(self, state: np.ndarray, time: float) -> bool:
        """Check if autopilot task is complete"""
        return self.autopilot.is_finished(time, state)
    
    def reset(self):
        """Reset autopilot to initial state"""
        if hasattr(self.autopilot, 'waypoint_index'):
            self.autopilot.waypoint_index = 0
        if hasattr(self.autopilot, 'done_time'):
            self.autopilot.done_time = 0.0
        initial_mode = 'Waypoint 1' if hasattr(self.autopilot, 'waypoints') else self.autopilot.mode
        self.autopilot.mode = initial_mode
        self.mode = initial_mode


def run_f16_env(env: F16Env, agent, tmax: float, print_mode_changes: bool = False) -> Dict[str, Any]:
    """
    Run F-16 simulation using environment and agent
    
    Args:
        env: F16Env instance
        agent: Agent with get_action(state, time) -> u_ref
        tmax: Maximum simulation time
        print_mode_changes: Whether to print mode transitions
        
    Returns:
        Dictionary with simulation results
    """
    state, info = env.reset()
    
    modes = [getattr(agent, 'mode', 'Unknown')]
    last_mode = modes[0]
    
    while env.time < tmax:
        u_ref = agent.get_action(state, env.time)
        state, reward, terminated, truncated, info = env.step(u_ref)
        
        current_mode = getattr(agent, 'mode', 'Unknown')
        modes.append(current_mode)
        
        if print_mode_changes and current_mode != last_mode:
            print(f"Mode transition {last_mode} -> {current_mode} at time {env.time}")
        last_mode = current_mode
        
        if terminated or truncated:
            break
        
        if hasattr(agent, 'is_done') and agent.is_done(state, env.time):
            break
    
    history = env.get_history()
    
    result = {
        'times': history['times'],
        'states': history['states'],
        'modes': modes,
        'runtime': history['wall_time'],
        'status': 'finished' if not terminated else 'terminated',
    }
    
    if env.extended_states:
        result['xd_list'] = history['xd_list']
        result['u_list'] = history['u_list']
        result['Nz_list'] = history['Nz_list']
        result['ps_list'] = history['ps_list']
        result['Ny_r_list'] = history['Ny_r_list']
    
    return result
