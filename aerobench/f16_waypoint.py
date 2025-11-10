"""
F16 Waypoint Environment - Python wrapper using C implementation

All simulation logic is in f16_waypoint.h (calls f16_model.h and raylib_renderer.h).
This is just a thin Gymnasium-compatible Python interface.
"""

import numpy as np
from typing import Tuple, Dict, Any
from aerobench import f16_waypoint_cy


class StateIndex:
    """State variable indices"""
    VT = VEL = 0
    ALPHA = 1
    BETA = 2
    PHI = 3
    THETA = 4
    PSI = 5
    P = 6
    Q = 7
    R = 8
    POSN = POS_N = 9
    POSE = POS_E = 10
    ALT = H = 11
    POW = 12


class F16Waypoint:
    """F16 Waypoint environment - thin wrapper over C implementation"""
    
    def __init__(self, step_size=1/30, time_limit=100.0, extended_states=True, random_seed=None):
        """Initialize environment"""
        self.step_size = step_size
        self.time_limit = time_limit
        self.extended_states = extended_states
        
        # Create C environment via Cython
        self._c_env = f16_waypoint_cy.F16WaypointEnv(
            step_size=step_size,
            time_limit=time_limit,
            random_seed=random_seed or 0
        )
    
    def reset(self, keep_position=False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment"""
        return self._c_env.reset(keep_position=keep_position)
    
    def step(self, u_ref: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment"""
        return self._c_env.step(u_ref)
    
    def render(self):
        """Render"""
        self._c_env.render()
    
    def should_close_window(self) -> bool:
        """Check if window should close"""
        return self._c_env.should_close_window()
    
    def close_window(self):
        """Close window"""
        self._c_env.close_window()
    
    def clear_trail(self):
        """Clear trail"""
        self._c_env.clear_trail()
    
    @property
    def state(self):
        """Get current state"""
        return self._c_env.state
    
    @property
    def waypoint(self):
        """Get current waypoint"""
        return self._c_env.waypoint
    
    @property
    def time(self):
        """Get current time"""
        return self._c_env.time


if __name__ == '__main__':
    """Speed test for F16 Waypoint environment"""
    import time
    
    print("F16 Waypoint Speed Test")
    print("=" * 50)
    
    # Create environment
    env = F16Waypoint(step_size=1/30, time_limit=100.0, random_seed=42)
    state, info = env.reset()
    
    # Pre-generate random actions for consistent testing
    CACHE = 1024
    actions = np.random.uniform(-1, 1, (CACHE, 4))
    actions[:, 0] = np.clip(actions[:, 0] * 2 + 1, -1, 4)  # Nz: -1 to 4
    actions[:, 1] = actions[:, 1] * 0.5  # ps: -0.5 to 0.5
    actions[:, 2] = actions[:, 2] * 0.1  # Ny_r: -0.1 to 0.1
    actions[:, 3] = np.clip(actions[:, 3] * 0.5 + 0.5, 0, 1)  # throttle: 0 to 1
    
    steps = 0
    i = 0
    
    print(f"Running for 5 seconds...")
    start = time.time()
    while time.time() - start < 5:
        state, reward, terminated, truncated, info = env.step(actions[i % CACHE])
        steps += 1
        i += 1
        
        if terminated or truncated:
            state, info = env.reset(keep_position=(info.get('termination_reason') == 'success'))
    
    elapsed = time.time() - start
    sps = int(steps / elapsed)
    
    print(f"Steps: {steps}")
    print(f"Time: {elapsed:.2f}s")
    print(f"F16 Waypoint SPS: {sps}")
    print("=" * 50)
