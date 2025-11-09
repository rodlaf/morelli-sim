"""
Agent wrapper for converting existing autopilots to work with F16Env

This allows existing autopilots to work with the new Gymnasium-style API
"""

from typing import Optional
import numpy as np


class AutopilotAgent:
    """
    Wrapper that converts existing Autopilot classes to work with F16Env
    
    This adapter allows the old autopilot interface to work with the new
    environment-based architecture.
    """
    
    def __init__(self, autopilot):
        """
        Initialize agent with an autopilot
        
        Args:
            autopilot: An Autopilot instance (e.g., WaypointAutopilot)
        """
        self.autopilot = autopilot
        self.mode = autopilot.mode
    
    def get_action(self, state: np.ndarray, time: float) -> np.ndarray:
        """
        Get control action from autopilot
        
        Args:
            state: Current state vector
            time: Current simulation time
            
        Returns:
            u_ref: Control reference [Nz_ref, ps_ref, Ny_r_ref, throttle]
        """
        # Update autopilot mode if needed
        self.autopilot.advance_discrete_mode(time, state)
        self.mode = self.autopilot.mode
        
        # Get control input
        u_ref = self.autopilot.get_u_ref(time, state)
        
        return np.array(u_ref, dtype=float)
    
    def is_done(self, state: np.ndarray, time: float) -> bool:
        """
        Check if autopilot task is complete
        
        Args:
            state: Current state vector
            time: Current simulation time
            
        Returns:
            True if autopilot has finished its task
        """
        return self.autopilot.is_finished(time, state)
    
    def reset(self):
        """Reset the autopilot to initial state"""
        # Most autopilots track internal state like waypoint_index
        # Reset to initial mode
        if hasattr(self.autopilot, 'waypoint_index'):
            self.autopilot.waypoint_index = 0
        
        if hasattr(self.autopilot, 'done_time'):
            self.autopilot.done_time = 0.0
        
        # Reset mode
        initial_mode = 'Waypoint 1' if hasattr(self.autopilot, 'waypoints') else self.autopilot.mode
        self.autopilot.mode = initial_mode
        self.mode = initial_mode
