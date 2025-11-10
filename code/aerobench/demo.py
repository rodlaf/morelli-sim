"""
F-16 Waypoint Demo

Demonstrates F-16 waypoint following with autopilot agent.
Run with: python -m aerobench.demo
"""

import time
from math import pi, atan2, sqrt, sin, cos, asin

import numpy as np
from numpy import deg2rad

from aerobench.f16_waypoint import F16Waypoint
from aerobench.util import StateIndex


class AutopilotAgent:
    """Simple waypoint-following autopilot for single waypoint tracking"""
    
    def __init__(self, target_waypoint, target_speed=550.0, max_bank_deg=65.0):
        """
        Initialize autopilot for single waypoint
        
        Args:
            target_waypoint: [east, north, altitude] position
            target_speed: Target airspeed in ft/s
            max_bank_deg: Maximum bank angle in degrees
        """
        self.waypoint = np.array(target_waypoint)
        self.target_speed = target_speed
        self.max_bank_rad = deg2rad(max_bank_deg)
    
    def update_waypoint(self, new_waypoint):
        """Update to new waypoint when captured"""
        self.waypoint = np.array(new_waypoint)
    
    def get_action(self, state: np.ndarray, time: float) -> np.ndarray:
        """
        Compute control action [Nz, ps, Ny_r, throttle]
        
        Uses simple proportional control laws for heading, altitude, and speed tracking.
        """
        # Heading to waypoint
        psi_cmd = self._heading_to_waypoint(state)
        phi_cmd = self._heading_error_to_roll(state, psi_cmd)
        ps_cmd = self._roll_rate_command(state, phi_cmd)
        
        # Altitude tracking
        nz_cmd = self._altitude_nz_command(state, phi_cmd)
        
        # Speed tracking
        throttle = self._throttle_command(state)
        
        # Clamp Nz
        nz_cmd = np.clip(nz_cmd, -1.0, 4.0)
        
        return np.array([nz_cmd, ps_cmd, 0.0, throttle], dtype=float)
    
    def _heading_to_waypoint(self, state: np.ndarray) -> float:
        """Calculate heading angle to waypoint"""
        e_pos, n_pos = state[StateIndex.POSE], state[StateIndex.POSN]
        de, dn = self.waypoint[0] - e_pos, self.waypoint[1] - n_pos
        return self._wrap_pi(pi/2 - atan2(dn, de))
    
    def _heading_error_to_roll(self, state: np.ndarray, psi_cmd: float) -> float:
        """Convert heading error to roll angle command (P control)"""
        psi = self._wrap_pi(state[StateIndex.PSI])
        psi_err = self._wrap_pi(psi_cmd - psi)
        r = state[StateIndex.R]
        
        # PD control: proportional on heading error, derivative on yaw rate
        phi_cmd = 5.0 * psi_err - 0.5 * r
        return np.clip(phi_cmd, -self.max_bank_rad, self.max_bank_rad)
    
    def _roll_rate_command(self, state: np.ndarray, phi_cmd: float) -> float:
        """Convert roll angle error to roll rate command (PD control)"""
        phi, p = state[StateIndex.PHI], state[StateIndex.P]
        return 0.75 * (phi_cmd - phi) - 0.5 * p
    
    def _altitude_nz_command(self, state: np.ndarray, phi_cmd: float) -> float:
        """Calculate Nz command to track altitude"""
        h_cmd = self.waypoint[2]
        h = state[StateIndex.ALT]
        vt = state[StateIndex.VT]
        
        # Altitude error and vertical rate
        h_err = h_cmd - h
        gamma = self._path_angle(state)
        h_dot = vt * sin(gamma)
        
        # PD control on altitude
        nz_alt = 0.005 * h_err - 0.02 * h_dot
        
        # Add compensation for bank angle (1/cos(phi) - 1)
        nz_turn = (1.0 / cos(phi_cmd) - 1.0) if abs(phi_cmd) > 0.01 else 0.0
        
        # Special handling for descent in bank (avoid negative Gs)
        if h_err < 0 and abs(phi_cmd) > deg2rad(15):
            return max(0.0, nz_alt + nz_turn)
        
        return nz_alt + nz_turn
    
    def _throttle_command(self, state: np.ndarray) -> float:
        """Simple proportional throttle control"""
        return 0.25 * (self.target_speed - state[StateIndex.VT])
    
    def _path_angle(self, state: np.ndarray) -> float:
        """Calculate flight path angle gamma"""
        alpha, beta = state[StateIndex.ALPHA], state[StateIndex.BETA]
        phi, theta = state[StateIndex.PHI], state[StateIndex.THETA]
        
        return asin((cos(alpha)*sin(theta) - sin(alpha)*cos(theta)*cos(phi))*cos(beta) - 
                    cos(theta)*sin(phi)*sin(beta))
    
    @staticmethod
    def _wrap_pi(angle: float) -> float:
        """Wrap angle to [-pi, pi]"""
        rv = angle % (2 * pi)
        if rv > pi:
            rv -= 2 * pi
        return rv


def run_simulation(env: F16Waypoint, agent: AutopilotAgent, 
                   playback_speed: float, render_fps: int):
    """
    Run F-16 simulation with live rendering (PufferLib-style)
    
    Args:
        env: F16Waypoint environment
        agent: AutopilotAgent instance
        playback_speed: Speed multiplier (1.0 = real-time, 2.0 = 2x speed, etc.)
        render_fps: Target frames per second for rendering
        
    Runs infinite loop with real-time rendering until window closed.
    """
    state, info = env.reset()
    
    print("Starting live F-16 simulation...")
    print(f"Playback speed: {playback_speed}x")
    print(f"Render FPS: {render_fps}")
    print(f"Simulation step size: {env.step_size:.6f}s")
    print(f"Current waypoint: E={env.waypoint[0]:7.1f} N={env.waypoint[1]:7.1f} Alt={env.waypoint[2]:7.1f}")
    print("Press ESC or close window to exit\n")
    
    frame_count = 0
    sim_time_elapsed = 0.0
    waypoint_count = 1
    
    # Frame timing
    target_frame_time = 1.0 / render_fps
    last_render_time = time.perf_counter()
    
    # Track waypoint for detecting changes
    last_waypoint = env.waypoint.copy()
    
    # Main loop
    env.render()
    
    while not env.should_close_window():
        current_time = time.perf_counter()
        
        # Determine how much simulation time should have elapsed
        real_time_elapsed = current_time - last_render_time
        target_sim_time = sim_time_elapsed + (real_time_elapsed * playback_speed)
        
        # Step simulation forward until we catch up to target time
        steps_taken = 0
        while env.time < target_sim_time:
            # Get action from agent
            u_ref = agent.get_action(state, env.time)
            
            # Step environment
            state, reward, terminated, truncated, info = env.step(u_ref)
            steps_taken += 1
            
            # Check if waypoint changed (waypoint was captured)
            if not np.array_equal(env.waypoint, last_waypoint):
                waypoint_count += 1
                print(f"Waypoint {waypoint_count - 1} reached at t={env.time:.1f}s")
                print(f"New waypoint {waypoint_count}: E={env.waypoint[0]:7.1f} N={env.waypoint[1]:7.1f} Alt={env.waypoint[2]:7.1f}")
                
                # Update agent to track new waypoint
                agent.update_waypoint(env.waypoint)
                last_waypoint = env.waypoint.copy()
            
            # Check for termination
            if terminated or truncated:
                break
        
        # Render current state
        env.render()
        
        # Update timing
        sim_time_elapsed = env.time
        last_render_time = current_time
        frame_count += 1
        
        # Reset on termination or truncation (physics bounds violated)
        if terminated or truncated:
            print(f"\nSimulation terminated at {env.time:.2f}s after {frame_count} frames")
            print(f"Waypoints reached: {waypoint_count - 1}")
            print(f"Final altitude: {state[StateIndex.ALT]:.1f} ft")
            print("Resetting...\n")
            
            # Reset environment (generates new waypoint)
            state, info = env.reset()
            env.reset_rendering()
            
            # Reset agent with new waypoint
            agent = AutopilotAgent(env.waypoint)
            
            print(f"New waypoint 1: E={env.waypoint[0]:7.1f} N={env.waypoint[1]:7.1f} Alt={env.waypoint[2]:7.1f}")
            print()
            
            frame_count = 0
            sim_time_elapsed = 0.0
            waypoint_count = 1
            last_waypoint = env.waypoint.copy()
            last_render_time = time.perf_counter()
        
        # Sleep to maintain target FPS
        frame_duration = time.perf_counter() - current_time
        sleep_time = target_frame_time - frame_duration
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Cleanup
    env.close_window()
    print("\nSimulation window closed")


def main():
    """Main demo function"""
    
    # Create environment with continuous waypoint generation
    env = F16Waypoint(
        step_size=1/120,  # Fine-grained simulation steps
        time_limit=300.0,  # Longer time limit for continuous task
        extended_states=True,
        random_seed=None  # Set to int for reproducibility
    )
    
    # Create simple autopilot agent for single waypoint tracking
    agent = AutopilotAgent(env.waypoint)
    
    # Run live simulation loop
    run_simulation(
        env, 
        agent,
        playback_speed=3.0,  # 1.0 = real-time, 2.0 = 2x speed, 0.5 = slow-mo
        render_fps=60        # Render at 60 FPS regardless of step_size
    )


if __name__ == '__main__':
    main()
