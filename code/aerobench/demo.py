"""
F-16 Waypoint Demo

Demonstrates F-16 waypoint following with autopilot agent.
Run with: python -m aerobench.demo
"""

import time
from math import pi, atan2, sqrt, sin, cos, asin

import numpy as np
from numpy import deg2rad, rad2deg

from aerobench.f16_waypoint import F16Waypoint
from aerobench.util import StateIndex


class AutopilotAgent:
    """Waypoint-following autopilot - ported from original MATLAB version"""
    
    def __init__(self, target_waypoint):
        """
        Initialize autopilot for single waypoint
        
        Args:
            target_waypoint: [east, north, altitude] position
        """
        self.waypoint = np.array(target_waypoint)
        
        # Waypoint config
        self.cfg_slant_range_threshold = 250
        
        # Gains for speed control
        self.cfg_k_vt = 0.25
        self.cfg_airspeed = 550
        
        # Gains for altitude tracking
        self.cfg_k_alt = 0.005
        self.cfg_k_h_dot = 0.02
        
        # Gains for heading tracking
        self.cfg_k_prop_psi = 5
        self.cfg_k_der_psi = 0.5
        
        # Gains for roll tracking
        self.cfg_k_prop_phi = 0.75
        self.cfg_k_der_phi = 0.5
        self.cfg_max_bank_deg = 65
        
        # Ranges for Nz
        self.cfg_max_nz_cmd = 4
        self.cfg_min_nz_cmd = -1
    
    def update_waypoint(self, new_waypoint):
        """Update to new waypoint when captured"""
        print(f'(AutopilotAgent) UPDATING WAYPOINT')
        self.waypoint = np.array(new_waypoint)
    
    def get_action(self, state: np.ndarray, time: float) -> np.ndarray:
        """Get control action [Nz, ps, Ny_r, throttle]"""
        
        # Get desired heading to waypoint
        psi_cmd = self._get_waypoint_heading(state)
        
        # Get desired roll angle given desired heading
        phi_cmd = self._get_phi_to_track_heading(state, psi_cmd)
        ps_cmd = self._track_roll_angle(state, phi_cmd)
        
        nz_cmd = self._track_altitude(state)
        throttle = self._track_airspeed(state)
        
        # Trim to limits
        nz_cmd = max(self.cfg_min_nz_cmd, min(self.cfg_max_nz_cmd, nz_cmd))
        
        return np.array([nz_cmd, ps_cmd, 0, throttle], dtype=float)
    
    def _get_waypoint_heading(self, state: np.ndarray) -> float:
        """Get heading to waypoint"""
        e_pos = state[StateIndex.POSE]
        n_pos = state[StateIndex.POSN]
        
        delta_e = self.waypoint[0] - e_pos
        delta_n = self.waypoint[1] - n_pos
        
        heading = self._wrap_to_pi(pi/2 - atan2(delta_n, delta_e))
        
        return heading
    
    def _get_phi_to_track_heading(self, state: np.ndarray, psi_cmd: float) -> float:
        """PD Control on heading angle using phi_cmd as control"""
        
        psi = self._wrap_to_pi(state[StateIndex.PSI])
        r = state[StateIndex.R]
        
        # Calculate PD control
        psi_err = self._wrap_to_pi(psi_cmd - psi)
        phi_cmd = psi_err * self.cfg_k_prop_psi - r * self.cfg_k_der_psi
        
        # Bound to acceptable bank angles
        max_bank_rad = np.deg2rad(self.cfg_max_bank_deg)
        phi_cmd = min(max(phi_cmd, -max_bank_rad), max_bank_rad)
        
        return phi_cmd
    
    def _track_roll_angle(self, state: np.ndarray, phi_cmd: float) -> float:
        """PD control on roll angle using stability roll rate"""
        
        phi = state[StateIndex.PHI]
        p = state[StateIndex.P]
        
        # Calculate PD control
        ps = (phi_cmd - phi) * self.cfg_k_prop_phi - p * self.cfg_k_der_phi
        
        return ps
    
    def _track_airspeed(self, state: np.ndarray) -> float:
        """Proportional control on airspeed using throttle"""
        
        vt_cmd = self.cfg_airspeed
        throttle = self.cfg_k_vt * (vt_cmd - state[StateIndex.VT])
        
        return throttle
    
    def _track_altitude(self, state: np.ndarray) -> float:
        """Get nz to track altitude, taking turning into account"""
        
        h_cmd = self.waypoint[2]
        h = state[StateIndex.ALT]
        phi = state[StateIndex.PHI]
        
        # Calculate altitude error (positive => below target alt)
        h_error = h_cmd - h
        nz_alt = self._track_altitude_wings_level(state)
        nz_roll = self._get_nz_for_level_turn(state)
        
        if h_error > 0:
            # Ascend wings level or banked
            nz = nz_alt + nz_roll
        elif abs(phi) < np.deg2rad(15):
            # Descend wings (close enough to) level
            nz = nz_alt + nz_roll
        else:
            # Descend in bank (no negative Gs)
            nz = max(0, nz_alt + nz_roll)
        
        return nz
    
    def _track_altitude_wings_level(self, state: np.ndarray) -> float:
        """Get nz to track altitude"""
        
        h_cmd = self.waypoint[2]
        vt = state[StateIndex.VT]
        h = state[StateIndex.ALT]
        
        # Proportional-Derivative Control
        h_error = h_cmd - h
        gamma = self._get_path_angle(state)
        h_dot = vt * sin(gamma)  # Calculated, not differentiated
        
        # Calculate Nz command
        nz = self.cfg_k_alt * h_error - self.cfg_k_h_dot * h_dot
        
        return nz
    
    def _get_nz_for_level_turn(self, state: np.ndarray) -> float:
        """Get nz to do a level turn - pull g's to maintain altitude during bank"""
        
        phi = state[StateIndex.PHI]
        
        if abs(phi):  # if cos(phi) ~= 0
            nz = 1 / cos(phi) - 1  # Keeps plane at altitude
        else:
            nz = 0
        
        return nz
    
    def _get_path_angle(self, state: np.ndarray) -> float:
        """Get the path angle gamma"""
        
        alpha = state[StateIndex.ALPHA]
        beta = state[StateIndex.BETA]
        phi = state[StateIndex.PHI]
        theta = state[StateIndex.THETA]
        
        gamma = asin((cos(alpha)*sin(theta) - 
                      sin(alpha)*cos(theta)*cos(phi))*cos(beta) - 
                     (cos(theta)*sin(phi))*sin(beta))
        
        return gamma
    
    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """Wrap angle to [-pi, pi]"""
        rv = angle % (2 * pi)
        if rv > pi:
            rv -= 2 * pi
        return rv


def run_simulation(env: F16Waypoint, agent: AutopilotAgent, 
                   playback_speed: float, render_fps: int):
    """
    Run F-16 simulation with live rendering
    
    Args:
        env: F16Waypoint environment (already reset)
        agent: AutopilotAgent instance
        playback_speed: Speed multiplier (1.0 = real-time, 2.0 = 2x speed, etc.)
        render_fps: Target frames per second for rendering
    """
    state = env.state.copy()
    
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
            # Check if waypoint changed (check BEFORE getting action)
            if not np.array_equal(env.waypoint, last_waypoint):
                waypoint_count += 1
                print(f"Waypoint {waypoint_count - 1} reached at t={env.time:.1f}s")
                print(f"New waypoint {waypoint_count}: E={env.waypoint[0]:7.1f} N={env.waypoint[1]:7.1f} Alt={env.waypoint[2]:7.1f}")
                
                # Update agent to track new waypoint
                agent.update_waypoint(env.waypoint)
                last_waypoint = env.waypoint.copy()
            
            # Get action from agent
            u_ref = agent.get_action(state, env.time)
            
            # Step environment (may change waypoint in _check_terminated)
            state, reward, terminated, truncated, info = env.step(u_ref)
            steps_taken += 1
            
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
            
            # Clear trail but keep window/camera where it is
            env.clear_trail()
            
            # Reset environment (generates new waypoint)
            state, info = env.reset()
            
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
        step_size=1/30,  # Fine-grained simulation steps
        time_limit=100.0,  # Longer time limit for continuous task
        extended_states=True,
        random_seed=None  # Set to int for reproducibility
    )
    
    # Reset environment first to get the initial waypoint
    state, info = env.reset()
    
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
