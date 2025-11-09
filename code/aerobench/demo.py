"""
F-16 Waypoint Demo

Demonstrates F-16 waypoint following with autopilot agent.
Run with: python -m aerobench.demo
"""

from math import pi, atan2, sqrt, sin, cos, asin

import numpy as np
from numpy import deg2rad

from aerobench.f16_waypoint import F16Waypoint
from aerobench.util import StateIndex


class AutopilotAgent:
    """Waypoint-following autopilot agent for F-16"""
    
    def __init__(self, waypoints, stdout=False):
        """
        Initialize waypoint autopilot
        
        Args:
            waypoints: List of 3-tuples (east, north, altitude)
            stdout: Whether to print mode transitions
        """
        self.waypoints = waypoints
        self.stdout = stdout
        self.waypoint_index = 0
        self.done_time = 0.0
        self.mode = 'Waypoint 1'
        
        # Waypoint config
        self.slant_range_threshold = 250
        
        # Speed control gains
        self.k_vt = 0.25
        self.airspeed = 550
        
        # Altitude tracking gains
        self.k_alt = 0.005
        self.k_h_dot = 0.02
        
        # Heading tracking gains
        self.k_prop_psi = 5
        self.k_der_psi = 0.5
        
        # Roll tracking gains
        self.k_prop_phi = 0.75
        self.k_der_phi = 0.5
        self.max_bank_deg = 65
        
        # Nz limits
        self.max_nz_cmd = 4
        self.min_nz_cmd = -1
    
    def get_action(self, state: np.ndarray, time: float) -> np.ndarray:
        """
        Get control action [Nz, ps, Ny_r, throttle]
        
        Args:
            state: F-16 state vector
            time: Current simulation time
            
        Returns:
            Control vector [Nz, ps, Ny_r, throttle]
        """
        # Advance discrete mode
        self._advance_mode(time, state)
        
        # Compute control based on current mode
        if self.mode != "Done":
            psi_cmd = self._get_waypoint_heading(state)
            phi_cmd = self._get_phi_to_track_heading(state, psi_cmd)
            ps_cmd = self._track_roll_angle(state, phi_cmd)
            nz_cmd = self._track_altitude(state)
            throttle = self._track_airspeed(state)
        else:
            # Waypoint following complete: fly level
            throttle = self._track_airspeed(state)
            ps_cmd = self._track_roll_angle(state, 0)
            nz_cmd = self._track_altitude_wings_level(state)
        
        # Clamp Nz to limits
        nz_cmd = max(self.min_nz_cmd, min(self.max_nz_cmd, nz_cmd))
        
        return np.array([nz_cmd, ps_cmd, 0, throttle], dtype=float)
    
    def is_done(self, state: np.ndarray, time: float) -> bool:
        """Check if all waypoints have been reached and settling time elapsed"""
        return self.waypoint_index >= len(self.waypoints) and self.done_time + 5.0 < time
    
    def _advance_mode(self, time: float, state: np.ndarray):
        """Advance discrete mode based on waypoint proximity"""
        if self.waypoint_index < len(self.waypoints):
            slant_range = self._get_slant_range(state)
            
            if slant_range < self.slant_range_threshold:
                self.waypoint_index += 1
                
                if self.waypoint_index >= len(self.waypoints):
                    self.done_time = time
        
        # Update mode string
        old_mode = self.mode
        if self.waypoint_index >= len(self.waypoints):
            self.mode = 'Done'
        else:
            self.mode = f'Waypoint {self.waypoint_index + 1}'
        
        # Print mode transitions
        if old_mode != self.mode:
            print(f"(AutopilotAgent) Mode transition {old_mode} -> {self.mode} at time {time:.2f}s")
    
    def _get_waypoint_heading(self, state: np.ndarray) -> float:
        """Get heading to current waypoint"""
        waypoint = self.waypoints[self.waypoint_index]
        
        e_pos = state[StateIndex.POSE]
        n_pos = state[StateIndex.POSN]
        
        delta_e = waypoint[0] - e_pos
        delta_n = waypoint[1] - n_pos
        
        heading = self._wrap_to_pi(pi/2 - atan2(delta_n, delta_e))
        return heading
    
    def _get_slant_range(self, state: np.ndarray) -> float:
        """Get slant range to current waypoint"""
        waypoint = self.waypoints[self.waypoint_index]
        
        e_pos = state[StateIndex.POSE]
        n_pos = state[StateIndex.POSN]
        alt = state[StateIndex.ALT]
        
        delta = [waypoint[i] - [e_pos, n_pos, alt][i] for i in range(3)]
        return sqrt(delta[0]**2 + delta[1]**2 + delta[2]**2)
    
    def _get_phi_to_track_heading(self, state: np.ndarray, psi_cmd: float) -> float:
        """PD control on heading angle using phi_cmd as control"""
        psi = self._wrap_to_pi(state[StateIndex.PSI])
        r = state[StateIndex.R]
        
        psi_err = self._wrap_to_pi(psi_cmd - psi)
        phi_cmd = psi_err * self.k_prop_psi - r * self.k_der_psi
        
        # Clamp to max bank angle
        max_bank_rad = np.deg2rad(self.max_bank_deg)
        phi_cmd = max(-max_bank_rad, min(max_bank_rad, phi_cmd))
        
        return phi_cmd
    
    def _track_roll_angle(self, state: np.ndarray, phi_cmd: float) -> float:
        """PD control on roll angle using stability roll rate"""
        phi = state[StateIndex.PHI]
        p = state[StateIndex.P]
        
        ps = (phi_cmd - phi) * self.k_prop_phi - p * self.k_der_phi
        return ps
    
    def _track_airspeed(self, state: np.ndarray) -> float:
        """Proportional control on airspeed using throttle"""
        throttle = self.k_vt * (self.airspeed - state[StateIndex.VT])
        return throttle
    
    def _track_altitude(self, state: np.ndarray) -> float:
        """Get Nz to track altitude, taking turning into account"""
        h_cmd = self.waypoints[self.waypoint_index][2]
        h = state[StateIndex.ALT]
        phi = state[StateIndex.PHI]
        
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
        """PD control for altitude tracking"""
        i = self.waypoint_index if self.waypoint_index < len(self.waypoints) else -1
        h_cmd = self.waypoints[i][2]
        
        vt = state[StateIndex.VT]
        h = state[StateIndex.ALT]
        
        h_error = h_cmd - h
        gamma = self._get_path_angle(state)
        h_dot = vt * sin(gamma)
        
        nz = self.k_alt * h_error - self.k_h_dot * h_dot
        return nz
    
    def _get_nz_for_level_turn(self, state: np.ndarray) -> float:
        """Get Nz to maintain altitude during bank based on trig"""
        phi = state[StateIndex.PHI]
        
        if abs(phi):  # if cos(phi) ~= 0
            nz = 1 / cos(phi) - 1
        else:
            nz = 0
        
        return nz
    
    def _get_path_angle(self, state: np.ndarray) -> float:
        """Calculate path angle gamma"""
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


def run_simulation(env: F16Waypoint, agent: AutopilotAgent):
    """
    Run F-16 simulation with live rendering (PufferLib-style)
    
    Args:
        env: F16Waypoint environment
        agent: AutopilotAgent instance
        
    Runs infinite loop with real-time rendering until window closed.
    """
    state, info = env.reset()
    
    print("Starting live F-16 simulation...")
    print(f"Waypoints: {len(env.waypoints)}")
    for i, wp in enumerate(env.waypoints):
        print(f"  WP{i+1}: E={wp[0]:7.1f} N={wp[1]:7.1f} Alt={wp[2]:7.1f}")
    print("Press ESC or close window to exit\n")
    
    frame_count = 0
    
    # Main loop - similar to PufferLib squared.h example
    env.render()
    
    while not env.should_close_window():
        # Get action from agent
        u_ref = agent.get_action(state, env.time)
        
        # Step environment
        state, reward, terminated, truncated, info = env.step(u_ref)
        
        # Render current state
        env.render()
        
        frame_count += 1
        
        # Reset on termination or completion
        if terminated or truncated or agent.is_done(state, env.time):
            print(f"\nSimulation complete at {env.time:.2f}s after {frame_count} frames")
            print(f"Final altitude: {state[12]:.1f} ft")
            print("Resetting with new waypoints...\n")
            
            # Reset environment (generates new waypoints)
            state, info = env.reset()
            env.reset_rendering()
            
            # Reinitialize agent with new waypoints
            agent.__init__(env.waypoints, stdout=agent.stdout)
            
            print(f"New waypoints: {len(env.waypoints)}")
            for i, wp in enumerate(env.waypoints):
                print(f"  WP{i+1}: E={wp[0]:7.1f} N={wp[1]:7.1f} Alt={wp[2]:7.1f}")
            print()
            
            frame_count = 0
    
    # Cleanup
    env.close_window()
    print("\nSimulation window closed")


def main():
    """Main demo function"""
    
    # Create environment with random waypoint generation
    env = F16Waypoint(
        num_waypoints=3,
        step_size=1/30,
        time_limit=150.0,
        extended_states=True,
        waypoint_radius=15000.0,
        altitude_range=(1000.0, 4000.0),
        random_seed=None  # Set to int for reproducibility
    )
    
    # Create autopilot agent with environment's generated waypoints
    agent = AutopilotAgent(env.waypoints, stdout=True)
    
    # Run live simulation loop
    run_simulation(env, agent)


if __name__ == '__main__':
    main()
