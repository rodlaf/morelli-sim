"""
F-16 Waypoint Demo

Demonstrates F-16 waypoint following with autopilot agent.
Run with: python -m aerobench.demo
"""

import numpy as np
from numpy import deg2rad

from aerobench.f16_waypoint import F16Waypoint
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot
from aerobench.visualize import anim3d


class AutopilotAgent:
    """Wrapper for autopilot to work with environment"""
    
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


def run_simulation(env: F16Waypoint, agent: AutopilotAgent, 
                   tmax: float, print_mode_changes: bool = True):
    """
    Run F-16 simulation with environment and agent
    
    Args:
        env: F16Waypoint environment
        agent: AutopilotAgent instance
        tmax: Maximum simulation time
        print_mode_changes: Whether to print mode transitions
        
    Returns:
        Dictionary with simulation results
    """
    state, info = env.reset()
    
    modes = [agent.mode]
    last_mode = modes[0]
    
    while env.time < tmax:
        u_ref = agent.get_action(state, env.time)
        state, reward, terminated, truncated, info = env.step(u_ref)
        
        modes.append(agent.mode)
        
        if print_mode_changes and agent.mode != last_mode:
            print(f"Mode transition {last_mode} -> {agent.mode} at time {env.time}")
        last_mode = agent.mode
        
        if terminated or truncated:
            break
        
        if agent.is_done(state, env.time):
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


def main():
    """Main demo function"""
    
    # Initial conditions
    power = 9
    alpha = deg2rad(2.1215)
    beta = 0
    alt = 1500
    vt = 540
    phi = 0
    theta = 0
    psi = 0
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    
    # Create environment (waypoints are fixed in F16Waypoint.WAYPOINTS)
    env = F16Waypoint(
        initial_state=np.array(init, dtype=float),
        step_size=1/30,
        time_limit=150.0,
        extended_states=True
    )
    
    # Create autopilot agent
    autopilot = WaypointAutopilot(F16Waypoint.WAYPOINTS, stdout=True)
    agent = AutopilotAgent(autopilot)
    
    # Run simulation
    print("Starting F-16 waypoint simulation...")
    result = run_simulation(env, agent, tmax=150.0, print_mode_changes=True)
    
    print(f"\nSimulation completed in {result['runtime']:.2f} seconds")
    print(f"Total frames: {len(result['times'])}")
    print(f"Time range: {result['times'][0]:.2f}s to {result['times'][-1]:.2f}s")
    print(f"Final altitude: {result['states'][-1][12]:.1f} ft")
    print(f"Status: {result['status']}")
    
    # Visualize with Raylib
    print("\nStarting 3D visualization...")
    anim3d.make_anim(result, '', F16Waypoint.WAYPOINTS)


if __name__ == '__main__':
    main()
