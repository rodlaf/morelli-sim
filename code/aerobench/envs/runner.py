"""
Runner for executing F-16 simulations with the Gymnasium-style API

Provides both the new environment-based interface and backward compatibility
with the old run_f16_sim function.
"""

import numpy as np
from typing import Dict, Any, Optional

from aerobench.envs.f16_env import F16Env
from aerobench.envs.agents import AutopilotAgent


def run_f16_env(env: F16Env,
                agent,
                tmax: float,
                print_mode_changes: bool = False) -> Dict[str, Any]:
    """
    Run F-16 simulation using environment and agent
    
    Args:
        env: F16Env instance
        agent: Agent that provides get_action(state, time) -> u_ref
                Can be an AutopilotAgent wrapper or custom agent
        tmax: Maximum simulation time
        print_mode_changes: Whether to print mode transitions
        
    Returns:
        Dictionary with simulation results matching old run_f16_sim format
    """
    # Reset environment
    state, info = env.reset()
    
    # Track modes if agent has mode attribute
    modes = [getattr(agent, 'mode', 'Unknown')]
    last_mode = modes[0]
    
    # Run simulation
    while env.time < tmax:
        # Get action from agent
        u_ref = agent.get_action(state, env.time)
        
        # Step environment
        state, reward, terminated, truncated, info = env.step(u_ref)
        
        # Track mode changes
        current_mode = getattr(agent, 'mode', 'Unknown')
        modes.append(current_mode)
        
        if print_mode_changes and current_mode != last_mode:
            print(f"Mode transition {last_mode} -> {current_mode} at time {env.time}")
        last_mode = current_mode
        
        # Check termination
        if terminated:
            break
        
        if truncated:
            break
        
        # Check if agent is done
        if hasattr(agent, 'is_done') and agent.is_done(state, env.time):
            break
    
    # Get history and build result dict
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


def run_f16_sim_compat(initial_state, 
                       tmax, 
                       autopilot,
                       step=1/30,
                       extended_states=False,
                       integrator_str='euler',
                       print_errors=True,
                       **kwargs):
    """
    Backward-compatible wrapper for run_f16_sim using new environment API
    
    This function maintains the old interface while using the new
    decoupled environment architecture underneath.
    
    Args:
        initial_state: Initial state vector
        tmax: Maximum simulation time
        autopilot: Autopilot instance
        step: Simulation step size
        extended_states: Whether to compute extended states
        integrator_str: Integration method
        print_errors: Whether to print integration errors
        
    Returns:
        Dictionary with same format as old run_f16_sim
    """
    # Create environment
    env = F16Env(
        initial_state=np.array(initial_state, dtype=float),
        step_size=step,
        integrator=integrator_str,
        time_limit=tmax,
        extended_states=extended_states
    )
    
    # Wrap autopilot as agent
    agent = AutopilotAgent(autopilot)
    
    # Run simulation
    result = run_f16_env(
        env=env,
        agent=agent,
        tmax=tmax,
        print_mode_changes=hasattr(autopilot, 'stdout') and autopilot.stdout
    )
    
    return result
