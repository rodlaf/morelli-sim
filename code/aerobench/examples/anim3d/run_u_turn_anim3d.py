'''
Stanley Bak

plots 3d animation for 'u_turn' scenario 
'''

import sys

import numpy as np
from numpy import deg2rad

from aerobench.envs.f16_env import F16Env
from aerobench.envs.agents import AutopilotAgent
from aerobench.envs.runner import run_f16_env
from aerobench.visualize import anim3d
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot

def simulate(filename):
    'simulate the system using environment/agent architecture'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 1500        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = 0           # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 150 # simulation time

    # make waypoint list
    waypoints = [[-5000, -7500, alt],
                 [-15000, -7500, alt-500],
                 [-15000, 6000, alt+2000]]

    # Create environment
    step = 1/30
    extended_states = True
    env = F16Env(
        initial_state=np.array(init, dtype=float),
        step_size=step,
        integrator='euler',
        time_limit=tmax,
        extended_states=extended_states
    )
    
    # Create agent from autopilot
    autopilot = WaypointAutopilot(waypoints, stdout=True)
    agent = AutopilotAgent(autopilot)
    
    # Run simulation
    res = run_f16_env(
        env=env,
        agent=agent,
        tmax=tmax,
        print_mode_changes=True
    )

    print(f"Waypoint simulation completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")

    return res, waypoints

def main():
    'main function'

    if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
        filename = sys.argv[1]
        print(f"saving result to '{filename}'")
    else:
        filename = ''
        print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")

    res, waypoints = simulate(filename)
        
    anim3d.make_anim(res, filename, waypoints)

if __name__ == '__main__':
    main()
