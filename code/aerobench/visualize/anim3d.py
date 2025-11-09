'''
3d plotting utilities for aerobench using Raylib
'''

import time

import pyray as rl

from aerobench.util import StateIndex
from aerobench.visualize import raylib_renderer
from aerobench.visualize.raylib_renderer import RenderState

# Simulation playback speed multiplier
# 1.0 = real-time (1 simulation second = 1 real second)
# 0.5 = half speed (slow motion)
# 2.0 = double speed (fast forward)
PLAYBACK_SPEED = 3.0


def make_anim(res, filename='', waypoints=None):
    '''
    Make a 3d animation of the F-16 maneuver using Raylib.

    Args:
        res: Simulation result dict (or list of dicts) from run_f16_sim
        filename: Not used (kept for compatibility). Video export not yet implemented.
        waypoints: Ignored - waypoints are configured in raylib_renderer.py as WAYPOINTS constant
    
    Note: All rendering parameters (window size, colors, camera settings, waypoints, etc.) 
    are configured in raylib_renderer.py using uppercase constants.
    '''

    start = time.time()

    # Normalize inputs to lists for multi-trajectory support
    if not isinstance(res, list):
        res = [res]

    # Extract trajectories - use ALL frames (no skipping)
    all_times = []
    all_states = []
    all_modes = []
    all_ps_list = []
    all_Nz_list = []

    for r in res:
        print(f"Simulation returned {len(r['times'])} frames")
        print(f"Time range: {r['times'][0]:.4f}s to {r['times'][-1]:.4f}s")
        
        # Use all frames - no skipping
        all_times.append(r['times'])
        all_states.append(r['states'])
        all_modes.append(r['modes'])
        all_ps_list.append(r['ps_list'])
        all_Nz_list.append(r['Nz_list'])

    # Calculate total frames
    total_frames = sum(len(t) for t in all_times)
    current_frame = 0
    
    # Calculate actual simulation step size from the data
    sim_duration = all_times[0][-1] - all_times[0][0]
    sim_fps = total_frames / sim_duration  # frames per simulation second

    print(f"Starting Raylib animation with {total_frames} frames")
    print(f"Simulation duration: {sim_duration:.2f}s")
    print(f"Simulation FPS: {sim_fps:.1f} frames/sim-second")
    print(f"Playback speed: {PLAYBACK_SPEED}x (1.0 = real-time)")

    # Calculate how many simulation frames to advance per render frame (60fps)
    # PLAYBACK_SPEED = 1.0 means 1 sim second per real second
    # At 60 render fps, we need to advance (sim_fps * PLAYBACK_SPEED / 60) frames per render
    frames_per_render = (sim_fps * PLAYBACK_SPEED) / 60.0
    accumulated_frames = 0.0
    
    print(f"Advancing {frames_per_render:.2f} sim frames per render frame")

    # Initialize the window before checking window_should_close
    raylib_renderer._initialize()
    
    # Main render loop - render at 60fps, advance through sim frames based on playback speed
    while not rl.window_should_close():
        # Determine which trajectory and frame we're in
        traj_frame_count = 0
        trajectory_index = 0
        frame_in_trajectory = 0
        
        for i, times in enumerate(all_times):
            if current_frame < traj_frame_count + len(times):
                trajectory_index = i
                frame_in_trajectory = current_frame - traj_frame_count
                break
            traj_frame_count += len(times)

        # Get current state data
        states = all_states[trajectory_index]
        times = all_times[trajectory_index]
        modes = all_modes[trajectory_index]
        Nz_list = all_Nz_list[trajectory_index]
        ps_list = all_ps_list[trajectory_index]

        # Clamp to valid range
        frame_in_trajectory = min(frame_in_trajectory, len(states) - 1)
        state = states[frame_in_trajectory]
        
        # Build RenderState
        render_state = RenderState(
            time_sec=float(times[frame_in_trajectory]),
            speed_fps=float(state[StateIndex.VT]),
            alpha_rad=float(state[StateIndex.ALPHA]),
            beta_rad=float(state[StateIndex.BETA]),
            phi_rad=float(state[StateIndex.PHI]),
            theta_rad=float(state[StateIndex.THETA]),
            psi_rad=float(state[StateIndex.PSI]),
            position_ft=(
                float(state[StateIndex.POS_E]),
                float(state[StateIndex.POS_N]),
                float(state[StateIndex.ALT])
            ),
            nz_g=float(Nz_list[frame_in_trajectory]),
            ps_rad_s=float(ps_list[frame_in_trajectory]),
            mode=modes[frame_in_trajectory]
        )

        raylib_renderer.render(render_state)
        
        # Advance frame based on playback speed
        # At 1.0x speed with step=1/120, we show 120 frames per real second (2 per render frame at 60fps)
        accumulated_frames += frames_per_render
        frames_to_advance = int(accumulated_frames)
        accumulated_frames -= frames_to_advance
        
        current_frame += frames_to_advance
        
        # Loop back to start when done
        if current_frame >= total_frames:
            current_frame = 0
            raylib_renderer.reset()
            print(f"Animation loop complete, restarting")
        
        # Print progress periodically
        if current_frame % 300 == 0 and current_frame > 0:
            sim_time = times[frame_in_trajectory]
            total_sim_time = all_times[-1][-1]
            print(f"Frame {current_frame}/{total_frames}, Sim time: {sim_time:.2f}s/{total_sim_time:.2f}s")

    raylib_renderer.close()

    elapsed = time.time() - start
    print(f"Visualization completed in {elapsed:.1f} seconds")

    # Note: Video/GIF saving not yet implemented in Raylib version
    if filename and filename != '':
        print(f"Note: Video/GIF export to '{filename}' not yet implemented in Raylib version")
        print("Run the visualization and use screen recording software to capture output")
