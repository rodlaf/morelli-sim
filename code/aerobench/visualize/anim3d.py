'''
3d plotting utilities for aerobench using Raylib
'''

import time

import pyray as rl

from aerobench.util import StateIndex
from aerobench.visualize.raylib_renderer import RaylibRenderer, RenderState

# Simulation playback speed multiplier
# 1.0 = real-time (1 simulation second = 1 real second)
# 0.5 = half speed (slow motion)
# 2.0 = double speed (fast forward)
PLAYBACK_SPEED = 1.0


def make_anim(res, filename, viewsize=1000, viewsize_z=1000, f16_scale=30, trail_pts=60,
              elev=30, azim=45, skip_frames=None, chase=False, fixed_floor=False,
              init_extra=None, update_extra=None, waypoints=None, chase_distance=1250.0):
    '''
    make a 3d plot of the F-16 maneuver using Raylib.

    see examples/anim3d folder for examples on usage
    
    NOTE: skip_frames parameter is ignored - all frames are always rendered for smooth playback.
    Use smaller step size in run_f16_sim() to control frame rate (e.g., step=1/120 for 120fps).
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
        print(f"DEBUG: Simulation returned {len(r['times'])} frames")
        print(f"DEBUG: Time range: {r['times'][0]:.4f}s to {r['times'][-1]:.4f}s")
        
        # Use all frames - no skipping
        all_times.append(r['times'])
        all_states.append(r['states'])
        all_modes.append(r['modes'])
        all_ps_list.append(r['ps_list'])
        all_Nz_list.append(r['Nz_list'])
        
    # Calculate time deltas to see if frames are evenly spaced
    if len(all_times[0]) > 1:
        deltas = [all_times[0][i+1] - all_times[0][i] for i in range(min(10, len(all_times[0])-1))]
        print(f"DEBUG: First 10 time deltas: {[f'{d:.6f}' for d in deltas]}")

    # Create renderer
    renderer = RaylibRenderer(
        width=1280,
        height=720,
        f16_scale=f16_scale if not isinstance(f16_scale, list) else f16_scale[0],
        trail_length=trail_pts if not isinstance(trail_pts, list) else trail_pts[0],
        view_size=viewsize if not isinstance(viewsize, list) else viewsize[0],
        chase_camera=chase if not isinstance(chase, list) else chase[0],
        chase_distance=chase_distance if not isinstance(chase_distance, list) else chase_distance[0],
        waypoints=waypoints,
    )

    # Calculate total frames
    total_frames = sum(len(t) for t in all_times)
    current_frame = 0

    print(f"Starting Raylib animation with {total_frames} frames")
    print(f"DEBUG: Simulation duration: {all_times[0][-1] - all_times[0][0]:.2f}s")
    print(f"DEBUG: Frames per sim second: {total_frames / (all_times[0][-1] - all_times[0][0]):.1f}")
    print(f"DEBUG: PLAYBACK_SPEED: {PLAYBACK_SPEED}x")

    # Simple playback: iterate through frames at controlled speed
    # PLAYBACK_SPEED controls how many simulation frames to advance per render frame
    frames_per_render = PLAYBACK_SPEED
    accumulated_frames = 0.0

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

        renderer.render(render_state)
        
        # Advance frame based on playback speed
        # At 1.0x speed with step=1/120, we show 120 frames per real second (2 per render frame at 60fps)
        accumulated_frames += frames_per_render
        frames_to_advance = int(accumulated_frames)
        accumulated_frames -= frames_to_advance
        
        current_frame += frames_to_advance
        
        # Loop back to start when done
        if current_frame >= total_frames:
            current_frame = 0
            renderer.trail.clear()
            print(f"DEBUG: Animation loop complete, restarting")
        
        # Print progress periodically
        if current_frame % 120 == 0:
            sim_time = times[frame_in_trajectory]
            total_sim_time = all_times[-1][-1]
            print(f"DEBUG: Frame {current_frame}/{total_frames}, Sim time: {sim_time:.2f}s/{total_sim_time:.2f}s")

    renderer.close()

    elapsed = time.time() - start
    print(f"Visualization completed in {elapsed:.1f} seconds")

    # Note: Video/GIF saving not yet implemented in Raylib version
    if filename and filename != '':
        print(f"Note: Video/GIF export to '{filename}' not yet implemented in Raylib version")
        print("Run the visualization and use screen recording software to capture output")
