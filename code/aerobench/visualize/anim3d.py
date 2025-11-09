'''
3d plotting utilities for aerobench using Raylib
'''

import time

import pyray as rl

from aerobench.util import StateIndex
from aerobench.visualize.raylib_renderer import RaylibRenderer, RenderState, PLAYBACK_SPEED

def make_anim(res, filename, viewsize=1000, viewsize_z=1000, f16_scale=30, trail_pts=60,
              elev=30, azim=45, skip_frames=None, chase=False, fixed_floor=False,
              init_extra=None, update_extra=None, waypoints=None):
    '''
    make a 3d plot of the F-16 maneuver using Raylib.

    see examples/anim3d folder for examples on usage
    '''

    start = time.time()

    # Normalize inputs to lists for multi-trajectory support
    if not isinstance(res, list):
        res = [res]

    # Apply defaults for skip_frames
    if skip_frames is None:
        skip_frames = [None] * len(res)
    elif not isinstance(skip_frames, list):
        skip_frames = [skip_frames]

    for i in range(len(skip_frames)):
        if skip_frames[i] is None:
            if filename == '':
                skip_frames[i] = 5
            elif filename.endswith('.gif'):
                skip_frames[i] = 2
            else:
                skip_frames[i] = 1

    # Extract and subsample trajectories
    all_times = []
    all_states = []
    all_modes = []
    all_ps_list = []
    all_Nz_list = []

    for r, skip in zip(res, skip_frames):
        t = r['times'][0::skip]
        s = r['states'][0::skip]
        m = r['modes'][0::skip]
        ps = r['ps_list'][0::skip]
        Nz = r['Nz_list'][0::skip]

        all_times.append(t)
        all_states.append(s)
        all_modes.append(m)
        all_ps_list.append(ps)
        all_Nz_list.append(Nz)

    # Create renderer
    renderer = RaylibRenderer(
        width=1280,
        height=720,
        f16_scale=f16_scale if not isinstance(f16_scale, list) else f16_scale[0],
        trail_length=trail_pts if not isinstance(trail_pts, list) else trail_pts[0],
        view_size=viewsize if not isinstance(viewsize, list) else viewsize[0],
        chase_camera=chase if not isinstance(chase, list) else chase[0],
        waypoints=waypoints,
    )

    # Calculate total frames
    total_frames = sum(len(t) for t in all_times)
    current_frame = 0
    trajectory_index = 0
    frame_in_trajectory = 0

    print(f"Starting Raylib animation with {total_frames} frames")

    # Timing for playback speed control
    last_frame_time = time.time()
    accumulated_time = 0.0

    # Main render loop
    while not rl.window_should_close():
        # Calculate delta time for smooth playback
        current_time = time.time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time
        
        # Accumulate time based on playback speed
        accumulated_time += delta_time * PLAYBACK_SPEED
        
        # Determine which trajectory we're in
        traj_frame_count = 0
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

        # Check if we should advance to the next frame
        should_advance = False
        if frame_in_trajectory < len(times) - 1:
            next_frame_time = times[frame_in_trajectory + 1]
            current_frame_time = times[frame_in_trajectory]
            frame_duration = next_frame_time - current_frame_time
            
            if accumulated_time >= frame_duration:
                should_advance = True
                accumulated_time -= frame_duration
        else:
            # At end of trajectory
            if accumulated_time >= 0.033:  # ~30fps fallback
                should_advance = True
                accumulated_time = 0.0
        
        if current_frame >= total_frames:
            # Loop back to start
            current_frame = 0
            trajectory_index = 0
            frame_in_trajectory = 0
            accumulated_time = 0.0
            renderer.trail.clear()
            continue

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

        # Advance frame only when enough time has accumulated
        if should_advance:
            current_frame += 1
        
        # Print progress periodically
        if current_frame % 30 == 0:
            print(f"Frame: {current_frame}/{total_frames}")

    renderer.close()

    elapsed = time.time() - start
    print(f"Visualization completed in {elapsed:.1f} seconds")

    # Note: Video/GIF saving not yet implemented in Raylib version
    if filename and filename != '':
        print(f"Note: Video/GIF export to '{filename}' not yet implemented in Raylib version")
        print("Run the visualization and use screen recording software to capture output")
