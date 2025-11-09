# cython: language_level=3
# Cython wrapper for raylib_renderer

cdef extern from "raylib_renderer.h":
    cdef int MAX_WAYPOINTS
    
    ctypedef struct RenderState:
        float time_sec
        float speed_fps
        float alpha_rad
        float beta_rad
        float phi_rad
        float theta_rad
        float psi_rad
        float pos_e
        float pos_n
        float altitude
        float nz_g
        float ps_rad_s
        int num_waypoints
        float waypoints[10][3]  # MAX_WAYPOINTS
    
    void raylib_renderer_render(RenderState* state)
    void raylib_renderer_close()
    void raylib_renderer_reset()
    bint raylib_renderer_should_close()

def render(state_dict):
    """Render a single frame of the simulation.
    
    Args:
        state_dict: Dict or object with attributes:
            time_sec, speed_fps, alpha_rad, beta_rad, phi_rad, theta_rad, psi_rad,
            position_ft (tuple of 3 floats: east, north, altitude),
            nz_g, ps_rad_s, waypoints (list of 3-tuples)
    """
    cdef RenderState state
    cdef int i
    
    # Extract from dict or object
    if isinstance(state_dict, dict):
        state.time_sec = state_dict['time_sec']
        state.speed_fps = state_dict['speed_fps']
        state.alpha_rad = state_dict['alpha_rad']
        state.beta_rad = state_dict['beta_rad']
        state.phi_rad = state_dict['phi_rad']
        state.theta_rad = state_dict['theta_rad']
        state.psi_rad = state_dict['psi_rad']
        state.pos_e = state_dict['position_ft'][0]
        state.pos_n = state_dict['position_ft'][1]
        state.altitude = state_dict['position_ft'][2]
        state.nz_g = state_dict['nz_g']
        state.ps_rad_s = state_dict['ps_rad_s']
        
        # Handle waypoints
        waypoints_list = state_dict.get('waypoints', [])
        state.num_waypoints = min(len(waypoints_list), 10)  # MAX_WAYPOINTS
        for i in range(state.num_waypoints):
            state.waypoints[i][0] = waypoints_list[i][0]  # east
            state.waypoints[i][1] = waypoints_list[i][1]  # north
            state.waypoints[i][2] = waypoints_list[i][2]  # altitude
    else:
        state.time_sec = state_dict.time_sec
        state.speed_fps = state_dict.speed_fps
        state.alpha_rad = state_dict.alpha_rad
        state.beta_rad = state_dict.beta_rad
        state.phi_rad = state_dict.phi_rad
        state.theta_rad = state_dict.theta_rad
        state.psi_rad = state_dict.psi_rad
        state.pos_e = state_dict.position_ft[0]
        state.pos_n = state_dict.position_ft[1]
        state.altitude = state_dict.position_ft[2]
        state.nz_g = state_dict.nz_g
        state.ps_rad_s = state_dict.ps_rad_s
        
        # Handle waypoints
        waypoints_list = getattr(state_dict, 'waypoints', [])
        state.num_waypoints = min(len(waypoints_list), 10)  # MAX_WAYPOINTS
        for i in range(state.num_waypoints):
            state.waypoints[i][0] = waypoints_list[i][0]  # east
            state.waypoints[i][1] = waypoints_list[i][1]  # north
            state.waypoints[i][2] = waypoints_list[i][2]  # altitude
    
    raylib_renderer_render(&state)

def close():
    """Close the renderer window."""
    raylib_renderer_close()

def reset():
    """Reset trail and markers (for animation loops)."""
    raylib_renderer_reset()

def window_should_close():
    """Check if window should close."""
    return raylib_renderer_should_close()
