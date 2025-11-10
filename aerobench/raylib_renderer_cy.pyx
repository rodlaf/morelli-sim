# cython: language_level=3
# Cython wrapper for raylib_renderer

cdef extern from "raylib_renderer.h":
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
        float waypoint_e
        float waypoint_n
        float waypoint_alt
        float waypoint_radius
        float world_bounds_e_min
        float world_bounds_e_max
        float world_bounds_n_min
        float world_bounds_n_max
        float world_bounds_alt_max
    
    void raylib_renderer_render(RenderState* state)
    void raylib_renderer_close()
    void raylib_renderer_clear_trail()
    bint raylib_renderer_should_close()

def render(state_dict):
    """Render a single frame of the simulation."""
    cdef RenderState state
    
    # Extract from dict
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
    
    # Single waypoint
    waypoint = state_dict['waypoint']
    state.waypoint_e = waypoint[0]
    state.waypoint_n = waypoint[1]
    state.waypoint_alt = waypoint[2]
    state.waypoint_radius = state_dict.get('waypoint_radius', 500.0)
    
    # World bounds
    state.world_bounds_e_min = state_dict['world_bounds_e_min']
    state.world_bounds_e_max = state_dict['world_bounds_e_max']
    state.world_bounds_n_min = state_dict['world_bounds_n_min']
    state.world_bounds_n_max = state_dict['world_bounds_n_max']
    state.world_bounds_alt_max = state_dict['world_bounds_alt_max']
    
    raylib_renderer_render(&state)

def close():
    """Close the renderer window."""
    raylib_renderer_close()

def clear_trail():
    """Clear trail/ribbon only (for episode boundaries without full reset)."""
    raylib_renderer_clear_trail()

def window_should_close():
    """Check if window should close."""
    return raylib_renderer_should_close()
