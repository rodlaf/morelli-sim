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
        const char* mode
    
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
            nz_g, ps_rad_s, mode
    """
    cdef RenderState state
    
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
        mode_str = state_dict.get('mode', '').encode('utf-8')
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
        mode_str = getattr(state_dict, 'mode', '').encode('utf-8')
    
    state.mode = mode_str
    
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
