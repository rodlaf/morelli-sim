import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import numpy as np
from pyray import *
from pyray import RL_PROJECTION, RL_MODELVIEW, MOUSE_BUTTON_LEFT, CAMERA_PERSPECTIVE

RAD2DEG = 180.0 / math.pi

# Rendering constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
TARGET_FPS = 60
CHASE_DISTANCE = 3000.0  # default camera distance from aircraft
CHASE_ELEVATION = 150.0  # camera height offset above aircraft
F16_SCALE = 150.0  # visual size of the F-16 model in feet
GRID_SPACING = 1000.0  # feet between ground grid lines
SKY_COLOR = Color(0, 0, 0, 255)  # Black sky
GRID_COLOR = Color(40, 120, 40, 180)  # Brighter semi-transparent green
TRAIL_COLOR = YELLOW  # Trail ribbon color
ALTITUDE_LINE_SPACING = 300.0  # feet between altitude markers
ALTITUDE_MARKER_COLOR = Color(255, 255, 0, 128)  # Dimmer yellow for altitude markers
RIBBON_WIDTH = 150.0  # Width of the trail ribbon in feet

# Waypoints configuration (must match run_u_turn_anim3d.py)
alt = 1500
WAYPOINTS = [
    [-5000, -7500, alt],
    [-15000, -7500, alt - 500],
    [-15000, 6000, alt + 2000]
]

# Module-level state
_camera = None
_near_plane = 0.1
_far_plane = 50000.0
_altitude_markers = []
_last_marker_pos = None
_font = None
_manual_camera_offset = [0.0, 0.0]  # azimuth, elevation
_manual_zoom = 0.0
_last_mouse_pos = None
_initialized = False


@dataclass
class RenderState:
    """Container for the simulation state needed by the renderer."""
    time_sec: float
    speed_fps: float
    alpha_rad: float
    beta_rad: float
    phi_rad: float
    theta_rad: float
    psi_rad: float
    position_ft: Tuple[float, float, float]  # (east, north, altitude)
    nz_g: float
    ps_rad_s: float
    mode: str


def _initialize():
    """Initialize the renderer (called automatically on first render)."""
    global _camera, _font, _initialized
    
    init_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Aerobench 3D")
    set_target_fps(TARGET_FPS)
    
    _camera = Camera3D(
        [0.0, CHASE_ELEVATION, -CHASE_DISTANCE],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        60.0,
        CAMERA_PERSPECTIVE,
    )
    
    _font = get_font_default()
    _initialized = True


def close():
    """Close the renderer window."""
    global _initialized
    if _initialized:
        close_window()
        _initialized = False


def reset():
    """Reset trail and markers (for animation loops)."""
    global _altitude_markers, _last_marker_pos
    _altitude_markers = []
    _last_marker_pos = None


def render(state: RenderState) -> None:
    """Render a single frame of the simulation."""
    global _altitude_markers, _last_marker_pos, _camera, _initialized
    
    if not _initialized:
        _initialize()
    
    _handle_camera_input()
    
    pos_e, pos_n, altitude = state.position_ft
    position = [float(pos_e), float(altitude), float(pos_n)]
    
    # Add altitude marker and trail point together at same spacing
    if _last_marker_pos is None:
        marker_data = {
            'pos': tuple(position),
            'roll': state.phi_rad,
            'pitch': state.theta_rad,
            'yaw': state.psi_rad
        }
        _altitude_markers.append(marker_data)
        _last_marker_pos = np.array(position)
    else:
        dist_from_last_marker = np.linalg.norm(np.array(position) - _last_marker_pos)
        if dist_from_last_marker >= ALTITUDE_LINE_SPACING:
            marker_data = {
                'pos': tuple(position),
                'roll': state.phi_rad,
                'pitch': state.theta_rad,
                'yaw': state.psi_rad
            }
            _altitude_markers.append(marker_data)
            _last_marker_pos = np.array(position)
    
    _update_camera(position)

    begin_drawing()
    clear_background(SKY_COLOR)
    begin_mode_3d(_camera)
    
    # Override projection for extended far clipping plane
    aspect = get_screen_width() / get_screen_height()
    top = _near_plane * math.tan(_camera.fovy * 0.5 * math.pi / 180.0)
    rl_matrix_mode(RL_PROJECTION)
    rl_load_identity()
    rl_frustum(-top * aspect, top * aspect, -top, top, _near_plane, _far_plane)
    rl_matrix_mode(RL_MODELVIEW)
    
    # Draw ground grid
    grid_extent = 50000.0
    for x in range(int((position[0] - grid_extent) // GRID_SPACING) * int(GRID_SPACING), 
                   int((position[0] + grid_extent) // GRID_SPACING) * int(GRID_SPACING) + 1, 
                   int(GRID_SPACING)):
        draw_line_3d([float(x), 0.0, position[2] - grid_extent],
                    [float(x), 0.0, position[2] + grid_extent], GRID_COLOR)
    
    for z in range(int((position[2] - grid_extent) // GRID_SPACING) * int(GRID_SPACING),
                   int((position[2] + grid_extent) // GRID_SPACING) * int(GRID_SPACING) + 1,
                   int(GRID_SPACING)):
        draw_line_3d([position[0] - grid_extent, 0.0, float(z)],
                    [position[0] + grid_extent, 0.0, float(z)], GRID_COLOR)
    
    # Draw altitude markers with horizontal orientation bars (T-shaped)
    hw = RIBBON_WIDTH * 0.5
    for i, m in enumerate(_altitude_markers):
        px, py, pz = m['pos']
        # Vertical line (ground to position)
        draw_line_3d([px, 0.0, pz], [px, py, pz], ALTITUDE_MARKER_COLOR)
        # Horizontal line (showing roll orientation) - precompute right vector
        sp, cp, st, ct, sy, cy = math.sin(m['roll']), math.cos(m['roll']), math.sin(m['pitch']), math.cos(m['pitch']), math.sin(m['yaw']), math.cos(m['yaw'])
        rx, ry, rz = sp * sy * st + cp * cy, -sp * ct, sp * cy * st - cp * sy
        norm = math.sqrt(rx*rx + ry*ry + rz*rz) + 1e-10
        rx, ry, rz = rx/norm * hw, ry/norm * hw, rz/norm * hw
        left_pt = [px - rx, py - ry, pz - rz]
        right_pt = [px + rx, py + ry, pz + rz]
        draw_line_3d(left_pt, right_pt, TRAIL_COLOR)
        # Center line and edge lines to next marker
        if i < len(_altitude_markers) - 1:
            next_m = _altitude_markers[i + 1]
            nx, ny, nz = next_m['pos']
            nsp, ncp, nst, nct, nsy, ncy = math.sin(next_m['roll']), math.cos(next_m['roll']), math.sin(next_m['pitch']), math.cos(next_m['pitch']), math.sin(next_m['yaw']), math.cos(next_m['yaw'])
            nrx, nry, nrz = nsp * nsy * nst + ncp * ncy, -nsp * nct, nsp * ncy * nst - ncp * nsy
            nnorm = math.sqrt(nrx*nrx + nry*nry + nrz*nrz) + 1e-10
            nrx, nry, nrz = nrx/nnorm * hw, nry/nnorm * hw, nrz/nnorm * hw
            next_left = [nx - nrx, ny - nry, nz - nrz]
            next_right = [nx + nrx, ny + nry, nz + nrz]
            draw_line_3d([px, py, pz], [nx, ny, nz], TRAIL_COLOR)  # center
            draw_line_3d(left_pt, next_left, TRAIL_COLOR)  # left edge
            draw_line_3d(right_pt, next_right, TRAIL_COLOR)  # right edge
    
    # Current position marker
    draw_line_3d([position[0], 0.0, position[2]], position, ALTITUDE_MARKER_COLOR)

    # Draw waypoints
    for wp in WAYPOINTS:
        wp_pos = [float(wp[0]), float(wp[2]), float(wp[1])]  # [east, altitude, north]
        draw_sphere(wp_pos, 50.0, PURPLE)
        draw_line_3d([wp_pos[0], 0.0, wp_pos[2]], wp_pos, PURPLE)

    # Draw plane
    _draw_simple_plane(position, state.phi_rad, state.theta_rad, state.psi_rad, F16_SCALE)

    end_mode_3d()

    _draw_hud(state)
    end_drawing()




def _draw_simple_plane(position: list, roll: float, pitch: float, yaw: float, size: float) -> None:
    """Draw F-16 using navigation equations."""
    px, py, pz = position
    sp, cp, st, ct, sy, cy = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch), math.sin(yaw), math.cos(yaw)
    
    # F16 navigation: body -> world (north, east, up) -> viz (East, Up, North)
    nx, ny, nz = ct * sy, st, ct * cy
    rx, ry, rz = sp * sy * st + cp * cy, -sp * ct, sp * cy * st - cp * sy
    ux, uy, uz = ny * rz - nz * ry, nz * rx - nx * rz, nx * ry - ny * rx  # cross product
    ul = math.sqrt(ux*ux + uy*uy + uz*uz)
    ux, uy, uz = ux/ul, uy/ul, uz/ul
    
    # Scaled components
    nose = [px + nx*size, py + ny*size, pz + nz*size]
    tail = [px - nx*size*0.4, py - ny*size*0.4, pz - nz*size*0.4]
    lwing = [px - rx*size*0.6, py - ry*size*0.6, pz - rz*size*0.6]
    rwing = [px + rx*size*0.6, py + ry*size*0.6, pz + rz*size*0.6]
    vtail = [tail[0] + ux*size*0.3, tail[1] + uy*size*0.3, tail[2] + uz*size*0.3]
    
    draw_line_3d(position, nose, RED)
    draw_sphere(nose, size * 0.08, RED)
    draw_line_3d(lwing, rwing, GREEN)
    draw_sphere(lwing, size * 0.08, GREEN)
    draw_sphere(rwing, size * 0.08, GREEN)
    draw_line_3d(position, tail, BLUE)
    draw_sphere(tail, size * 0.08, BLUE)
    draw_line_3d(tail, vtail, SKYBLUE)
    draw_sphere(vtail, size * 0.08, SKYBLUE)
    draw_sphere(position, size * 0.12, LIGHTGRAY)


def _handle_camera_input() -> None:
    """Handle mouse input for camera control."""
    global _manual_zoom, _manual_camera_offset, _last_mouse_pos
    
    wheel = get_mouse_wheel_move()
    if wheel != 0:
        _manual_zoom += wheel * 200.0
        min_zoom = -(20000.0 - CHASE_DISTANCE)
        max_zoom = CHASE_DISTANCE - 50.0
        _manual_zoom = max(min_zoom, min(max_zoom, _manual_zoom))
    
    if is_mouse_button_down(MOUSE_BUTTON_LEFT):
        mouse_pos = get_mouse_position()
        if _last_mouse_pos is not None:
            delta_x, delta_y = mouse_pos.x - _last_mouse_pos.x, mouse_pos.y - _last_mouse_pos.y
            _manual_camera_offset[0] -= delta_x * 0.01
            _manual_camera_offset[1] += delta_y * 0.01
            _manual_camera_offset[1] = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, _manual_camera_offset[1]))
        _last_mouse_pos = mouse_pos
    else:
        _last_mouse_pos = None


def _update_camera(position: list) -> None:
    """Update camera position and target."""
    global _camera
    
    _camera.target = position
    _camera.up = [0.0, 1.0, 0.0]
    
    d = max(50.0, min(20000.0, CHASE_DISTANCE - _manual_zoom))
    az, el = _manual_camera_offset
    ce, se, ca, sa = math.cos(el), math.sin(el), math.cos(az), math.sin(az)
    
    _camera.position = [
        position[0] + d * ce * sa,
        position[1] + d * se,
        position[2] + d * ce * ca
    ]


def _draw_hud(state: RenderState) -> None:
    """Draw heads-up display with telemetry."""
    p, lh = 12, 22
    texts = [
        f"t = {state.time_sec:.2f} s",
        f"h = {state.position_ft[2]:.1f} ft",
        f"V = {state.speed_fps:.1f} ft/s",
        f"alpha = {state.alpha_rad * RAD2DEG:.1f} deg",
        f"beta = {state.beta_rad * RAD2DEG:.1f} deg",
        f"Nz = {state.nz_g:.2f} g",
        f"ps = {state.ps_rad_s * RAD2DEG:.1f} deg/s",
        f"phi/theta/psi = {state.phi_rad * RAD2DEG:.1f}/{state.theta_rad * RAD2DEG:.1f}/{state.psi_rad * RAD2DEG:.1f} deg",
        "Left click + drag: rotate camera",
        "Mouse wheel: zoom in/out"
    ]
    for i, text in enumerate(texts):
        draw_text_ex(_font, text, [p, p + i * lh], 20, 1, RAYWHITE)



