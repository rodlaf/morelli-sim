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
CHASE_DISTANCE = 2000.0  # default camera distance from aircraft
CHASE_ELEVATION = 150.0  # camera height offset above aircraft
F16_SCALE = 150.0  # visual size of the F-16 model in feet
GRID_SPACING = 1000.0  # feet between ground grid lines
SKY_COLOR = Color(0, 0, 0, 255)  # Black sky
GRID_COLOR = Color(40, 120, 40, 180)  # Brighter semi-transparent green
TRAIL_COLOR = YELLOW  # Trail ribbon color
ALTITUDE_LINE_SPACING = 1500.0  # feet between altitude markers
ALTITUDE_MARKER_COLOR = Color(255, 255, 0, 128)  # Dimmer yellow for altitude markers
RIBBON_WIDTH = 100.0  # Width of the trail ribbon in feet
TRAIL_MIN_DISTANCE = 50.0  # Minimum distance between trail points in feet


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


class RaylibRenderer:
    """Immediate-mode renderer that draws a single frame per call to render()."""

    def __init__(self, waypoints=None) -> None:
        init_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Aerobench 3D")
        set_target_fps(TARGET_FPS)

        self.camera = Camera3D(
            [0.0, CHASE_ELEVATION, -CHASE_DISTANCE],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            60.0,
            CAMERA_PERSPECTIVE,
        )
        
        self.near_plane = 0.1
        self.far_plane = 50000.0
        self.trail: list = []
        self.altitude_markers: list = []
        self.distance_since_last_marker = 0.0
        self.chase_distance = CHASE_DISTANCE
        self.chase_elevation = CHASE_ELEVATION
        self.font = get_font_default()
        self.waypoints = waypoints or []
        self.manual_camera_offset = [0.0, 0.0]  # azimuth, elevation
        self.manual_zoom = 0.0
        self.last_mouse_pos = None

    def close(self) -> None:
        close_window()

    def render(self, state: RenderState) -> None:
        self._handle_camera_input()
        
        pos_e, pos_n, altitude = state.position_ft
        position = [float(pos_e), float(altitude), float(pos_n)]
        
        # Only add trail point if we've moved far enough from the last point
        should_add = len(self.trail) == 0
        if len(self.trail) > 0:
            dist = np.linalg.norm(np.array(position) - np.array(self.trail[-1]['pos']))
            should_add = dist >= TRAIL_MIN_DISTANCE
        
        if should_add:
            self.trail.append({
                'pos': tuple(position),
                'roll': state.phi_rad,
                'pitch': state.theta_rad,
                'yaw': state.psi_rad
            })
        
        # Add altitude marker every ALTITUDE_LINE_SPACING feet
        if len(self.trail) > 1:
            segment_dist = np.linalg.norm(np.array(position) - np.array(self.trail[-2]['pos']))
            self.distance_since_last_marker += segment_dist
            
            if self.distance_since_last_marker >= ALTITUDE_LINE_SPACING:
                self.altitude_markers.append(tuple(position))
                self.distance_since_last_marker = 0.0
        
        self._update_camera(position)

        begin_drawing()
        clear_background(SKY_COLOR)
        begin_mode_3d(self.camera)
        
        # Override projection for extended far clipping plane
        aspect = get_screen_width() / get_screen_height()
        top = self.near_plane * math.tan(self.camera.fovy * 0.5 * math.pi / 180.0)
        rl_matrix_mode(RL_PROJECTION)
        rl_load_identity()
        rl_frustum(-top * aspect, top * aspect, -top, top, self.near_plane, self.far_plane)
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
        
        # Draw altitude markers
        for marker_pos in self.altitude_markers:
            draw_line_3d([marker_pos[0], 0.0, marker_pos[2]], marker_pos, ALTITUDE_MARKER_COLOR)
        draw_line_3d([position[0], 0.0, position[2]], position, ALTITUDE_MARKER_COLOR)

        # Draw waypoints
        for wp in self.waypoints:
            wp_pos = [float(wp[0]), float(wp[2]), float(wp[1])]  # [east, altitude, north]
            draw_sphere(wp_pos, 50.0, PURPLE)
            draw_line_3d([wp_pos[0], 0.0, wp_pos[2]], wp_pos, PURPLE)

        # Draw plane and ribbon trail
        self._draw_simple_plane(position, state.phi_rad, state.theta_rad, state.psi_rad, F16_SCALE)
        self._draw_ribbon_trail()

        end_mode_3d()

        self._draw_hud(state)
        end_drawing()

    def _draw_simple_plane(self, position: list, roll: float, pitch: float, yaw: float, size: float) -> None:
        """Draw F-16 using navigation equations."""
        pos = np.array(position)
        sp, cp, st, ct, sy, cy = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch), math.sin(yaw), math.cos(yaw)
        
        # F16 navigation: body -> world (north, east, up) -> viz (East, Up, North)
        nose_dir = np.array([ct * sy, st, ct * cy])
        right_dir = np.array([sp * sy * st + cp * cy, -sp * ct, sp * cy * st - cp * sy])
        up_dir = np.cross(nose_dir, right_dir)
        up_dir /= np.linalg.norm(up_dir)
        
        nose, tail = pos + nose_dir * size, pos - nose_dir * size * 0.4
        left_wing, right_wing = pos - right_dir * size * 0.6, pos + right_dir * size * 0.6
        
        draw_line_3d(position, nose.tolist(), RED)
        draw_sphere(nose.tolist(), size * 0.08, RED)
        draw_line_3d(left_wing.tolist(), right_wing.tolist(), GREEN)
        draw_sphere(left_wing.tolist(), size * 0.08, GREEN)
        draw_sphere(right_wing.tolist(), size * 0.08, GREEN)
        draw_line_3d(position, tail.tolist(), BLUE)
        draw_sphere(tail.tolist(), size * 0.08, BLUE)
        draw_line_3d(tail.tolist(), (tail + up_dir * size * 0.3).tolist(), SKYBLUE)
        draw_sphere((tail + up_dir * size * 0.3).tolist(), size * 0.08, SKYBLUE)
        draw_sphere(position, size * 0.12, LIGHTGRAY)

    def _draw_ribbon_trail(self) -> None:
        """Draw trail as three lines (left, center, right) showing aircraft orientation."""
        if len(self.trail) < 2:
            return
        
        half_width = RIBBON_WIDTH / 2.0
        left_edges, right_edges = [], []
        
        for pt in self.trail:
            pos = np.array(pt['pos'])
            sp, cp, st, ct, sy, cy = math.sin(pt['roll']), math.cos(pt['roll']), math.sin(pt['pitch']), math.cos(pt['pitch']), math.sin(pt['yaw']), math.cos(pt['yaw'])
            r = np.array([sp * sy * st + cp * cy, -sp * ct, sp * cy * st - cp * sy]) / (np.linalg.norm(np.array([sp * sy * st + cp * cy, -sp * ct, sp * cy * st - cp * sy])) + 1e-10)
            left_edges.append(pos - r * half_width)
            right_edges.append(pos + r * half_width)
        
        # Draw three continuous lines: left edge, center, right edge
        for i in range(len(self.trail) - 1):
            draw_line_3d(left_edges[i].tolist(), left_edges[i + 1].tolist(), TRAIL_COLOR)
            draw_line_3d(right_edges[i].tolist(), right_edges[i + 1].tolist(), TRAIL_COLOR)
            draw_line_3d(self.trail[i]['pos'], self.trail[i + 1]['pos'], TRAIL_COLOR)

    def _handle_camera_input(self) -> None:
        """Handle mouse input for camera control."""
        wheel = get_mouse_wheel_move()
        if wheel != 0:
            self.manual_zoom += wheel * 200.0
            # Clamp to prevent "stuck" feeling at zoom limits
            min_zoom = -(20000.0 - CHASE_DISTANCE)
            max_zoom = CHASE_DISTANCE - 50.0
            self.manual_zoom = max(min_zoom, min(max_zoom, self.manual_zoom))
        
        if is_mouse_button_down(MOUSE_BUTTON_LEFT):
            mouse_pos = get_mouse_position()
            if self.last_mouse_pos is not None:
                delta_x, delta_y = mouse_pos.x - self.last_mouse_pos.x, mouse_pos.y - self.last_mouse_pos.y
                self.manual_camera_offset[0] -= delta_x * 0.01
                self.manual_camera_offset[1] += delta_y * 0.01
                self.manual_camera_offset[1] = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.manual_camera_offset[1]))
            self.last_mouse_pos = mouse_pos
        else:
            self.last_mouse_pos = None

    def _update_camera(self, position: list) -> None:
        self.camera.target = position
        self.camera.up = [0.0, 1.0, 0.0]
        
        distance = max(50.0, min(20000.0, self.chase_distance - self.manual_zoom))
        az, el = self.manual_camera_offset
        
        offset = np.array([
            distance * math.cos(el) * math.sin(az),
            distance * math.sin(el),
            distance * math.cos(el) * math.cos(az)
        ])
        
        self.camera.position = [float(position[0] + offset[0]), float(position[1] + offset[1]), float(position[2] + offset[2])]

    def _draw_hud(self, state: RenderState) -> None:
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
            draw_text_ex(self.font, text, [p, p + i * lh], 20, 1, RAYWHITE)


