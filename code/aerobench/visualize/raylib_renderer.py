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
CHASE_DISTANCE = 1250.0  # default camera distance from aircraft
CHASE_ELEVATION = 150.0  # camera height offset above aircraft
F16_SCALE = 100.0  # visual size of the F-16 model in feet
ALTITUDE_LINE_SPACING = 300.0  # feet between altitude markers
GRID_SPACING = 1000.0  # feet between ground grid lines
SKY_COLOR = Color(0, 0, 0, 255)  # Black sky
GRID_COLOR = Color(40, 120, 40, 180)  # Brighter semi-transparent green
TRAIL_COLOR = YELLOW  # Trail and altitude marker color


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
        
        self.trail.append(tuple(position))
        
        # Add altitude marker every ALTITUDE_LINE_SPACING feet
        if len(self.trail) > 1:
            segment_dist = np.linalg.norm(np.array(position) - np.array(self.trail[-2]))
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
            draw_line_3d([marker_pos[0], 0.0, marker_pos[2]], marker_pos, TRAIL_COLOR)
        draw_line_3d([position[0], 0.0, position[2]], position, TRAIL_COLOR)

        # Draw waypoints
        for wp in self.waypoints:
            wp_pos = [float(wp[0]), float(wp[2]), float(wp[1])]  # [east, altitude, north]
            draw_sphere(wp_pos, 50.0, PURPLE)
            draw_line_3d([wp_pos[0], 0.0, wp_pos[2]], wp_pos, PURPLE)

        # Draw plane and trail
        self._draw_simple_plane(position, state.phi_rad, state.theta_rad, state.psi_rad, F16_SCALE)
        for i in range(len(self.trail) - 1):
            draw_line_3d(self.trail[i], self.trail[i + 1], TRAIL_COLOR)

        end_mode_3d()

        self._draw_hud(state)
        end_drawing()

    def _draw_simple_plane(self, position: list, roll: float, pitch: float, yaw: float, size: float) -> None:
        """Draw simple F-16 representation using Euler angles and F16 navigation equations."""
        pos = np.array(position)
        
        # Compute trig values
        sphi, cphi = math.sin(roll), math.cos(roll)
        stheta, ctheta = math.sin(pitch), math.cos(pitch)
        spsi, cpsi = math.sin(yaw), math.cos(yaw)
        
        # F16 navigation equations: body axes -> world (north, east, up)
        forward_north, forward_east, forward_up = ctheta * cpsi, ctheta * spsi, stheta
        right_north = sphi * cpsi * stheta - cphi * spsi
        right_east = sphi * spsi * stheta + cphi * cpsi
        right_up = sphi * ctheta
        
        # Convert to visualization frame: (East, Up, North)
        nose_dir = np.array([forward_east, forward_up, forward_north])
        right_dir = np.array([right_east, -right_up, right_north])
        up_dir = np.cross(nose_dir, right_dir)
        up_dir /= np.linalg.norm(up_dir)
        
        # Scale and draw components
        forward, right, up = nose_dir * size, right_dir * size * 0.6, up_dir * size * 0.3
        nose, tail = pos + forward, pos - forward * 0.4
        right_wing, left_wing = pos + right, pos - right
        
        draw_line_3d(position, nose.tolist(), RED)
        draw_sphere(nose.tolist(), size * 0.08, RED)
        draw_line_3d(left_wing.tolist(), right_wing.tolist(), GREEN)
        draw_sphere(left_wing.tolist(), size * 0.08, GREEN)
        draw_sphere(right_wing.tolist(), size * 0.08, GREEN)
        draw_line_3d(position, tail.tolist(), BLUE)
        draw_sphere(tail.tolist(), size * 0.08, BLUE)
        draw_line_3d(tail.tolist(), (tail + up).tolist(), SKYBLUE)
        draw_sphere((tail + up).tolist(), size * 0.08, SKYBLUE)
        draw_sphere(position, size * 0.12, LIGHTGRAY)

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

        # TUNABLE: Camera distance limits (min, max in world units)
        min_distance = 50.0
        max_distance = 20000.0
        distance = max(min_distance, min(max_distance, self.chase_distance - self.manual_zoom))
        
        # Calculate spherical coordinates around target
        azimuth = self.manual_camera_offset[0]
        elevation = self.manual_camera_offset[1]
        
        # Convert spherical to cartesian offset from target
        # Pure spherical coordinates - maintains constant distance
        offset = np.array([
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            distance * math.cos(elevation) * math.cos(azimuth)
        ])
        
        self.camera.position = [
            float(position[0] + offset[0]),
            float(position[1] + offset[1]),
            float(position[2] + offset[2])
        ]

    def _draw_hud(self, state: RenderState) -> None:
        padding = 12
        line_height = 22

        draw_text_ex(
            self.font,
            f"t = {state.time_sec:.2f} s",
            [padding, padding],
            20,
            1,
            RAYWHITE,
        )

        draw_text_ex(
            self.font,
            f"h = {state.position_ft[2]:.1f} ft",
            [padding, padding + line_height],
            20,
            1,
            RAYWHITE,
        )
        draw_text_ex(
            self.font,
            f"V = {state.speed_fps:.1f} ft/s",
            [padding, padding + 2 * line_height],
            20,
            1,
            RAYWHITE,
        )

        draw_text_ex(
            self.font,
            f"alpha = {state.alpha_rad * RAD2DEG:.1f} deg",
            [padding, padding + 3 * line_height],
            20,
            1,
            RAYWHITE,
        )
        draw_text_ex(
            self.font,
            f"beta = {state.beta_rad * RAD2DEG:.1f} deg",
            [padding, padding + 4 * line_height],
            20,
            1,
            RAYWHITE,
        )

        draw_text_ex(
            self.font,
            f"Nz = {state.nz_g:.2f} g",
            [padding, padding + 5 * line_height],
            20,
            1,
            RAYWHITE,
        )
        draw_text_ex(
            self.font,
            f"ps = {state.ps_rad_s * RAD2DEG:.1f} deg/s",
            [padding, padding + 6 * line_height],
            20,
            1,
            RAYWHITE,
        )

        angles = (
            f"phi/theta/psi = "
            f"{state.phi_rad * RAD2DEG:.1f}/"
            f"{state.theta_rad * RAD2DEG:.1f}/"
            f"{state.psi_rad * RAD2DEG:.1f} deg"
        )
        draw_text_ex(
            self.font,
            angles,
            [padding, padding + 7 * line_height],
            20,
            1,
            RAYWHITE,
        )

        draw_text_ex(
            self.font,
            f"Left click + drag: rotate camera",
            [padding, padding + 8 * line_height],
            20,
            1,
            RAYWHITE,
        )

        draw_text_ex(
            self.font,
            f"Mouse wheel: zoom in/out",
            [padding, padding + 9 * line_height],
            20,
            1,
            RAYWHITE,
        )


