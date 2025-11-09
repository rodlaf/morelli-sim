import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import numpy as np
from pyray import *
from pyray import RL_PROJECTION, RL_MODELVIEW, MOUSE_BUTTON_LEFT, CAMERA_PERSPECTIVE
from scipy.io import loadmat

RAD2DEG = 180.0 / math.pi

# Simulation playback speed multiplier, percentage of real-time speed
PLAYBACK_SPEED = 10.0


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

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        title: str = "Aerobench 3D",
        f16_scale: float = 25.0,
        trail_length: int = 200,
        view_size: float = 1000.0,
        chase_camera: bool = True,
        chase_distance: float = 400.0,
        chase_elevation: float = 150.0,
        waypoints=None,
    ) -> None:
        init_window(width, height, title)
        set_target_fps(60)

        self.camera = Camera3D(
            [0.0, chase_elevation, -chase_distance],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            60.0,
            CAMERA_PERSPECTIVE,
        )
        
        # Set camera near/far planes for large viewing distances
        # Note: We'll manually handle this in begin_mode_3d by using rlgl functions if needed
        self.near_plane = 0.1
        self.far_plane = 50000.0  # Much larger to handle zoomed out views

        self.model = self._load_f16_model(f16_scale)
        # Handle infinite trail length (None means unlimited)
        maxlen = None if (isinstance(trail_length, float) and math.isinf(trail_length)) else trail_length
        self.trail: Deque[Tuple[float, float, float]] = deque(maxlen=maxlen)
        self.view_size = view_size
        self.chase_camera = chase_camera
        self.chase_distance = chase_distance
        self.chase_elevation = chase_elevation
        self.font = get_font_default()
        self.waypoints = waypoints if waypoints is not None else []

        # Camera control state
        self.manual_camera_offset = [0.0, 0.0]  # azimuth, elevation offsets in radians
        self.manual_zoom = 0.0  # zoom offset
        self.last_mouse_pos = None

        self.mode_colors: Dict[str, Tuple[int, int, int, int]] = {
            "gcas": (230, 41, 55, 255),  # RED
            "recovery": (0, 228, 48, 255),  # GREEN
            "normal": (102, 191, 255, 255),  # SKYBLUE
        }

    def close(self) -> None:
        close_window()

    @staticmethod
    def _load_f16_model(scale: float):
        base_dir = os.path.dirname(__file__)
        mat_path = os.path.join(base_dir, "f-16.mat")
        data = loadmat(mat_path)

        vertices = np.asarray(data["V"], dtype=np.float32)
        faces = np.asarray(data["F"], dtype=np.int32) - 1

        scaled = vertices * np.array([-scale, scale, scale], dtype=np.float32)

        normals = np.zeros_like(scaled)
        for f in faces:
            v0, v1, v2 = scaled[f[0]], scaled[f[1]], scaled[f[2]]
            face_normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(face_normal)
            if norm > 1e-6:
                face_normal /= norm
            normals[f[0]] += face_normal
            normals[f[1]] += face_normal
            normals[f[2]] += face_normal

        norms = np.linalg.norm(normals, axis=1)
        norms[norms < 1e-6] = 1.0
        normals /= norms[:, None]

        mesh = Mesh()
        mesh.vertexCount = scaled.shape[0]
        mesh.triangleCount = faces.shape[0]

        # Convert to ffi arrays
        flat_vertices = scaled.flatten().tolist()
        flat_normals = normals.flatten().tolist()
        flat_indices = faces.astype(np.uint16).flatten().tolist()

        # Pyray mesh expects ffi pointers
        mesh.vertices = ffi.new(f"float[{len(flat_vertices)}]", flat_vertices)
        mesh.normals = ffi.new(f"float[{len(flat_normals)}]", flat_normals)
        mesh.indices = ffi.new(f"unsigned short[{len(flat_indices)}]", flat_indices)

        upload_mesh(mesh, False)
        model = load_model_from_mesh(mesh)
        return model

    def render(self, state: RenderState) -> None:
        # Handle camera controls
        self._handle_camera_input()
        
        pos_e, pos_n, altitude = state.position_ft
        position = [float(pos_e), float(altitude), float(pos_n)]
        self.trail.append(tuple(position))

        rotation_matrix, basis = self._build_rotation(state.theta_rad, state.psi_rad, state.phi_rad)
        
        self._update_camera(position, basis)

        begin_drawing()
        clear_background(BLACK)

        begin_mode_3d(self.camera)
        
        # Override projection to extend far clipping plane (must be after begin_mode_3d)
        aspect = get_screen_width() / get_screen_height()
        top = self.near_plane * math.tan(self.camera.fovy * 0.5 * math.pi / 180.0)
        
        rl_matrix_mode(RL_PROJECTION)
        rl_load_identity()
        rl_frustum(-top * aspect, top * aspect, -top, top, self.near_plane, self.far_plane)
        rl_matrix_mode(RL_MODELVIEW)
        
        # Draw ground and grid
        draw_plane([position[0], 0.0, position[2]], [self.view_size, self.view_size], DARKGRAY)
        draw_grid(40, int(self.view_size / 20))

        # Draw plane
        self._draw_simple_plane(position, basis, 500.0)

        # Draw waypoints
        for i, wp in enumerate(self.waypoints):
            wp_pos = [float(wp[0]), float(wp[2]), float(wp[1])]  # (east, altitude, north)
            draw_sphere(wp_pos, 50.0, BLUE)
            draw_sphere_wires(wp_pos, 50.0, 8, 8, SKYBLUE)

        # Draw trail
        trail_pts = list(self.trail)
        for i in range(len(trail_pts) - 1):
            draw_line_3d(trail_pts[i], trail_pts[i + 1], YELLOW)

        end_mode_3d()

        self._draw_hud(state)
        end_drawing()

    def _draw_simple_plane(self, position: list, basis: np.ndarray, size: float) -> None:
        """Draw a simple plane using oriented axes and shapes."""
        pos = np.array(position)
        
        # basis vectors from rotation matrix (already normalized)
        forward = basis[:, 0] * size
        right = basis[:, 1] * size * 0.6
        up = basis[:, 2] * size * 0.3
        
        # Nose (forward direction) - RED
        nose = pos + forward
        draw_line_3d(position, nose.tolist(), RED)
        draw_sphere(nose.tolist(), size * 0.12, RED)
        
        # Wings - GREEN
        left_wing = pos - right
        right_wing = pos + right
        draw_line_3d(left_wing.tolist(), right_wing.tolist(), GREEN)
        draw_sphere(left_wing.tolist(), size * 0.08, GREEN)
        draw_sphere(right_wing.tolist(), size * 0.08, GREEN)
        
        # Tail (backward) - BLUE  
        tail = pos - forward * 0.4
        draw_line_3d(position, tail.tolist(), BLUE)
        
        # Vertical stabilizer (up) - SKYBLUE
        tail_up = tail + up
        draw_line_3d(tail.tolist(), tail_up.tolist(), SKYBLUE)
        
        # Body sphere
        draw_sphere(position, size * 0.12, LIGHTGRAY)

    def _handle_camera_input(self) -> None:
        """Handle mouse input for camera control."""
        # Zoom with mouse wheel
        wheel = get_mouse_wheel_move()
        if wheel != 0:
            # TUNABLE: Zoom sensitivity (units per wheel tick)
            self.manual_zoom += wheel * 200.0
        
        # Rotate camera with left mouse drag
        if is_mouse_button_down(MOUSE_BUTTON_LEFT):
            mouse_pos = get_mouse_position()
            if self.last_mouse_pos is not None:
                delta_x = mouse_pos.x - self.last_mouse_pos.x
                delta_y = mouse_pos.y - self.last_mouse_pos.y
                
                # TUNABLE: Rotation sensitivity (radians per pixel)
                rotation_sensitivity = 0.01
                
                # Flip both axes: negative signs reverse rotation direction
                self.manual_camera_offset[0] -= delta_x * rotation_sensitivity  # azimuth (horizontal)
                self.manual_camera_offset[1] += delta_y * rotation_sensitivity  # elevation (vertical)
                
                # Clamp elevation to prevent camera flipping upside down
                self.manual_camera_offset[1] = max(-math.pi/2 + 0.1, min(math.pi/2 - 0.1, self.manual_camera_offset[1]))
            
            self.last_mouse_pos = mouse_pos
        else:
            self.last_mouse_pos = None

    def _update_camera(self, position: list, basis: np.ndarray) -> None:
        self.camera.target = position
        self.camera.up = [0.0, 1.0, 0.0]

        if self.chase_camera:
            # TUNABLE: Camera distance limits (min, max in world units)
            min_distance = 50.0
            max_distance = 20000.0
            distance = max(min_distance, min(max_distance, self.chase_distance - self.manual_zoom))
            
            # Calculate spherical coordinates around target
            azimuth = self.manual_camera_offset[0]
            elevation = self.manual_camera_offset[1]
            
            # Convert spherical to cartesian offset from target
            # Default: behind the plane at base elevation
            offset = np.array([
                distance * math.cos(elevation) * math.sin(azimuth),
                distance * math.sin(elevation) + self.chase_elevation,
                distance * math.cos(elevation) * math.cos(azimuth)
            ])
            
            self.camera.position = [
                float(position[0] + offset[0]),
                float(position[1] + offset[1]),
                float(position[2] + offset[2])
            ]

    @staticmethod
    def _build_rotation(theta: float, psi: float, phi: float):
        psi_adj = psi - math.pi / 2.0
        phi_adj = -phi

        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        sin_psi = math.sin(psi_adj)
        cos_psi = math.cos(psi_adj)
        sin_phi = math.sin(phi_adj)
        cos_phi = math.cos(phi_adj)

        transform = np.array(
            [
                [cos_psi * cos_theta, -sin_psi * cos_theta, sin_theta],
                [cos_psi * sin_theta * sin_phi + sin_psi * cos_phi,
                 -sin_psi * sin_theta * sin_phi + cos_psi * cos_phi,
                 -cos_theta * sin_phi],
                [-cos_psi * sin_theta * cos_phi + sin_psi * sin_phi,
                 sin_psi * sin_theta * cos_phi + cos_psi * sin_phi,
                 cos_theta * cos_phi],
            ],
            dtype=np.float32,
        )

        matrix = Matrix(
            transform[0, 0], transform[1, 0], transform[2, 0], 0.0,
            transform[0, 1], transform[1, 1], transform[2, 1], 0.0,
            transform[0, 2], transform[1, 2], transform[2, 2], 0.0,
            0.0, 0.0, 0.0, 1.0
        )

        return matrix, transform

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

        mode_color = self.mode_colors.get(state.mode.lower(), (245, 245, 245, 255))
        draw_text_ex(
            self.font,
            f"Mode: {state.mode}",
            [padding, padding + line_height],
            20,
            1,
            mode_color,
        )

        draw_text_ex(
            self.font,
            f"h = {state.position_ft[2]:.1f} ft",
            [padding, padding + 2 * line_height],
            20,
            1,
            RAYWHITE,
        )
        draw_text_ex(
            self.font,
            f"V = {state.speed_fps:.1f} ft/s",
            [padding, padding + 3 * line_height],
            20,
            1,
            RAYWHITE,
        )

        draw_text_ex(
            self.font,
            f"alpha = {state.alpha_rad * RAD2DEG:.1f} deg",
            [padding, padding + 4 * line_height],
            20,
            1,
            RAYWHITE,
        )
        draw_text_ex(
            self.font,
            f"beta = {state.beta_rad * RAD2DEG:.1f} deg",
            [padding, padding + 5 * line_height],
            20,
            1,
            RAYWHITE,
        )

        draw_text_ex(
            self.font,
            f"Nz = {state.nz_g:.2f} g",
            [padding, padding + 6 * line_height],
            20,
            1,
            RAYWHITE,
        )
        draw_text_ex(
            self.font,
            f"ps = {state.ps_rad_s * RAD2DEG:.1f} deg/s",
            [padding, padding + 7 * line_height],
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
            [padding, padding + 8 * line_height],
            20,
            1,
            RAYWHITE,
        )
