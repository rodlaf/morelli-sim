import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import numpy as np
from pyray import *
from scipy.io import loadmat

RAD2DEG = 180.0 / math.pi


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

        self.model = self._load_f16_model(f16_scale)
        # Handle infinite trail length (None means unlimited)
        maxlen = None if (isinstance(trail_length, float) and math.isinf(trail_length)) else trail_length
        self.trail: Deque[Tuple[float, float, float]] = deque(maxlen=maxlen)
        self.view_size = view_size
        self.chase_camera = chase_camera
        self.chase_distance = chase_distance
        self.chase_elevation = chase_elevation
        self.font = get_font_default()

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
        pos_e, pos_n, altitude = state.position_ft
        position = [float(pos_e), float(altitude), float(pos_n)]
        self.trail.append(position)

        rotation_matrix, basis = self._build_rotation(state.theta_rad, state.psi_rad, state.phi_rad)
        self.model.transform = rotation_matrix

        self._update_camera(position, basis)

        begin_drawing()
        clear_background(BLACK)

        begin_mode_3d(self.camera)
        ground_size = self.view_size
        draw_plane([position[0], 0.0, position[2]], [ground_size, ground_size], DARKGRAY)
        draw_grid(40, int(ground_size / 20))

        draw_model(self.model, position, 1.0, LIGHTGRAY)

        # Draw trail
        trail_pts = list(self.trail)
        for i in range(len(trail_pts) - 1):
            draw_line_3d(trail_pts[i], trail_pts[i + 1], RED)

        end_mode_3d()

        self._draw_hud(state)
        end_drawing()

    def _update_camera(self, position: list, basis: np.ndarray) -> None:
        self.camera.target = position
        up = basis[:, 2]
        self.camera.up = [float(up[0]), float(up[1]), float(up[2])]

        if self.chase_camera:
            forward = basis[:, 0]
            eye = np.array(position)
            eye -= forward * self.chase_distance
            eye += up * self.chase_elevation
            self.camera.position = [float(eye[0]), float(eye[1]), float(eye[2])]

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
