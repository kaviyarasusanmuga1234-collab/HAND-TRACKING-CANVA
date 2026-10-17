import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime

# =====================================================================
# 3D MESH & BLENDER TOOL GENERATOR WITH AUTO-MERGE
# =====================================================================

class Mesh3D:
    def __init__(self, name="Object3D", color=(0, 230, 255)):
        self.name = name
        self.vertices = np.zeros((0, 3), dtype=np.float32)
        self.faces = []
        self.color = color
        self.pos = np.array([0.0, 0.0, 500.0], dtype=np.float32)
        self.rot = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale = 1.0
        self.selected = True
        self.auto_spin = False
        self.wireframe_only = False

    def translate(self, dx, dy, dz=0.0):
        self.pos[0] += dx
        self.pos[1] += dy
        self.pos[2] += dz

    def rotate(self, dpitch, dyaw, droll=0.0):
        self.rot[0] += dpitch
        self.rot[1] += dyaw
        self.rot[2] += droll

    def zoom(self, factor):
        self.scale = np.clip(self.scale * factor, 0.15, 6.0)

    def get_rotation_matrix(self):
        p, y, r = self.rot
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(p), -np.sin(p)],
            [0, np.sin(p), np.cos(p)]
        ], dtype=np.float32)

        Ry = np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ], dtype=np.float32)

        Rz = np.array([
            [np.cos(r), -np.sin(r), 0],
            [np.sin(r), np.cos(r), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        return Rz @ Ry @ Rx

    def merge_with(self, other_mesh):
        if len(other_mesh.vertices) == 0:
            return

        R_self = self.get_rotation_matrix()
        R_other = other_mesh.get_rotation_matrix()

        world_other = (other_mesh.vertices * other_mesh.scale @ R_other.T) + other_mesh.pos
        local_in_self = ((world_other - self.pos) @ R_self) / self.scale

        vertex_offset = len(self.vertices)
        new_faces = [[idx + vertex_offset for idx in face] for face in other_mesh.faces]

        self.vertices = np.vstack([self.vertices, local_in_self])
        self.faces.extend(new_faces)
        self.name = f"{self.name}+{other_mesh.name}"

    def get_screen_center(self, w, h, focal_length=600.0):
        cx, cy = w / 2.0, h / 2.0
        z = max(50.0, self.pos[2])
        proj_x = int((self.pos[0] * focal_length / z) + cx)
        proj_y = int((self.pos[1] * focal_length / z) + cy)
        return (proj_x, proj_y)

    def render(self, img, focal_length=600.0):
        if len(self.vertices) == 0:
            return

        h, w, _ = img.shape
        cx, cy = w / 2.0, h / 2.0

        R = self.get_rotation_matrix()
        scaled_verts = self.vertices * self.scale
        transformed = (scaled_verts @ R.T) + self.pos

        z = transformed[:, 2]
        if np.any(z <= 50.0):
            transformed[:, 2] = np.maximum(transformed[:, 2], 50.0)
            z = transformed[:, 2]

        proj_x = (transformed[:, 0] * focal_length / z) + cx
        proj_y = (transformed[:, 1] * focal_length / z) + cy
        screen_pts = np.stack([proj_x, proj_y], axis=1).astype(np.int32)

        light_dir = np.array([0.4, -0.6, -1.0], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        face_depths = []
        for i, face in enumerate(self.faces):
            avg_z = np.mean(transformed[face, 2])
            face_depths.append((avg_z, i))
        face_depths.sort(key=lambda item: item[0], reverse=True)

        for avg_z, f_idx in face_depths:
            face = self.faces[f_idx]
            pts = screen_pts[face]

            if not self.wireframe_only and len(face) >= 3:
                v0 = transformed[face[0]]
                v1 = transformed[face[1]]
                v2 = transformed[face[2]]
                norm = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(norm)
                if norm_len > 1e-5:
                    norm /= norm_len
                    intensity = np.clip(np.dot(-norm, light_dir), 0.22, 1.0)
                    shade_col = tuple(int(c * intensity) for c in self.color)
                    cv2.fillPoly(img, [pts], shade_col, cv2.LINE_AA)

            edge_col = (0, 255, 255) if self.selected else (90, 90, 90)
            cv2.polylines(img, [pts], isClosed=True, color=edge_col, thickness=1, lineType=cv2.LINE_AA)

        if self.selected:
            center_x, center_y = self.get_screen_center(w, h, focal_length)
            cv2.drawMarker(img, (center_x, center_y), (0, 255, 255), cv2.MARKER_CROSS, 16, 1)


class Blender3DToolbox:
    @staticmethod
    def create_cube(size=120):
        mesh = Mesh3D(name="Cube", color=(0, 200, 255))
        s = size / 2.0
        mesh.vertices = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s,  s], [s, -s,  s], [s, s,  s], [-s, s,  s]
        ], dtype=np.float32)
        mesh.faces = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
        ]
        return mesh

    @staticmethod
    def create_sphere(radius=80, rings=10, sectors=14):
        mesh = Mesh3D(name="Sphere", color=(0, 255, 120))
        verts = []
        for i in range(rings + 1):
            theta = (i * np.pi) / rings
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            for j in range(sectors):
                phi = (j * 2 * np.pi) / sectors
                verts.append([radius * sin_t * np.cos(phi), radius * cos_t, radius * sin_t * np.sin(phi)])
        mesh.vertices = np.array(verts, dtype=np.float32)

        faces = []
        for i in range(rings):
            for j in range(sectors):
                p0 = i * sectors + j
                p1 = p0 + sectors
                next_j = (j + 1) % sectors
                p2 = (i + 1) * sectors + next_j
                p3 = i * sectors + next_j
                faces.append([p0, p1, p2, p3])
        mesh.faces = faces
        return mesh

    @staticmethod
    def create_cylinder(radius=60, height=140, segments=14):
        mesh = Mesh3D(name="Cylinder", color=(255, 120, 0))
        h2 = height / 2.0
        verts = []
        for y_pos in [-h2, h2]:
            for j in range(segments):
                ang = (j * 2 * np.pi) / segments
                verts.append([radius * np.cos(ang), y_pos, radius * np.sin(ang)])
        mesh.vertices = np.array(verts, dtype=np.float32)

        faces = []
        for j in range(segments):
            next_j = (j + 1) % segments
            faces.append([j, next_j, next_j + segments, j + segments])
        faces.append(list(range(segments)))
        faces.append(list(range(segments, segments * 2)))
        mesh.faces = faces
        return mesh

    @staticmethod
    def create_pyramid(base_size=120, height=130):
        mesh = Mesh3D(name="Pyramid", color=(255, 0, 180))
        s = base_size / 2.0
        h2 = height / 2.0
        mesh.vertices = np.array([
            [-s, h2, -s], [s, h2, -s], [s, h2, s], [-s, h2, s], [0, -h2, 0]
        ], dtype=np.float32)
        mesh.faces = [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
        return mesh

    @staticmethod
    def create_torus(R=75, r=26, rings=14, sides=10):
        mesh = Mesh3D(name="Torus", color=(0, 230, 255))
        verts = []
        for i in range(rings):
            u = (i * 2 * np.pi) / rings
            for j in range(sides):
                v = (j * 2 * np.pi) / sides
                x = (R + r * np.cos(v)) * np.cos(u)
                y = (R + r * np.cos(v)) * np.sin(u)
                z = r * np.sin(v)
                verts.append([x, y, z])
        mesh.vertices = np.array(verts, dtype=np.float32)

        faces = []
        for i in range(rings):
            next_i = (i + 1) % rings
            for j in range(sides):
                next_j = (j + 1) % sides
                faces.append([i * sides + j, next_i * sides + j, next_i * sides + next_j, i * sides + next_j])
        mesh.faces = faces
        return mesh

    @staticmethod
    def extrude_line_to_3d_tube(stroke_2d, radius=22, segments=8):
        if len(stroke_2d) < 3:
            return None
        mesh = Mesh3D(name="ExtrudedTube", color=(0, 255, 230))
        pts = np.array(stroke_2d, dtype=np.float32)
        center_2d = np.mean(pts, axis=0)
        pts[:, 0] -= center_2d[0]
        pts[:, 1] -= center_2d[1]

        step = max(1, len(pts) // 16)
        sampled = pts[::step]
        if len(sampled) < 2:
            return None

        verts = []
        for i in range(len(sampled)):
            p = sampled[i]
            tangent = (sampled[i + 1] - p) if (i < len(sampled) - 1) else (p - sampled[i - 1])
            t_norm = np.linalg.norm(tangent)
            tangent = np.array([1.0, 0.0], dtype=np.float32) if t_norm < 1e-4 else (tangent / t_norm)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

            for s in range(segments):
                ang = (s * 2 * np.pi) / segments
                verts.append([p[0] + normal[0] * np.cos(ang) * radius,
                              p[1] + normal[1] * np.cos(ang) * radius,
                              np.sin(ang) * radius])

        mesh.vertices = np.array(verts, dtype=np.float32)
        faces = []
        for i in range(len(sampled) - 1):
            for s in range(segments):
                next_s = (s + 1) % segments
                faces.append([i * segments + s, (i + 1) * segments + s, (i + 1) * segments + next_s, i * segments + next_s])
        mesh.faces = faces
        return mesh

    @staticmethod
    def circle_to_3d_sphere(stroke_2d):
        pts = np.array(stroke_2d, dtype=np.float32)
        dim = np.max(pts, axis=0) - np.min(pts, axis=0)
        radius = float(np.clip(np.mean(dim) / 2.0, 35.0, 180.0))
        return Blender3DToolbox.create_sphere(radius=radius)


# =====================================================================
# 2D CANVAS OBJECT WITH AUTO-MERGE
# =====================================================================

class Canvas2DObject:
    def __init__(self, color, thickness):
        self.strokes = []
        self.color = color
        self.thickness = thickness
        self.selected = False
        self.bbox = (0, 0, 0, 0)

    def add_stroke(self, points):
        if points and len(points) > 1:
            self.strokes.append(list(points))
            self.update_bbox()

    def update_bbox(self):
        all_pts = [p for s in self.strokes for p in s]
        if not all_pts:
            self.bbox = (0, 0, 0, 0)
            return
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        pad = max(25, self.thickness + 15)
        self.bbox = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    def distance_to_point(self, x, y):
        if not self.strokes:
            return float('inf')
        x1, y1, x2, y2 = self.bbox
        if not (x1 - 40 <= x <= x2 + 40 and y1 - 40 <= y <= y2 + 40):
            return np.hypot(x - (x1 + x2) / 2, y - (y1 + y2) / 2)
        min_d = float('inf')
        for stroke in self.strokes:
            for px, py in stroke:
                d = (px - x) ** 2 + (py - y) ** 2
                if d < min_d:
                    min_d = d
        return np.sqrt(min_d)

    def translate(self, dx, dy):
        self.strokes = [[(x + dx, y + dy) for (x, y) in s] for s in self.strokes]
        self.update_bbox()

    def merge_with(self, other_obj):
        self.strokes.extend(other_obj.strokes)
        self.update_bbox()

    def draw(self, img):
        for stroke in self.strokes:
            if len(stroke) < 2:
                continue
            for i in range(1, len(stroke)):
                cv2.line(img, stroke[i - 1], stroke[i], self.color, self.thickness, cv2.LINE_AA)

        if self.selected:
            x1, y1, x2, y2 = self.bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)


# =====================================================================
# DUSTBIN (TRASH CAN) OBJECT
# =====================================================================

class Dustbin:
    def __init__(self, x=1150, y=550, w=100, h=125):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.is_open = False
        self.flash_timer = 0

    def contains(self, px, py):
        pad = 40
        return (self.x - pad <= px <= self.x + self.w + pad) and (self.y - pad <= py <= self.y + self.h + pad)

    def trigger_delete_effect(self):
        self.flash_timer = time.time() + 0.8

    def draw(self, img):
        is_flashing = time.time() < self.flash_timer
        border_color = (0, 0, 255) if (self.is_open or is_flashing) else (140, 140, 140)
        fill_color = (0, 0, 160) if is_flashing else (25, 25, 25)

        bx, by, bw, bh = self.x, self.y, self.w, self.h

        pts = np.array([
            [bx + 10, by + 32],
            [bx + bw - 10, by + 32],
            [bx + bw - 20, by + bh],
            [bx + 20, by + bh]
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], fill_color, cv2.LINE_AA)
        cv2.polylines(img, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

        for i in range(1, 4):
            frac = i / 4.0
            x_top = int(bx + 10 + (bw - 20) * frac)
            x_bot = int(bx + 20 + (bw - 40) * frac)
            cv2.line(img, (x_top, by + 38), (x_bot, by + bh - 6), border_color, 1, cv2.LINE_AA)

        if self.is_open or is_flashing:
            lid_pts = np.array([
                [bx - 8, by + 12],
                [bx + bw - 8, by - 16],
                [bx + bw - 4, by - 4],
                [bx - 4, by + 24]
            ], dtype=np.int32)
            cv2.fillPoly(img, [lid_pts], (0, 0, 220), cv2.LINE_AA)
            cv2.polylines(img, [lid_pts], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(img, "RELEASE TO DELETE", (bx - 40, by - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            lid_rect = (bx - 4, by + 18, bw + 8, 14)
            cv2.rectangle(img, (lid_rect[0], lid_rect[1]),
                          (lid_rect[0] + lid_rect[2], lid_rect[1] + lid_rect[3]), (45, 45, 45), -1)
            cv2.rectangle(img, (lid_rect[0], lid_rect[1]),
                          (lid_rect[0] + lid_rect[2], lid_rect[1] + lid_rect[3]), border_color, 2)
            cv2.rectangle(img, (bx + bw // 2 - 12, by + 10), (bx + bw // 2 + 12, by + 18), border_color, 2)

        label_col = (0, 0, 255) if self.is_open else (160, 160, 160)
        cv2.putText(img, "DUSTBIN", (bx + 14, by + bh + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, label_col, 2 if self.is_open else 1, cv2.LINE_AA)


# =====================================================================
# CORE ENGINE - CLEAN PROFESSIONAL MINIMAL UI
# =====================================================================

class HandTrackerState:
    def __init__(self, label, default_color):
        self.label = label
        self.color = default_color
        self.current_stroke = []
        self.smoothed_pos = None
        self.is_grabbing = False
        self.grabbed_3d_obj = None
        self.grabbed_2d_obj = None
        self.prev_pinch_pos = None
        self.stroke_pause_counter = 0


class AirCanvasBlenderApp:
    def __init__(self):
        self.window_name = "Air Canvas Pro"
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

        self.objects_2d = []
        self.objects_3d = []
        self.active_3d_index = 0

        self.objects_3d.append(Blender3DToolbox.create_sphere(radius=85))
        self.dustbin = Dustbin(x=1140, y=540, w=105, h=130)

        self.hand_states = {
            "Right": HandTrackerState("Right", (255, 200, 0)),
            "Left": HandTrackerState("Left", (0, 140, 255))
        }

        self.mouse_pos = (0, 0)
        self.mouse_left_down = False
        self.mouse_right_down = False
        self.mouse_stroke = []
        self.mouse_prev_pos = None
        self.mouse_grabbed_3d = None
        self.mouse_grabbed_2d = None

        self.pencil_size = 6
        self.size_presets = [("S", 3), ("M", 7), ("L", 14), ("XL", 26)]

        self.active_tool = "DRAW_2D"
        self.pure_canvas = False
        self.show_about = False
        self.notification = ""
        self.notification_timer = 0

        self.blender_tools = [
            ("2D PEN", "DRAW_2D"),
            ("3D TUBE", "BLENDER_TUBE"),
            ("3D SPHERE", "BLENDER_CIRCLE"),
            ("CUBE", "SPAWN_CUBE"),
            ("CYLINDER", "SPAWN_CYLINDER"),
            ("PYRAMID", "SPAWN_PYRAMID"),
            ("TORUS", "SPAWN_TORUS"),
            ("MOVE/ROTATE", "MOVE_3D")
        ]

    def set_notification(self, text, duration=2.0):
        self.notification = text
        self.notification_timer = time.time() + duration

    def set_pencil_size(self, new_size):
        self.pencil_size = int(np.clip(new_size, 2, 38))
        self.set_notification(f"Size: {self.pencil_size}px")

    def get_active_3d_object(self):
        if self.objects_3d and 0 <= self.active_3d_index < len(self.objects_3d):
            return self.objects_3d[self.active_3d_index]
        return None

    def on_mouse_event(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        active_mesh = self.get_active_3d_object()

        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                if active_mesh:
                    active_mesh.zoom(1.10)
                    self.set_notification(f"Zoom: {int(active_mesh.scale * 100)}%")
            else:
                if active_mesh:
                    active_mesh.zoom(0.90)
                    self.set_notification(f"Zoom: {int(active_mesh.scale * 100)}%")

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_left_down = True
            self.mouse_prev_pos = (x, y)

            clicked_ui = self.handle_toolbar_click(x, y, param)
            if not clicked_ui and not self.show_about:
                if active_mesh:
                    cx_3d, cy_3d = active_mesh.get_screen_center(1280, 720)
                    if np.hypot(x - cx_3d, y - cy_3d) < 120:
                        self.mouse_grabbed_3d = active_mesh

                if self.mouse_grabbed_3d is None:
                    for obj_2d in reversed(self.objects_2d):
                        if obj_2d.distance_to_point(x, y) < 60:
                            self.mouse_grabbed_2d = obj_2d
                            break

                if self.mouse_grabbed_3d is None and self.mouse_grabbed_2d is None:
                    self.mouse_stroke = [(x, y)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dustbin.contains(x, y):
                self.dustbin.is_open = True

            if self.mouse_left_down and self.mouse_prev_pos is not None:
                dx = x - self.mouse_prev_pos[0]
                dy = y - self.mouse_prev_pos[1]

                if self.mouse_grabbed_3d:
                    self.mouse_grabbed_3d.translate(dx=dx * 0.8, dy=dy * 0.8)
                elif self.mouse_grabbed_2d:
                    self.mouse_grabbed_2d.translate(dx, dy)
                elif not self.show_about and len(self.mouse_stroke) > 0:
                    self.mouse_stroke.append((x, y))

                self.mouse_prev_pos = (x, y)

            elif self.mouse_right_down and self.mouse_prev_pos is not None and active_mesh:
                dx = x - self.mouse_prev_pos[0]
                dy = y - self.mouse_prev_pos[1]
                active_mesh.rotate(dyaw=dx * 0.015, dpitch=-dy * 0.015)
                self.mouse_prev_pos = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_left_down = False

            if self.mouse_grabbed_3d:
                if self.dustbin.contains(x, y):
                    if self.mouse_grabbed_3d in self.objects_3d:
                        self.objects_3d.remove(self.mouse_grabbed_3d)
                        self.active_3d_index = max(0, len(self.objects_3d) - 1)
                        self.dustbin.trigger_delete_effect()
                        self.set_notification(f"Deleted: {self.mouse_grabbed_3d.name}")
                else:
                    self.check_and_merge_3d_objects(self.mouse_grabbed_3d)
                self.mouse_grabbed_3d = None

            elif self.mouse_grabbed_2d:
                if self.dustbin.contains(x, y):
                    if self.mouse_grabbed_2d in self.objects_2d:
                        self.objects_2d.remove(self.mouse_grabbed_2d)
                        self.dustbin.trigger_delete_effect()
                        self.set_notification("Deleted: 2D Object")
                else:
                    self.check_and_merge_2d_objects(self.mouse_grabbed_2d)
                self.mouse_grabbed_2d = None

            elif len(self.mouse_stroke) > 2:
                self.commit_stroke(self.mouse_stroke, (0, 255, 255))
                self.mouse_stroke = []

            self.mouse_prev_pos = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_right_down = True
            self.mouse_prev_pos = (x, y)

        elif event == cv2.EVENT_RBUTTONUP:
            self.mouse_right_down = False
            self.mouse_prev_pos = None

    def analyze_hand(self, lms, w, h):
        pts = {}
        for idx in range(21):
            pts[idx] = (int(lms[idx].x * w), int(lms[idx].y * h))

        wrist = pts[0]
        palm_size = max(25, int(np.hypot(pts[9][0] - wrist[0], pts[9][1] - wrist[1])))

        def is_finger_extended(tip_idx, pip_idx):
            d_tip = np.hypot(pts[tip_idx][0] - wrist[0], pts[tip_idx][1] - wrist[1])
            d_pip = np.hypot(pts[pip_idx][0] - wrist[0], pts[pip_idx][1] - wrist[1])
            return d_tip > (d_pip * 1.15)

        index_ext = is_finger_extended(8, 6)
        middle_ext = is_finger_extended(12, 10)

        pinch_dist = np.hypot(pts[8][0] - pts[4][0], pts[8][1] - pts[4][1])
        pinch_ratio = pinch_dist / palm_size
        pinch_center = ((pts[8][0] + pts[4][0]) // 2, (pts[8][1] + pts[4][1]) // 2)

        return {
            "pts": pts,
            "index_tip": pts[8],
            "pinch_center": pinch_center,
            "pinch_ratio": pinch_ratio,
            "index_ext": index_ext,
            "middle_ext": middle_ext,
            "palm_size": palm_size
        }

    def check_and_merge_3d_objects(self, active_obj):
        if not active_obj or len(self.objects_3d) < 2:
            return

        merge_distance = 120.0
        i = 0
        while i < len(self.objects_3d):
            other = self.objects_3d[i]
            if other != active_obj:
                dist = np.linalg.norm(active_obj.pos - other.pos)
                if dist < merge_distance:
                    active_obj.merge_with(other)
                    self.objects_3d.pop(i)
                    self.active_3d_index = self.objects_3d.index(active_obj)
                    self.set_notification(f"Merged with {other.name}")
                    continue
            i += 1

    def check_and_merge_2d_objects(self, active_obj):
        if not active_obj or len(self.objects_2d) < 2:
            return

        i = 0
        while i < len(self.objects_2d):
            other = self.objects_2d[i]
            if other != active_obj:
                x1, y1, x2, y2 = active_obj.bbox
                ox1, oy1, ox2, oy2 = other.bbox
                if not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2):
                    active_obj.merge_with(other)
                    self.objects_2d.pop(i)
                    self.set_notification("2D Objects Merged")
                    continue
            i += 1

    def commit_stroke(self, stroke, color):
        if len(stroke) < 3:
            return

        active_obj = self.get_active_3d_object()

        if self.active_tool == "BLENDER_TUBE":
            tube_radius = max(12, int(self.pencil_size * 2.2))
            mesh = Blender3DToolbox.extrude_line_to_3d_tube(stroke, radius=tube_radius)
            if mesh:
                if active_obj:
                    active_obj.selected = False
                self.objects_3d.append(mesh)
                self.active_3d_index = len(self.objects_3d) - 1
                self.set_notification("3D Tube Created")

        elif self.active_tool == "BLENDER_CIRCLE":
            mesh = Blender3DToolbox.circle_to_3d_sphere(stroke)
            if mesh:
                if active_obj:
                    active_obj.selected = False
                self.objects_3d.append(mesh)
                self.active_3d_index = len(self.objects_3d) - 1
                self.set_notification("3D Sphere Created")

        elif self.active_tool == "DRAW_2D" or self.active_tool == "MOVE_3D":
            new_2d = Canvas2DObject(color, self.pencil_size)
            new_2d.add_stroke(stroke)
            self.objects_2d.append(new_2d)

    def draw_top_toolbar(self, frame):
        h, w, _ = frame.shape
        ui_h = 70
        cv2.rectangle(frame, (0, 0), (w, ui_h), (18, 18, 18), -1)
        cv2.line(frame, (0, ui_h), (w, ui_h), (0, 200, 220), 2)

        # 1. Tools
        btn_w = 84
        btn_h = 38
        btn_y = 16

        for i, (label, mode_id) in enumerate(self.blender_tools):
            bx = 10 + i * (btn_w + 5)
            is_active = (self.active_tool == mode_id)

            bg_col = (0, 150, 240) if is_active else (40, 40, 40)
            cv2.rectangle(frame, (bx, btn_y), (bx + btn_w, btn_y + btn_h), bg_col, -1)
            cv2.rectangle(frame, (bx, btn_y), (bx + btn_w, btn_y + btn_h), (0, 230, 255) if is_active else (70, 70, 70), 1)
            cv2.putText(frame, label, (bx + 6, btn_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Pencil Size Controller
        size_bar_x = 10 + len(self.blender_tools) * (btn_w + 5) + 6

        cv2.rectangle(frame, (size_bar_x, btn_y), (size_bar_x + 26, btn_y + btn_h), (45, 45, 45), -1)
        cv2.putText(frame, "-", (size_bar_x + 6, btn_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        preset_start_x = size_bar_x + 30
        p_w = 28
        for idx, (p_name, p_val) in enumerate(self.size_presets):
            px = preset_start_x + idx * (p_w + 3)
            is_cur = abs(self.pencil_size - p_val) <= 2
            p_bg = (0, 190, 190) if is_cur else (40, 40, 40)
            cv2.rectangle(frame, (px, btn_y), (px + p_w, btn_y + btn_h), p_bg, -1)
            cv2.putText(frame, p_name, (px + 6, btn_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (0, 0, 0) if is_cur else (255, 255, 255), 2 if is_cur else 1)

        inc_x = preset_start_x + len(self.size_presets) * (p_w + 3)
        cv2.rectangle(frame, (inc_x, btn_y), (inc_x + 26, btn_y + btn_h), (45, 45, 45), -1)
        cv2.putText(frame, "+", (inc_x + 6, btn_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        badge_x = inc_x + 32
        cv2.putText(frame, f"{self.pencil_size}px", (badge_x, btn_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

        # 3. 3D Zoom Buttons
        zoom_btn_x = badge_x + 46
        cv2.rectangle(frame, (zoom_btn_x, btn_y), (zoom_btn_x + 32, btn_y + btn_h), (50, 50, 50), -1)
        cv2.putText(frame, "Z-", (zoom_btn_x + 4, btn_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        cv2.rectangle(frame, (zoom_btn_x + 36, btn_y), (zoom_btn_x + 68, btn_y + btn_h), (50, 50, 50), -1)
        cv2.putText(frame, "Z+", (zoom_btn_x + 40, btn_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

        # 4. Action Buttons (Clean & Minimal)
        ab_x = w - 235
        ab_col = (160, 90, 10) if self.show_about else (90, 50, 15)
        cv2.rectangle(frame, (ab_x, btn_y), (ab_x + 62, btn_y + btn_h), ab_col, -1)
        cv2.putText(frame, "ABOUT", (ab_x + 7, btn_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 2)

        cx = w - 168
        cv2.rectangle(frame, (cx, btn_y), (cx + 64, btn_y + btn_h), (35, 35, 160), -1)
        cv2.putText(frame, "CLEAR", (cx + 8, btn_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)

        sx = w - 98
        cv2.rectangle(frame, (sx, btn_y), (sx + 72, btn_y + btn_h), (25, 140, 45), -1)
        cv2.putText(frame, "SAVE", (sx + 14, btn_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)

        # Subtle toast notification (only when action occurs)
        if time.time() < self.notification_timer and self.notification:
            cv2.rectangle(frame, (w // 2 - 140, 80), (w // 2 + 140, 112), (20, 20, 20), -1)
            cv2.rectangle(frame, (w // 2 - 140, 80), (w // 2 + 140, 112), (0, 255, 120), 1)
            cv2.putText(frame, self.notification, (w // 2 - 120, 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

    def draw_about_modal(self, frame):
        h, w, _ = frame.shape
        mw, mh = 840, 530
        mx1 = (w - mw) // 2
        my1 = (h - mh) // 2
        mx2 = mx1 + mw
        my2 = my1 + mh

        overlay = frame.copy()
        cv2.rectangle(overlay, (mx1, my1), (mx2, my2), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)

        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 230, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (mx1, my1), (mx2, my1 + 48), (32, 32, 40), -1)
        cv2.line(frame, (mx1, my1 + 48), (mx2, my1 + 48), (0, 230, 255), 2)

        cv2.putText(frame, "AIR CANVAS PRO - HOW TO USE GUIDE",
                    (mx1 + 25, my1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)

        close_btn = (mx2 - 45, my1 + 10, 35, 28)
        cv2.rectangle(frame, (close_btn[0], close_btn[1]),
                      (close_btn[0] + close_btn[2], close_btn[1] + close_btn[3]), (0, 0, 180), -1)
        cv2.putText(frame, "X", (close_btn[0] + 11, close_btn[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        lines = [
            ("1. MOUSE CONTROLS", (0, 255, 200)),
            ("   * Left-Click & Drag: Draw lines or letters directly on the canvas.", (220, 220, 220)),
            ("   * Left-Click on Object: Drag and move any 2D drawing or 3D model.", (220, 220, 220)),
            ("   * Right-Click & Drag: Orbit and rotate the active 3D model in all directions.", (220, 220, 220)),
            ("   * Scroll Wheel: Zoom in (scroll up) and Zoom out (scroll down).", (220, 220, 220)),
            ("   * Dustbin Delete: Drag any object into the bottom-right DUSTBIN and release.", (220, 220, 220)),
            ("", (0, 0, 0)),
            ("2. AIR HAND TRACKING", (0, 255, 200)),
            ("   * Raise Index Finger: Draw smoothly in air without breaks.", (220, 220, 220)),
            ("   * Pinch (Thumb + Index): Grab and drag any object; drop into DUSTBIN to delete.", (220, 220, 220)),
            ("   * Index + Middle Fingers: Turn hand to rotate active 3D model.", (220, 220, 220)),
            ("", (0, 0, 0)),
            ("3. 3D BLENDER TOOLS & MERGE", (0, 255, 200)),
            ("   * 3D TUBE: Draw line -> Extrudes into a 3D cylindrical pipe.", (220, 220, 220)),
            ("   * 3D SPHERE: Draw circle -> Converts into a rotating 3D UV sphere.", (220, 220, 220)),
            ("   * AUTO-MERGE: Bring two 3D models or 2D drawings close together to merge them!", (220, 220, 220)),
            ("", (0, 0, 0)),
            ("4. SIZES, SAVE & CLEAR", (0, 255, 200)),
            ("   * Tap or Click [-] / [+] or S, M, L, XL to change pencil thickness.", (220, 220, 220)),
            ("   * Tap or Click SAVE to export high-res PNG to 'saved_canvas/' folder.", (220, 220, 220)),
            ("   * Tap or Click CLEAR to wipe all objects on canvas." , (220, 220, 220))
        ]

        text_y = my1 + 72
        for heading, col in lines:
            if heading:
                cv2.putText(frame, heading, (mx1 + 30, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1 if col != (0, 255, 200) else 2, cv2.LINE_AA)
            text_y += 20

        cv2.putText(frame, "Click [X], click [ABOUT], or press 'H' / 'Esc' to dismiss.",
                    (mx1 + 220, my2 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

    def handle_toolbar_click(self, x, y, canvas_img):
        w = 1280
        btn_w = 84
        btn_h = 38
        btn_y = 16

        if self.show_about:
            mw, mh = 840, 530
            mx2 = (w - mw) // 2 + mw
            my1 = (720 - mh) // 2
            if (mx2 - 45) <= x <= (mx2 - 10) and (my1 + 10) <= y <= (my1 + 40):
                self.show_about = False
                self.set_notification("Guide Closed")
                return True

        if y > 70:
            return False

        ab_x = w - 235
        if ab_x <= x <= ab_x + 62 and btn_y <= y <= btn_y + btn_h:
            self.show_about = not self.show_about
            self.set_notification("Guide Toggled" if self.show_about else "Guide Closed")
            return True

        for i, (label, mode_id) in enumerate(self.blender_tools):
            bx = 10 + i * (btn_w + 5)
            if bx <= x <= bx + btn_w and btn_y <= y <= btn_y + btn_h:
                active_obj = self.get_active_3d_object()
                if active_obj:
                    active_obj.selected = False

                if mode_id == "SPAWN_CUBE":
                    self.objects_3d.append(Blender3DToolbox.create_cube())
                    self.active_3d_index = len(self.objects_3d) - 1
                    self.set_notification("Created 3D Cube")
                elif mode_id == "SPAWN_CYLINDER":
                    self.objects_3d.append(Blender3DToolbox.create_cylinder())
                    self.active_3d_index = len(self.objects_3d) - 1
                    self.set_notification("Created 3D Cylinder")
                elif mode_id == "SPAWN_PYRAMID":
                    self.objects_3d.append(Blender3DToolbox.create_pyramid())
                    self.active_3d_index = len(self.objects_3d) - 1
                    self.set_notification("Created 3D Pyramid")
                elif mode_id == "SPAWN_TORUS":
                    self.objects_3d.append(Blender3DToolbox.create_torus())
                    self.active_3d_index = len(self.objects_3d) - 1
                    self.set_notification("Created 3D Torus")
                else:
                    self.active_tool = mode_id
                    self.set_notification(f"Tool: {label}")
                return True

        size_bar_x = 10 + len(self.blender_tools) * (btn_w + 5) + 6

        if size_bar_x <= x <= size_bar_x + 26 and btn_y <= y <= btn_y + btn_h:
            self.set_pencil_size(self.pencil_size - 2)
            return True

        preset_start_x = size_bar_x + 30
        p_w = 28
        for idx, (p_name, p_val) in enumerate(self.size_presets):
            px = preset_start_x + idx * (p_w + 3)
            if px <= x <= px + p_w and btn_y <= y <= btn_y + btn_h:
                self.set_pencil_size(p_val)
                return True

        inc_x = preset_start_x + len(self.size_presets) * (p_w + 3)
        if inc_x <= x <= inc_x + 26 and btn_y <= y <= btn_y + btn_h:
            self.set_pencil_size(self.pencil_size + 2)
            return True

        badge_x = inc_x + 32
        zoom_btn_x = badge_x + 46
        act_mesh = self.get_active_3d_object()

        if zoom_btn_x <= x <= zoom_btn_x + 32 and btn_y <= y <= btn_y + btn_h:
            if act_mesh:
                act_mesh.zoom(0.85)
                self.set_notification(f"Zoom: {int(act_mesh.scale * 100)}%")
            return True

        if zoom_btn_x + 36 <= x <= zoom_btn_x + 68 and btn_y <= y <= btn_y + btn_h:
            if act_mesh:
                act_mesh.zoom(1.15)
                self.set_notification(f"Zoom: {int(act_mesh.scale * 100)}%")
            return True

        cx = w - 168
        if cx <= x <= cx + 64 and btn_y <= y <= btn_y + btn_h:
            self.objects_2d.clear()
            self.objects_3d.clear()
            self.set_notification("Cleared All Objects")
            return True

        sx = w - 98
        if sx <= x <= sx + 72 and btn_y <= y <= btn_y + btn_h:
            os.makedirs("saved_canvas", exist_ok=True)
            filename = datetime.now().strftime("saved_canvas/canvas_3d_%Y%m%d_%H%M%S.png")
            if canvas_img is not None:
                cv2.imwrite(filename, canvas_img)
            self.set_notification(f"SAVED: {os.path.basename(filename)}", duration=3.0)
            return True

        return False

    def set_window_icon(self):
        try:
            import ctypes
            ico_path = os.path.abspath("app_logo.ico")
            if os.path.exists(ico_path):
                hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
                if hwnd:
                    IMAGE_ICON = 1
                    LR_LOADFROMFILE = 0x00000010
                    LR_DEFAULTSIZE = 0x00000040
                    hicon = ctypes.windll.user32.LoadImageW(
                        0, ico_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE
                    )
                    if hicon:
                        WM_SETICON = 0x0080
                        ICON_SMALL = 0
                        ICON_BIG = 1
                        ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
                        ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
        except Exception:
            pass

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self.set_window_icon()
        cv2.setMouseCallback(self.window_name, self.on_mouse_event)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            canvas_img = np.zeros((h, w, 3), dtype=np.uint8)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            detected_hands = {}
            self.dustbin.is_open = False

            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_handedness in enumerate(results.multi_handedness):
                    orig_label = hand_handedness.classification[0].label
                    label = "Right" if orig_label == "Left" else "Left"

                    hand_lms = results.multi_hand_landmarks[idx]
                    analysis = self.analyze_hand(hand_lms.landmark, w, h)
                    detected_hands[label] = analysis

                    itip = analysis["index_tip"]
                    col = (0, 255, 255) if label == "Right" else (255, 140, 0)

                    # Subtle clean cursor dot (no text clutter)
                    cv2.circle(frame, itip, max(3, self.pencil_size // 2), col, -1, cv2.LINE_AA)
                    cv2.circle(frame, itip, max(5, self.pencil_size // 2 + 2), (255, 255, 255), 1, cv2.LINE_AA)

            active_mesh = self.get_active_3d_object()

            # -------------------------------------------------------------
            # HAND TRACKING PIPELINE
            # -------------------------------------------------------------
            for label in ["Right", "Left"]:
                hstate = self.hand_states[label]

                if label in detected_hands:
                    data = detected_hands[label]
                    ix, iy = data["index_tip"]
                    px, py = data["pinch_center"]
                    pinch_ratio = data["pinch_ratio"]
                    index_ext = data["index_ext"]
                    middle_ext = data["middle_ext"]

                    if hstate.smoothed_pos is None:
                        hstate.smoothed_pos = (float(ix), float(iy))
                    else:
                        alpha = 0.70
                        hstate.smoothed_pos = (
                            alpha * ix + (1 - alpha) * hstate.smoothed_pos[0],
                            alpha * iy + (1 - alpha) * hstate.smoothed_pos[1]
                        )
                    six, siy = int(hstate.smoothed_pos[0]), int(hstate.smoothed_pos[1])

                    if self.dustbin.contains(px, py) or self.dustbin.contains(six, siy):
                        self.dustbin.is_open = True

                    # 1. UI Toolbar Tap
                    if (iy <= 70 or self.show_about) and index_ext and not middle_ext and pinch_ratio >= 0.35:
                        self.handle_toolbar_click(ix, iy, canvas_img)
                        if len(hstate.current_stroke) > 0:
                            self.commit_stroke(hstate.current_stroke, hstate.color)
                            hstate.current_stroke = []

                    # 2. 3D ROTATION: Index + Middle extended
                    elif index_ext and middle_ext and active_mesh and not hstate.is_grabbing and not self.show_about:
                        if len(hstate.current_stroke) > 0:
                            self.commit_stroke(hstate.current_stroke, hstate.color)
                            hstate.current_stroke = []

                        if hstate.prev_pinch_pos is not None:
                            dx = six - hstate.prev_pinch_pos[0]
                            dy = siy - hstate.prev_pinch_pos[1]
                            active_mesh.rotate(dyaw=dx * 0.015, dpitch=-dy * 0.015)
                        hstate.prev_pinch_pos = (six, siy)
                        cv2.circle(frame, (six, siy), 12, (0, 255, 180), 1, cv2.LINE_AA)

                    # 3. PINCH TO GRAB, DRAG & DROP INTO DUSTBIN
                    elif (pinch_ratio < 0.32 or hstate.is_grabbing) and not self.show_about:
                        if not hstate.is_grabbing:
                            hstate.is_grabbing = True
                            hstate.prev_pinch_pos = (px, py)
                            if len(hstate.current_stroke) > 0:
                                self.commit_stroke(hstate.current_stroke, hstate.color)
                                hstate.current_stroke = []

                            if active_mesh:
                                cx_3d, cy_3d = active_mesh.get_screen_center(w, h)
                                if np.hypot(px - cx_3d, py - cy_3d) < 140:
                                    hstate.grabbed_3d_obj = active_mesh

                            if hstate.grabbed_3d_obj is None:
                                for obj_2d in reversed(self.objects_2d):
                                    if obj_2d.distance_to_point(px, py) < 80:
                                        hstate.grabbed_2d_obj = obj_2d
                                        break

                        cv2.circle(frame, (px, py), 10, (0, 255, 255), -1)

                        if self.dustbin.contains(px, py):
                            self.dustbin.is_open = True

                        if hstate.prev_pinch_pos is not None:
                            dx = px - hstate.prev_pinch_pos[0]
                            dy = py - hstate.prev_pinch_pos[1]

                            if hstate.grabbed_3d_obj:
                                hstate.grabbed_3d_obj.translate(dx=dx * 0.8, dy=dy * 0.8)
                            elif hstate.grabbed_2d_obj:
                                hstate.grabbed_2d_obj.translate(dx, dy)

                        hstate.prev_pinch_pos = (px, py)

                        if pinch_ratio >= 0.45:
                            if self.dustbin.contains(px, py):
                                if hstate.grabbed_3d_obj in self.objects_3d:
                                    self.objects_3d.remove(hstate.grabbed_3d_obj)
                                    self.active_3d_index = max(0, len(self.objects_3d) - 1)
                                    self.dustbin.trigger_delete_effect()
                                    self.set_notification(f"Deleted: {hstate.grabbed_3d_obj.name}")

                                if hstate.grabbed_2d_obj in self.objects_2d:
                                    self.objects_2d.remove(hstate.grabbed_2d_obj)
                                    self.dustbin.trigger_delete_effect()
                                    self.set_notification("Deleted: 2D Object")
                            else:
                                if hstate.grabbed_3d_obj:
                                    self.check_and_merge_3d_objects(hstate.grabbed_3d_obj)
                                elif hstate.grabbed_2d_obj:
                                    self.check_and_merge_2d_objects(hstate.grabbed_2d_obj)

                            hstate.is_grabbing = False
                            hstate.grabbed_3d_obj = None
                            hstate.grabbed_2d_obj = None
                            hstate.prev_pinch_pos = None

                    # 4. CONTINUOUS UNBROKEN DRAWING
                    else:
                        hstate.prev_pinch_pos = None
                        hstate.is_grabbing = False
                        hstate.grabbed_3d_obj = None
                        hstate.grabbed_2d_obj = None

                        if index_ext and not middle_ext and not self.show_about:
                            hstate.stroke_pause_counter = 0
                            if len(hstate.current_stroke) == 0:
                                hstate.current_stroke.append((six, siy))
                            else:
                                last_pt = hstate.current_stroke[-1]
                                dist = np.hypot(six - last_pt[0], siy - last_pt[1])
                                if dist > 2:
                                    hstate.current_stroke.append((six, siy))

                            cv2.circle(frame, (six, siy), max(3, self.pencil_size // 2), hstate.color, -1)
                        else:
                            if len(hstate.current_stroke) > 0:
                                hstate.stroke_pause_counter += 1
                                if hstate.stroke_pause_counter > 6:
                                    self.commit_stroke(hstate.current_stroke, hstate.color)
                                    hstate.current_stroke = []
                                    hstate.stroke_pause_counter = 0

                else:
                    if len(hstate.current_stroke) > 0:
                        hstate.stroke_pause_counter += 1
                        if hstate.stroke_pause_counter > 6:
                            self.commit_stroke(hstate.current_stroke, hstate.color)
                            hstate.current_stroke = []
                            hstate.stroke_pause_counter = 0
                    hstate.is_grabbing = False
                    hstate.grabbed_3d_obj = None
                    hstate.grabbed_2d_obj = None
                    hstate.prev_pinch_pos = None

            # Render 2D Canvas Objects
            for obj in self.objects_2d:
                obj.draw(canvas_img)

            # In-flight hand strokes
            for hstate in self.hand_states.values():
                if len(hstate.current_stroke) > 1:
                    for i in range(1, len(hstate.current_stroke)):
                        cv2.line(canvas_img, hstate.current_stroke[i - 1], hstate.current_stroke[i],
                                 hstate.color, self.pencil_size, cv2.LINE_AA)

            # In-flight mouse stroke
            if len(self.mouse_stroke) > 1:
                for i in range(1, len(self.mouse_stroke)):
                    cv2.line(canvas_img, self.mouse_stroke[i - 1], self.mouse_stroke[i],
                             (0, 255, 255), self.pencil_size, cv2.LINE_AA)

            # Render 3D Meshes
            for i, mesh in enumerate(self.objects_3d):
                mesh.selected = (i == self.active_3d_index)
                mesh.render(canvas_img)

            gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            _, inv_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
            inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)

            combined = cv2.bitwise_and(frame, inv_mask)
            combined = cv2.bitwise_or(combined, canvas_img)

            self.draw_top_toolbar(combined)
            self.dustbin.draw(combined)

            if self.show_about:
                self.draw_about_modal(combined)

            display_view = canvas_img if self.pure_canvas else combined
            cv2.imshow(self.window_name, display_view)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                if self.show_about:
                    self.show_about = False
                else:
                    break
            elif key == ord('h') or key == ord('?'):
                self.show_about = not self.show_about
            elif key == ord('c'):
                self.objects_2d.clear()
                self.objects_3d.clear()
                self.set_notification("Cleared All Objects")
            elif key == ord('s'):
                os.makedirs("saved_canvas", exist_ok=True)
                filename = datetime.now().strftime("saved_canvas/canvas_3d_%Y%m%d_%H%M%S.png")
                cv2.imwrite(filename, canvas_img)
                self.set_notification(f"SAVED: {os.path.basename(filename)}")
            elif key == ord('1'):
                self.set_pencil_size(3)
            elif key == ord('2'):
                self.set_pencil_size(7)
            elif key == ord('3'):
                self.set_pencil_size(14)
            elif key == ord('4'):
                self.set_pencil_size(26)
            elif key == ord('+') or key == ord('='):
                self.set_pencil_size(self.pencil_size + 2)
            elif key == ord('-') or key == ord('_'):
                self.set_pencil_size(self.pencil_size - 2)
            elif key == ord('w'):
                if active_mesh:
                    active_mesh.wireframe_only = not active_mesh.wireframe_only
            elif key == ord('['):
                if active_mesh:
                    active_mesh.zoom(0.85)
            elif key == ord(']'):
                if active_mesh:
                    active_mesh.zoom(1.15)
            elif key == 9:
                self.pure_canvas = not self.pure_canvas

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AirCanvasBlenderApp()
    app.run()
