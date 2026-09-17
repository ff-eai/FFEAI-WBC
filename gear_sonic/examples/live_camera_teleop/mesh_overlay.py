# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live in-camera SOMA mesh overlay for the webcam bridge.

Renders the SOMA body mesh over the incoming camera frames in a *separate
process* so the GEM-X -> SONIC stream loop is never blocked by rendering.
Shading and compositing mirror GEM-X's offline ``render_incam`` in
``scripts/demo/demo_soma_onnx.py`` (Open3D offscreen renderer, soft-shadow
lighting, depth-mask alpha composite), so the window looks like the
``*_1_incam.mp4`` that demo writes.

Hand-over is a depth-1 queue with drop-on-full semantics: when the renderer is
slower than the stream it skips frames instead of building a backlog, and the
stream loop only ever pays for one non-blocking ``put``. The vertices come from
the same per-frame SOMA decode that is streamed to SONIC, posed in the camera
frame, so the window shows exactly the body the robot is asked to follow.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

import numpy as np

MESH_COLOR = (0.4, 0.8, 0.4)  # same green as the offline in-camera render


def _render_worker(q, stop_event, error_event, W, H, K, faces, scale, show, save_path, fps, window_name):
    """Child process entry: report any failure, never die silently."""
    try:
        _render_loop(q, stop_event, W, H, K, faces, scale, show, save_path, fps, window_name)
    except Exception:
        import sys
        import traceback

        print("[mesh_overlay] render process failed:", file=sys.stderr)
        traceback.print_exc()
        error_event.set()


def _render_loop(q, stop_event, W, H, K, faces, scale, show, save_path, fps, window_name):
    """Own an Open3D offscreen renderer and composite frames until a None arrives."""
    import cv2
    import open3d as o3d

    rw, rh = int(round(W * scale)), int(round(H * scale))
    Ks = np.asarray(K, dtype=np.float64).copy()
    Ks[:2] *= scale  # fx, fy, cx, cy scale with the image

    rendering = o3d.visualization.rendering
    renderer = rendering.OffscreenRenderer(rw, rh)
    renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])
    renderer.scene.set_lighting(renderer.scene.LightingProfile.SOFT_SHADOWS, np.array([0.0, 0.7, 0.7]))
    renderer.scene.camera.set_projection(Ks, 0.01, 100.0, float(rw), float(rh))
    # Camera at the origin looking down +Z with image y pointing down (OpenCV frame).
    renderer.scene.camera.look_at([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0])

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.9, 0.9, 0.9, 1.0]
    triangles = o3d.utility.Vector3iVector(np.ascontiguousarray(faces, dtype=np.int32))
    vcolors = o3d.utility.Vector3dVector(np.tile(np.asarray(MESH_COLOR, dtype=np.float64), (int(faces.max()) + 1, 1)))

    writer = None
    if save_path:
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (rw, rh))

    have_geom = False
    n_rendered = 0
    t_fps = time.time()
    render_fps = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if show and (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    stop_event.set()
                continue
            if item is None:
                break
            frame_bgr, verts, meta = item

            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts.astype(np.float64)), triangles)
            mesh.compute_vertex_normals()
            mesh.vertex_colors = vcolors
            if have_geom:
                renderer.scene.remove_geometry("body")
            renderer.scene.add_geometry("body", mesh, mat)
            have_geom = True

            rendered = np.asarray(renderer.render_to_image())  # RGB uint8
            depth = np.asarray(renderer.render_to_depth_image())  # 1.0 where nothing was hit
            mask = cv2.GaussianBlur((depth < 1.0).astype(np.float32), (5, 5), sigmaX=1.0)[..., None]
            comp = rendered[..., ::-1].astype(np.float32) * mask + frame_bgr.astype(np.float32) * (1.0 - mask)
            comp = comp.clip(0, 255).astype(np.uint8)

            n_rendered += 1
            if n_rendered % 10 == 0:
                now = time.time()
                render_fps = 10.0 / max(1e-6, now - t_fps)
                t_fps = now
            label = (
                f"stream frame {meta.get('frame', 0)} @ {meta.get('stream_fps', 0.0):4.1f} fps | "
                f"overlay {render_fps:4.1f} fps | dropped {meta.get('dropped', 0)}"
            )
            cv2.putText(comp, label, (8, 22), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(comp, label, (8, 22), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(comp)
            if show:
                cv2.imshow(window_name, comp)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    stop_event.set()
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


class MeshOverlay:
    """Parent-side handle: spawns the render process and feeds it frames.

    Parameters
    ----------
    W, H : int
        Camera frame size (pixels).
    K : (3, 3) array
        Full-image intrinsics used by GEM-X for this stream (``estimate_K``).
    faces : (F, 3) int array
        SOMA mesh triangles (``SomaLayer.faces``).
    scale : float
        Render/display scale (0.5 renders at half resolution: ~4x cheaper).
    show : bool
        Open a cv2 window (needs a display). ``q`` or ESC in it requests a stop.
    save_path : str | None
        Optional mp4 path for the overlay video.
    """

    def __init__(self, W, H, K, faces, scale=1.0, show=True, save_path=None, fps=30.0, window_name="GEM mesh overlay"):
        if not show and not save_path:
            raise ValueError("MeshOverlay needs show=True and/or a save_path")
        self.scale = float(scale)
        self.W, self.H = int(W), int(H)
        self.dropped = 0
        self.submitted = 0
        ctx = mp.get_context("spawn")  # never fork a CUDA/warp-initialised parent
        self.q = ctx.Queue(maxsize=1)
        # The parent must never block on this queue: if the child dies with a frame
        # still buffered, a normal exit would hang flushing it to a dead reader.
        self.q.cancel_join_thread()
        self.stop_event = ctx.Event()
        self.error_event = ctx.Event()
        self._warned_dead = False
        self.proc = ctx.Process(
            target=_render_worker,
            args=(
                self.q,
                self.stop_event,
                self.error_event,
                self.W,
                self.H,
                np.asarray(K, dtype=np.float64),
                np.asarray(faces, dtype=np.int32),
                self.scale,
                bool(show),
                save_path,
                float(fps),
                window_name,
            ),
            daemon=True,
        )
        self.proc.start()

    def submit(self, frame_bgr, verts, frame_idx=0, stream_fps=0.0):
        """Non-blocking hand-over of one frame; drops it if the renderer is busy.

        ``frame_bgr`` is copied (or downscaled), so the caller may keep drawing
        on its own buffer afterwards.
        """
        import cv2

        if not self.alive():
            if not self._warned_dead:
                self._warned_dead = True
                print(
                    "\n[mesh_overlay] render process is gone (see traceback above); "
                    "continuing to stream without the overlay.",
                    flush=True,
                )
            return
        self.submitted += 1
        if self.scale != 1.0:
            rw, rh = int(round(self.W * self.scale)), int(round(self.H * self.scale))
            frame = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_AREA)
        else:
            frame = frame_bgr.copy()
        meta = {"frame": int(frame_idx), "stream_fps": float(stream_fps), "dropped": self.dropped}
        try:
            self.q.put_nowait((frame, np.ascontiguousarray(verts, dtype=np.float32), meta))
        except queue.Full:
            self.dropped += 1

    def alive(self):
        return self.proc.is_alive() and not self.error_event.is_set()

    def stop_requested(self):
        """True when the user pressed q/ESC in the overlay window (never on child failure)."""
        return self.stop_event.is_set()

    def close(self, timeout=5.0):
        if self.proc.is_alive():
            try:
                self.q.put(None, timeout=1.0)
            except Exception:
                pass
            self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(1.0)
        self.q.close()
