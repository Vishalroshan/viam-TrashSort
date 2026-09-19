"""
table_reconstruct.py — orbit scan + Poisson reconstruction + render.

Designed to be called from cleanup_ui.py's Robot class, which keeps its own
open machine connection. Everything here accepts `machine` as a parameter
rather than opening a new connection.

Public API used by cleanup_ui.py:

    clouds = await orbit_scan(machine, center, n_poses, radius, height,
                              on_progress=None)
        Drive the arm on N viewpoints around `center` (world-frame mm) and
        return a list of world-frame point-cloud arrays. on_progress(i, n)
        fires after each capture.

    mesh = await asyncio.to_thread(poisson_reconstruct, clouds)
        CPU-heavy Poisson step — runs off the event loop so the UI stays live.
        Returns an open3d TriangleMesh in world-frame mm.

    jpeg_bytes = render_mesh(mesh, grasp_points=None)
        Render the mesh to JPEG bytes, optionally marking grasp points as red
        spheres with orange approach arrows. Primary renderer is Open3D's
        OffscreenRenderer; falls back to matplotlib when EGL is unavailable.

    ReconstructResult
        Dataclass returned from the high-level reconstruct() helper.

Setup:
    .venv/bin/pip install viam-sdk anthropic python-dotenv numpy pillow
    .venv/bin/pip install open3d
"""

from __future__ import annotations

import asyncio
import io
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

import table_scan as ts
import table_cleanup as tc
from table_orbit_scan import (
    orbit_poses,
    transit_pose,
    deproject_frame,
    voxel_downsample,
    save_ply,
    ORBIT_RADIUS_MM,
    ORBIT_HEIGHT_MM,
    TRANSIT_HEIGHT_MM,
    SETTLE_S,
    VOXEL_MM,
    _move,
)

# ---------------------------------------------------------------------------
# Reconstruction parameters
# ---------------------------------------------------------------------------

DEFAULT_N_POSES = 12

POISSON_DEPTH = 8
POISSON_TRIM_QUANTILE = 0.05

SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO = 2.0

NORMAL_RADIUS_MM = 15.0
NORMAL_MAX_NN = 30

GRASP_SPHERE_RADIUS_MM = 8.0

ARROW_CYLINDER_RADIUS = 3.0
ARROW_CONE_RADIUS = 6.0
ARROW_CYLINDER_HEIGHT = 40.0
ARROW_CONE_HEIGHT = 15.0
ARROW_TOTAL_HEIGHT = ARROW_CYLINDER_HEIGHT + ARROW_CONE_HEIGHT

RENDER_WIDTH = 900
RENDER_HEIGHT = 700
RENDER_FOV = 55.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ReconstructResult:
    mesh_ply: str
    cloud_ply: str
    render_jpeg: bytes
    n_views: int
    n_points: int
    n_triangles: int
    grasp_points: list[tuple] = field(default_factory=list)  # (x,y,z,theta) per object


# ---------------------------------------------------------------------------
# Orbit scan
# ---------------------------------------------------------------------------

async def orbit_scan(
    machine,
    center: list[float],
    n_poses: int = DEFAULT_N_POSES,
    radius: float = ORBIT_RADIUS_MM,
    height: float = ORBIT_HEIGHT_MM,
    on_progress: Optional[Callable[[int, int], None]] = None,
    return_pose=None,
) -> list[np.ndarray]:
    """
    Drive the arm around center and collect depth clouds.

    center is a world-frame [x, y, z] mm point — typically the mean centroid
    of all located objects on the table, or a fixed workspace midpoint.

    Afterwards the arm goes to return_pose (an arm Pose, e.g. where it was
    before the orbit), or to tc.HOME_POSE when that is None.

    Uses the already-open machine connection. Returns a list of (N, 3) float32
    arrays in world-frame mm.
    """
    from viam.components.arm import Arm
    from viam.services.motion import Motion
    from viam.proto.common import Pose, PoseInFrame

    arm = await ts.resolve(machine, Arm, tc.ARM_NAME)
    motion = await ts.resolve(machine, Motion, tc.MOTION_NAME)
    poses = orbit_poses(center, n=n_poses, radius=radius, height=height)
    clouds: list[np.ndarray] = []

    for i, pose in enumerate(poses):
        az = math.degrees(math.atan2(pose.y - center[1], pose.x - center[0]))
        print(f"  [orbit {i+1}/{n_poses}] az={az:+.0f}°  "
              f"pos=({pose.x:.0f}, {pose.y:.0f}, {pose.z:.0f})")

        tr = transit_pose(pose)
        if not await _move(motion, tr, f"transit {i+1}"):
            print(f"    transit failed — skipping", file=sys.stderr)
            if on_progress:
                on_progress(i + 1, n_poses)
            continue

        if not await _move(motion, pose, f"orbit {i+1}"):
            print(f"    orbit move failed — skipping", file=sys.stderr)
            if on_progress:
                on_progress(i + 1, n_poses)
            continue

        await asyncio.sleep(SETTLE_S)

        try:
            frame = await ts.capture(machine)
            pts = deproject_frame(frame)
            if len(pts) > 0:
                clouds.append(pts)
                print(f"    {len(pts):,} world-frame points")
            else:
                print(f"    no depth this viewpoint", file=sys.stderr)
        except Exception as e:
            print(f"    capture failed: {e}", file=sys.stderr)

        if on_progress:
            on_progress(i + 1, n_poses)

    # Return to where the orbit started (or home), confirming the arm got there.
    end = return_pose or tc.HOME_POSE
    tr_end = Pose(x=end.x, y=end.y, z=TRANSIT_HEIGHT_MM,
                  o_x=0, o_y=0, o_z=-1, theta=0)
    await _move(motion, tr_end, "transit back")
    rig = tc.Rig(motion=motion, arm=arm, gripper=None)
    if not await tc.return_to_start(rig, end, None):
        print("  !! could not get back to the starting position; the arm is "
              "stopped where it is", file=sys.stderr)
    return clouds


# ---------------------------------------------------------------------------
# Poisson reconstruction
# ---------------------------------------------------------------------------

def poisson_reconstruct(
    clouds: list[np.ndarray],
) -> tuple["open3d.geometry.TriangleMesh", int]:
    """
    Merge all clouds and run Poisson reconstruction on the full scene.

    No per-object cropping — the entire merged cloud goes to Poisson.
    Returns (mesh, n_points_used).

    CPU-heavy; always call via asyncio.to_thread().
    """
    import open3d as o3d

    if not clouds:
        raise RuntimeError("No depth clouds to reconstruct from.")

    merged = np.vstack(clouds)
    if VOXEL_MM > 0:
        merged = voxel_downsample(merged, VOXEL_MM)

    print(f"  Merged cloud: {len(merged):,} points")

    if len(merged) < 100:
        raise RuntimeError(
            f"Only {len(merged)} points in cloud — too sparse for Poisson. "
            "Try more orbit poses."
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged.astype(np.float64))

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
    print(f"  After outlier removal: {len(pcd.points):,} points")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS_MM, max_nn=NORMAL_MAX_NN))
    pcd.orient_normals_consistent_tangent_plane(k=20)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH)
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.quantile(densities, POISSON_TRIM_QUANTILE))
    mesh.compute_vertex_normals()

    print(f"  Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    return mesh, len(merged)


# ---------------------------------------------------------------------------
# Render: mesh + optional grasp-point markers → JPEG bytes
# ---------------------------------------------------------------------------

def render_mesh(
    mesh: "open3d.geometry.TriangleMesh",
    grasp_points: Optional[list[tuple]] = None,
) -> bytes:
    """
    Render the mesh to JPEG bytes.

    grasp_points is an optional list of (x, y, z, theta_deg) tuples — one
    per object that the cleanup code would attempt to pick. Each gets a red
    sphere and orange approach arrow in the render.
    """
    try:
        return _render_open3d(mesh, grasp_points or [])
    except Exception as e:
        print(f"  Open3D render failed ({e}); using matplotlib fallback", file=sys.stderr)
        return _render_matplotlib(mesh, grasp_points or [])


def _build_grasp_geometries(grasp_points: list[tuple]) -> list:
    """Build Open3D sphere + arrow for each grasp point."""
    import open3d as o3d

    geoms = []
    for gx, gy, gz, theta_deg in grasp_points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=GRASP_SPHERE_RADIUS_MM)
        sphere.translate(np.array([gx, gy, gz]))
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.95, 0.15, 0.15])

        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=ARROW_CYLINDER_RADIUS,
            cone_radius=ARROW_CONE_RADIUS,
            cylinder_height=ARROW_CYLINDER_HEIGHT,
            cone_height=ARROW_CONE_HEIGHT,
        )
        R_flip = arrow.get_rotation_matrix_from_xyz((math.pi, 0, 0))
        arrow.rotate(R_flip, center=[0, 0, 0])
        R_roll = arrow.get_rotation_matrix_from_xyz(
            (0, 0, math.radians(theta_deg)))
        arrow.rotate(R_roll, center=[0, 0, 0])
        arrow.translate(np.array([gx, gy, gz + ARROW_TOTAL_HEIGHT]))
        arrow.compute_vertex_normals()
        arrow.paint_uniform_color([1.0, 0.55, 0.0])

        geoms.extend([sphere, arrow])
    return geoms


def _render_open3d(
    mesh: "open3d.geometry.TriangleMesh",
    grasp_points: list[tuple],
) -> bytes:
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    rend = rendering.OffscreenRenderer(RENDER_WIDTH, RENDER_HEIGHT)
    rend.scene.set_background([0.12, 0.12, 0.18, 1.0])

    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultLit"
    mat_mesh.base_color = [0.65, 0.68, 0.80, 1.0]
    rend.scene.add_geometry("mesh", mesh, mat_mesh)

    for i, geom in enumerate(_build_grasp_geometries(grasp_points)):
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        is_sphere = i % 2 == 0
        mat.base_color = ([0.95, 0.15, 0.15, 1.0] if is_sphere
                          else [1.0, 0.55, 0.0, 1.0])
        rend.scene.add_geometry(f"grasp_{i}", geom, mat)

    bounds = mesh.get_axis_aligned_bounding_box()
    centre = np.asarray(bounds.get_center())
    extent = bounds.get_max_extent()
    eye = centre + np.array([0.0, -extent * 0.85, extent * 0.65])
    rend.setup_camera(RENDER_FOV, centre.tolist(), eye.tolist(), [0.0, 0.0, 1.0])

    rend.scene.scene.set_sun_light([0.6, -1.0, -0.5], [1.0, 1.0, 1.0], 80000)
    rend.scene.scene.enable_sun_light(True)
    rend.scene.scene.set_ambient_light([0.3, 0.3, 0.35], 15000)

    img = rend.render_to_image()
    buf = io.BytesIO()
    from PIL import Image as PILImage
    PILImage.fromarray(np.asarray(img)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _render_matplotlib(
    mesh: "open3d.geometry.TriangleMesh",
    grasp_points: list[tuple],
) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)

    fig = plt.figure(figsize=(RENDER_WIDTH / 100, RENDER_HEIGHT / 100),
                     facecolor="#1a1a2e")
    ax = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")
    ax.set_box_aspect([1, 1, 1])

    if len(tris) > 8000:
        idx = np.random.default_rng(0).choice(len(tris), 8000, replace=False)
        tris_draw = tris[idx]
    else:
        tris_draw = tris

    poly = Poly3DCollection(verts[tris_draw], alpha=0.65, linewidths=0)
    poly.set_facecolor([0.40, 0.45, 0.70])
    poly.set_edgecolor("none")
    ax.add_collection3d(poly)

    for gx, gy, gz, theta_deg in grasp_points:
        ax.scatter([gx], [gy], [gz], color="#f03030", s=160, zorder=10)
        ax.quiver(gx, gy, gz + 45, 0, 0, -45,
                  color="#ff8c00", arrow_length_ratio=0.35, linewidth=2.0)
        rad = math.radians(theta_deg)
        half = 18.0
        dx, dy = half * math.cos(rad), half * math.sin(rad)
        ax.plot([gx - dx, gx + dx], [gy - dy, gy + dy], [gz, gz],
                color="#ff8c00", linewidth=2.5)

    ax.tick_params(colors="#aaa", labelsize=7)
    for attr in ("xaxis", "yaxis", "zaxis"):
        getattr(ax, attr).label.set_color("#aaa")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")

    for axis, col in zip(["x", "y", "z"],
                         [verts[:, 0], verts[:, 1], verts[:, 2]]):
        getattr(ax, f"set_{axis}lim")(col.min() - 10, col.max() + 10)

    ax.view_init(elev=28, azim=-55)

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", dpi=100, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------

async def reconstruct(
    machine,
    objects: list[dict],
    n_poses: int = DEFAULT_N_POSES,
    radius: float = ORBIT_RADIUS_MM,
    height: float = ORBIT_HEIGHT_MM,
    on_progress: Optional[Callable[[int, int], None]] = None,
    return_pose=None,
) -> ReconstructResult:
    """
    Full pipeline: compute scene center → orbit → fuse → Poisson → render.

    objects is the current scan's object list (used to compute orbit center
    and to annotate grasp points on the render). The arm orbits the mean
    centroid of all located objects, so the whole scene stays in frame.
    """
    import open3d as o3d

    located = [o for o in objects
               if o.get("position") and "centroid" in o["position"]]
    if not located:
        raise RuntimeError("No located objects to compute scene center from.")

    center = np.mean([o["position"]["centroid"] for o in located],
                     axis=0).tolist()
    print(f"  Scene center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) mm "
          f"(mean of {len(located)} located objects)")

    clouds = await orbit_scan(machine, center, n_poses, radius, height, on_progress,
                              return_pose)

    if not clouds:
        raise RuntimeError("No depth data captured.")

    mesh, n_pts = await asyncio.to_thread(poisson_reconstruct, clouds)

    # Collect grasp points for all pickable objects.
    grasp_points = []
    for o in objects:
        if tc.gate(o["position"]) is None:
            gx, gy, gz = tc.grasp_point(o["position"])
            theta = tc.grasp_theta(o["position"])
            grasp_points.append((gx, gy, gz, theta))

    render_jpeg = await asyncio.to_thread(render_mesh, mesh, grasp_points)

    merged = np.vstack(clouds)
    cloud_path = "orbit_cloud.ply"
    save_ply(voxel_downsample(merged, VOXEL_MM) if VOXEL_MM > 0 else merged,
             cloud_path)

    mesh_path = "orbit_mesh.ply"
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    return ReconstructResult(
        mesh_ply=mesh_path,
        cloud_ply=cloud_path,
        render_jpeg=render_jpeg,
        n_views=len(clouds),
        n_points=n_pts,
        n_triangles=len(mesh.triangles),
        grasp_points=grasp_points,
    )
