"""
table_orbit_scan.py — orbit the arm around a detected object and build a
dense 3D point cloud by fusing depth frames from multiple viewpoints.

How it works
------------
The world-frame centroid from a prior scan (table_objects.json) anchors a
circular orbit path. The arm visits N_POSES evenly-spaced positions around
that centroid at a fixed radius and height, capturing colour + depth at each
stop. Because ts.capture() calls _cam_to_world() at the time of capture, the
transform from camera pixels to world-frame millimetres is always exact — the
arm can be anywhere and the depth cloud lands in the right place.

Each depth frame is deprojected and merged into a single world-frame point
cloud that is denser and more complete than any single overhead view.

The orbit orientation vector points the arm's tool axis from each orbit
position toward the centroid: the camera looks at the object from the side,
not just from above. Viam's orientation_vector encodes this as the unit
"look-at" direction (o_x, o_y, o_z) plus a roll theta around it.

Outputs
-------
  orbit_NN.jpg          — colour frame from each viewpoint (NN = pose index)
  orbit_cloud.ply       — full merged point cloud, world frame, mm
  orbit_object.ply      — points within ±CROP_PAD_MM of the object's bounding
                           box (strips the table and background)
  orbit_mesh.ply        — Poisson surface mesh (only with --reconstruct)

Setup:
    .venv/bin/pip install viam-sdk anthropic python-dotenv numpy pillow
    .venv/bin/pip install open3d          # only needed for --reconstruct

Run:
    # use the objects from the last scan
    .venv/bin/python table_orbit_scan.py

    # specify a different scan file and a specific object
    .venv/bin/python table_orbit_scan.py --objects table_objects_1.json --id obj1

    # scan the table live first, then orbit the first trash object
    .venv/bin/python table_orbit_scan.py --live-scan

    # print the planned orbit poses and exit — no arm motion
    .venv/bin/python table_orbit_scan.py --dry-run

    # build a mesh too
    .venv/bin/python table_orbit_scan.py --reconstruct
"""

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

import table_scan as ts
import table_cleanup as tc
from viam.components.arm import Arm
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import Motion

# ---------------------------------------------------------------------------
# Orbit geometry
# ---------------------------------------------------------------------------

# Number of evenly-spaced viewpoints around the object.
N_POSES = 8

# Horizontal distance (mm) from the centroid to each orbit position.
# Keep this large enough that the arm doesn't clip nearby objects, but
# small enough that the object fills a reasonable portion of the depth frame.
ORBIT_RADIUS_MM = 280.0

# Height above the centroid (mm). This plus the centroid's world z gives the
# arm's z at each orbit point. 250 mm puts the camera well above a bottle on
# the table while still seeing its sides.
ORBIT_HEIGHT_MM = 250.0

# Roll around the look-at vector. 0 is fine for a depth camera; change this
# if the camera image comes out sideways.
ORBIT_THETA_DEG = 0.0

# Transit height (mm) used between orbit poses to avoid sweeping close to
# objects on the way to each new viewpoint. The arm goes up to this height
# before arcing to the next orbit position.
TRANSIT_HEIGHT_MM = 350.0

# Settle time (s) after each move, before capturing. The xArm7 can ring a
# little after a move completes; a short wait avoids motion blur in the depth.
SETTLE_S = 0.4

# ---------------------------------------------------------------------------
# Point-cloud crop
# ---------------------------------------------------------------------------

# The object's bounding box from the scan, grown by this much on each side,
# is used to extract the object-only cloud from the merged full-scene cloud.
CROP_PAD_MM = 60.0

# Voxel size (mm) used to downsample the merged cloud before saving.
# Set to 0 to skip downsampling.
VOXEL_MM = 2.0


# ---------------------------------------------------------------------------
# Orbit pose math
# ---------------------------------------------------------------------------

def _look_at_vector(p: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Unit vector from p toward target — the arm's look-at direction."""
    d = target - p
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return (0.0, 0.0, -1.0)
    d /= norm
    return float(d[0]), float(d[1]), float(d[2])


def orbit_poses(centroid: list[float],
                n: int = N_POSES,
                radius: float = ORBIT_RADIUS_MM,
                height: float = ORBIT_HEIGHT_MM) -> list[Pose]:
    """
    N evenly-spaced arm poses on a circle around the centroid.

    Each pose sits at (centroid + radial offset, centroid_z + height) and
    has its tool axis pointing from that position toward the centroid, so the
    camera looks at the object rather than off to the side.

    The azimuth starts at 0 (positive world-x direction) and sweeps
    counter-clockwise when viewed from above.
    """
    cx, cy, cz = centroid
    target = np.array([cx, cy, cz])
    poses = []
    for i in range(n):
        az = 2.0 * math.pi * i / n
        px = cx + radius * math.cos(az)
        py = cy + radius * math.sin(az)
        pz = cz + height
        ox, oy, oz = _look_at_vector(np.array([px, py, pz]), target)
        poses.append(Pose(
            x=px, y=py, z=pz,
            o_x=ox, o_y=oy, o_z=oz,
            theta=ORBIT_THETA_DEG,
        ))
    return poses


def transit_pose(pose: Pose) -> Pose:
    """Safe-height transit above an orbit pose (arm pointing down)."""
    return Pose(
        x=pose.x, y=pose.y, z=TRANSIT_HEIGHT_MM,
        o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0,
    )


# ---------------------------------------------------------------------------
# Depth → world point cloud
# ---------------------------------------------------------------------------

def deproject_frame(frame: ts.Frame) -> np.ndarray:
    """
    All valid depth pixels deprojected into world-frame xyz (mm).

    Returns (N, 3) float32. Returns empty array if no depth or no transform.
    The camera intrinsics and cam_to_world transform stored in the Frame are
    those measured at the moment of capture, so this is always exact.
    """
    if frame.depth_mm is None or frame.cam_to_world is None or frame.intrinsics is None:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy, cx, cy = frame.intrinsics
    depth = frame.depth_mm
    v_idx, u_idx = np.nonzero(depth > 0)
    if len(v_idx) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth[v_idx, u_idx].astype(np.float64)
    x_cam = (u_idx.astype(np.float64) - cx) * z / fx
    y_cam = (v_idx.astype(np.float64) - cy) * z / fy
    pts_cam = np.column_stack([x_cam, y_cam, z])

    rot = frame.cam_to_world[:3, :3]
    shift = frame.cam_to_world[:3, 3]
    pts_world = (pts_cam @ rot.T + shift).astype(np.float32)
    return pts_world


def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Fast voxel grid downsampling — keeps one point per cell."""
    if voxel_size <= 0 or len(pts) == 0:
        return pts
    mins = pts.min(axis=0)
    indices = ((pts - mins) / voxel_size).astype(np.int32)
    keys = indices[:, 0] * (1 << 20) + indices[:, 1] * (1 << 10) + indices[:, 2]
    _, first = np.unique(keys, return_index=True)
    return pts[first]


# ---------------------------------------------------------------------------
# PLY output
# ---------------------------------------------------------------------------

def save_ply(points: np.ndarray, path: str, colors: np.ndarray | None = None) -> None:
    """Write a binary little-endian PLY point cloud, optionally with RGB."""
    n = len(points)
    has_color = colors is not None and len(colors) == n
    if has_color:
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        ).encode()
        data = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
        data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
        data["r"], data["g"], data["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    else:
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"end_header\n"
        ).encode()
        data = points.astype("<f4")
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())
    print(f"  Saved {n:,} points → {path}")


# ---------------------------------------------------------------------------
# Arm motion helpers
# ---------------------------------------------------------------------------

async def _move(motion: Motion, pose: Pose, label: str,
                component: str = tc.ARM_NAME) -> bool:
    """Move one component to a world-frame pose; return True on success."""
    from grpclib.exceptions import GRPCError
    try:
        ok = await motion.move(
            component_name=component,
            destination=PoseInFrame(reference_frame="world", pose=pose),
            timeout=tc.MOVE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        print(f"  !! move to {label} timed out after {tc.MOVE_TIMEOUT_S:.0f}s",
              file=sys.stderr)
        return False
    except GRPCError as e:
        print(f"  !! move to {label} failed: {e.message}", file=sys.stderr)
        return False
    if not ok:
        print(f"  !! move to {label} returned false", file=sys.stderr)
    return bool(ok)


# ---------------------------------------------------------------------------
# Main orbit routine
# ---------------------------------------------------------------------------

async def orbit_and_scan(
    centroid: list[float],
    obj_box: dict,
    n_poses: int,
    radius: float,
    height: float,
    dry_run: bool,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Drive the arm around the centroid, capturing a depth cloud at each stop.

    Returns (list_of_point_clouds, list_of_colour_image_paths).
    """
    poses = orbit_poses(centroid, n=n_poses, radius=radius, height=height)
    cx, cy, cz = centroid

    print(f"\nOrbit plan: {n_poses} poses, radius={radius:.0f} mm, "
          f"height above centroid={height:.0f} mm")
    for i, p in enumerate(poses):
        az_deg = math.degrees(math.atan2(p.y - cy, p.x - cx))
        print(f"  [{i+1:02d}] az={az_deg:+6.1f}°  "
              f"pos=({p.x:.0f}, {p.y:.0f}, {p.z:.0f}) mm  "
              f"look-at=({p.o_x:+.3f}, {p.o_y:+.3f}, {p.o_z:+.3f})")

    if dry_run:
        print("\n--dry-run: no arm motion.")
        return [], []

    machine = await ts.connect()
    clouds: list[np.ndarray] = []
    colour_paths: list[str] = []
    try:
        arm = await ts.resolve(machine, Arm, tc.ARM_NAME)
        motion = await ts.resolve(machine, Motion, tc.MOTION_NAME)

        start_pos = await arm.get_end_position()
        print(f"\nArm at start: ({start_pos.x:.1f}, {start_pos.y:.1f}, "
              f"{start_pos.z:.1f}) mm")

        for i, pose in enumerate(poses):
            print(f"\n── Viewpoint {i+1}/{n_poses} ──")

            # 1. Rise to transit height (arm pointing down — always safe).
            tr = transit_pose(pose)
            print(f"   Transit: ({tr.x:.0f}, {tr.y:.0f}, {tr.z:.0f}) mm ↑")
            if not await _move(motion, tr, f"transit {i+1}"):
                print(f"   Skipping viewpoint {i+1}.", file=sys.stderr)
                continue

            # 2. Arc to orbit pose (camera angled toward object).
            print(f"   Orbit:   ({pose.x:.0f}, {pose.y:.0f}, {pose.z:.0f}) mm →")
            if not await _move(motion, pose, f"orbit {i+1}"):
                print(f"   Skipping viewpoint {i+1}.", file=sys.stderr)
                continue

            # 3. Settle, then capture.
            await asyncio.sleep(SETTLE_S)
            print("   Capturing colour + depth...")
            try:
                frame = await ts.capture(machine)
            except Exception as e:
                print(f"   Capture failed: {e} — skipping", file=sys.stderr)
                continue

            # 4. Save colour image.
            ext = ts.SUPPORTED_IMAGE_TYPES.get(frame.media_type, ".jpg")
            cpath = f"orbit_{i+1:02d}{ext}"
            with open(cpath, "wb") as f:
                f.write(frame.color)
            colour_paths.append(cpath)
            print(f"   Colour → {cpath}")

            # 5. Deproject depth to world frame.
            pts = deproject_frame(frame)
            print(f"   Depth:   {len(pts):,} world-frame points captured")
            if len(pts) > 0:
                clouds.append(pts)

        # Return home.
        print(f"\n── Returning home ──")
        tr_home = Pose(x=tc.HOME_POSE.x, y=tc.HOME_POSE.y, z=TRANSIT_HEIGHT_MM,
                       o_x=0, o_y=0, o_z=-1, theta=0)
        await _move(motion, tr_home, "transit home")
        await _move(motion, tc.HOME_POSE, "home", component=tc.ARM_NAME)

    except BaseException:
        print("\nInterrupted — stopping arm.", file=sys.stderr)
        try:
            await arm.stop()
        except Exception as e:
            print(f"  arm stop failed: {e}", file=sys.stderr)
        raise
    finally:
        await machine.close()

    return clouds, colour_paths


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def merge_and_save(clouds: list[np.ndarray], obj: dict) -> tuple[str, str | None]:
    """Merge clouds, downsample, save full and object-cropped PLY."""
    merged = np.vstack(clouds)
    print(f"\nMerged: {len(merged):,} points from {len(clouds)} viewpoint(s)")

    if VOXEL_MM > 0:
        merged = voxel_downsample(merged, VOXEL_MM)
        print(f"After voxel downsampling ({VOXEL_MM} mm): {len(merged):,} points")

    full_path = "orbit_cloud.ply"
    save_ply(merged, full_path)

    # Crop to object bounding box ± padding.
    obj_path = None
    if "min" in obj["position"] and "max" in obj["position"]:
        lo = np.array(obj["position"]["min"]) - CROP_PAD_MM
        hi = np.array(obj["position"]["max"]) + CROP_PAD_MM
        mask = np.all((merged >= lo) & (merged <= hi), axis=1)
        obj_cloud = merged[mask]
        print(f"Object crop (±{CROP_PAD_MM:.0f} mm): {len(obj_cloud):,} points")
        if len(obj_cloud) >= 50:
            obj_path = "orbit_object.ply"
            save_ply(obj_cloud, obj_path)

    return full_path, obj_path


def reconstruct_mesh(ply_path: str) -> str | None:
    """Poisson surface reconstruction via Open3D. Returns mesh path or None."""
    try:
        import open3d as o3d
    except ImportError:
        print("\nopen3d not installed — skipping mesh reconstruction.")
        print("Install it with:  pip install open3d  (or  .venv/bin/pip install open3d)")
        return None

    print("\nRunning Open3D Poisson surface reconstruction...")
    pcd = o3d.geometry.PointCloud()
    raw = np.load(ply_path) if ply_path.endswith(".npy") else None
    # Load from PLY.
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"  Loaded {len(pcd.points):,} points")

    # Remove outliers.
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  After outlier removal: {len(pcd.points):,} points")

    # Estimate normals pointing away from the centroid of the cloud.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15.0, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=20)

    # Poisson reconstruction.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    densities = np.asarray(densities)
    # Trim low-density artefacts at the boundary.
    to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(to_remove)
    mesh.compute_vertex_normals()

    mesh_path = "orbit_mesh.ply"
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"  Mesh → {mesh_path} "
          f"({len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles)")
    return mesh_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def pick_object(data: dict, obj_id: str | None) -> dict:
    """Select the object to orbit from the scan results."""
    candidates = [o for o in data["objects"]
                  if o.get("position") and "centroid" in o["position"]]
    if not candidates:
        raise SystemExit("No localized objects in scan data. "
                         "Run a scan with depth first (not --image).")
    if obj_id:
        obj = next((o for o in candidates if o["id"] == obj_id), None)
        if obj is None:
            ids = [o["id"] for o in candidates]
            raise SystemExit(f"Object {obj_id!r} not found. "
                             f"Available ids with 3D positions: {ids}")
        return obj
    # Default: first trash object, then first object overall.
    trash = [o for o in candidates if o["is_trash"]]
    return (trash or candidates)[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Orbit the arm around a detected object and build a 3D point cloud.")
    ap.add_argument("--objects", default="table_objects.json",
                    help="Path to table_objects.json from a prior scan "
                         "(default: table_objects.json)")
    ap.add_argument("--id", dest="obj_id", default=None,
                    help="Object id to orbit (default: first trash object)")
    ap.add_argument("--poses", type=int, default=N_POSES,
                    help=f"Number of orbit viewpoints (default {N_POSES})")
    ap.add_argument("--radius", type=float, default=ORBIT_RADIUS_MM,
                    help=f"Orbit radius in mm (default {ORBIT_RADIUS_MM})")
    ap.add_argument("--height", type=float, default=ORBIT_HEIGHT_MM,
                    help=f"Height above centroid in mm (default {ORBIT_HEIGHT_MM})")
    ap.add_argument("--live-scan", action="store_true",
                    help="Run a live table scan first, then orbit the result")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the orbit plan; do not move the arm")
    ap.add_argument("--reconstruct", action="store_true",
                    help="Run Poisson mesh reconstruction after scanning "
                         "(requires open3d)")
    args = ap.parse_args()

    # --- Live scan (optional) -----------------------------------------------
    if args.live_scan:
        print("Running live table scan first...")
        frame = asyncio.run(ts.grab_frame())
        ext = ts.SUPPORTED_IMAGE_TYPES[frame.media_type]
        with open("frame" + ext, "wb") as f:
            f.write(frame.color)
        result, marked = ts.scan(frame)
        if marked:
            with open("frame_marked.jpg", "wb") as f:
                f.write(marked)
        with open(args.objects, "w") as f:
            json.dump(result, f, indent=2)
        n_trash = sum(o["is_trash"] for o in result["objects"])
        print(f"Scan complete: {len(result['objects'])} objects, {n_trash} trash")
        data = result
    else:
        objects_path = Path(args.objects)
        if not objects_path.exists():
            raise SystemExit(
                f"{args.objects} not found.\n"
                "Run table_scan.py first, or use --live-scan to scan now.")
        with open(objects_path) as f:
            data = json.load(f)

    # --- Select object -------------------------------------------------------
    obj = pick_object(data, args.obj_id)
    centroid = obj["position"]["centroid"]
    print(f"\nTarget object: {obj['name']}  (id={obj['id']})")
    print(f"  centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) mm")
    print(f"  size:     {obj['position']['size']} mm")
    print(f"  is_trash: {obj['is_trash']}")

    # Warn about sparse depth — common for clear plastic.
    cov = obj["position"].get("coverage", 1.0)
    if cov < 0.10:
        print(f"\n  ⚠  Depth coverage is only {cov:.0%} (clear/shiny surface).")
        print("     More orbit poses (--poses 16) and a slightly lower height")
        print("     will help fill in the cloud from different angles.\n")

    # --- Orbit ---------------------------------------------------------------
    clouds, colour_paths = asyncio.run(orbit_and_scan(
        centroid, obj["position"],
        n_poses=args.poses,
        radius=args.radius,
        height=args.height,
        dry_run=args.dry_run,
    ))

    if not clouds:
        if not args.dry_run:
            print("\nNo depth clouds captured — nothing to save.")
        return

    # --- Merge + save --------------------------------------------------------
    full_path, obj_path = merge_and_save(clouds, obj)

    # --- Optionally reconstruct mesh -----------------------------------------
    if args.reconstruct:
        src = obj_path or full_path
        reconstruct_mesh(src)

    print("\nDone.")
    print(f"  Colour frames: {colour_paths}")
    print(f"  Full cloud:    {full_path}")
    if obj_path:
        print(f"  Object cloud:  {obj_path}")
    print("\nTo view in MeshLab, CloudCompare, or any PLY viewer:")
    print(f"  open {obj_path or full_path}")


if __name__ == "__main__":
    main()
