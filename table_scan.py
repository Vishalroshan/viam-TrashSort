"""
table_scan.py — grab a frame from the Viam camera, identify every object on the
table, classify each as trash / not trash, and localize it in 3D.

Where things are and what they are come from different places, each doing
what it is good at:
  - WHERE: the machine's `obstacles-pointcloud` vision service removes the
    table plane and clusters what's left; group_objects() merges its
    fragments into one point set per object.
  - WHAT: Claude sees the photo plus a copy with a numbered box drawn around
    each of those objects, and names and classifies each number. It never
    has to produce coordinates, which it does only approximately.
Without a segmenter configured, it falls back to Claude's own 2D boxes cut
out of the depth frame (localize()).

Results are reported in the `world` frame: the camera is mounted on the arm,
so camera-frame coordinates move whenever the arm does and are not a stable
description of where a thing is.

Output: table_objects.json  (plus frame.jpg and frame_marked.jpg for debugging)

Setup:
    .venv/bin/pip install viam-sdk anthropic python-dotenv numpy pillow
    cp .env.example .env    # then fill it in

Run:
    .venv/bin/python table_scan.py
    .venv/bin/python table_scan.py --image test.jpeg   # prompt only, no 3D
"""

import argparse
import asyncio
import base64
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import anthropic
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5"

# Mime types Claude's vision input accepts, keyed by what the camera reports.
# Anything else (depth frames, point clouds) is not a still image we can send.
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

SYSTEM_PROMPT = """You are the perception stage of a tabletop tidying robot.

You will be given a photo of a table, usually followed by a second copy of the
same photo with numbered magenta boxes drawn on it. Identify EVERY distinct
physical object resting on the table surface. Then decide, for each one,
whether it is trash.

Rules for the numbered marks:
- Each box is an object a depth sensor found; its number is the object's
  `mark`. The boxes are accurate; judge what is inside each one from the
  unmarked photo, where the drawing does not hide anything.
- Every mark number must be used by at least one entry. If a box holds
  something that is not an object (a shadow, glare, a piece of the robot),
  still list it, say what it is, and set is_trash false.
- If one box clearly holds two or more separate objects, list each object
  with that same mark number.
- Objects with no box get mark 0. The depth sensor often misses clear
  plastic and very shiny things, so look for those especially.
- When there is no marked copy, every object gets mark 0.

Rules for identifying objects:
- One entry per physical object. If there are three identical bottle caps,
  emit three entries, not one.
- Do not list the table itself, the tabletop surface, shadows, reflections,
  or anything a person is holding.
- Do not list parts of a larger object separately (a cup and its lid, while
  attached, are one object).
- Use a specific, concrete name: "crumpled paper napkin", not "paper".
- If you cannot tell what something is, name it by its appearance
  ("small white cylindrical object") and set confidence low.

Rules for the trash judgment — this is about STATE and USE, not object class:
- Trash: consumed, spent, damaged, or discarded items with no remaining
  function. Food waste, wrappers, used napkins, empty containers, crumpled
  paper, broken pieces, disposable cutlery that has been used.
- Not trash: tools, electronics, cables, personal belongings, unopened or
  partially full containers, reusable dishes, anything with obvious remaining
  value or function.
- Ambiguous cases default to is_trash: false. A false negative leaves a mess;
  a false positive throws away someone's keys. Prefer leaving the mess.
- The `reason` field must state what you actually saw that drove the call,
  not a restatement of the label.

`bbox_norm` is the object's bounding box in normalized image coordinates,
origin at top-left, ordered [x_min, y_min, x_max, y_max], each value in [0, 1].
Approximate is fine; for marked objects the mark's box is what gets used."""

# The API enforces this shape, so the prompt above does not have to ask for it.
OBJECTS_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "is_trash": {"type": "boolean"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                    "mark": {"type": "integer"},
                    # Structured outputs rejects minItems/maxItems above 1, so
                    # the 4-element length is enforced by _valid_bbox instead.
                    "bbox_norm": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "required": [
                    "id",
                    "name",
                    "is_trash",
                    "reason",
                    "confidence",
                    "mark",
                    "bbox_norm",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["objects"],
    "additionalProperties": False,
}


DEPTH_MIME = "image/vnd.viam.dep"

# Viam's depth frames are uint16 big-endian millimetres behind a 24-byte header
# (8 bytes magic, then width and height as 8-byte big-endian ints).
DEPTH_HEADER_BYTES = 24


@dataclass
class Frame:
    """One capture: the colour image, aligned depth, and how to place it."""

    color: bytes
    media_type: str
    depth_mm: Optional[np.ndarray]  # (h, w) float32, 0 where the sensor saw nothing
    intrinsics: Optional[tuple]  # (fx, fy, cx, cy)
    cam_to_world: Optional[np.ndarray]  # 4x4, millimetres
    # Objects from the plane-removing segmenter, each (n, 3) camera-frame mm.
    # None when no segmenter is configured.
    segments: Optional[list] = None


def parse_depth(data: bytes) -> np.ndarray:
    width = int.from_bytes(data[8:16], "big")
    height = int.from_bytes(data[16:24], "big")
    flat = np.frombuffer(data[DEPTH_HEADER_BYTES:], dtype=">u2")
    return flat.reshape(height, width).astype(np.float32)


def parse_pcd(data: bytes) -> np.ndarray:
    """XYZ points from a PCD blob, as (n, 3) millimetres.

    Viam writes PCD in metres; everything else here is millimetres.
    """
    header, pos = {}, 0
    while True:
        nl = data.index(b"\n", pos)
        key, _, val = data[pos:nl].decode().partition(" ")
        pos = nl + 1
        if key and not key.startswith("#"):
            header[key] = val.split()
        if key == "DATA":
            break

    fields = header["FIELDS"]
    if header["DATA"][0] == "binary":
        kinds = {"F": "f", "U": "u", "I": "i"}
        dtype = np.dtype([(f, f"<{kinds[t]}{s}") for f, t, s in
                          zip(fields, header["TYPE"], header["SIZE"])])
        rec = np.frombuffer(data, dtype=dtype, count=int(header["POINTS"][0]),
                            offset=pos)
        xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
    elif header["DATA"][0] == "ascii":
        rows = np.loadtxt(data[pos:].decode().splitlines(), ndmin=2)
        xyz = rows[:, [fields.index(k) for k in "xyz"]]
    else:
        raise ValueError(f"unsupported PCD encoding {header['DATA'][0]!r}")
    return xyz[np.isfinite(xyz).all(axis=1)] * 1000.0


async def _cam_to_world(machine, camera_name: str) -> np.ndarray:
    """Build the camera->world affine by measuring where four points land.

    Rather than converting Viam's orientation-vector representation into a
    rotation matrix by hand, ask the frame system where the camera origin and
    three unit axes end up in world coordinates, and read the basis off the
    answers. Four round trips once per scan, and no conversion to get wrong.
    """
    from viam.proto.common import Pose, PoseInFrame

    async def to_world(x, y, z):
        q = PoseInFrame(
            reference_frame=camera_name,
            pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=1, theta=0),
        )
        p = (await machine.transform_pose(q, "world")).pose
        return np.array([p.x, p.y, p.z], dtype=np.float64)

    origin = await to_world(0, 0, 0)
    unit = 1000.0  # a metre out along each axis, in Viam's millimetres
    axes = [(await to_world(*v) - origin) / unit for v in
            ((unit, 0, 0), (0, unit, 0), (0, 0, unit))]

    m = np.eye(4)
    m[:3, :3] = np.column_stack(axes)
    m[:3, 3] = origin
    return m


async def connect():
    """Open one RobotClient connection to the configured Viam machine.

    Callers that only need a single frame can use grab_frame(), which opens
    and closes its own connection. Callers that go on to drive the arm and
    gripper after the scan (table_cleanup.py) should call this once and pass
    the same `machine` to capture() and to every subsequent client, rather
    than reconnecting per step.
    """
    from viam.robot.client import RobotClient

    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    address = os.environ.get("VIAM_ADDRESS", "armfarm15-main.310sld03v2.viam.cloud")
    return await RobotClient.at_address(address, opts)


async def resolve(machine, cls, name: str, retries: int = 3, delay: float = 0.5):
    """`cls.from_robot(machine, name)`, retrying through a known SDK race.

    RobotClient.refresh() (run once automatically on connect) silently drops
    a resource from the local registry if its client registration hiccups
    during that one pass — the server-side resource_names list still shows
    it, but from_robot() raises ResourceNotFoundError anyway. Observed
    intermittently against this exact machine: same resource, same code,
    alternating pass/fail. A fresh refresh() and retry clears it.
    """
    from viam.errors import ResourceNotFoundError

    for attempt in range(retries):
        try:
            return cls.from_robot(machine, name)
        except ResourceNotFoundError:
            if attempt == retries - 1:
                raise
            print(
                f"{name!r} not yet in local registry (attempt {attempt + 1}/{retries}), "
                "refreshing...",
                file=sys.stderr,
            )
            await machine.refresh()
            await asyncio.sleep(delay)


async def segment(machine, camera_name: str) -> Optional[list]:
    """Objects above the table from the `obstacles-pointcloud` vision service.

    The service fits the table plane with RANSAC, removes it, and clusters
    what's left — so each segment is part or all of one object's own points,
    with no table mixed in. Returns None when the service isn't configured or
    fails, and scan() falls back to cutting objects out of the depth frame.
    """
    from grpclib.exceptions import GRPCError
    from viam.errors import ResourceNotFoundError
    from viam.services.vision import Vision

    name = os.environ.get("VIAM_SEGMENTER", "table-segmenter")
    try:
        seg = await resolve(machine, Vision, name)
        objs = await seg.get_object_point_clouds(camera_name)
    except (ResourceNotFoundError, GRPCError) as e:
        print(f"segmenter {name!r} unavailable ({e}); using depth boxes only",
              file=sys.stderr)
        return None
    return [parse_pcd(o.point_cloud) for o in objs]


async def capture(machine, camera_name: Optional[str] = None) -> Frame:
    """Pull one capture off the camera on an already-open connection.

    The camera hands back several images (colour alongside depth), so select
    by mime type rather than assuming an order.
    """
    from viam.components.camera import Camera

    camera_name = camera_name or os.environ.get("VIAM_CAMERA", "cam")
    cam = await resolve(machine, Camera, camera_name)
    images, _ = await cam.get_images()

    color = next(
        (i for i in images if str(i.mime_type) in SUPPORTED_IMAGE_TYPES), None
    )
    if color is None:
        got = ", ".join(f"{i.name}={i.mime_type}" for i in images) or "nothing"
        raise RuntimeError(
            f"camera {camera_name!r} returned no JPEG or PNG image (got: {got})"
        )

    depth_img = next((i for i in images if str(i.mime_type) == DEPTH_MIME), None)
    depth = parse_depth(depth_img.data) if depth_img else None
    if depth is None:
        print(
            f"camera {camera_name!r} returned no depth frame; "
            "objects will have no 3D position.",
            file=sys.stderr,
        )
        return Frame(color.data, str(color.mime_type), None, None, None)

    props = await cam.get_properties()
    k = props.intrinsic_parameters
    intrinsics = (k.focal_x_px, k.focal_y_px, k.center_x_px, k.center_y_px)

    if (depth.shape[1], depth.shape[0]) != (color.width, color.height):
        raise RuntimeError(
            f"depth {depth.shape[1]}x{depth.shape[0]} does not match colour "
            f"{color.width}x{color.height}; the 2D boxes would not line up"
        )

    # The segmenter takes its own point cloud a moment after get_images();
    # both describe the same scene as long as the arm holds still between.
    return Frame(
        color.data,
        str(color.mime_type),
        depth,
        intrinsics,
        await _cam_to_world(machine, camera_name),
        await segment(machine, camera_name),
    )


async def grab_frame() -> Frame:
    """Connect, pull one capture, and disconnect. See capture() for callers
    that need to keep the connection open past the scan."""
    machine = await connect()
    try:
        return await capture(machine)
    finally:
        await machine.close()


def _image_block(data: bytes, media_type: str) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(data).decode(),
        },
    }


def classify(image_bytes: bytes, media_type: str,
             marked: Optional[bytes] = None, n_marks: int = 0) -> dict:
    """Send the frame (and its numbered copy, if any) to Claude and parse the
    structured response. `marked` is a JPEG from draw_marks()."""
    content = [_image_block(image_bytes, media_type)]
    if marked:
        content.append(_image_block(marked, "image/jpeg"))
        content.append({"type": "text", "text":
                        f"Scan this table. The second image has {n_marks} "
                        f"numbered mark(s), 1 to {n_marks}."})
    else:
        content.append({"type": "text", "text":
                        "Scan this table. There is no marked copy this time."})

    # An API key that is not scoped to a workspace must name one explicitly.
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace} if workspace else None
    client = anthropic.Anthropic(default_headers=headers)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={
            "format": {"type": "json_schema", "schema": OBJECTS_SCHEMA},
        },
        messages=[{"role": "user", "content": content}],
    )
    if resp.stop_reason == "max_tokens":
        print("Response was truncated; raise max_tokens.", file=sys.stderr)
    raw = next(b.text for b in resp.content if b.type == "text")
    return json.loads(raw)


# A 2D box always catches some table around the object. Split the depth values
# inside it into a near cluster (the object) and a far one (the surface behind
# it) — but only when the two are actually far enough apart to be different
# things, otherwise the box is all object and splitting it invents a boundary.
MIN_CLUSTER_SEPARATION_MM = 20.0
MIN_DEPTH_POINTS = 50


def _otsu(values: np.ndarray, bins: int = 64) -> tuple[float, float, float]:
    """Threshold that best splits values in two. Returns (cut, near_mean, far_mean)."""
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w_near = np.cumsum(hist)
    w_far = hist.sum() - w_near
    csum = np.cumsum(hist * centers)
    mean_near = csum / np.maximum(w_near, 1)
    mean_far = (csum[-1] - csum) / np.maximum(w_far, 1)
    splittable = (w_near > 0) & (w_far > 0)
    if not splittable.any():
        # Every value fell in one bin — a uniform patch. There is no boundary
        # to find, and pretending otherwise discards the whole region.
        return float("inf"), 0.0, 0.0
    between = np.where(
        splittable, w_near * w_far * (mean_near - mean_far) ** 2, -1.0
    )
    i = int(np.argmax(between))
    return float(edges[i + 1]), float(mean_near[i]), float(mean_far[i])


# The machine's `table` obstacle is a 200mm-tall box whose frame sits at
# z=-123, so its top surface — where every object rests — is at z=-23.
TABLE_TOP_Z_MM = -23.0

# Fragments whose world-frame footprints come within this distance of each
# other are one object. The segmenter over-splits: a crushed can came back as
# ~14 fragments and a lying cylinder as 3 strips, because its clustering only
# joins neighbouring cells of near-equal height. The cost: objects closer
# than this also merge, and are then reported rather than picked.
MERGE_GAP_MM = 10.0
MIN_OBJECT_POINTS = 300


def _project(frame: Frame, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points to colour-image pixels.

    The camera's point cloud is its colour-aligned depth deprojected with the
    colour intrinsics, so this lands on exactly the pixels each point came from.
    """
    fx, fy, cx, cy = frame.intrinsics
    return fx * pts[:, 0] / pts[:, 2] + cx, fy * pts[:, 1] / pts[:, 2] + cy


def group_objects(frame: Frame) -> list[np.ndarray]:
    """Merge the segmenter's fragments into one camera-frame point set per object.

    Ordered left to right in the image, so mark numbers read naturally.
    """
    rot, shift = frame.cam_to_world[:3, :3], frame.cam_to_world[:3, 3]
    pieces, footprints = [], []
    for s in frame.segments:
        s = s[s[:, 2] > 0]
        if len(s) == 0:
            continue
        w = s @ rot.T + shift
        # Only the biggest plane (the table) is removed, so the floor beyond
        # its edge comes back as dozens of fragments far below the tabletop.
        if np.median(w[:, 2]) < TABLE_TOP_Z_MM:
            continue
        pieces.append(s)
        footprints.append((w[:, :2].min(axis=0), w[:, :2].max(axis=0)))

    parent = list(range(len(pieces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            (lo_a, hi_a), (lo_b, hi_b) = footprints[i], footprints[j]
            gap = np.maximum(lo_b - hi_a, lo_a - hi_b)
            if (gap <= MERGE_GAP_MM).all():
                parent[root(i)] = root(j)

    groups: dict[int, list] = {}
    for i, s in enumerate(pieces):
        groups.setdefault(root(i), []).append(s)
    objects = [np.vstack(g) for g in groups.values()]
    objects = [o for o in objects if len(o) >= MIN_OBJECT_POINTS]
    return sorted(objects, key=lambda o: float(np.median(_project(frame, o)[0])))


def draw_marks(frame: Frame, objects: list[np.ndarray]) -> bytes:
    """The colour image with a numbered box around each object, as JPEG."""
    import io

    from PIL import Image, ImageDraw, ImageFont

    colour = (255, 0, 255)  # magenta: rare on a table, loud on white
    im = Image.open(io.BytesIO(frame.color)).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(20, im.height // 28))
    for n, pts in enumerate(objects, 1):
        u, v = _project(frame, pts)
        box = (u.min() - 4, v.min() - 4, u.max() + 4, v.max() + 4)
        draw.rectangle(box, outline=colour, width=3)
        # Label sits just above the box's corner, on a filled tag so it
        # reads over any background.
        label = str(n)
        l, t, r, b = draw.textbbox((0, 0), label, font=font)
        x, y = box[0], max(box[1] - (b - t) - 10, 0)
        draw.rectangle((x, y, x + (r - l) + 8, y + (b - t) + 8), fill=colour)
        draw.text((x + 4 - l, y + 4 - t), label, fill="white", font=font)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _object_position(frame: Frame, pts: np.ndarray) -> dict:
    u, v = _project(frame, pts)
    area = max(float((u.max() - u.min()) * (v.max() - v.min())), 1.0)
    return _world_box(frame, pts, min(len(pts) / area, 1.0), "segmenter")


def scan(frame: Frame) -> tuple[dict, Optional[bytes]]:
    """Identify, classify and place every object on the table.

    Returns the validated result, each object carrying a `position`, and the
    marked-up image Claude saw (None when there were no marks to draw).
    """
    objects = group_objects(frame) if frame.segments is not None else None
    marked = draw_marks(frame, objects) if objects else None
    n = len(objects or [])
    result = validate(classify(frame.color, frame.media_type, marked, n), n)

    claims = Counter(o["mark"] for o in result["objects"])
    for obj in result["objects"]:
        m = obj["mark"]
        if objects is None:
            # No segmenter: fall back to Claude's own box on the depth frame.
            obj["position"] = localize(frame, obj["bbox_norm"])
        elif m == 0:
            # Claude's own box is too loose to grasp on (seen 80-100px off),
            # so an object the segmenter didn't find gets no position.
            obj["position"] = {"points": 0,
                               "reason": "not found by the depth segmenter"}
        elif claims[m] > 1:
            obj["position"] = {"points": len(objects[m - 1]),
                               "reason": f"mark {m} holds more than one object"}
        else:
            obj["position"] = _object_position(frame, objects[m - 1])
    return result, marked


def localize(frame: Frame, bbox_norm) -> Optional[dict]:
    """Turn one normalized 2D box into a world-frame 3D box and centroid by
    cutting the object out of the depth frame. The fallback when no
    segmenter is configured.

    Returns None when the sensor gave us too little to work with — which is a
    real outcome, not an error: the RealSense projects infrared, and clear
    plastic and shiny metal return little or none of it.
    """
    if frame.depth_mm is None or bbox_norm is None:
        return None

    h, w = frame.depth_mm.shape
    x0, x1 = sorted((int(bbox_norm[0] * w), int(bbox_norm[2] * w)))
    y0, y1 = sorted((int(bbox_norm[1] * h), int(bbox_norm[3] * h)))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(max(x1, x0 + 1), w), min(max(y1, y0 + 1), h)
    return _localize_depth_box(frame, x0, y0, x1, y1)


def _world_box(frame: Frame, pts_cam: np.ndarray, coverage: float, source: str) -> dict:
    """World-frame box and centroid of camera-frame points (mm)."""
    pts = pts_cam @ frame.cam_to_world[:3, :3].T + frame.cam_to_world[:3, 3]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    # Centre of the box rather than the mean of the points: we only ever see the
    # camera-facing shell, so the mean sits on the near surface, not inside.
    centre = (lo + hi) / 2.0

    # The footprint's long axis, seen from above — what a gripper has to line
    # up against. Percentile extents so a few stray points don't widen it.
    xy = pts[:, :2] - pts[:, :2].mean(axis=0)
    _, axes = np.linalg.eigh(np.cov(xy.T))
    major, minor = axes[:, 1], axes[:, 0]
    yaw = math.degrees(math.atan2(major[1], major[0]))
    yaw = (yaw + 90.0) % 180.0 - 90.0  # an axis, not a direction: [-90, 90)
    extent = lambda d: float(np.subtract(*np.percentile(xy @ d, [99, 1])))  # noqa: E731

    r3 = lambda v: [round(float(c), 1) for c in v]  # noqa: E731
    return {
        "frame": "world",
        "units": "mm",
        "source": source,
        "centroid": r3(centre),
        "min": r3(lo),
        "max": r3(hi),
        "size": r3(hi - lo),
        "long_axis_deg": round(yaw, 1),  # world yaw of the footprint's long side
        "length_mm": round(extent(major), 1),
        "width_mm": round(extent(minor), 1),
        "points": len(pts),
        "coverage": round(float(coverage), 3),
    }


def _localize_depth_box(frame: Frame, x0: int, y0: int, x1: int, y1: int) -> Optional[dict]:
    """Cut the object out of the depth pixels inside its 2D box."""
    sub = frame.depth_mm[y0:y1, x0:x1]
    mask = sub > 0
    if int(mask.sum()) < MIN_DEPTH_POINTS:
        return {"points": int(mask.sum()), "reason": "no usable depth in box"}

    cut, near_mean, far_mean = _otsu(sub[mask])
    if abs(far_mean - near_mean) >= MIN_CLUSTER_SEPARATION_MM:
        mask &= sub <= cut
    if int(mask.sum()) < MIN_DEPTH_POINTS:
        return {"points": int(mask.sum()), "reason": "near cluster too small"}

    # Drop stragglers: speckle on the object's silhouette reads as depth
    # halfway between the object and whatever is behind it.
    kept = sub[mask]
    median = float(np.median(kept))
    mad = float(np.median(np.abs(kept - median)))
    if mad > 0:
        mask &= np.abs(sub - median) <= 3.0 * 1.4826 * mad
    if int(mask.sum()) < MIN_DEPTH_POINTS:
        return {"points": int(mask.sum()), "reason": "too few depth points after filtering"}

    rows, cols = np.nonzero(mask)
    z = sub[mask].astype(np.float64)
    fx, fy, cx, cy = frame.intrinsics
    x_cam = (cols + x0 - cx) * z / fx
    y_cam = (rows + y0 - cy) * z / fy

    pts = np.column_stack([x_cam, y_cam, z])
    return _world_box(frame, pts, mask.sum() / sub.size, "depth-box")


def _valid_bbox(bbox) -> bool:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return False
    if not all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in bbox):
        return False
    x_min, y_min, x_max, y_max = bbox
    return x_min < x_max and y_min < y_max


def validate(payload: dict, n_marks: int = 0) -> dict:
    """Reject malformed entries rather than letting them reach the arm.

    The schema already guarantees the shape, so this is a second line of
    defence — it catches values that are well-typed but unusable.
    """
    clean, dropped = [], 0
    for i, obj in enumerate(payload.get("objects", [])):
        if not isinstance(obj.get("name"), str) or not obj["name"].strip():
            dropped += 1
            continue
        if not isinstance(obj.get("is_trash"), bool):
            dropped += 1
            continue
        obj.setdefault("id", f"obj_{i + 1:02d}")
        obj.setdefault("confidence", 0.0)
        obj.setdefault("reason", "")
        if not _valid_bbox(obj.get("bbox_norm")):
            obj["bbox_norm"] = None
        # A mark that doesn't exist could only point the arm at the wrong
        # object; treat it as unmarked.
        if not (isinstance(obj.get("mark"), int) and 0 <= obj["mark"] <= n_marks):
            obj["mark"] = 0
        clean.append(obj)
    if dropped:
        print(f"Dropped {dropped} malformed object(s).", file=sys.stderr)

    # Every mark is something physical on the table. One Claude skipped still
    # has to be kept — as an obstacle the arm avoids, never as trash.
    used = {o["mark"] for o in clean}
    for m in range(1, n_marks + 1):
        if m not in used:
            clean.append({
                "id": f"mark_{m}", "name": "unidentified object", "is_trash": False,
                "reason": "found by the depth segmenter but not identified",
                "confidence": 0.0, "mark": m, "bbox_norm": None,
            })
    return {"objects": clean}


def media_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    for media_type, suffix in SUPPORTED_IMAGE_TYPES.items():
        if ext == suffix or (media_type == "image/jpeg" and ext == ".jpeg"):
            return media_type
    raise SystemExit(f"unsupported image type {ext!r}; use .jpg/.jpeg or .png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="use a local image instead of the camera")
    ap.add_argument("--out", default="table_objects.json")
    args = ap.parse_args()

    if args.image:
        media_type = media_type_for(args.image)
        with open(args.image, "rb") as f:
            frame = Frame(f.read(), media_type, None, None, None)
        print("--image has no depth; objects will have no 3D position.",
              file=sys.stderr)
    else:
        frame = asyncio.run(grab_frame())
        frame_path = "frame" + SUPPORTED_IMAGE_TYPES[frame.media_type]
        with open(frame_path, "wb") as f:
            f.write(frame.color)

    result, marked = scan(frame)
    if marked:
        with open("frame_marked.jpg", "wb") as f:
            f.write(marked)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    trash = sum(1 for o in result["objects"] if o["is_trash"])
    located = sum(1 for o in result["objects"] if o["position"] and "centroid" in o["position"])
    print(f"{len(result['objects'])} objects, {trash} trash, "
          f"{located} localized -> {args.out}")
    for o in result["objects"]:
        verdict = "TRASH" if o["is_trash"] else "keep "
        tag = f"#{o['mark']}" if o["mark"] else "--"
        print(f"  [{verdict}] {tag:>3} {o['name']}  ({o['confidence']:.2f})  {o['reason']}")
        pos = o["position"]
        if pos and "centroid" in pos:
            c, s = pos["centroid"], pos["size"]
            print(f"          centroid {c} mm  size {s} mm  "
                  f"({pos['points']} pts, {pos['coverage']:.0%} of box, "
                  f"{pos['source']})")
        elif pos:
            print(f"          no 3D position: {pos['reason']} ({pos['points']} pts)")


if __name__ == "__main__":
    main()
