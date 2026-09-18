"""
table_scan.py — grab a frame from the Viam camera, identify every object on the
table, classify each as trash / not trash, and localize it in 3D.

The RealSense returns a depth frame pixel-aligned with the colour frame in the
same get_images() call, so each 2D box Claude returns can be turned into a real
3D box by deprojecting the depth pixels inside it. Results are reported in the
`world` frame: the camera is mounted on the arm, so camera-frame coordinates
move whenever the arm does and are not a stable description of where a thing is.

Output: table_objects.json  (plus frame.jpg for debugging)

Setup:
    .venv/bin/pip install viam-sdk anthropic python-dotenv numpy
    cp .env.example .env    # then fill it in

Run:
    .venv/bin/python table_scan.py
    .venv/bin/python table_scan.py --image test.jpeg   # prompt only, no 3D
"""

import argparse
import asyncio
import base64
import json
import os
import sys
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

You will be given one photo of a table. Identify EVERY distinct physical object
resting on the table surface. Then decide, for each one, whether it is trash.

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
Approximate is fine."""

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


def parse_depth(data: bytes) -> np.ndarray:
    width = int.from_bytes(data[8:16], "big")
    height = int.from_bytes(data[16:24], "big")
    flat = np.frombuffer(data[DEPTH_HEADER_BYTES:], dtype=">u2")
    return flat.reshape(height, width).astype(np.float32)


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

    return Frame(
        color.data,
        str(color.mime_type),
        depth,
        intrinsics,
        await _cam_to_world(machine, camera_name),
    )


async def grab_frame() -> Frame:
    """Connect, pull one capture, and disconnect. See capture() for callers
    that need to keep the connection open past the scan."""
    machine = await connect()
    try:
        return await capture(machine)
    finally:
        await machine.close()


def classify(image_bytes: bytes, media_type: str) -> dict:
    """Send the frame to Claude and parse the structured response."""
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
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(image_bytes).decode(),
                        },
                    },
                    {"type": "text", "text": "Scan this table."},
                ],
            },
        ],
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


def localize(frame: Frame, bbox_norm) -> Optional[dict]:
    """Turn one normalized 2D box into a world-frame 3D box and centroid.

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
    pts = pts @ frame.cam_to_world[:3, :3].T + frame.cam_to_world[:3, 3]

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    # Centre of the box rather than the mean of the points: we only ever see the
    # camera-facing shell, so the mean sits on the near surface, not inside.
    centre = (lo + hi) / 2.0
    r3 = lambda v: [round(float(c), 1) for c in v]  # noqa: E731
    return {
        "frame": "world",
        "units": "mm",
        "centroid": r3(centre),
        "min": r3(lo),
        "max": r3(hi),
        "size": r3(hi - lo),
        "points": int(mask.sum()),
        "coverage": round(float(mask.sum()) / sub.size, 3),
    }


def _valid_bbox(bbox) -> bool:
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return False
    if not all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in bbox):
        return False
    x_min, y_min, x_max, y_max = bbox
    return x_min < x_max and y_min < y_max


def validate(payload: dict) -> dict:
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
        clean.append(obj)
    if dropped:
        print(f"Dropped {dropped} malformed object(s).", file=sys.stderr)
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

    result = validate(classify(frame.color, frame.media_type))

    for obj in result["objects"]:
        obj["position"] = localize(frame, obj["bbox_norm"])

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    trash = sum(1 for o in result["objects"] if o["is_trash"])
    located = sum(1 for o in result["objects"] if o["position"] and "centroid" in o["position"])
    print(f"{len(result['objects'])} objects, {trash} trash, "
          f"{located} localized -> {args.out}")
    for o in result["objects"]:
        mark = "TRASH" if o["is_trash"] else "keep "
        print(f"  [{mark}] {o['name']}  ({o['confidence']:.2f})  {o['reason']}")
        pos = o["position"]
        if pos and "centroid" in pos:
            c, s = pos["centroid"], pos["size"]
            print(f"          centroid {c} mm  size {s} mm  "
                  f"({pos['points']} pts, {pos['coverage']:.0%} of box)")
        elif pos:
            print(f"          no 3D position: {pos['reason']} ({pos['points']} pts)")


if __name__ == "__main__":
    main()
