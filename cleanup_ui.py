"""
cleanup_ui.py — a PyQt5 window for choosing what the arm throws away, by chat.

Scan the table, then tell it what to remove in plain language ("throw away
the crushed can", "get rid of all the trash except the paper"). Claude maps
the request onto the objects from the latest scan and the window shows which
ones it picked. Nothing moves until you confirm, with the Execute button or
by replying "go". After the picks it scans again so the list stays current.

To move an object instead: left-click a spot on the live feed (right-click
clears it), then say which object goes there ("put the block there"). The
object is picked up and set down centred on that spot, at the same height and
heading it was picked up with.

To push objects apart (e.g. when a spot is too crowded to place into): drag a
path across the live feed, starting on clear table. The closed gripper
follows it slowly, SWEEP_CLEARANCE_MM above the table surface fitted from the
depth camera (never below the configured tabletop), with its flat finger face
leading, then lifts straight up.

3D Scan: orbits the arm around the scene's centroid, fuses depth frames into
a Poisson mesh, and displays the result in the 3D view tab with all pickable
objects' grasp points marked as red spheres + orange approach arrows.

Stop halts the arm immediately, wherever it is. Clear arm error recovers the
xArm after a collision fault.

Setup: see README.md; the UI additionally needs PyQt5 and open3d.

Run, from the repo root:
    python cleanup_ui.py
"""

import asyncio
import json
import math
import os
import re
import sys
import threading
import traceback

import anthropic
import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import WorldState
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import Motion

from trashsort import table_cleanup as tc
from trashsort import table_scan as ts
from trashsort import table_reconstruct as tr
from trashsort.paths import out_path as _out

INTERPRET_PROMPT = """You are the chat interface of a tabletop robot arm. The arm
can pick objects up off the table and either drop them in a trash bin or set
them down at a point the user has clicked on the live camera feed. You never
move the arm yourself: you choose which objects the user means, and the
program shows that plan and waits for the user to confirm it.

You are given the latest scan of the table — every object with its `id`,
`mark` (its number in the marked photo, 0 if it has none), name, whether the
scan judged it trash, and whether the arm can reach it — plus the marked photo
itself, whether a target point is currently selected, and any plan waiting for
confirmation. You cannot see where the point is; the program handles that.

Choose `action`:
- "pick": the user wants specific objects thrown away. `target_ids` holds
  exactly the objects they mean and nothing else. "All the trash" means every
  object whose is_trash is true. Include objects they asked for even when
  pickable is false; the program will explain why those are skipped. If the
  request could mean more than one set of objects (e.g. "the can" when there
  are two cans), do not guess: use "chat" and ask which one.
- "place": the user wants one object moved to the selected point ("put the
  can there", "move the block to the point"). `target_ids` holds exactly that
  one object. If no point is selected, use "chat" and ask them to click a
  point on the live feed first. If they name several objects, use "chat" and
  explain that only one object can go to the point at a time.
- "sweep": the user wants the arm to push along the path they drew on the
  live feed ("push along the path", "sweep", "clear them apart"). The closed
  gripper drags slowly just above the table and shoves whatever is in the way.
  `target_ids` is empty. If no path is drawn, use "chat" and tell them to drag
  a path across the live feed first.
- "confirm": the user approves the plan that is waiting ("yes", "go", "do
  it"). Only valid when a plan is waiting; otherwise use "chat".
- "cancel": the user drops the plan that is waiting.
- "scan": the user asks to look at the table again.
- "chat": anything else, such as questions about what is on the table.

`target_ids` is empty for every action except "pick" and "place". A typical
flow: "place" is refused because the spot is too close to other objects, and
the user draws a path to sweep them apart first. `reply` is
a short message shown to the user: for "pick" or "place", say which objects
you chose, naming them the way the photo shows them; for "chat", answer or
ask."""

LOCATE_WINDOW_PX = 4


def shifted(obj: dict, dx: float, dy: float) -> dict:
    pos = dict(obj["position"])
    for key in ("centroid", "min", "max"):
        pos[key] = [pos[key][0] + dx, pos[key][1] + dy, pos[key][2]]
    return {**obj, "position": pos}


def blocking_object(obj: dict, point: tuple, objects: list[dict]) -> dict | None:
    r = obj["position"]["length_mm"] / 2 + tc.OBSTACLE_PADDING_MM
    for o in located(objects):
        if o is not obj and _box_gap(point, o) < r:
            return o
    return None


SWEEP_CLEARANCE_MM = 20.0

SWEEP_SPEED_DEGS_PER_SEC = 8.0
SWEEP_ACCEL_DEGS_PER_SEC2 = 50.0
ARM_NORMAL_SPEED_DEGS_PER_SEC = 60.0
ARM_NORMAL_ACCEL_DEGS_PER_SEC2 = 381.67

SWEEP_MAX_WAYPOINTS = 15
SWEEP_SIMPLIFY_MM = 5.0
SWEEP_MIN_LENGTH_MM = 20.0
# Longest single straight move while sweeping. A line-constrained move is
# only ever tried "direct" (joints interpolated to the goal), and that bows
# away from the true line more the longer the move and the further out the
# arm reaches; past SWEEP_LINE's 2mm the planner refuses it outright. Seen
# on this arm: 105-161mm segments failed, 30-48mm ones passed, one 43mm one
# failed far out at ~570mm reach.
SWEEP_STEP_MM = 20.0
SWEEP_START_CLEARANCE_MM = 25.0
DRAG_THRESHOLD_PX = 10

SWEEP_LINE = Constraints(linear_constraint=[
    LinearConstraint(line_tolerance_mm=2.0, orientation_tolerance_degs=3.0)])

# Fallbacks for one short sweep step whose straight-line move is refused.
# A line-constrained move is only ever tried "direct" (joints interpolated
# to the goal), so where the arm has to swing a joint to keep the gripper
# pointing down — near a joint limit or an awkward configuration — even a
# 19mm step gets refused. First relax the line; as a last resort let the
# planner search freely, which on a step this short still stays close to
# the line and still avoids the table from the machine config.
SWEEP_LINE_LOOSE = Constraints(linear_constraint=[
    LinearConstraint(line_tolerance_mm=6.0, orientation_tolerance_degs=8.0)])
SWEEP_STEP_FALLBACKS = [(SWEEP_LINE_LOOSE, "looser line (6 mm)"),
                        (None, "free plan")]

# Reconstruction defaults (overridable via the UI in a future iteration).
RECONSTRUCT_N_POSES = 12
RECONSTRUCT_RADIUS_MM = 280.0
RECONSTRUCT_HEIGHT_MM = 250.0


async def sweep_move(rig, pose, label: str, world, constraints=SWEEP_LINE) -> bool:
    from grpclib.exceptions import GRPCError
    from viam.proto.common import PoseInFrame

    try:
        ok = await rig.motion.move(
            component_name=tc.GRIPPER_NAME,
            destination=PoseInFrame(reference_frame="world", pose=pose),
            world_state=world, constraints=constraints,
            timeout=tc.MOVE_TIMEOUT_S * 3,
        )
    except asyncio.TimeoutError:
        await rig.arm.stop()
        print(f"    {label} timed out", file=sys.stderr)
        return False
    except GRPCError as e:
        print(f"    {label} failed: {e.message}", file=sys.stderr)
        return False
    if not ok:
        print(f"    {label} failed (planner returned false)", file=sys.stderr)
    return ok


def fit_table_plane(pts: np.ndarray) -> np.ndarray:
    bins = np.arange(pts[:, 2].min(), pts[:, 2].max() + 5.0, 5.0)
    hist, edges = np.histogram(pts[:, 2], bins=bins)
    mode = edges[hist.argmax()] + 2.5
    keep = np.abs(pts[:, 2] - mode) < 30.0
    for _ in range(3):
        a = np.column_stack([pts[keep, 0], pts[keep, 1], np.ones(keep.sum())])
        coef = np.linalg.lstsq(a, pts[keep, 2], rcond=None)[0]
        resid = pts[:, 2] - (coef[0] * pts[:, 0] + coef[1] * pts[:, 1] + coef[2])
        keep = np.abs(resid) < 8.0
    return coef


def ray_to_plane(origin: np.ndarray, direction: np.ndarray, coef) -> np.ndarray | None:
    a, b, c = coef
    n = np.array([-a, -b, 1.0])
    denom = n @ direction
    if abs(denom) < 1e-9:
        return None
    t = (c - n @ origin) / denom
    return origin + t * direction if t > 0 else None


def simplify_path(pts: list, tol: float) -> list:
    if len(pts) < 3:
        return list(pts)
    p0, p1 = np.array(pts[0][:2]), np.array(pts[-1][:2])
    seg = p1 - p0
    seg_len = np.linalg.norm(seg)
    best_i, best_d = 0, -1.0
    for i in range(1, len(pts) - 1):
        q = np.array(pts[i][:2])
        d = (abs(seg[0] * (q - p0)[1] - seg[1] * (q - p0)[0]) / seg_len
             if seg_len > 1e-9 else np.linalg.norm(q - p0))
        if d > best_d:
            best_i, best_d = i, d
    if best_d <= tol:
        return [pts[0], pts[-1]]
    left = simplify_path(pts[:best_i + 1], tol)
    return left[:-1] + simplify_path(pts[best_i:], tol)


def path_length(path: list) -> float:
    return sum(math.dist(p[:2], q[:2]) for p, q in zip(path, path[1:]))


def sweep_theta(start, end) -> float:
    yaw = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    theta = 90.0 - yaw + tc.JAW_AXIS_OFFSET_DEG
    lo = tc.HOME_POSE.theta - 90.0
    return lo + (theta - lo) % 180.0


def sweep_thetas(start, end) -> list[float]:
    """Gripper rolls that sweep this path, preferred first.

    The closed gripper leads with a flat finger face either way round, so
    theta and theta + 180 push the same. They put the wrist camera on opposite
    sides, though, and near a wall or at some wrist angles only one of them
    is collision-free.
    """
    th = sweep_theta(start, end)
    return [th, th + 180.0 if th < tc.HOME_POSE.theta else th - 180.0]


def _box_gap(point, obj: dict) -> float:
    lo, hi = obj["position"]["min"], obj["position"]["max"]
    dx = max(lo[0] - point[0], 0, point[0] - hi[0])
    dy = max(lo[1] - point[1], 0, point[1] - hi[1])
    return math.hypot(dx, dy)


def located(objects: list[dict]) -> list[dict]:
    return [o for o in objects if tc.gate(o["position"]) is None]


def path_crosses(path: list, objects: list[dict]) -> list[dict]:
    hits = []
    for o in located(objects):
        for p, q in zip(path, path[1:]):
            n = max(int(math.dist(p[:2], q[:2]) / 5.0), 1)
            if any(_box_gap((p[0] + (q[0] - p[0]) * i / n,
                             p[1] + (q[1] - p[1]) * i / n), o)
                   < SWEEP_START_CLEARANCE_MM for i in range(n + 1)):
                hits.append(o)
                break
    return hits


INTERPRET_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string",
                   "enum": ["pick", "place", "sweep", "confirm", "cancel",
                            "scan", "chat"]},
        "target_ids": {"type": "array", "items": {"type": "string"}},
        "reply": {"type": "string"},
    },
    "required": ["action", "target_ids", "reply"],
    "additionalProperties": False,
}


def interpret(message: str, objects: list[dict], marked: bytes | None,
              pending: dict | None, has_point: bool, has_path: bool,
              history: list[dict]) -> dict:
    listing = [{
        "id": o["id"], "mark": o["mark"], "name": o["name"],
        "is_trash": o["is_trash"], "reason": o["reason"],
        "pickable": tc.gate(o["position"]) is None,
        "why_not_pickable": tc.gate(o["position"]),
    } for o in objects]
    waiting = (f"{pending['kind']} {[o['id'] for o in pending['targets']]}"
               if pending else "none")
    state = (f"Latest scan:\n{json.dumps(listing, indent=1)}\n\n"
             f"Target point selected: {'yes' if has_point else 'no'}\n"
             f"Sweep path drawn: {'yes' if has_path else 'no'}\n"
             f"Plan waiting for confirmation: {waiting}\n\n"
             f"User: {message}")
    content = []
    if marked:
        content.append(ts._image_block(marked, "image/jpeg"))
    content.append({"type": "text", "text": state})

    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace} if workspace else None
    client = anthropic.Anthropic(default_headers=headers)
    resp = client.messages.create(
        model=ts.MODEL,
        max_tokens=2000,
        system=INTERPRET_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": INTERPRET_SCHEMA}},
        messages=history + [{"role": "user", "content": content}],
    )
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


class Robot:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.machine = None
        self.arm = None
        # Arm pose when the latest scan was taken; every action returns here.
        self.scan_pose = None
        self._connect_lock = asyncio.Lock()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def _connected(self):
        async with self._connect_lock:
            if self.machine is None:
                print("Connecting to the machine...")
                self.machine = await ts.connect()
                self.arm = await ts.resolve(self.machine, Arm, tc.ARM_NAME)
        return self.machine

    async def feed(self, show, is_on):
        cam, source = None, None
        while True:
            if not is_on():
                await asyncio.sleep(0.3)
                continue
            try:
                if cam is None:
                    cam = await ts.resolve(await self._connected(), Camera,
                                           os.environ.get("VIAM_CAMERA", "cam"))
                    images, _ = await cam.get_images()
                    source = next(i.name for i in images
                                  if str(i.mime_type) in ts.SUPPORTED_IMAGE_TYPES)
                images, _ = await cam.get_images(filter_source_names=[source])
                show(images[0].data)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"live feed: {e}; retrying", file=sys.stderr)
                cam = None
                await asyncio.sleep(2.0)

    async def scan(self):
        machine = await self._connected()
        try:
            frame = await ts.capture(machine)
            self.scan_pose = await self.arm.get_end_position()
        except Exception:
            self.machine = None
            raise
        with open(_out("frame" + ts.SUPPORTED_IMAGE_TYPES[frame.media_type]), "wb") as f:
            f.write(frame.color)
        result, marked = await asyncio.to_thread(ts.scan, frame)
        if marked:
            with open(_out("frame_marked.jpg"), "wb") as f:
                f.write(marked)
        with open(_out("table_objects.json"), "w") as f:
            json.dump(result, f, indent=2)
        return result, marked or frame.color

    async def locate(self, u: float, v: float, img_w: int, img_h: int) -> tuple:
        depth, (fx, fy, cx, cy), c2w = await self._depth_view()
        h, w = depth.shape
        du, dv = int(u * w / img_w), int(v * h / img_h)
        r = LOCATE_WINDOW_PX
        patch = depth[max(dv - r, 0):dv + r + 1, max(du - r, 0):du + r + 1]
        valid = patch[patch > 0]
        if len(valid) < 10:
            raise RuntimeError("no depth reading at that spot; click somewhere else")
        z = float(np.median(valid))
        p_cam = np.array([(du - cx) * z / fx, (dv - cy) * z / fy, z])
        p = c2w[:3, :3] @ p_cam + c2w[:3, 3]
        print(f"target point: pixel ({du}, {dv}) -> world "
              f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm")
        return tuple(round(float(c), 1) for c in p)

    async def _depth_view(self):
        machine = await self._connected()
        name = os.environ.get("VIAM_CAMERA", "cam")
        cam = await ts.resolve(machine, Camera, name)
        images, _ = await cam.get_images()
        dep = next((i for i in images if str(i.mime_type) == ts.DEPTH_MIME), None)
        if dep is None:
            raise RuntimeError("camera returned no depth frame")
        depth = ts.parse_depth(dep.data)
        h, w = depth.shape
        k = (await cam.get_properties()).intrinsic_parameters
        intr = (k.focal_x_px * w / k.width_px, k.focal_y_px * h / k.height_px,
                k.center_x_px * w / k.width_px, k.center_y_px * h / k.height_px)
        # Same per-frame levelling as ts.capture(), so clicked points and sweep
        # paths agree with the scan on where the table is.
        c2w, note = ts.level_to_table(depth, intr, await ts._cam_to_world(machine, name))
        print(note)
        return depth, intr, c2w

    async def trace_path(self, pixels: list, img_w: int, img_h: int) -> list:
        depth, (fx, fy, cx, cy), c2w = await self._depth_view()
        h, w = depth.shape
        v, u = np.nonzero(depth > 0)
        z = depth[v, u].astype(np.float64)
        rot, origin = c2w[:3, :3], c2w[:3, 3]
        world = np.column_stack([(u - cx) * z / fx, (v - cy) * z / fy, z]) @ rot.T + origin
        coef = fit_table_plane(world[::10])
        print(f"table plane: z = {coef[0]:.4f}x + {coef[1]:.4f}y + {coef[2]:.1f} mm")

        pts = []
        for pu, pv in pixels:
            du, dv = pu * w / img_w, pv * h / img_h
            p = ray_to_plane(origin, rot @ np.array([(du - cx) / fx, (dv - cy) / fy, 1.0]),
                             coef)
            if p is not None:
                # Never below the configured tabletop, which was checked by
                # hand; the camera's view of it only ever raises the path.
                pts.append((float(p[0]), float(p[1]), max(float(p[2]), tc.TABLE_TOP_Z_MM)))
        path = simplify_path(pts, SWEEP_SIMPLIFY_MM)
        tol = SWEEP_SIMPLIFY_MM
        while len(path) > SWEEP_MAX_WAYPOINTS:
            tol *= 1.5
            path = simplify_path(pts, tol)
        if len(path) < 2 or path_length(path) < SWEEP_MIN_LENGTH_MM:
            raise RuntimeError("path too short; drag a longer line")
        print("sweep path: " + " -> ".join(f"({x:.0f},{y:.0f},{z:.0f})" for x, y, z in path))
        return [tuple(round(c, 1) for c in p) for p in path]

    async def _rig(self) -> tc.Rig:
        machine = await self._connected()
        rig = tc.Rig(
            motion=await ts.resolve(machine, Motion, tc.MOTION_NAME),
            arm=self.arm,
            gripper=await ts.resolve(machine, Gripper, tc.GRIPPER_NAME),
        )
        if (await rig.gripper.is_holding_something()).is_holding_something:
            raise RuntimeError("gripper already reports holding something; "
                               "clear it before running")
        await rig.gripper.open()
        # Where this action returns the arm to at the end: the pose of the
        # latest scan, or (with no scan yet) wherever the arm is now.
        self.return_pose = self.scan_pose or await rig.arm.get_end_position()
        return rig

    async def _go_back(self, rig: tc.Rig, world: WorldState) -> list[str]:
        """Return the arm to where the latest scan was taken. [] if it got
        there, else a message for the chat."""
        s = self.return_pose
        print(f"Returning to the scan position ({s.x:.0f}, {s.y:.0f}, {s.z:.0f}).")
        if await tc.return_to_start(rig, s, world):
            return []
        return ["Could not get back to the scan position: the arm is stopped "
                "where it is. Press Clear arm error first if it faulted."]

    @staticmethod
    async def _guarded(rig: tc.Rig, coro):
        try:
            return await coro
        except BaseException:
            print("Interrupted — stopping arm.", file=sys.stderr)
            try:
                await rig.arm.stop()
            except Exception as e:
                print(f"  arm stop failed: {e}", file=sys.stderr)
            raise

    async def pick(self, targets: list[dict], objects: list[dict]) -> list[str]:
        rig = await self._rig()
        return await self._guarded(rig, self._pick(rig, targets, objects))

    async def _pick(self, rig, targets, objects) -> list[str]:
        on_table = [o for o in objects if tc.gate(o["position"]) is None]
        outcomes = []
        for obj in targets:
            print(f"  -> {obj['name']}")
            outcome, holding = await tc.pick_and_drop(
                rig, obj, tc.world_state([o for o in on_table if o is not obj]))
            print(f"     {outcome}")
            outcomes.append(f"{obj['name']}: {outcome}")
            if outcome == "dropped" or holding:
                on_table.remove(obj)
            if holding:
                outcomes.append("Stopped: the gripper is still holding an "
                                "object; remaining picks skipped.")
                break
        return outcomes + await self._go_back(rig, tc.world_state(on_table))

    async def place(self, obj: dict, point: tuple, objects: list[dict]) -> list[str]:
        rig = await self._rig()
        return await self._guarded(rig, self._place(rig, obj, point, objects))

    async def _place(self, rig, obj, point, objects) -> list[str]:
        others = [o for o in objects
                  if o is not obj and tc.gate(o["position"]) is None]
        world = tc.world_state(others)
        x, y, z = tc.grasp_point(obj["position"])
        tx, ty = point[0], point[1]
        th = tc.grasp_theta(obj["position"])
        release_z = z
        print(f"  -> {obj['name']} to ({tx:.1f}, {ty:.1f})")

        if not await tc.move(rig, tc.down_at(x, y, tc.SAFE_HEIGHT_MM, th),
                             "transit above object", world):
            return (["move to transit-above-object failed"]
                    + await self._go_back(rig, world))
        if not await tc.move(rig, tc.down_at(x, y, z, th), "grasp", world, straight=True):
            # It may have stopped part way down beside the object; rise
            # straight up before heading back.
            await tc.move(rig, tc.down_at(x, y, tc.SAFE_HEIGHT_MM, th),
                          "retreat after failed descent", world, straight=True)
            return (["descent to grasp point failed"]
                    + await self._go_back(rig, world))

        grabbed = await rig.gripper.grab()
        holding = (await rig.gripper.is_holding_something()).is_holding_something
        if not grabbed or not holding:
            await rig.gripper.open()
            await tc.move(rig, tc.down_at(x, y, tc.SAFE_HEIGHT_MM, th),
                          "retreat after failed grab", world, straight=True)
            return ([f"grab failed (grab()={grabbed}, holding={holding})"]
                    + await self._go_back(rig, world))

        steps = [
            (tc.down_at(x, y, tc.SAFE_HEIGHT_MM, th), "lift", True),
            (tc.down_at(tx, ty, tc.SAFE_HEIGHT_MM, th), "transit above target", False),
            (tc.down_at(tx, ty, release_z, th), "descent to target", True),
        ]
        for pose, label, straight in steps:
            if not await tc.move(rig, pose, label, world, straight=straight):
                return [f"{label} failed — the gripper is still holding "
                        f"{obj['name']}; clear it by hand"]

        await rig.gripper.open()
        still = (await rig.gripper.is_holding_something()).is_holding_something
        await tc.move(rig, tc.down_at(tx, ty, tc.SAFE_HEIGHT_MM, th),
                      "retreat from target", world, straight=True)
        if still:
            return ["release did not let go (still reports holding)"]

        moved = shifted(obj, tx - obj["position"]["centroid"][0],
                        ty - obj["position"]["centroid"][1])
        return ([f"{obj['name']}: placed at ({tx:.0f}, {ty:.0f}) mm"]
                + await self._go_back(rig, tc.world_state(others + [moved])))

    async def sweep(self, path: list, objects: list[dict]) -> list[str]:
        rig = await self._rig()
        await rig.gripper.grab()
        if (await rig.gripper.is_holding_something()).is_holding_something:
            await rig.gripper.open()
            raise RuntimeError("gripper reports holding something after closing; "
                               "check it's empty")
        try:
            return await self._guarded(rig, self._sweep(rig, path, objects))
        finally:
            try:
                await rig.arm.do_command({
                    "set_speed": ARM_NORMAL_SPEED_DEGS_PER_SEC,
                    "set_acceleration": ARM_NORMAL_ACCEL_DEGS_PER_SEC2})
                print(f"arm speed restored to {ARM_NORMAL_SPEED_DEGS_PER_SEC:.0f} deg/s")
            except Exception as e:
                print(f"!! could not restore arm speed: {e}", file=sys.stderr)

    async def _sweep(self, rig, path, objects) -> list[str]:
        lift = tc.CLAW_REACH_MM + SWEEP_CLEARANCE_MM
        sx, sy, sz = path[0]
        free = WorldState()
        world = tc.world_state(located(objects))

        th = None
        for cand in sweep_thetas(path[0], path[-1]):
            if await tc.move(rig, tc.down_at(sx, sy, tc.SAFE_HEIGHT_MM, cand),
                             f"transit above path start (theta {cand:.0f})", world):
                th = cand
                break
        if th is None:
            # Nothing was executed: the gripper is still up where it started,
            # so there is nothing to lift out of — just go back.
            return (["move above the path start failed with the gripper either "
                     "way round (see the terminal for what it would hit). Try "
                     "starting the path further from the walls."]
                    + await self._go_back(rig, world))

        await rig.arm.do_command({"set_speed": SWEEP_SPEED_DEGS_PER_SEC,
                                  "set_acceleration": SWEEP_ACCEL_DEGS_PER_SEC2})
        print(f"arm speed set to {SWEEP_SPEED_DEGS_PER_SEC:.0f} deg/s for the sweep")
        if not await sweep_move(rig, tc.down_at(sx, sy, sz + lift, th), "descent to path", free):
            return await self._sweep_home(rig, th, "descent to the path start failed")
        for i, (a, b) in enumerate(zip(path, path[1:]), 1):
            # Split into short straight steps, each well inside the planner's
            # 2mm line tolerance (see SWEEP_STEP_MM).
            steps = max(1, math.ceil(math.dist(a[:2], b[:2]) / SWEEP_STEP_MM))
            for k in range(1, steps + 1):
                x, y, z = (a[j] + (b[j] - a[j]) * k / steps for j in range(3))
                pose = tc.down_at(x, y, z + lift, th)
                label = f"path waypoint {i}/{len(path) - 1} step {k}/{steps}"
                if await sweep_move(rig, pose, label, free):
                    continue
                # Log where the arm is, to see which joint is at its limit.
                joints = (await rig.arm.get_joint_positions()).values
                print(f"    joints (deg) at ({x:.0f}, {y:.0f}): "
                      + ", ".join(f"{j:.0f}" for j in joints), file=sys.stderr)
                for constraints, how in SWEEP_STEP_FALLBACKS:
                    print(f"    retrying {label} with {how}", file=sys.stderr)
                    if await sweep_move(rig, pose, f"{label} ({how})", free,
                                        constraints):
                        break
                else:
                    return await self._sweep_home(
                        rig, th, f"sweep stopped on the way to waypoint "
                        f"{i}/{len(path) - 1}")
        return await self._sweep_home(rig, th, f"swept {path_length(path):.0f} mm")

    async def _sweep_home(self, rig, th: float, outcome: str) -> list[str]:
        p = (await rig.motion.get_pose(tc.GRIPPER_NAME, "world")).pose
        # Only lift if the gripper is actually down below transit height; a
        # "lift" from higher up would be a descent, and fails the line rule.
        if p.z < tc.SAFE_HEIGHT_MM - 1.0 and not await sweep_move(
                rig, tc.down_at(p.x, p.y, tc.SAFE_HEIGHT_MM, th),
                "lift after sweep", WorldState()):
            return [outcome, "lift afterwards failed: the gripper is still down at "
                    "the table. If the arm faulted, press Clear arm error, then "
                    "jog it up before anything else."]
        await rig.gripper.open()
        # Back to normal speed before the long move back: at sweep speed it
        # crawls, and can run out the move timeout and be stopped part way.
        # (sweep()'s finally restores it again, in case we never get here.)
        await rig.arm.do_command({"set_speed": ARM_NORMAL_SPEED_DEGS_PER_SEC,
                                  "set_acceleration": ARM_NORMAL_ACCEL_DEGS_PER_SEC2})
        return [outcome] + await self._go_back(rig, WorldState())

    async def reconstruct(self, objects: list[dict], bridge) -> tr.ReconstructResult:
        """Orbit the scene, build Poisson mesh, mark all grasp points in render."""
        machine = await self._connected()

        def on_progress(i: int, n: int):
            bridge.reconstruct_progress.emit(i, n)

        return await tr.reconstruct(
            machine,
            objects,
            n_poses=RECONSTRUCT_N_POSES,
            radius=RECONSTRUCT_RADIUS_MM,
            height=RECONSTRUCT_HEIGHT_MM,
            on_progress=on_progress,
            # Come back to where the latest scan was taken.
            return_pose=self.scan_pose or await self.arm.get_end_position(),
        )

    async def clear_error(self) -> list[str]:
        await self._connected()
        resp = await self.arm.do_command({"clear_error": True})
        return [f"clear_error sent to the arm ({resp or 'ok'})"]

    async def stop_arm(self):
        if self.arm is not None:
            await self.arm.stop()

    def close(self):
        if self.machine is not None:
            try:
                self.submit(self.machine.close()).result(timeout=5)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)


MOVING = ("pick", "place", "sweep", "reconstruct")

CONFIRM_HINT = "<br>Press <b>Execute plan</b> or reply <b>go</b> to start."


class ClickLabel(QLabel):
    clicked = pyqtSignal(int, int, int)
    stroking = pyqtSignal(list)
    stroked = pyqtSignal(list)

    def __init__(self, *args):
        super().__init__(*args)
        self._stroke = None

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._stroke = [(e.x(), e.y())]
        else:
            self.clicked.emit(e.x(), e.y(), int(e.button()))

    def mouseMoveEvent(self, e):
        if self._stroke is not None:
            self._stroke.append((e.x(), e.y()))
            if self._is_drag():
                self.stroking.emit(list(self._stroke))

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.LeftButton or self._stroke is None:
            return
        stroke, self._stroke = self._stroke, None
        if self._is_drag(stroke):
            self.stroked.emit(stroke)
        else:
            self.clicked.emit(stroke[0][0], stroke[0][1], int(Qt.LeftButton))

    def _is_drag(self, stroke=None) -> bool:
        s = stroke or self._stroke
        x0, y0 = s[0]
        return any(math.hypot(x - x0, y - y0) >= DRAG_THRESHOLD_PX for x, y in s)


class Bridge(QObject):
    log = pyqtSignal(str)
    done = pyqtSignal(str, object, object)
    frame = pyqtSignal(bytes)
    reconstruct_progress = pyqtSignal(int, int)   # i, n_poses


class LogStream:
    def __init__(self, bridge: Bridge, original):
        self.bridge, self.original = bridge, original

    ANSI = re.compile(r"\x1b\[[0-9;]*m")

    def write(self, text):
        if self.original:
            self.original.write(text)
        if text:
            self.bridge.log.emit(self.ANSI.sub("", text))

    def flush(self):
        if self.original:
            self.original.flush()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Table Cleanup")
        self.resize(1400, 850)

        self.robot = Robot()
        self.bridge = Bridge()
        self.bridge.log.connect(self._append_log)
        self.bridge.done.connect(self._finished)
        self.bridge.reconstruct_progress.connect(self._reconstruct_progress)
        sys.stdout = LogStream(self.bridge, sys.__stdout__)
        sys.stderr = LogStream(self.bridge, sys.__stderr__)

        self.objects: list[dict] = []
        self.marked: bytes | None = None
        self.pending: dict | None = None
        self.target: dict | None = None
        self.path_px: list | None = None
        self.path: list | None = None
        self.history: list[dict] = []
        self.busy = None
        self.job = None
        self.last_reconstruct: tr.ReconstructResult | None = None

        self._build()
        self._say("robot", "Hi. Press <b>Scan</b> (or just tell me what to "
                  "throw away) and I'll look at the table first. To move "
                  "something instead, click a spot on the live feed, then "
                  "tell me which object to put there. To push things aside, "
                  "drag a path across the live feed. Press <b>3D Scan</b> to "
                  "orbit the scene and build a Poisson mesh with grasp points marked.")

        self.live_pix: QPixmap | None = None
        self.bridge.frame.connect(self._show_live)
        self.live_enabled = True
        self.live_on.toggled.connect(lambda on: setattr(self, "live_enabled", on))
        self.feed = self.robot.submit(self.robot.feed(
            self.bridge.frame.emit, lambda: self.live_enabled))

    # ---- layout ----------------------------------------------------------

    @staticmethod
    def _picture(placeholder: str, cls=QLabel) -> QLabel:
        label = cls(placeholder)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(320, 180)
        label.setStyleSheet("background:#222; color:#aaa;")
        return label

    def _build(self):
        self.live = self._picture("Connecting to camera...", ClickLabel)
        self.live.setToolTip("Click: set the target point.  Drag: draw a sweep path.  "
                             "Right-click: clear both.")
        self.live.setCursor(Qt.CrossCursor)
        self.live.clicked.connect(self._live_clicked)
        self.live.stroking.connect(self._live_stroking)
        self.live.stroked.connect(self._live_stroked)
        self.live_on = QCheckBox("Live feed")
        self.live_on.setChecked(True)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Object", "Scan says", "Arm"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)

        self.scan_btn = QPushButton("Scan")
        self.reconstruct_btn = QPushButton("3D Scan")
        self.reconstruct_btn.setToolTip(
            "Orbit the arm around the scene, run Poisson mesh reconstruction, "
            "and display grasp points for all pickable objects in the 3D view tab."
        )
        self.exec_btn = QPushButton("Execute plan")
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setStyleSheet(
            "QPushButton{background:#c62828;color:white;font-weight:bold;padding:6px 18px}"
            "QPushButton:disabled{background:#7a4a4a;color:#ddd}")
        self.clear_btn = QPushButton("Clear arm error")
        self.clear_btn.setToolTip("After a collision fault (e.g. the fingers touched "
                                  "the table), remove the cause, then press this.")
        self.scan_btn.clicked.connect(lambda: self._start_scan())
        self.reconstruct_btn.clicked.connect(self._start_reconstruct)
        self.exec_btn.clicked.connect(self._execute)
        self.stop_btn.clicked.connect(self._stop)
        self.clear_btn.clicked.connect(
            lambda: self._run("clear", self.robot.clear_error()))
        buttons = QHBoxLayout()
        for b in (self.scan_btn, self.reconstruct_btn, self.exec_btn, self.clear_btn):
            buttons.addWidget(b)
        buttons.addWidget(self.live_on)
        buttons.addStretch()
        buttons.addWidget(self.stop_btn)

        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        self.input = QLineEdit()
        self.input.setPlaceholderText(
            "e.g. throw away the crushed can  /  move the block to the point")
        self.input.returnPressed.connect(self._send)
        send = QPushButton("Send")
        send.clicked.connect(self._send)
        row = QHBoxLayout()
        row.addWidget(self.input)
        row.addWidget(send)
        self.send_btn = send

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(QFont("monospace", 9))

        # ── Bottom-right: tabbed panel with flat scan + 3D view ──
        self.image_tabs = QTabWidget()
        self.image_tabs.setDocumentMode(True)

        # Tab 0: flat scan image
        self.image = self._picture("No scan yet")
        self.image_tabs.addTab(self.image, "Scan")

        # Tab 1: 3D reconstruction view
        mesh_container = QWidget()
        mesh_vbox = QVBoxLayout(mesh_container)
        mesh_vbox.setContentsMargins(0, 0, 0, 0)
        mesh_vbox.setSpacing(4)

        self.mesh_view = self._picture("No 3D scan yet")
        self.mesh_progress = QProgressBar()
        self.mesh_progress.setRange(0, RECONSTRUCT_N_POSES)
        self.mesh_progress.setTextVisible(True)
        self.mesh_progress.setFormat("Orbit: %v / %m viewpoints")
        self.mesh_progress.setMaximumHeight(18)
        self.mesh_progress.setVisible(False)
        self.mesh_info = QLabel("")
        self.mesh_info.setAlignment(Qt.AlignCenter)
        self.mesh_info.setStyleSheet("font-size: 11px; color: #aaa; padding: 2px;")

        mesh_vbox.addWidget(self.mesh_view)
        mesh_vbox.addWidget(self.mesh_progress)
        mesh_vbox.addWidget(self.mesh_info)
        self.image_tabs.addTab(mesh_container, "3D view")

        # Left pane
        controls = QWidget()
        bv = QVBoxLayout(controls)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.addWidget(self.table)
        bv.addLayout(buttons)
        left = QSplitter(Qt.Vertical)
        left.addWidget(self.live)
        left.addWidget(controls)
        left.addWidget(self.log)
        left.setSizes([450, 250, 150])

        # Right pane
        chat_box = QWidget()
        cv = QVBoxLayout(chat_box)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.addWidget(self.chat)
        cv.addLayout(row)
        right = QSplitter(Qt.Vertical)
        right.addWidget(chat_box)
        right.addWidget(self.image_tabs)
        right.setSizes([450, 400])
        for s in (left, right):
            s.splitterMoved.connect(lambda *_: self._rescale())

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([800, 600])
        self.setCentralWidget(split)
        self._refresh_controls()

    # ---- helpers ---------------------------------------------------------

    def _say(self, who: str, html: str):
        colour = {"you": "#1565c0", "robot": "#2e7d32", "error": "#c62828"}[who]
        label = {"you": "You", "robot": "Robot", "error": "Error"}[who]
        self.chat.append(f'<p><b style="color:{colour}">{label}:</b> {html}</p>')

    def _append_log(self, text: str):
        self.log.moveCursor(self.log.textCursor().End)
        self.log.insertPlainText(text)
        self.log.moveCursor(self.log.textCursor().End)

    def _refresh_controls(self):
        idle = self.busy is None
        self.scan_btn.setEnabled(idle)
        self.reconstruct_btn.setEnabled(idle and bool(self.objects))
        self.clear_btn.setEnabled(idle)
        self.exec_btn.setEnabled(idle and bool(self.pending))
        self.input.setEnabled(idle)
        self.send_btn.setEnabled(idle)
        self.stop_btn.setEnabled(self.busy in MOVING)
        status = {
            None: "Ready",
            "scan": "Scanning the table...",
            "chat": "Thinking...",
            "locate": "Locating the clicked point...",
            "trace": "Tracing the path onto the table...",
            "clear": "Clearing the arm error...",
            "pick": "Arm moving — STOP halts it",
            "place": "Arm moving — STOP halts it",
            "sweep": "Sweeping slowly — STOP halts it",
            "reconstruct": f"Orbiting + reconstructing ({RECONSTRUCT_N_POSES} poses) — STOP halts arm",
        }[self.busy]
        if self.target and idle:
            x, y, _ = self.target["world"]
            status += f"   |   target point ({x:.0f}, {y:.0f}) mm"
        self.statusBar().showMessage(status)

    def _run(self, kind: str, coro, context=None):
        self.busy = kind
        self._refresh_controls()
        self.job = self.robot.submit(coro)

        def report(fut):
            if fut.cancelled():
                self.bridge.done.emit(kind, context, "cancelled")
            elif fut.exception():
                e = fut.exception()
                traceback.print_exception(type(e), e, e.__traceback__)
                self.bridge.done.emit(kind, context, e)
            else:
                self.bridge.done.emit(kind, (fut.result(), context), None)
        self.job.add_done_callback(report)

    @staticmethod
    def _fit(label: QLabel, pix: QPixmap):
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation))

    def _with_marker(self, pix: QPixmap) -> QPixmap:
        if not self.target and not self.path_px:
            return pix
        pix = pix.copy()
        r = max(pix.width() // 60, 8)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        if self.path_px and len(self.path_px) > 1:
            pts = [(int(u), int(v)) for u, v in self.path_px]
            style = Qt.SolidLine if self.path else Qt.DashLine
            for colour, width in ((QColor("black"), 8), (QColor("#ff9100"), 4)):
                p.setPen(QPen(colour, width, style, Qt.RoundCap, Qt.RoundJoin))
                for a, b in zip(pts, pts[1:]):
                    p.drawLine(a[0], a[1], b[0], b[1])
            p.setPen(QPen(QColor("black"), 2))
            p.setBrush(QColor("#ff9100"))
            p.drawEllipse(pts[0][0] - r // 2, pts[0][1] - r // 2, r, r)
            (x0, y0), (x1, y1) = pts[max(len(pts) - 6, 0)], pts[-1]
            ang = math.atan2(y1 - y0, x1 - x0)
            for side in (-1, 1):
                a = ang + math.pi - side * math.radians(28)
                p.setPen(QPen(QColor("#ff9100"), 4, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(x1, y1, int(x1 + 2 * r * math.cos(a)), int(y1 + 2 * r * math.sin(a)))
        if self.target:
            u, v = self.target["px"]
            p.setBrush(Qt.NoBrush)
            for colour, width in ((QColor("black"), 7), (QColor("#00e5ff"), 3)):
                p.setPen(QPen(colour, width))
                p.drawEllipse(int(u - r), int(v - r), 2 * r, 2 * r)
                p.drawLine(int(u - 2 * r), int(v), int(u - r // 2), int(v))
                p.drawLine(int(u + r // 2), int(v), int(u + 2 * r), int(v))
                p.drawLine(int(u), int(v - 2 * r), int(u), int(v - r // 2))
                p.drawLine(int(u), int(v + r // 2), int(u), int(v + 2 * r))
        p.end()
        return pix

    def _show_live(self, data: bytes):
        pix = QPixmap()
        if pix.loadFromData(data):
            self.live_pix = pix
            self._fit(self.live, self._with_marker(pix))

    def _rescale(self):
        if self.live_pix is not None:
            self._fit(self.live, self._with_marker(self.live_pix))
        if self.marked:
            pix = QPixmap()
            pix.loadFromData(self.marked)
            self._fit(self.image, self._with_marker(pix))
        if self.last_reconstruct is not None:
            pix = QPixmap()
            if pix.loadFromData(self.last_reconstruct.render_jpeg):
                self._fit(self.mesh_view, pix)

    def _to_image(self, x: float, y: float, clamp: bool = False):
        shown = self.live.pixmap()
        if self.live_pix is None or shown is None or shown.isNull():
            return None
        ox = (self.live.width() - shown.width()) / 2
        oy = (self.live.height() - shown.height()) / 2
        if clamp:
            x = min(max(x, ox), ox + shown.width() - 1)
            y = min(max(y, oy), oy + shown.height() - 1)
        elif not (ox <= x < ox + shown.width() and oy <= y < oy + shown.height()):
            return None
        return ((x - ox) * self.live_pix.width() / shown.width(),
                (y - oy) * self.live_pix.height() / shown.height())

    def _can_mark(self) -> bool:
        if self.busy:
            self.statusBar().showMessage("Wait until the current job finishes "
                                         "before marking the feed.", 4000)
            return False
        return self.live_pix is not None

    def _live_clicked(self, x: int, y: int, button: int):
        if button == Qt.RightButton:
            cleared = [n for n, v in (("target point", self.target),
                                      ("sweep path", self.path_px)) if v]
            if cleared and not self.busy:
                self.target = self.path_px = self.path = None
                if self.pending and self.pending["kind"] in ("place", "sweep"):
                    self.pending = None
                self._say("robot", " and ".join(cleared).capitalize() + " cleared.")
                self._rescale()
                self._show_scan()
                self._refresh_controls()
            return
        uv = self._to_image(x, y)
        if uv is None or not self._can_mark():
            return
        self._run("locate", self.robot.locate(*uv, self.live_pix.width(),
                                              self.live_pix.height()), uv)

    def _live_stroking(self, stroke: list):
        if self.busy or self.live_pix is None:
            return
        self.path = None
        self.path_px = [self._to_image(x, y, clamp=True) for x, y in stroke]
        self._rescale()

    def _live_stroked(self, stroke: list):
        if not self._can_mark():
            return
        self.path = None
        self.path_px = [self._to_image(x, y, clamp=True) for x, y in stroke]
        if self.pending and self.pending["kind"] == "sweep":
            self.pending = None
        self._rescale()
        self._run("trace", self.robot.trace_path(
            self.path_px, self.live_pix.width(), self.live_pix.height()))

    def _show_scan(self):
        self._rescale()
        pending_ids = {o["id"] for o in (self.pending or {}).get("targets", [])}
        self.table.setRowCount(len(self.objects))
        for r, o in enumerate(self.objects):
            why = tc.gate(o["position"])
            cells = [str(o["mark"]) if o["mark"] else "–", o["name"],
                     "trash" if o["is_trash"] else "keep",
                     "reachable" if why is None else why]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(o["reason"] if c < 3 else text)
                if o["id"] in pending_ids:
                    item.setBackground(QColor("#ffe082"))
                self.table.setItem(r, c, item)
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(2)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    # ---- actions ---------------------------------------------------------

    def _start_scan(self, then_message: str | None = None):
        self.pending = None
        self._run("scan", self.robot.scan(), then_message)

    def _start_reconstruct(self):
        if self.busy or not self.objects:
            return
        n_located = sum(1 for o in self.objects
                        if o.get("position") and "centroid" in o["position"])
        if n_located == 0:
            self._say("error", "Run a scan first so I have object positions to orbit.")
            return
        n_pickable = sum(1 for o in self.objects if tc.gate(o["position"]) is None)
        self._say(
            "robot",
            f"Starting 3D scene scan — {RECONSTRUCT_N_POSES} orbit viewpoints around "
            f"the mean centroid of {n_located} located object(s). "
            f"{n_pickable} pickable object(s) will be marked in the render. "
            "Press <b>STOP</b> to halt the arm."
        )
        self.mesh_progress.setMaximum(RECONSTRUCT_N_POSES)
        self.mesh_progress.setValue(0)
        self.mesh_progress.setVisible(True)
        self.mesh_info.setText("Starting orbit…")
        self.image_tabs.setCurrentIndex(1)
        self._run("reconstruct", self.robot.reconstruct(self.objects, self.bridge))

    def _reconstruct_progress(self, i: int, n: int):
        self.mesh_progress.setValue(i)
        if i < n:
            self.mesh_info.setText(f"Capturing viewpoint {i}/{n}…")
        else:
            self.mesh_info.setText("Running Poisson reconstruction…")

    def _send(self):
        text = self.input.text().strip()
        if not text or self.busy:
            return
        self.input.clear()
        self._say("you", text)
        if not self.objects:
            self._say("robot", "Let me look at the table first.")
            self._start_scan(then_message=text)
            return
        self._ask(text)

    def _ask(self, text: str):
        pending, has_point = self.pending, self.target is not None
        has_path = self.path is not None

        async def job():
            return await asyncio.to_thread(
                interpret, text, self.objects, self.marked, pending, has_point,
                has_path, list(self.history))
        self._run("chat", job(), text)

    def _execute(self):
        if not self.pending or self.busy:
            return
        plan, self.pending = self.pending, None
        names = ", ".join(o["name"] for o in plan["targets"])
        if plan["kind"] == "sweep":
            self._say("robot", "Sweeping along the path. Press <b>STOP</b> to halt "
                      "the arm.")
            self._run("sweep", self.robot.sweep(plan["path"], self.objects))
        elif plan["kind"] == "place":
            x, y, _ = plan["point"]
            self._say("robot", f"Moving {names} to ({x:.0f}, {y:.0f}) mm. "
                      "Press <b>STOP</b> to halt the arm.")
            self._run("place", self.robot.place(plan["targets"][0], plan["point"],
                                                self.objects))
        else:
            self._say("robot", f"Throwing away: {names}. "
                      "Press <b>STOP</b> to halt the arm.")
            self._run("pick", self.robot.pick(plan["targets"], self.objects))

    def _stop(self):
        self.robot.submit(self.robot.stop_arm())
        if self.job is not None:
            self.robot.loop.call_soon_threadsafe(self.job.cancel)
        self._say("error", "STOP pressed — arm halted where it is.")

    def _plan(self, ids: list[str]) -> str:
        by_id = {o["id"]: o for o in self.objects}
        chosen = [by_id[i] for i in ids if i in by_id]
        ok = [o for o in chosen if tc.gate(o["position"]) is None]
        lines = []
        for o in chosen:
            why = tc.gate(o["position"])
            tag = f"#{o['mark']} " if o["mark"] else ""
            if why:
                lines.append(f"✗ {tag}{o['name']} — can't pick: {why}")
            else:
                note = "" if o["is_trash"] else " <i>(the scan judged this not trash)</i>"
                lines.append(f"✓ {tag}{o['name']}{note}")
        self.pending = {"kind": "pick", "targets": ok, "point": None} if ok else None
        if not chosen:
            return "I couldn't match that to anything from the last scan."
        tail = (CONFIRM_HINT if ok
                else "<br>None of these can be picked, so there's nothing to run.")
        return "<br>".join(lines) + tail

    def _plan_sweep(self) -> str:
        self.pending = None
        if not self.path:
            return "Drag a path across the live feed first."
        start = self.path[0]
        for o in located(self.objects):
            if _box_gap(start, o) < SWEEP_START_CLEARANCE_MM:
                tag = f"#{o['mark']} " if o["mark"] else ""
                return (f"✗ The path starts on or right next to {tag}{o['name']}; "
                        "the gripper would come down on top of it. Start the path "
                        "on clear table and drag through the objects.")
        hits = path_crosses(self.path, self.objects)
        crossing = (", ".join((f"#{o['mark']} " if o["mark"] else "") + o["name"]
                              for o in hits) if hits else "nothing from the last scan")
        self.pending = {"kind": "sweep", "targets": hits, "path": self.path}
        return (f"✓ Sweep {path_length(self.path):.0f} mm through "
                f"{len(self.path)} waypoints, gripper closed, fingertips "
                f"{SWEEP_CLEARANCE_MM:.0f} mm above the table, at "
                f"{SWEEP_SPEED_DEGS_PER_SEC:.0f} deg/s. It will push: {crossing}."
                + CONFIRM_HINT)

    def _plan_place(self, ids: list[str]) -> str:
        self.pending = None
        by_id = {o["id"]: o for o in self.objects}
        chosen = [by_id[i] for i in ids if i in by_id]
        if not self.target:
            return "Click the spot on the live feed where it should go first."
        if len(chosen) != 1:
            return ("I couldn't match that to exactly one object from the last scan."
                    if not chosen else "Only one object can go to the point at a time.")
        obj = chosen[0]
        tag = f"#{obj['mark']} " if obj["mark"] else ""
        why = tc.gate(obj["position"])
        if why:
            return f"✗ {tag}{obj['name']} — can't pick: {why}"
        point = self.target["world"]
        blocker = blocking_object(obj, point, self.objects)
        if blocker:
            btag = f"#{blocker['mark']} " if blocker["mark"] else ""
            return (f"✗ The point is too close to {btag}{blocker['name']} for "
                    f"{obj['name']} ({obj['position']['length_mm']:.0f} mm long) "
                    "to fit. Pick a clearer spot.")
        self.pending = {"kind": "place", "targets": [obj], "point": point}
        return (f"✓ Move {tag}{obj['name']} to ({point[0]:.0f}, {point[1]:.0f}) mm, "
                "set down at the height it's picked from." + CONFIRM_HINT)

    def _finished(self, kind: str, payload, error):
        self.busy, self.job = None, None
        if error is not None:
            if error == "cancelled":
                self._say("error", "Stopped.")
            else:
                self._say("error", f"{kind} failed: {error}")
            if kind == "trace":
                self.path_px = None
                self._rescale()
            if kind == "reconstruct":
                self.mesh_progress.setVisible(False)
                self.mesh_info.setText("Reconstruction failed — see log.")
            self._refresh_controls()
            return

        result, context = payload
        if kind == "trace":
            self.path = result
            self._say("robot", self._plan_sweep())
            self._rescale()
            self._show_scan()
        elif kind == "clear":
            self._say("robot", "<br>".join(result))
        elif kind == "locate":
            self.target = {"px": context, "world": result}
            x, y, z = result
            self._say("robot", f"Target point set at ({x:.0f}, {y:.0f}) mm. "
                      "Now tell me which object to move there.")
            if self.pending and self.pending["kind"] == "place":
                self.pending = None
                self._say("robot", "(The plan waiting for confirmation used the "
                          "old point, so I dropped it — ask again.)")
            self._rescale()
            self._show_scan()
        elif kind == "scan":
            scan, self.marked = result
            self.objects = scan["objects"]
            self.history.clear()
            n_trash = sum(o["is_trash"] for o in self.objects)
            self._say("robot", f"I see {len(self.objects)} object(s), "
                      f"{n_trash} of them look like trash.")
            self._show_scan()
            if context:
                self._refresh_controls()
                self._ask(context)
                return
        elif kind == "reconstruct":
            result: tr.ReconstructResult
            self.last_reconstruct = result
            pix = QPixmap()
            if pix.loadFromData(result.render_jpeg):
                self._fit(self.mesh_view, pix)
            self.mesh_progress.setVisible(False)
            grasp_summary = (
                f"{len(result.grasp_points)} grasp point(s) marked"
                if result.grasp_points else "no pickable objects to mark"
            )
            self.mesh_info.setText(
                f"{result.n_triangles:,} triangles · "
                f"{result.n_views}/{RECONSTRUCT_N_POSES} views · "
                f"{grasp_summary}"
            )
            self._say(
                "robot",
                f"3D scan complete: <b>{result.n_triangles:,} triangles</b> from "
                f"{result.n_views} viewpoints, {result.n_points:,} points. "
                f"{grasp_summary.capitalize()}. "
                f"Mesh saved to <code>{result.mesh_ply}</code>."
            )
        elif kind == "chat":
            action, reply = result["action"], result["reply"]
            self.history += [{"role": "user", "content": context},
                             {"role": "assistant", "content": json.dumps(result)}]
            self._say("robot", reply)
            if action == "pick":
                self._say("robot", self._plan(result["target_ids"]))
                self._show_scan()
            elif action == "place":
                self._say("robot", self._plan_place(result["target_ids"]))
                self._show_scan()
            elif action == "sweep":
                self._say("robot", self._plan_sweep())
                self._show_scan()
            elif action == "cancel":
                self.pending = None
                self._show_scan()
            elif action == "scan":
                self._start_scan()
                return
            elif action == "confirm":
                if self.pending:
                    self._execute()
                    return
                self._say("robot", "There's no plan waiting to confirm.")
        elif kind in MOVING:
            if kind == "place" and "placed at" in result[-1]:
                self.target = None
            if kind == "sweep":
                self.path_px = self.path = None
            self._say("robot", "<br>".join(result) + "<br>Scanning again...")
            self._start_scan()
            return
        self._refresh_controls()

    def closeEvent(self, e):
        if self.busy in MOVING:
            self._stop()
        self.robot.loop.call_soon_threadsafe(self.feed.cancel)
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        self.robot.close()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
