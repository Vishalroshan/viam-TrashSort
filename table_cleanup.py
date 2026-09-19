"""
table_cleanup.py — scan the table, then pick up and drop every object
classified as trash into a fixed drop location.

Reuses table_scan's connect/capture/scan to identify what's on the table and
where each object sits in world-frame mm. For each
object marked trash, if its 3D position is trustworthy and it fits between
the jaws, the gripper:
  1. moves to a safe transit height above the object, turned so the jaws
     will close across the object's narrow side,
  2. descends in a straight line to the grasp point,
  3. grabs, and confirms it actually holds something,
  4. lifts straight back up, moves over the drop pose, descends, releases,
     and confirms the release actually let go,
  5. retreats straight back up to transit height.

Every move targets the `gripper` frame (not the arm flange), and every plan
is given the other localized objects as obstacles, so the planner routes
around things that are staying on the table. The table, walls and ceiling
are already obstacles in the machine's own frame system config.

Objects whose depth signal is too sparse to trust (a common failure mode for
clear plastic and shiny metal under IR depth sensing — the water bottle on
this table is the running example) are skipped and reported rather than
risking a blind grasp.

There is no confirmation prompt: this script scans and immediately acts on
whatever it classifies as trash. Ctrl-C stops the arm where it is.

Setup: same as table_scan.py — .env populated, .venv/bin/pip install
       viam-sdk anthropic python-dotenv numpy pillow

Run:
    .venv/bin/python table_cleanup.py             # scan, then act
    .venv/bin/python table_cleanup.py --dry-run    # scan, print the plan, touch nothing physical
"""

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass

from grpclib.exceptions import GRPCError
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Vector3,
    WorldState,
)
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import Motion

import table_scan as ts

# --- Frame-system facts, read off this machine's config --------------------

# The `gripper` frame sits 150mm out along the arm's tool axis, and its
# "claws" geometry reaches 50mm beyond that origin. Pointing straight down, a
# gripper target at height z puts the fingertips at z - CLAW_REACH_MM.
CLAW_REACH_MM = 50.0

TABLE_TOP_Z_MM = ts.TABLE_TOP_Z_MM

# --- Motion constants -------------------------------------------------------
# All heights below are for the gripper frame, not the arm flange.

SAFE_HEIGHT_MM = 200.0            # transit height; fingertips at 150mm
GRASP_TABLE_CLEARANCE_MM = 10.0   # never plan fingertips closer than this to the table
OBSTACLE_PADDING_MM = 10.0        # grow each object's box by this on every side
MOVE_TIMEOUT_S = 60.0

# Every synthetic pick/drop pose points the end effector straight down
# (o_z=-1); theta is the roll about that axis and is chosen per object.
DOWN = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)

# Pointing straight down, the gripper frame's y axis lies at world yaw
# 90 - theta (checked against the machine's frame system at several thetas).
# The jaws close along the gripper's y axis, confirmed on the real arm. If
# they ever close along x instead, set this to 90.
JAW_AXIS_OFFSET_DEG = 0.0

# Inner gap between the fully open jaws, measured on the real gripper.
# Objects wider than this across their narrow side are skipped, not attempted.
GRIPPER_MAX_OPENING_MM = 85.0

# Vertical approaches and retreats run close to neighbouring objects, so they
# go in a straight line rather than wherever the planner's path wanders.
STRAIGHT = Constraints(
    linear_constraint=[
        LinearConstraint(line_tolerance_mm=5.0, orientation_tolerance_degs=5.0)
    ]
)

# The previous arm-flange drop pose (z=206.2) less the 150mm gripper offset.
DROP_POSE = Pose(x=292.5, y=-399.9, z=56.2, **DOWN)

# Verbatim from viam_move_to_centroid.py — an *arm* pose the user measured by
# jogging the real arm there, not one this script can re-derive.
HOME_POSE = Pose(
    x=263.16, y=-39.11, z=442.47,
    o_x=-0.0067, o_y=-0.1893, o_z=-0.9819, theta=85.94,
)

ARM_NAME = "arm"
GRIPPER_NAME = "gripper"
MOTION_NAME = "builtin"

# --- Depth-confidence gate --------------------------------------------------
# A 2D box always catches some of the object's real depth signal, but not
# always enough to trust for a physical grasp. Clear plastic and shiny metal
# in particular return little or no IR.
#
# Loosened deliberately (was 0.15 / 200 / 10-400mm): on this table's actual
# test scene — two aluminum cans, a crumpled paper ball, a clear bottle — the
# stricter gate skipped all four. The looser numbers below let a real-but-
# sparse reading through (a can read at 13% coverage, a bottle at 5%) but
# still reject pure noise. What this does NOT fix: a specular can can return
# a geometrically wrong box even at good coverage (one read 8661 points at
# 28% coverage but only 8.9mm thick, versus a real ~66mm-diameter can) — the
# centroid from a reading like that is still only an approximation of the
# object's near-facing surface, not a verified true center. Loosening the
# gate is a deliberate accuracy-for-coverage tradeoff, not a fix to the
# underlying sparsity.
MIN_COVERAGE = 0.03
MIN_POINTS = 100
SIZE_BOUNDS_MM = (5.0, 400.0)


def gate(position: dict | None) -> str | None:
    """Return None if position is trustworthy enough to act on, else why not."""
    if position is None or "centroid" not in position:
        reason = position["reason"] if position else "no depth frame"
        return f"no usable 3D position ({reason})"
    if position["coverage"] < MIN_COVERAGE:
        return f"depth coverage too low ({position['coverage']:.0%} of box)"
    if position["points"] < MIN_POINTS:
        return f"too few depth points ({position['points']})"
    lo, hi = SIZE_BOUNDS_MM
    if not all(lo <= s <= hi for s in position["size"]):
        return f"implausible size {position['size']} mm"
    if position["max"][2] <= TABLE_TOP_Z_MM:
        return f"reads as below the table surface (top z={position['max'][2]} mm)"
    if position["width_mm"] > GRIPPER_MAX_OPENING_MM:
        return (f"too wide for the gripper ({position['width_mm']:.0f} mm across "
                f"its narrow side, jaws open {GRIPPER_MAX_OPENING_MM:.0f} mm)")
    return None


def grasp_theta(position: dict) -> float:
    """Gripper roll that closes the jaws across the object's narrow side.

    Jaws close along world yaw 90 - theta + JAW_AXIS_OFFSET_DEG; that must be
    perpendicular to the long axis, i.e. long_axis + 90. Solving gives
    theta = JAW_AXIS_OFFSET_DEG - long_axis. The jaws are symmetric, so theta
    and theta + 180 grip the same way; keep the one within 90 degrees of the
    home roll, so the wrist never spins half a turn for nothing.
    """
    theta = JAW_AXIS_OFFSET_DEG - position["long_axis_deg"]
    lo = HOME_POSE.theta - 90.0
    return lo + (theta - lo) % 180.0


def resting_box(position: dict) -> tuple[list, list]:
    """The object's world-frame box, extended down to the table.

    Looking down, depth only sees an object's top surface — a standing can
    read as 13mm thick at z=101..114. Everything seen is resting on the
    table, so the part the camera can't see runs down to the tabletop.
    """
    lo, hi = list(position["min"]), list(position["max"])
    lo[2] = min(lo[2], TABLE_TOP_Z_MM)
    return lo, hi


def grasp_point(position: dict) -> tuple[float, float, float]:
    """Gripper-frame target: the middle of the resting box, kept clear of the table."""
    lo, hi = resting_box(position)
    x, y, _ = position["centroid"]
    z = max((lo[2] + hi[2]) / 2.0,
            TABLE_TOP_Z_MM + CLAW_REACH_MM + GRASP_TABLE_CLEARANCE_MM)
    return x, y, z


def world_state(objs: list[dict]) -> WorldState:
    """Every object still on the table, as padded boxes the planner must avoid."""
    p = OBSTACLE_PADDING_MM
    geoms = []
    for o in objs:
        lo, hi = resting_box(o["position"])
        c = [(a + b) / 2.0 for a, b in zip(lo, hi)]
        d = [b - a + 2 * p for a, b in zip(lo, hi)]
        geoms.append(Geometry(
            # o_z=1 is the identity orientation; an all-zero one is invalid.
            center=Pose(x=c[0], y=c[1], z=c[2], o_z=1.0),
            box=RectangularPrism(dims_mm=Vector3(x=d[0], y=d[1], z=d[2])),
            label=o["id"],
        ))
    return WorldState(
        obstacles=[GeometriesInFrame(reference_frame="world", geometries=geoms)]
    )


def down_at(x: float, y: float, z: float, theta: float = 0.0) -> Pose:
    return Pose(x=x, y=y, z=z, **{**DOWN, "theta": theta})


@dataclass
class Rig:
    motion: Motion
    arm: Arm
    gripper: Gripper


async def move(rig: Rig, pose: Pose, label: str, world: WorldState, *,
               component: str = GRIPPER_NAME, straight: bool = False) -> bool:
    try:
        ok = await rig.motion.move(
            component_name=component,
            destination=PoseInFrame(reference_frame="world", pose=pose),
            world_state=world,
            constraints=STRAIGHT if straight else None,
            timeout=MOVE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        # The deadline only ends the RPC; stop the arm in case execution didn't.
        await rig.arm.stop()
        print(f"    move to {label} timed out after {MOVE_TIMEOUT_S:.0f}s",
              file=sys.stderr)
        return False
    except GRPCError as e:
        # A plan the solver can't satisfy at all (e.g. every IK solution
        # collides with an obstacle) raises rather than returning False.
        print(f"    move to {label} failed: {e.message}", file=sys.stderr)
        return False
    if not ok:
        print(f"    move to {label} failed (planner returned false)", file=sys.stderr)
    return ok


async def pick_and_drop(rig: Rig, obj: dict, world: WorldState) -> tuple[str, bool]:
    """Attempt the full pick/drop sequence for one object.

    Returns (outcome, holding): holding is True when the sequence stopped
    with the object still in the gripper.
    """
    x, y, z = grasp_point(obj["position"])
    # Turn to the grasp roll up at transit height, then hold it all the way
    # through: the descent is a straight line that allows only 5 degrees of
    # rotation, and there's no reason to twist a held object before the drop.
    th = grasp_theta(obj["position"])

    if not await move(rig, down_at(x, y, SAFE_HEIGHT_MM, th), "transit above object", world):
        return "move to transit-above-object failed", False
    if not await move(rig, down_at(x, y, z, th), "grasp", world, straight=True):
        return "descent to grasp point failed", False

    grabbed = await rig.gripper.grab()
    holding = (await rig.gripper.is_holding_something()).is_holding_something
    if not grabbed or not holding:
        # Reopen so the next object isn't approached with closed jaws.
        await rig.gripper.open()
        await move(rig, down_at(x, y, SAFE_HEIGHT_MM, th), "retreat after failed grab",
                   world, straight=True)
        return f"grab failed (grab()={grabbed}, holding={holding})", False

    if not await move(rig, down_at(x, y, SAFE_HEIGHT_MM, th), "lift", world, straight=True):
        return "lift after grab failed (still holding object)", True
    if not await move(rig, down_at(DROP_POSE.x, DROP_POSE.y, SAFE_HEIGHT_MM, th),
                      "transit above drop", world):
        return "transit to drop failed (still holding object)", True
    if not await move(rig, down_at(DROP_POSE.x, DROP_POSE.y, DROP_POSE.z, th), "drop",
                      world, straight=True):
        return "descent to drop pose failed (still holding object)", True

    await rig.gripper.open()
    still_holding = (await rig.gripper.is_holding_something()).is_holding_something
    await move(rig, down_at(DROP_POSE.x, DROP_POSE.y, SAFE_HEIGHT_MM, th),
               "retreat from drop", world, straight=True)
    if still_holding:
        return "release did not let go (still reports holding)", True

    return "dropped", False


# How close the arm must get back to its starting pose to count as returned.
RETURN_TOLERANCE_MM = 5.0
RETURN_TOLERANCE_DEG = 3.0


def pose_error(a: Pose, b: Pose) -> tuple[float, float]:
    """(position error mm, orientation error deg) between two arm poses."""
    dist = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
    u, v = (a.o_x, a.o_y, a.o_z), (b.o_x, b.o_y, b.o_z)
    cos = sum(p * q for p, q in zip(u, v)) / (math.hypot(*u) * math.hypot(*v))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
    roll = abs((a.theta - b.theta + 180.0) % 360.0 - 180.0)
    return dist, max(tilt, roll)


async def return_to_start(rig: Rig, start: Pose, world: WorldState) -> bool:
    """Move the arm back to the pose it started in, and confirm it got there.

    One retry: a plan can fail or stop short for reasons that don't repeat.
    """
    for attempt in (1, 2):
        await move(rig, start, "starting position", world, component=ARM_NAME)
        dist, ang = pose_error(await rig.arm.get_end_position(), start)
        if dist <= RETURN_TOLERANCE_MM and ang <= RETURN_TOLERANCE_DEG:
            print("Back at the starting position.")
            return True
        print(f"  not at the starting position after attempt {attempt}/2: "
              f"{dist:.1f} mm and {ang:.1f} deg off", file=sys.stderr)
    return False


async def run(dry_run: bool, out_path: str) -> None:
    machine = await ts.connect()
    try:
        arm = await ts.resolve(machine, Arm, ARM_NAME)
        # Where the arm is now is where it goes back to at the end.
        start = await arm.get_end_position()
        frame = await ts.capture(machine)
        # World-frame targets should not depend on where the arm was at capture;
        # logging it makes a drift between runs diagnosable.
        p = await arm.get_end_position()
        print(f"arm at capture: ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")
        with open("frame" + ts.SUPPORTED_IMAGE_TYPES[frame.media_type], "wb") as f:
            f.write(frame.color)

        result, marked = ts.scan(frame)
        if marked:
            with open("frame_marked.jpg", "wb") as f:
                f.write(marked)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        trash = [o for o in result["objects"] if o["is_trash"]]
        print(f"{len(result['objects'])} objects, {len(trash)} trash -> {out_path}")

        plan = []
        for obj in trash:
            reason = gate(obj["position"])
            plan.append((obj, reason))
            if reason:
                print(f"  [SKIP ] {obj['name']}: {reason}")
            else:
                pos = obj["position"]
                g = ", ".join(f"{v:.1f}" for v in grasp_point(pos))
                print(f"  [PICK ] {obj['name']}  grasp at ({g}) mm  "
                      f"theta {grasp_theta(pos):.0f}  closing on {pos['width_mm']:.0f} mm "
                      f"(long side {pos['length_mm']:.0f} mm at {pos['long_axis_deg']:.0f} deg)  "
                      f"[{pos['source']}]")

        # Only objects with a trustworthy box become obstacles; a bad reading
        # could just as easily wall off the whole table.
        on_table = [o for o in result["objects"] if gate(o["position"]) is None]
        for o in result["objects"]:
            if not o["is_trash"] and o not in on_table:
                print(f"  [WARN ] {o['name']} (keep) has no trustworthy position; "
                      "the planner cannot avoid it")

        attempt = [obj for obj, reason in plan if reason is None]
        if not attempt:
            print("Nothing to pick up.")
            return
        if dry_run:
            print(f"--dry-run: would attempt {len(attempt)} pick(s) around "
                  f"{len(on_table)} modelled object(s); no arm motion.")
            return

        rig = Rig(
            motion=await ts.resolve(machine, Motion, MOTION_NAME),
            arm=arm,
            gripper=await ts.resolve(machine, Gripper, GRIPPER_NAME),
        )

        # Opening a gripper that holds something would drop it wherever the
        # arm happens to be.
        if (await rig.gripper.is_holding_something()).is_holding_something:
            raise SystemExit("gripper already reports holding something; "
                             "clear it before running")
        await rig.gripper.open()

        try:
            for obj in attempt:
                print(f"  -> {obj['name']}")
                outcome, holding = await pick_and_drop(
                    rig, obj, world_state([o for o in on_table if o is not obj]))
                print(f"     {outcome}")
                if outcome == "dropped" or holding:
                    on_table.remove(obj)
                if holding:
                    print("  Stopping: the gripper is still holding an object; "
                          "remaining picks skipped.", file=sys.stderr)
                    break
        except BaseException:
            # Ctrl-C or an unexpected error mid-motion: halt the arm where it
            # is rather than planning a move nobody asked for.
            print("Interrupted — stopping arm.", file=sys.stderr)
            try:
                await arm.stop()
            except Exception as e:  # best-effort — log, don't mask the real error
                print(f"  arm stop failed: {e}", file=sys.stderr)
            raise

        print(f"Returning to the starting position "
              f"({start.x:.1f}, {start.y:.1f}, {start.z:.1f}).")
        if not await return_to_start(rig, start, world_state(on_table)):
            print("Could not get back to the starting position; the arm is "
                  "stopped where it is.", file=sys.stderr)
    finally:
        await machine.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="scan and print the plan; do not move the arm")
    ap.add_argument("--out", default="table_objects.json")
    args = ap.parse_args()
    asyncio.run(run(args.dry_run, args.out))


if __name__ == "__main__":
    main()
