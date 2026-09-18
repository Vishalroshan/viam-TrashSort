"""
table_cleanup.py — scan the table, then pick up and drop every object
classified as trash into a fixed drop location.

Reuses table_scan's connect/capture/classify/validate/localize to identify
what's on the table and where each object sits in world-frame mm. For each
object marked trash, if its 3D position is trustworthy, the arm:
  1. moves to a safe transit height,
  2. approaches from APPROACH_CLEARANCE_MM above the object's centroid,
  3. grabs, and confirms it actually holds something,
  4. lifts back to transit height, moves over the drop pose, descends,
     releases, and confirms the release actually let go,
  5. retreats back up to transit height.

Objects whose depth signal is too sparse to trust (a common failure mode for
clear plastic and shiny metal under IR depth sensing — the water bottle on
this table is the running example) are skipped and reported rather than
risking a blind grasp.

There is no confirmation prompt: this script scans and immediately acts on
whatever it classifies as trash.

Setup: same as table_scan.py — .env populated, .venv/bin/pip install
       viam-sdk anthropic python-dotenv numpy

Run:
    .venv/bin/python table_cleanup.py             # scan, then act
    .venv/bin/python table_cleanup.py --dry-run    # scan, print the plan, touch nothing physical
"""

import argparse
import asyncio
import json
import sys

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

import table_scan as ts

# --- Motion constants, generalized from viam_move_to_centroid.py ----------

SAFE_HEIGHT_MM = 300.0            # world z for every transit move
APPROACH_CLEARANCE_MM = 150.0     # grab from this far above the centroid z

# Top-down orientation used for every synthetic pick/drop pose in the
# reference script — the end effector points straight down (o_z=-1),
# no roll (theta=0).
DOWN = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)

DROP_POSE = Pose(x=292.5, y=-399.9, z=206.2, **DOWN)

# Verbatim from viam_move_to_centroid.py — a pose the user measured by
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
    """Return None if position is trustworthy enough to grasp on, else why not."""
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
    return None


async def move(motion: MotionClient, x: float, y: float, z: float, label: str) -> bool:
    ok = await motion.move(
        component_name=ARM_NAME,
        destination=PoseInFrame(
            reference_frame="world", pose=Pose(x=x, y=y, z=z, **DOWN)
        ),
    )
    if not ok:
        print(f"    move to {label} failed (planner returned false)", file=sys.stderr)
    return ok


async def pick_and_drop(motion: MotionClient, gripper: Gripper, obj: dict) -> str:
    """Attempt the full pick/drop sequence for one object. Returns an outcome string."""
    x, y, z = obj["position"]["centroid"]
    name = obj["name"]

    if not await move(motion, x, y, SAFE_HEIGHT_MM, "transit above object"):
        return "move to transit-above-object failed"
    if not await move(motion, x, y, z + APPROACH_CLEARANCE_MM, "approach"):
        return "move to approach height failed"

    grabbed = await gripper.grab()
    holding = (await gripper.is_holding_something()).is_holding_something
    if not grabbed or not holding:
        await move(motion, x, y, SAFE_HEIGHT_MM, "retreat after failed grab")
        return f"grab failed (grab()={grabbed}, holding={holding})"

    if not await move(motion, x, y, SAFE_HEIGHT_MM, "lift"):
        return "lift after grab failed"
    if not await move(motion, DROP_POSE.x, DROP_POSE.y, SAFE_HEIGHT_MM, "transit above drop"):
        return "transit to drop failed (still holding object)"
    if not await move(motion, DROP_POSE.x, DROP_POSE.y, DROP_POSE.z, "drop"):
        return "descent to drop pose failed (still holding object)"

    await gripper.open()
    still_holding = (await gripper.is_holding_something()).is_holding_something
    await move(motion, DROP_POSE.x, DROP_POSE.y, SAFE_HEIGHT_MM, "retreat from drop")
    if still_holding:
        return "release did not let go (still reports holding)"

    return "dropped"


async def run(dry_run: bool, out_path: str) -> None:
    machine = await ts.connect()
    try:
        frame = await ts.capture(machine)
        with open("frame" + ts.SUPPORTED_IMAGE_TYPES[frame.media_type], "wb") as f:
            f.write(frame.color)

        result = ts.validate(ts.classify(frame.color, frame.media_type))
        for obj in result["objects"]:
            obj["position"] = ts.localize(frame, obj["bbox_norm"])
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
                c = obj["position"]["centroid"]
                print(f"  [PICK ] {obj['name']}  centroid {c} mm")

        attempt = [obj for obj, reason in plan if reason is None]
        if not attempt:
            print("Nothing to pick up.")
            return
        if dry_run:
            print(f"--dry-run: would attempt {len(attempt)} pick(s); no arm motion.")
            return

        arm = await ts.resolve(machine, Arm, ARM_NAME)
        gripper = await ts.resolve(machine, Gripper, GRIPPER_NAME)
        motion = await ts.resolve(machine, MotionClient, MOTION_NAME)

        try:
            for obj in attempt:
                print(f"  -> {obj['name']}")
                outcome = await pick_and_drop(motion, gripper, obj)
                print(f"     {outcome}")
        finally:
            print("Returning home.")
            try:
                await arm.move_to_position(HOME_POSE)
            except Exception as e:  # best-effort — log, don't mask the real error
                print(f"  home move failed: {e}", file=sys.stderr)
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
