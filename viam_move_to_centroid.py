import asyncio
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient
from viam.services.motion import MotionClient
from viam.components.arm import ArmClient
from viam.components.gripper import GripperClient
from viam.proto.common import Pose, PoseInFrame

async def main():
    robot = await RobotClient.at_address(
        'armfarm15-main.310sld03v2.viam.cloud',
        RobotClient.Options.with_api_key(
            api_key='VIAM_API_KEY_REDACTED',
            api_key_id='VIAM_API_KEY_ID_REDACTED'
        )
    )

    arm = ArmClient.from_robot(robot, "arm")
    motion = MotionClient.from_robot(robot, "builtin")
    vision = VisionClient.from_robot(robot, "green-segmenter")

    segments = await vision.get_object_point_clouds("cam")
    if not segments:
        print("No segments found")
        await robot.close()
        return

    c = segments[0].geometries.geometries[0].center
    print(f"Centroid (cam frame): x={c.x:.1f} y={c.y:.1f} z={c.z:.1f}")

    pif = PoseInFrame(
        reference_frame="cam",
        pose=Pose(x=c.x, y=c.y, z=c.z, o_x=0, o_y=0, o_z=-1, theta=0)
    )

    transformed = await robot.transform_pose(pif, "world")
    print("Centroid (world frame):", transformed)

    current = await arm.get_end_position()
    print("Arm now (world frame):", current)

    c = segments[0].geometries.geometries[0].center

    pif = PoseInFrame(
        reference_frame="cam",
        pose=Pose(x=c.x, y=c.y, z=c.z, o_x=0, o_y=0, o_z=1, theta=0)
    )

    w = (await robot.transform_pose(pif, "world")).pose
    print(f"Target (world): x={w.x:.1f} y={w.y:.1f} z={w.z:.1f}")

        # Safe transit height
    await motion.move(
        component_name="arm",
        destination=PoseInFrame(
            reference_frame="world",
            pose=Pose(x=308.5, y=-25.3, z=300,
                      o_x=0, o_y=0, o_z=-1, theta=0)
        )
    )
    print("At safe height")

    # Approach above block
    await motion.move(
        component_name="arm",
        destination=PoseInFrame(
            reference_frame="world",
            pose=Pose(x=308.5, y=-25.3, z=114.9+150,
                      o_x=0, o_y=0, o_z=-1, theta=0)
        )
    )
    print("At approach")

    # Close gripper
    gripper = GripperClient.from_robot(robot, "gripper")
    await gripper.grab()
    print("Grabbed!")

    # Lift back to safe height
    await motion.move(
        component_name="arm",
        destination=PoseInFrame(
            reference_frame="world",
            pose=Pose(x=w.x, y=w.y, z=300,
                      o_x=0, o_y=0, o_z=-1, theta=0)
        )
    )
    print("Lifted!")

    # Return home
    await arm.move_to_position(
        Pose(
            x=263.16,
            y=-39.11,
            z=442.47,
            o_x=-0.0067,
            o_y=-0.1893,
            o_z=-0.9819,
            theta=85.94
        )
    )
    print("Home!")

    await robot.close()

asyncio.run(main())