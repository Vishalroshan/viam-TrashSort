import asyncio

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.vision import VisionClient
from viam.services.generic import Generic as GenericService

async def connect():
    opts = RobotClient.Options.with_api_key(
         
        api_key='smeizcjrwe5reik85jxcjp8ammyr0za8',
        
        api_key_id='66b63da0-fe7d-42fd-b54a-3f45a6f8ecf4'
    )
    
    return await RobotClient.at_address('armfarm15-main.310sld03v2.viam.cloud', opts)

async def main():
    async with await connect() as machine:
        print('Resources:')
        print(machine.resource_names)
        
        # arm
        arm = Arm.from_robot(machine, "arm")
        arm_return_value = await arm.get_end_position()
        print(f"arm get_end_position return value: {arm_return_value}")

        # cam
        cam = Camera.from_robot(machine, "cam")
        cam_return_value = await cam.get_images()
        print(f"cam get_images return value: {cam_return_value}")

        # gripper
        gripper = Gripper.from_robot(machine, "gripper")
        gripper_return_value = await gripper.is_moving()
        print(f"gripper is_moving return value: {gripper_return_value}")

        # table
        table = Gripper.from_robot(machine, "table")
        table_return_value = await table.is_moving()
        print(f"table is_moving return value: {table_return_value}")

        # wall-front
        wall_front = Gripper.from_robot(machine, "wall-front")
        wall_front_return_value = await wall_front.is_moving()
        print(f"wall-front is_moving return value: {wall_front_return_value}")

        # wall-side
        wall_side = Gripper.from_robot(machine, "wall-side")
        wall_side_return_value = await wall_side.is_moving()
        print(f"wall-side is_moving return value: {wall_side_return_value}")

        # ceiling
        ceiling = Gripper.from_robot(machine, "ceiling")
        ceiling_return_value = await ceiling.is_moving()
        print(f"ceiling is_moving return value: {ceiling_return_value}")

        # qwen-vision
        qwen_vision = VisionClient.from_robot(machine, "qwen-vision")
        qwen_vision_return_value = await qwen_vision.get_properties()
        print(f"qwen-vision get_properties return value: {qwen_vision_return_value}")

        # green-detector
        green_detector = VisionClient.from_robot(machine, "green-detector")
        green_detector_return_value = await green_detector.get_properties()
        print(f"green-detector get_properties return value: {green_detector_return_value}")

        # green-segmenter
        green_segmenter = VisionClient.from_robot(machine, "green-segmenter")
        green_segmenter_return_value = await green_segmenter.get_properties()
        print(f"green-segmenter get_properties return value: {green_segmenter_return_value}")

if __name__ == '__main__':
    asyncio.run(main())