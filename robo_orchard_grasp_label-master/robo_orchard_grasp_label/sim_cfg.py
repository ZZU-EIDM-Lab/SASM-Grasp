# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import isaaclab.sim as sim_utils
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera import Camera
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg

image_width = 1280
image_height = 720
focal_length = 1.93
focus_distance = 1.0

fx = 912.758056640625
fy = 912.8134155273438
cx, cy = 1280 / 2.0, 720 / 2.0

intrinsic_matrix = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

D435I_CFG = CameraCfg(
    prim_path="/World/Camera",
    class_type=Camera,
    offset=CameraCfg.OffsetCfg(
        pos=(0.5, 0, 1),
        rot=rot_utils.euler_angles_to_quats(
            np.array([0, 180, 90]), degrees=True
        ),
    ),
    height=image_height,
    width=image_width,
    data_types=[
        "rgb",
        "distance_to_image_plane",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ],
    colorize_semantic_segmentation=False,
    colorize_instance_id_segmentation=False,
    colorize_instance_segmentation=False,
    spawn=PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=intrinsic_matrix,
        width=image_width,
        height=image_height,
    ),
)


class SceneCfg(InteractiveSceneCfg):
    light: AssetBaseCfg | None = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75), intensity=3000.0
        ),
    )

    plane: AssetBaseCfg | None = None

    cameras: CameraCfg | None = None

    table: AssetBaseCfg | None = None

    objects: dict | AssetBaseCfg = None
