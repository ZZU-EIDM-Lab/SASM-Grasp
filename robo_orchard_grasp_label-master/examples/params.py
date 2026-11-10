# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
from robo_orchard_grasp_label.datatypes import (
    GraspParam,
    GripperParam,
    SceneParam,
)

grasp_param = GraspParam(
    num_views=300,
    num_angles=12,
    num_depths=4,
)

gripper_param = GripperParam(
    height=0.02,
    depth_base=0.02,
    finger_width=0.01,
    bottom_thickness=0.1,
    empty_thresh=10,
    max_width=0.12,
    collision_thresh=0.01,
)


# graspnet training scene
scene_params = [
    SceneParam(id=0, name="scene_0000"),
]
