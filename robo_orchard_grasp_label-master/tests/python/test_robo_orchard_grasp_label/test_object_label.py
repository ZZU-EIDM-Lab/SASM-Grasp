# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import os

import pytest

from robo_orchard_grasp_label.datatypes import (
    GraspParam,
    GripperParam,
    ObjectParam,
)
from robo_orchard_grasp_label.object_label import (
    ObjectGraspLabelGenerator,
)


def test_object_label(ROBO_ORCHARD_TEST_WORKSPACE: str):
    grasp_param = GraspParam(num_views=1, num_angles=1, num_depths=1)
    gripper_param = GripperParam(
        height=0.02,
        depth_base=0.02,
        finger_width=0.01,
        bottom_thickness=0.1,
        empty_thresh=10,
        max_width=0.12,
        collision_thresh=0.01,
    )
    obj_params = [
        ObjectParam(
            name="SCHOOL_BUS",
            id=0,
            sample_voxel_size=0.005,
            model_voxel_size=0.001,
            max_sample_points=10000,
            max_model_points=10000,
        )
    ]
    obj_param = obj_params[0]

    dummy_data_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/robo_orchard_grasp_label_ut_data/models",
    )

    object_label_genereator = ObjectGraspLabelGenerator(
        obj_param=obj_param,
        grasp_param=grasp_param,
        gripper_param=gripper_param,
        model_dir=dummy_data_path,
        log_interval=100,
        mp_batch_size=100,
        mp_proc=2,
    )
    object_label_genereator.run()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
