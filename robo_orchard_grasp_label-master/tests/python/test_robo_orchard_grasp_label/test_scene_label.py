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
    SceneParam,
)
from robo_orchard_grasp_label.scene_label import SceneGraspLabelGenerator


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
        ),
    ]
    scene_params = [SceneParam(id=0, name="scene_0000")]
    scene_param = scene_params[0]

    dummy_data_path = os.path.join(
        ROBO_ORCHARD_TEST_WORKSPACE,
        "robo_orchard_workspace/robo_orchard_grasp_label_ut_data",
    )
    model_dir = os.path.join(dummy_data_path, "models")
    scene_dir = os.path.join(dummy_data_path, "scenes")
    grasp_label_dir = os.path.join(dummy_data_path, "grasp_labels")

    scene_label_generator = SceneGraspLabelGenerator(
        scene_param=scene_param,
        grasp_param=grasp_param,
        gripper_param=gripper_param,
        obj_params=obj_params,
        output_dir=dummy_data_path,
        model_dir=model_dir,
        scene_dir=scene_dir,
        grasp_label_dir=grasp_label_dir,
        log_interval=100,
        mp_batch_size=100,
        mp_proc=2,
        save_result=False,
        cal_collision=False,
        cal_graspness=False,
    )
    scene_label_generator.run()


if __name__ == "__main__":
    pytest.main(["-s", __file__])
