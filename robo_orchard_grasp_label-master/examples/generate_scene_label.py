# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import argparse
import json
import os

from params import grasp_param, gripper_param, scene_params

from robo_orchard_grasp_label.datatypes import ObjectParam
from robo_orchard_grasp_label.scene_label import SceneGraspLabelGenerator

parser = argparse.ArgumentParser(description="Generate scene grasp labels.")
parser.add_argument(
    "--scene_dir",
    type=str,
    help="Directory containing scene data.",
)
parser.add_argument(
    "--model_dir",
    type=str,
    help="Directory containing mesh data.",
)
parser.add_argument(
    "--grasp_label_dir",
    type=str,
    help="Directory containing grasp label data.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./data",
    help="Directory to save the output labels.",
)
parser.add_argument(
    "--pipeline_test",
    action="store_true",
    help="Run a pipeline test with limited data.",
)
args = parser.parse_args()


def main(
    grasp_param,
    gripper_param,
    obj_params,
    scene_params,
):
    scene_dir = args.scene_dir
    model_dir = args.model_dir
    grasp_label_dir = args.grasp_label_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.pipeline_test:
        scene_params = scene_params[:1]
        grasp_param.num_views = 2
        grasp_param.num_angles = 2
        grasp_param.num_depths = 2

    for scene_param in scene_params:
        scene_label_generator = SceneGraspLabelGenerator(
            scene_param=scene_param,
            grasp_param=grasp_param,
            gripper_param=gripper_param,
            obj_params=obj_params,
            scene_dir=scene_dir,
            model_dir=model_dir,
            grasp_label_dir=grasp_label_dir,
            output_dir=output_dir,
            cal_collision=True,
            cal_graspness=True,
            mp_batch_size=100,
            mp_proc=2,
        )
        scene_label_generator.run()


if __name__ == "__main__":
    object_param_file = os.path.join(args.model_dir, "../object_params.json")
    if not os.path.exists(object_param_file):
        raise FileNotFoundError(
            f"Object parameter file not found: {object_param_file}, \
                please run mesh_to_usd.py first."
        )
    with open(object_param_file, "r") as f:
        data = json.load(f)
    obj_params = [ObjectParam(**item) for item in data]
    main(grasp_param, gripper_param, obj_params, scene_params)
