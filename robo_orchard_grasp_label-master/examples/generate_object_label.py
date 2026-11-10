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

from params import grasp_param, gripper_param

from robo_orchard_grasp_label.datatypes import ObjectParam
from robo_orchard_grasp_label.object_label import (
    ObjectGraspLabelGenerator,
    ObjectGraspLabelOutput,
)

parser = argparse.ArgumentParser(description="Generate object grasp labels.")
parser.add_argument(
    "--model_dir",
    type=str,
    default="./data/",
    help="Root directory of the dataset.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./data/",
    help="Directory to save the generated labels.",
)
parser.add_argument(
    "--pipeline_test",
    action="store_true",
    help="Run a pipeline test with reduced parameters.",
)
args = parser.parse_args()


def main(grasp_param, gripper_param, obj_params):
    model_dir = args.model_dir
    output_dir = args.output_dir

    if args.pipeline_test:
        obj_params = obj_params[:1]
        grasp_param.num_views = 2
        grasp_param.num_angles = 2
        grasp_param.num_depths = 2

    for obj_param in obj_params:
        object_label_genereator = ObjectGraspLabelGenerator(
            obj_param=obj_param,
            grasp_param=grasp_param,
            gripper_param=gripper_param,
            model_dir=model_dir,
            log_interval=10000,
            mp_batch_size=100,
        )
        object_label_output: ObjectGraspLabelOutput = (
            object_label_genereator.run()
        )
        object_label_output.save_to_file(
            os.path.join(output_dir, f"{obj_param.name}_labels.npz")
        )


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
    main(grasp_param, gripper_param, obj_params)
