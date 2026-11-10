# ruff: noqa: E402

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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Args for appLauncher and capture manager"
)
parser.add_argument(
    "--num_objects", type=int, required=True, help="Number of objects to use"
)
parser.add_argument(
    "--asset_dir", type=str, required=True, help="Directory to load data from"
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save outputs",
)
parser.add_argument(
    "--table_asset_path",
    type=str,
    required=True,
    help="Path to the table asset",
)
parser.add_argument(
    "--t_cam_to_table_path",
    type=str,
    required=True,
    help="Path to the camera to table transform file",
)
parser.add_argument("--scene_id", type=int, default=0)

AppLauncher.add_app_launcher_args(parser)

args = parser.parse_args()

filtered_vars = filter(
    lambda item: item[0] in AppLauncher._APPLAUNCHER_CFG_INFO,
    vars(args).items(),
)
launcher_args = argparse.Namespace(**dict(filtered_vars))
print("launcher_args:", launcher_args)
app_launcher = AppLauncher(launcher_args)
simulation_app = app_launcher.app


from robo_orchard_grasp_label.capture_manager import (
    ViewCaptureManager,
)
from robo_orchard_grasp_label.datatypes import ObjectParam


def main():
    output_dir = os.path.join(
        args.output_dir, f"scene_{str(args.scene_id).zfill(4)}"
    )
    table_asset_path = os.path.abspath(args.table_asset_path)
    asset_dir = os.path.abspath(args.asset_dir)
    object_params_path = os.path.join(asset_dir, "../object_params.json")
    if not os.path.exists(object_params_path):
        raise FileNotFoundError(
            f"Object parameters file not found at {object_params_path}, \
                please run mesh_to_usd.py first."
        )
    with open(object_params_path, "r") as f:
        data = json.load(f)
    obj_params = [ObjectParam(**item) for item in data]

    sim_manager = ViewCaptureManager(
        simulation_app,
        num_objects=args.num_objects,
        asset_dir=asset_dir,
        output_dir=output_dir,
        table_asset_path=table_asset_path,
        t_cam_to_table_path=args.t_cam_to_table_path,
        obj_params=obj_params,
    )
    sim_manager.setup()
    sim_manager.run_simulation()


if __name__ == "__main__":
    main()
    simulation_app.close()
