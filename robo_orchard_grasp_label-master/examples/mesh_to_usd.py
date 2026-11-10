# ruff: noqa: E402

# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import argparse
import glob
import json
import os
import shutil

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

parser = argparse.ArgumentParser()
parser.add_argument(
    "--in_path",
    type=str,
    required=True,
    help="The input path containing the mesh files, e.g., 'data/meshes'",
)
parser.add_argument(
    "--out_path",
    type=str,
    required=True,
    help="The output path for USD files, e.g., 'data/usd'",
)
args = parser.parse_args()

from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg

from robo_orchard_grasp_label.datatypes import ObjectParam
from robo_orchard_grasp_label.mesh_converter import MeshConverter


def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
        print(f"Copied file {src} to {dst}.")
    except IOError as e:
        print(f"Unable to copy file {src} to {dst}. {e}")


def main():
    """Traverse the input path and convert all mesh files to USD format."""
    mesh_files = glob.glob(f"{args.in_path}/**/*.obj", recursive=True)
    mesh_files.sort()
    out_path = os.path.abspath(args.out_path)

    object_params = []
    for i, mesh_path_i in enumerate(mesh_files):
        mesh_path_i = os.path.abspath(mesh_path_i)

        # change this if the mesh file structure is different
        mesh_name_i = mesh_path_i.split("/")[-3]
        # change this if the texture file name is different
        texture_name_i = "texture.png"

        # make sure the texture is in the same folder as the mesh file
        texture_file = os.path.join(
            os.path.dirname(mesh_path_i), texture_name_i
        )
        if not os.path.exists(texture_file):
            texture_file = glob.glob(
                f"{os.path.dirname(mesh_path_i)}/../**/{texture_name_i}",
                recursive=True,
            )
            assert len(texture_file) == 1, (
                f"Found wrong texture file for {mesh_path_i}."
            )
            copy_file(
                texture_file[0],
                os.path.join(os.path.dirname(mesh_path_i), texture_name_i),
            )

        usd_path_i = os.path.join(out_path, mesh_name_i)
        os.makedirs(usd_path_i, exist_ok=True)

        mesh_converter_cfg = MeshConverterCfg(
            asset_path=mesh_path_i,  # The absolute path to the asset file.
            usd_dir=usd_path_i,  # The absolute path to store the usd.
            force_usd_conversion=True,
        )
        MeshConverter(mesh_converter_cfg)
        print("Finish converting", mesh_path_i, "to", usd_path_i)

        object_param = ObjectParam(name=mesh_name_i, id=int(i))
        object_params.append(object_param)

    # dump object parameters to a JSON file
    json_file_path = os.path.join(out_path, "../object_params.json")
    print(f"Saving object parameters to {json_file_path}...")
    with open(json_file_path, "w") as f:
        json.dump([obj.dict() for obj in object_params], f, indent=4)


if __name__ == "__main__":
    main()
