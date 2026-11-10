# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import argparse

from robo_orchard_grasp_label.utils.sdfgen.sdf_modules import sdf

parser = argparse.ArgumentParser(
    description="Generate SDF for a given object mesh file."
)
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to the input mesh file.",
)
parser.add_argument(
    "--dim", type=int, default=100, help="Dimension of the SDF grid."
)
parser.add_argument(
    "--padding",
    type=int,
    default=5,
    help="Padding around the object in the SDF grid.",
)
args = parser.parse_args()


def main():
    sdf(args.input_file, args.dim, args.padding)


if __name__ == "__main__":
    main()
