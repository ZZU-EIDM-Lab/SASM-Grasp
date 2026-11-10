# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import random

import numpy as np


def add_random_position_noise(
    pos, x_range=(0.0, 0.0), y_range=(0.0, 0.0), z_range=(0.0, 0.0), dim="xyz"
):
    """Add random noise to the position.

    Args:
        pos (list): Original position [x, y, z].
        x_range (tuple): Range for x-axis noise.
        y_range (tuple): Range for y-axis noise.
        z_range (tuple): Range for z-axis noise.
        dim (str): Dimensions to apply noise to.

    Returns:
        list: New position with noise added.
    """
    x_range = tuple(x_range)
    y_range = tuple(y_range)
    z_range = tuple(z_range)

    x, y, z = pos[0], pos[1], pos[2]
    if "x" in dim:
        x = pos[0] + random.uniform(*x_range)
    if "y" in dim:
        y = pos[1] + random.uniform(*y_range)
    if "z" in dim:
        z = pos[2] + random.uniform(*z_range)
    return [x, y, z]


def add_random_rotation_noise(
    quat, delta_range=(0.0, 0.0), scalar_first=False
):
    """Add random noise to the rotation quaternion.

    Args:
        quat (list): Original quaternion [x, y, z, w] or [w, x, y, z].
        delta_range (tuple): Range for rotation noise in degrees.
        scalar_first (bool): If True, in [w, x, y, z] order.

    Returns:
        list: New quaternion with noise added.
    """
    from scipy.spatial.transform import Rotation as R

    rotation = R.from_quat(quat, scalar_first=scalar_first)
    delta_rotation = R.from_euler(
        "xyz",
        np.random.uniform(delta_range[0], delta_range[1], size=3),
        degrees=True,
    )

    update_rotation = rotation * delta_rotation
    update_quat = update_rotation.as_quat()  # xyzw

    if scalar_first:
        update_quat = np.roll(update_quat, shift=1)
    return update_quat
