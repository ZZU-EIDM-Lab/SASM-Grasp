# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import os

import pytest


@pytest.fixture(scope="module")
def ROBO_ORCHARD_TEST_WORKSPACE() -> str:
    return os.environ["ROBO_ORCHARD_TEST_WORKSPACE"]
