# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class GripperParam(BaseModel):
    height: float = Field(0.02, ge=0.0, description="Gripper height in meters")
    finger_width: float = Field(
        0.01, ge=0.0, description="Gripper width in meters"
    )
    depth_base: float = Field(
        0.02, ge=0.0, description="Gripper depth in meters"
    )
    bottom_thickness: float = Field(
        0.1, ge=0.0, description="Gripper bottom thickness in meters"
    )
    empty_thresh: float = Field(10, ge=0, description="Empty grasp threshold")
    max_width: float = Field(0.12, ge=0, description="Max gripper width")
    collision_thresh: float = Field(
        0.01, ge=0, description="Collision threshold"
    )


class GraspParam(BaseModel):
    num_views: int = Field(
        300, ge=0.0, description="Number of approaching directions"
    )
    num_angles: int = Field(
        12, ge=0.0, description="Number of in-plane rotation angles in 0~pi"
    )
    num_depths: int = Field(
        5, gt=0, le=5, description="Number of depth levels, must be in 1~5"
    )


class ObjectParam(BaseModel):
    name: str = Field(..., description="Name of the object")
    id: int = Field(..., description="Id of the object")
    sample_voxel_size: float = Field(
        0.005,
        description="Downsample voxel size in meters for sample points",
    )
    max_sample_points: int = Field(
        10000, description="Maximum number of sample points"
    )
    model_voxel_size: float = Field(
        0.001,
        description="Downsample voxel size in meters for model points",
    )
    max_model_points: int = Field(
        10000, description="Maximum number of model points"
    )
    model_config = ConfigDict(protected_namespaces=())


class SceneParam(BaseModel):
    id: int = Field(..., description="Id of the scene")
    name: str = Field(None, description="Name of the scene")

    @model_validator(mode="after")
    def set_default_name(self):
        if self.name is None:
            self.name = f"scene_{str(self.id).zfill(4)}"
        return self


class ForceClosureConfig(BaseModel):
    quality_method: str = Field(
        "force_closure", description="Method to compute force closure"
    )
    num_cone_faces: int = Field(
        8, ge=3, description="Number of cone faces for force closure"
    )
    soft_fingers: int = Field(
        1, ge=0, description="Use soft fingers for force closure"
    )
    quality_type: str = Field(
        "quasi_static", description="Type of quality metric"
    )
    all_contacts_required: int = Field(
        1, ge=0, description="Require all contacts for force closure"
    )
    check_approach: bool = Field(
        False, description="Check approach direction for force closure"
    )
    torque_scaling: float = Field(
        0.01, ge=0.0, description="Scaling factor for torque"
    )
    wrench_norm_thresh: float = Field(
        0.001, ge=0.0, description="Threshold for wrench norm"
    )
    wrench_regularizer: float = Field(
        1e-10, ge=0.0, description="Regularizer for wrench computation"
    )

    def to_config_dict(self) -> ConfigDict:
        return {"metrics": {"force_closure": self.dict()}}


class GameraParam(BaseModel):
    width: int = Field(ge=1, description="Width of the camera image")
    height: int = Field(ge=1, description="Height of the camera image")
    fx: float = Field(ge=0.0, description="Focal length in x direction")
    fy: float = Field(ge=0.0, description="Focal length in y direction")
    cx: float = Field(ge=0.0, description="Principal point x coordinate")
    cy: float = Field(ge=0.0, description="Principal point y coordinate")
    scale: float = Field(ge=0.0, description="Scale factor for depth values")
