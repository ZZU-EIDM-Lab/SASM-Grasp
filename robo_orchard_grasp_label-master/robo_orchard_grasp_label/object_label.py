# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import functools
import logging
import multiprocessing as mp
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from numpydantic import NDArray
from pydantic import BaseModel, ConfigDict, Field

from robo_orchard_grasp_label.datatypes import (
    ForceClosureConfig,
    GraspParam,
    GripperParam,
    ObjectParam,
)
from robo_orchard_grasp_label.label_utils import (
    analyze_width,
    batch_viewpoint_params_to_matrix,
    generate_views,
    get_grasp_score,
    load_dexnet_model,
    load_mesh_as_pointcloud,
    matrix_to_dexnet_params,
)
from robo_orchard_grasp_label.utils.dexnet.grasping.grasp import (
    ParallelJawPtGrasp3D,
)
from robo_orchard_grasp_label.utils.dexnet.grasping.grasp_quality_config import (  # noqa: E501
    GraspQualityConfigFactory,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ObjectGraspLabelGenerator(BaseModel):
    obj_param: ObjectParam
    grasp_param: GraspParam
    gripper_param: GripperParam
    model_dir: str
    log_interval: int = 10000
    mp_batch_size: int = 10000
    mp_proc: int = 4
    model_config = ConfigDict(extra="allow")

    def run(self) -> "ObjectGraspLabelOutput":
        sample_points = load_mesh_as_pointcloud(
            path=os.path.join(
                self.model_dir,
                self.obj_param.name,
                "meshes/model.obj",
            ),
            voxel_size=self.obj_param.sample_voxel_size,
        )  # (N, 3)
        self.num_sample_points = sample_points.shape[0]
        self.num_grasps = (
            self.num_sample_points
            * self.grasp_param.num_views
            * self.grasp_param.num_angles
            * self.grasp_param.num_depths
        )
        logger.info(
            f" Processing {self.num_sample_points} \
            points with {self.num_grasps} grasps"
        )

        collision_points = load_mesh_as_pointcloud(
            path=os.path.join(
                self.model_dir,
                self.obj_param.name,
                "meshes/model.obj",
            ),
            voxel_size=self.obj_param.model_voxel_size,
        )

        dexnet_model = load_dexnet_model(
            os.path.join(
                self.model_dir,
                self.obj_param.name,
                "meshes/model",
            )
        )

        grasp_points, rotations, angles, depths = self._gen_grasp_pose(
            sample_points
        )

        dexgrasps, widths, collisions = self._gen_widths_collisions_dexgrasps(
            grasp_points=grasp_points,
            collision_points=collision_points,
            rotations=rotations,
            depths=depths,
            gripper_param=self.gripper_param,
        )

        scores = self._gen_grasp_scores(dexnet_model, dexgrasps)

        widths = widths.reshape(
            self.num_sample_points,
            self.grasp_param.num_views,
            self.grasp_param.num_angles,
            self.grasp_param.num_depths,
            1,
        )
        scores = scores.reshape(
            self.num_sample_points,
            self.grasp_param.num_views,
            self.grasp_param.num_angles,
            self.grasp_param.num_depths,
        )
        collisions = collisions.reshape(
            self.num_sample_points,
            self.grasp_param.num_views,
            self.grasp_param.num_angles,
            self.grasp_param.num_depths,
        )

        return ObjectGraspLabelOutput(
            grasp_points=sample_points,
            scores=scores,
            collisions=collisions,
            dexgrasps=dexgrasps,
            rotations=rotations,
            angles=angles,
            depths=depths,
            widths=widths,
        )

    def _gen_grasp_pose(self, sample_points):
        grasp_points = sample_points[:, np.newaxis, np.newaxis, np.newaxis, :]
        grasp_points = np.tile(
            grasp_points,
            [
                1,
                self.grasp_param.num_views,
                self.grasp_param.num_angles,
                self.grasp_param.num_depths,
                1,
            ],
        )  # (N, num_views, num_angles, num_depths, 3)

        angles = np.array(
            [
                np.pi / self.grasp_param.num_angles * i
                for i in range(self.grasp_param.num_angles)
            ],
            dtype=np.float32,
        )
        views = generate_views(self.grasp_param.num_views)  # (num_views, 3)
        views_repeat = views.repeat(self.grasp_param.num_angles, 0)
        angles_repeat = (
            angles.reshape(1, self.grasp_param.num_angles)
            .repeat(self.grasp_param.num_views, 0)
            .reshape(-1)
        )

        grasp_rots = batch_viewpoint_params_to_matrix(
            -views_repeat, angles_repeat
        ).reshape(self.grasp_param.num_views, self.grasp_param.num_angles, 9)
        grasp_rots = grasp_rots[np.newaxis, :, :, np.newaxis, :]
        grasp_rots = np.tile(
            grasp_rots,
            [self.num_sample_points, 1, 1, self.grasp_param.num_depths, 1],
        )  # (N, num_views, num_angles, num_depths, 9)

        grasp_angles = angles[
            np.newaxis, np.newaxis, :, np.newaxis, np.newaxis
        ]
        grasp_angles = np.tile(
            grasp_angles,
            [
                self.num_sample_points,
                self.grasp_param.num_views,
                1,
                self.grasp_param.num_depths,
                1,
            ],
        )  # (N, num_veiws, num_angles, grasp_depths, 1)

        depths = [0.01, 0.02, 0.03, 0.04][: self.grasp_param.num_depths]
        if self.grasp_param.num_depths == 5:
            depths.insert(0, 0.005)
        depths = np.array(depths, dtype=np.float32)
        grasp_depths = depths[
            np.newaxis, np.newaxis, np.newaxis, :, np.newaxis
        ]
        grasp_depths = np.tile(
            grasp_depths,
            [
                self.num_sample_points,
                self.grasp_param.num_views,
                self.grasp_param.num_angles,
                1,
                1,
            ],
        )  # (N, num_views, num_angles, num_depths, 1)

        return grasp_points, grasp_rots, grasp_angles, grasp_depths

    def _batch_compute_collision(
        self,
        start_id,
        batch_size,
        grasp_points,
        grasp_rots,
        grasp_depths,
        model,
        gripper_param,
        shared_grasp_widths,
        shared_grasp_collisions,
    ):
        max_width = gripper_param.max_width
        empty_thresh = gripper_param.empty_thresh
        height = gripper_param.height
        depth_base = gripper_param.depth_base
        finger_width = gripper_param.finger_width
        bottom_thickness = gripper_param.bottom_thickness

        dexgrasps = list()
        for grasp_id in range(
            start_id, min(start_id + batch_size, self.num_grasps)
        ):
            if (grasp_id + 1) % self.log_interval == 0:
                current_time = datetime.now()
                logger.info(
                    f"\t{current_time} , collision detection for grasp idx: \
                        [{grasp_id + 1}|{self.num_grasps}]"
                )
            grasp_point = grasp_points[grasp_id]
            R = grasp_rots[grasp_id]
            depth = grasp_depths[grasp_id]
            target = model - grasp_point
            target = np.matmul(target, R)  # gripper coordinate

            mask1 = (target[:, 2] > -height / 2) & (target[:, 2] < height / 2)
            mask2 = (target[:, 0] > -depth_base) & (target[:, 0] < depth)
            mask7 = (target[:, 0] > -(depth_base + bottom_thickness)) & (
                target[:, 0] < -depth_base
            )
            mask4_max_width = target[:, 1] < -max_width / 2
            mask6_max_width = target[:, 1] > max_width / 2

            inner_mask = (
                mask1 & mask2 & (~mask4_max_width) & (~mask6_max_width)
            )
            empty_flag = np.sum(inner_mask) < empty_thresh

            points_in_gripper = target[inner_mask][:, 1]  # just y
            width = analyze_width(points_in_gripper)
            shared_grasp_widths[grasp_id] = width

            mask3 = target[:, 1] > -(width / 2 + finger_width)
            mask5 = target[:, 1] < (width / 2 + finger_width)
            mask4 = target[:, 1] < -width / 2
            mask6 = target[:, 1] > width / 2

            left_mask = (
                mask1 & mask2 & mask3 & mask4
            )  # point collision with left gripper finger
            right_mask = (
                mask1 & mask2 & mask5 & mask6
            )  # point collision with right gripper finger
            bottom_mask = (
                mask1 & mask3 & mask5 & mask7
            )  # point collision with bottom gripper
            inner_mask_with_width = (
                mask1 & mask2 & (~mask4) & (~mask6)
            )  # inner mask with calculated width
            collision_flag = np.any((left_mask | right_mask | bottom_mask))
            empty_flag_with_width = (
                np.sum(inner_mask_with_width) < empty_thresh
            )
            collision_flag = collision_flag | empty_flag_with_width
            shared_grasp_collisions[grasp_id] = int(collision_flag)
            if collision_flag or empty_flag:
                dexgrasps.append(None)
            else:
                # gripper coordinate
                gripper_offset = np.array(
                    [float(depth), 0, 0], dtype=np.float32
                )
                center = grasp_point + R @ gripper_offset
                center = center[np.newaxis, :]  # object coordinate
                binormal, approach_angle = matrix_to_dexnet_params(R)
                grasp = ParallelJawPtGrasp3D(
                    ParallelJawPtGrasp3D.configuration_from_params(
                        center, binormal, width, approach_angle
                    ),
                    depth,
                )
                dexgrasps.append(grasp)
        return dexgrasps

    def _gen_widths_collisions_dexgrasps(
        self, grasp_points, collision_points, rotations, depths, gripper_param
    ):
        grasp_points = grasp_points.reshape([-1, 3])
        grasp_rots = rotations.reshape([-1, 3, 3])
        grasp_depths = depths.reshape([-1, 1])

        dexgrasps = list()
        with mp.Manager() as manager:
            shared_grasp_widths = manager.list([0.0] * grasp_depths.size)
            shared_grasp_collisions = manager.list([0] * grasp_depths.size)

            batch_compute_collision_partial = functools.partial(
                self._batch_compute_collision,
                grasp_points=grasp_points,
                grasp_rots=grasp_rots,
                grasp_depths=grasp_depths,
                model=collision_points,
                gripper_param=gripper_param,
                shared_grasp_widths=shared_grasp_widths,
                shared_grasp_collisions=shared_grasp_collisions,
            )
            logger.info(
                "Start batch collision computation with {}".format(
                    self.mp_batch_size
                )
            )
            with Pool(processes=self.mp_proc) as pool:
                all_dexgrasps = pool.starmap(
                    batch_compute_collision_partial,
                    [
                        (i, self.mp_batch_size)
                        for i in range(0, self.num_grasps, self.mp_batch_size)
                    ],
                )
                dexgrasps = [
                    dexgrasp for batch in all_dexgrasps for dexgrasp in batch
                ]

            np_shared_grasp_widths = np.array(
                shared_grasp_widths, dtype=np.float32
            )
            np_shared_grasp_collisions = np.array(
                shared_grasp_collisions, dtype=bool
            )

        return dexgrasps, np_shared_grasp_widths, np_shared_grasp_collisions

    def _batch_compute_score(
        self, start_id, batch_size, dexgrasps, dexnet_model, num_grasps
    ):
        batch_scores = []
        for grasp_id in range(
            start_id, min(start_id + batch_size, num_grasps)
        ):
            if (grasp_id + 1) % self.log_interval == 0:
                current_time = datetime.now()
                logger.info(
                    f"\t{current_time} , scoring grasp idx: \
                        [{grasp_id + 1}|{num_grasps}]"
                )
            grasp = dexgrasps[grasp_id]
            if grasp is None:
                score = -1.0
            else:
                config = ForceClosureConfig().to_config_dict()
                force_closure_quality_config = dict()
                fc_list = np.array(
                    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                )
                for value_fc in fc_list:
                    value_fc = round(value_fc, 2)
                    config["metrics"]["force_closure"]["friction_coef"] = (
                        value_fc
                    )
                    force_closure_quality_config[value_fc] = (
                        GraspQualityConfigFactory.create_config(
                            config["metrics"]["force_closure"]
                        )
                    )
                score = get_grasp_score(
                    grasp, dexnet_model, fc_list, force_closure_quality_config
                )
            batch_scores.append(score)
        return batch_scores

    def _gen_grasp_scores(self, dexnet_model, dexgrasps):
        batch_compute_score_partial = functools.partial(
            self._batch_compute_score,
            dexgrasps=dexgrasps,
            dexnet_model=dexnet_model,
            num_grasps=self.num_grasps,
        )
        print(
            "Start batch score generation with {}".format(self.mp_batch_size)
        )
        with Pool(processes=self.mp_proc) as pool:
            all_scores = pool.starmap(
                batch_compute_score_partial,
                [
                    (i, self.mp_batch_size)
                    for i in range(0, self.num_grasps, self.mp_batch_size)
                ],
            )
            scores = [score for batch in all_scores for score in batch]
        scores = np.array(scores).astype(np.float32)
        return scores


class ObjectGraspLabelOutput(BaseModel):
    grasp_points: NDArray = Field(
        ...,
        description="grasp points downsampled from mesh, (N,3)",
    )
    scores: NDArray = Field(
        ...,
        description="quality scores, (N, num_views, num_angles, num_depths)",
    )
    collisions: NDArray = Field(
        ...,
        description="collision label, (N, num_views, num_angles, num_depths)",
    )
    dexgrasps: NDArray = Field(
        ...,
        description="dexgrasp label, (N, num_views, num_angles, num_depths)",
    )
    rotations: NDArray = Field(
        ..., description="rotation matrix, (N, num_views, 3, 3)"
    )
    angles: NDArray = Field(
        ...,
        description="grasp angles, (N, num_views, num_angles, num_depths, 1)",
    )
    depths: NDArray = Field(
        ...,
        description="grasp depths, (N, num_views, num_angles, num_depths, 1)",
    )
    widths: NDArray = Field(
        ...,
        description="grasp width, (N, num_views, num_angles, num_depths, 1)",
    )

    def save_to_file(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        logger.info(f"Saving grasp labels to {path}")
        offsets = np.concatenate(
            [self.angles, self.depths, self.widths], axis=-1
        )  # (N, num_views, num_angles, num_depths, 3)
        np.savez(
            path,
            points=self.grasp_points,
            scores=self.scores,
            offsets=offsets,
            collision=self.collisions,
        )
