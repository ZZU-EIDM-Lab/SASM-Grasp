# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import functools
import json
import logging
import multiprocessing as mp
import os
from datetime import datetime
from multiprocessing import Pool
from typing import List

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict

from robo_orchard_grasp_label.datatypes import (
    GameraParam,
    GraspParam,
    GripperParam,
    ObjectParam,
    SceneParam,
)
from robo_orchard_grasp_label.label_utils import (
    batch_viewpoint_params_to_matrix,
    create_point_cloud_from_depth_image,
    create_table_points,
    generate_views,
    get_workspace_mask,
    load_mesh_as_pointcloud,
    transform_point_cloud,
)
from robo_orchard_grasp_label.utils.knn.knn_modules import knn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SceneGraspLabelGenerator(BaseModel):
    scene_param: SceneParam
    grasp_param: GraspParam
    gripper_param: GripperParam
    obj_params: List[ObjectParam]
    scene_dir: str
    model_dir: str
    grasp_label_dir: str
    output_dir: str
    log_interval: int = 100
    mp_proc: int = 4
    mp_batch_size: int = 100
    use_ing: bool = True
    save_result: bool = True
    cal_collision: bool = True
    cal_graspness: bool = True
    model_config = ConfigDict(extra="allow")

    def run(self):
        self.obj_list = self._get_object_param()
        if self.cal_collision:
            scene_points = self._gen_scene_point()
            views, angles, depths, widths = self._gen_grasp_pose_template()
            self._gen_scene_collision(
                views, angles, depths, widths, scene_points
            )
        if self.cal_graspness:
            self._gen_graspness()

    def _gen_grasp_pose_template(self):
        views = generate_views(self.grasp_param.num_views)
        angles = np.array(
            [
                np.pi / self.grasp_param.num_angles * i
                for i in range(self.grasp_param.num_angles)
            ],
            dtype=np.float32,
        )
        depths = [0.01, 0.02, 0.03, 0.04][: self.grasp_param.num_depths]
        if self.grasp_param.num_depths == 5:
            depths.insert(0, 0.005)

        widths = [0.01 * x for x in range(1, 16, 1)]
        return views, angles, depths, widths

    def _get_object_param(self) -> List[ObjectParam]:
        obj_names_path = os.path.join(
            self.scene_dir,
            self.scene_param.name,
            "object_name_list.json",
        )
        if not os.path.exists(obj_names_path):
            raise FileNotFoundError(
                "Object name list not found in the scene directory."
            )
        with open(obj_names_path, "r") as f:
            obj_names = json.load(f)

        obj_name_to_param = {obj.name: obj for obj in self.obj_params}
        obj_list = []
        for obj_name in obj_names:
            obj_list.append(obj_name_to_param[obj_name])
        return obj_list

    def _get_object_pose(self, ann_id=0):
        obj_poses_path = os.path.join(
            self.scene_dir,
            self.scene_param.name,
            "poses",
            str(ann_id).zfill(4) + ".npz",
        )
        if not os.path.exists(obj_poses_path):
            raise FileNotFoundError(
                f"Object poses for annotation {ann_id} not found."
            )
        obj_poses = np.load(obj_poses_path)
        obj_pose_list = []
        for obj in self.obj_list:
            obj_pose = obj_poses[obj.name]
            obj_pose_list.append(obj_pose)
        return obj_pose_list

    def _load_object_point(self, obj, voxel_size, max_points) -> np.ndarray:
        points = load_mesh_as_pointcloud(
            os.path.join(self.model_dir, obj, "meshes/model.obj"),
            voxel_size,
            max_points,
        )
        return points

    def _gen_scene_point(self) -> np.ndarray:
        logger.info("Generating scene points")

        # load object point cloud and downsample
        obj_point_list = []
        for obj in self.obj_list:
            points = self._load_object_point(
                obj.name, obj.model_voxel_size, obj.max_model_points
            )
            obj_point_list.append(points)

        # transform object points to camera coordinate
        trans_point_list = []
        for point, pose in zip(
            obj_point_list, self._get_object_pose(), strict=False
        ):
            point_trans = transform_point_cloud(point, pose)
            trans_point_list.append(point_trans)

        # crreat table points and transform to camera coordinate
        table_point = create_table_points(
            1.0, 1.0, 0.05, dx=-0.1, dy=-0.5, dz=-0.05, grid_size=0.008
        )
        cam_wrt_world = np.load(
            os.path.join(
                self.scene_dir,
                self.scene_param.name,
                "T_cam_to_world.npy",
            )
        )[0]
        table_point_trans = transform_point_cloud(
            table_point, np.linalg.inv(cam_wrt_world)
        )

        trans_point_list.append(table_point_trans)
        scene_points = np.concatenate(trans_point_list)

        if self.save_result:
            os.makedirs(
                os.path.join(self.output_dir, "scene_points"), exist_ok=True
            )
            save_path = os.path.join(
                self.output_dir,
                "scene_points",
                "{}.npy".format(self.scene_param.name),
            )
            logger.info(f"Saving scene points to {save_path}")
            np.save(save_path, scene_points)
        return scene_points

    def _batch_scene_collision(
        self,
        start_id,
        batch_size,
        shared_collision,
        points_sampled,
        scene_points,
        views,
        angles,
        depths,
        widths,
        num_points,
    ):
        for point_ind in range(
            start_id, min(start_id + batch_size, num_points)
        ):
            if (point_ind + 1) % self.log_interval == 0:
                current_time = datetime.now()
                logger.info(
                    f"\t{current_time} , point idx: \
                        [{point_ind + 1}|{num_points}]"
                )
            point_sampled = points_sampled[point_ind]
            batch_collision = np.ones(
                [
                    self.grasp_param.num_views,
                    self.grasp_param.num_angles,
                    self.grasp_param.num_depths,
                ],
                dtype=bool,
            )  # has collision: 1, no collision:0
            for j, view in enumerate(views):
                for k, angle in enumerate(angles):
                    # transform model to gripper frame
                    R = batch_viewpoint_params_to_matrix(-view, angle)
                    points_centered = scene_points - point_sampled.reshape(
                        1, -1
                    )
                    points_rotated = np.dot(R.T, points_centered.T).T
                    # check height
                    mask1 = points_rotated[:, 2] > (
                        -self.gripper_param.height / 2.0
                    )
                    mask2 = points_rotated[:, 2] < (
                        self.gripper_param.height / 2.0
                    )
                    target = points_rotated[mask1 & mask2]
                    for d, depth in enumerate(depths):
                        points_cropped = target[target[:, 0] < depth].copy()
                        # compute width
                        points_in_gripper = None
                        for w in widths:
                            mask3 = points_cropped[:, 1] > (-w / 2.0)
                            mask4 = points_cropped[:, 1] < (w / 2.0)
                            mask = mask3 & mask4
                            inner_points = points_cropped[mask]
                            outer_points = points_cropped[~mask]
                            # check depth
                            if np.any(
                                inner_points[:, 0]
                                < -self.gripper_param.depth_base
                            ):
                                continue
                            # check inner space
                            if len(inner_points) < 10:
                                continue
                            # check outer collision
                            ymin = inner_points[:, 1].min()
                            ymax = inner_points[:, 1].max()
                            half_width = max(abs(ymin), abs(ymax))
                            outer_mask1 = outer_points[:, 1] >= -(
                                half_width
                                + self.gripper_param.collision_thresh
                            )
                            outer_mask2 = outer_points[:, 1] <= (
                                half_width
                                + self.gripper_param.collision_thresh
                            )
                            if np.any(outer_mask1 & outer_mask2):
                                continue
                            points_in_gripper = inner_points
                            break
                        if points_in_gripper is None:
                            continue

                        ymin = points_in_gripper[:, 1].min()
                        ymax = points_in_gripper[:, 1].max()
                        # width = 2 * max(abs(ymin), abs(ymax))
                        batch_collision[j, k, d] = False

            shared_collision[point_ind] = batch_collision

    def _gen_scene_collision(
        self, views, angles, depths, widths, scene_points
    ):
        for obj, obj_pose in zip(
            self.obj_list, self._get_object_pose(), strict=False
        ):
            logger.info(f"Generating scene collision for {obj.name}")
            points_sampled = self._load_object_point(
                obj.name, obj.sample_voxel_size, obj.max_sample_points
            )

            # transform scene point to object coordinate
            scene_points_trans = transform_point_cloud(
                scene_points, np.linalg.inv(obj_pose)
            )
            # crop scene point to speed up
            xmin, xmax = points_sampled[:, 0].min(), points_sampled[:, 0].max()
            ymin, ymax = points_sampled[:, 1].min(), points_sampled[:, 1].max()
            zmin, zmax = points_sampled[:, 2].min(), points_sampled[:, 2].max()
            outlier = 0.05
            mask_x = (scene_points_trans[:, 0] > xmin - outlier) & (
                scene_points_trans[:, 0] < xmax + outlier
            )
            mask_y = (scene_points_trans[:, 1] > ymin - outlier) & (
                scene_points_trans[:, 1] < ymax + outlier
            )
            mask_z = (scene_points_trans[:, 2] > zmin - outlier) & (
                scene_points_trans[:, 2] < zmax + outlier
            )
            mask = mask_x & mask_y & mask_z
            scene_points_trans = scene_points_trans[mask]

            num_points = len(points_sampled)
            with mp.Manager() as manager:
                shared_collision = manager.dict()
                batch_scene_collision_partial = functools.partial(
                    self._batch_scene_collision,
                    shared_collision=shared_collision,
                    points_sampled=points_sampled,
                    scene_points=scene_points_trans,
                    views=views,
                    angles=angles,
                    depths=depths,
                    widths=widths,
                    num_points=num_points,
                )
                logger.info(
                    "Start batch scene collision computation with {}".format(
                        self.mp_batch_size
                    )
                )
                with Pool(processes=self.mp_proc) as pool:
                    pool.starmap(
                        batch_scene_collision_partial,
                        [
                            (i, self.mp_batch_size)
                            for i in range(0, num_points, self.mp_batch_size)
                        ],
                    )
                collisions = [None for _ in range(num_points)]
                for i in range(len(points_sampled)):
                    collisions[i] = shared_collision[i]
                collisions = np.array(collisions).astype(bool)

                if self.save_result:
                    os.makedirs(
                        os.path.join(
                            self.output_dir,
                            "collision_label",
                            self.scene_param.name,
                        ),
                        exist_ok=True,
                    )
                    save_path = os.path.join(
                        self.output_dir,
                        "collision_label",
                        self.scene_param.name,
                        f"{obj.name}_collision.npy",
                    )
                    logger.info(f"Saving collision label to {save_path}")
                    np.save(save_path, collisions)

    def _load_depth(self, img_name):
        depth_path = os.path.join(
            self.scene_dir,
            self.scene_param.name,
            "depth",
            img_name + ".png",
        )
        if not os.path.exists(depth_path):
            logger.error(f"Depth image {depth_path} not found")
            return None
        depth = np.array(Image.open(depth_path))
        return depth

    def _load_label(self, img_name):
        label_path = os.path.join(
            self.scene_dir,
            self.scene_param.name,
            "label",
            img_name + ".png",
        )
        if not os.path.exists(label_path):
            logger.error(f"Label image {label_path} not found")
            return None
        label = np.array(Image.open(label_path))
        return label

    def _load_grasp_label(self, obj_prefix):
        grasp_label_path = os.path.join(
            self.grasp_label_dir, obj_prefix + "_labels.npz"
        )
        if not os.path.exists(grasp_label_path):
            logger.error(f"Grasp label {grasp_label_path} not found")
            return None
        logger.info(f"\tLoad grasp_label from : {grasp_label_path}")
        grasp_label = np.load(grasp_label_path)
        return grasp_label

    def _gen_graspness(self):
        logger.info("Start generating graspness")

        fric_coef_thresh = 0.8
        point_grasp_num = (
            self.grasp_param.num_views
            * self.grasp_param.num_angles
            * self.grasp_param.num_depths
        )
        collision_dump = []
        grasp_labels = {}
        for obj in self.obj_list:
            file = self._load_grasp_label(obj.name)
            grasp_labels[obj.name] = (
                file["points"].astype(np.float32),
                file["offsets"].astype(np.float32),
                file["scores"].astype(np.float32),
            )
            collision_path = os.path.join(
                self.output_dir,
                "collision_label",
                self.scene_param.name,
                "{}_collision.npy".format(obj.name),
            )
            if not os.path.exists(collision_path):
                raise FileNotFoundError(
                    f"Collision label {collision_path} not found"
                )
            logger.info(f"\tLoad collision_label from : {collision_path}")
            collision_dump.append(np.load(collision_path))

        cam_wrt_world = np.load(
            os.path.join(
                self.scene_dir,
                "scene_" + str(self.scene_param.id).zfill(4),
                "T_cam_to_world.npy",
            )
        )

        meta = np.load(
            os.path.join(self.scene_dir, self.scene_param.name, "meta.npz")
        )
        factor_depth = meta["factor_depth"]
        intrinsic = meta["intrinsic_matrix"]
        camera = GameraParam(
            width=1280,
            height=720,
            fx=intrinsic[0][0],
            fy=intrinsic[1][1],
            cx=intrinsic[0][2],
            cy=intrinsic[1][2],
            scale=factor_depth,
        )

        for ann_id in range(256):
            # get scene point cloud
            logger.info(
                f"\tgenerating scene: {self.scene_param.id} ann: {ann_id}"
            )
            depth = self._load_depth(str(ann_id).zfill(4))
            seg = self._load_label(str(ann_id).zfill(4))

            cloud = create_point_cloud_from_depth_image(
                depth, camera, organized=True
            )

            # remove outlier and get objectness label
            depth_mask = depth > 0
            trans = cam_wrt_world[ann_id]
            workspace_mask = get_workspace_mask(
                cloud, seg, trans=trans, organized=True, outlier=0.02
            )
            mask = depth_mask & workspace_mask
            cloud_masked = cloud[mask]
            # objectness_label = seg[mask]

            grasp_points = []
            grasp_points_graspness = []
            for i, (obj, trans_) in enumerate(
                zip(self.obj_list, self._get_object_pose(ann_id), strict=False)
            ):
                sampled_points, offsets, fric_coefs = grasp_labels[obj.name]
                collision = collision_dump[
                    i
                ]  # num_points * num_views * num_angles * num_depths
                num_points = sampled_points.shape[0]

                valid_grasp_mask = (
                    (fric_coefs <= fric_coef_thresh)
                    & (fric_coefs > 0)
                    & ~collision
                )
                valid_grasp_mask = valid_grasp_mask.reshape(num_points, -1)
                graspness = np.sum(valid_grasp_mask, axis=1) / point_grasp_num
                if self.use_ing:  # Instance-normalized graspness
                    graspness = (graspness - graspness.min()) / (
                        graspness.max() - graspness.min()
                    )
                target_points = transform_point_cloud(sampled_points, trans_)
                grasp_points.append(target_points)
                grasp_points_graspness.append(graspness.reshape(num_points, 1))
            grasp_points = np.vstack(grasp_points)
            grasp_points_graspness = np.vstack(grasp_points_graspness)

            grasp_points = torch.from_numpy(grasp_points).cuda()
            grasp_points_graspness = torch.from_numpy(
                grasp_points_graspness
            ).cuda()
            grasp_points = (
                grasp_points.transpose(0, 1).contiguous().unsqueeze(0)
            )

            masked_points_num = cloud_masked.shape[0]
            cloud_masked_graspness = np.zeros((masked_points_num, 1))
            part_num = int(masked_points_num / 10000)
            for i in range(1, part_num + 2):  # lack of cuda memory
                if i == part_num + 1:
                    cloud_masked_partial = cloud_masked[10000 * part_num :]
                    if len(cloud_masked_partial) == 0:
                        break
                else:
                    cloud_masked_partial = cloud_masked[
                        10000 * (i - 1) : (i * 10000)
                    ]
                cloud_masked_partial = torch.from_numpy(
                    cloud_masked_partial
                ).cuda()
                cloud_masked_partial = (
                    cloud_masked_partial.transpose(0, 1)
                    .contiguous()
                    .unsqueeze(0)
                )
                nn_inds = (
                    knn(grasp_points, cloud_masked_partial, k=1).squeeze() - 1
                )
                cloud_masked_graspness[10000 * (i - 1) : (i * 10000)] = (
                    torch.index_select(grasp_points_graspness, 0, nn_inds)
                    .cpu()
                    .numpy()
                )

            # cloud_masked_graspness[~foreground_mask] = 0.0
            max_graspness = np.max(cloud_masked_graspness)
            min_graspness = np.min(cloud_masked_graspness)
            cloud_masked_graspness = (
                cloud_masked_graspness - min_graspness
            ) / (max_graspness - min_graspness)

            if self.save_result:
                os.makedirs(
                    os.path.join(
                        self.output_dir, "graspness", self.scene_param.name
                    ),
                    exist_ok=True,
                )
                save_path = os.path.join(
                    self.output_dir,
                    "graspness",
                    self.scene_param.name,
                    str(ann_id).zfill(4) + ".npy",
                )
                logger.info(f"\tSave graspness to  : {save_path}")
                np.save(save_path, cloud_masked_graspness)
