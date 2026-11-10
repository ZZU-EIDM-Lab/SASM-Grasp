# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

# This file was originally copied from the [EconomicGrasp] repository:
# https://github.com/iSEE-Laboratory/EconomicGrasp
# Modifications have been made to fit the needs of this project.

import os

import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from torch.utils.data import Dataset
import json
from robo_orchard_lab.utils.geometry import depth_to_range_image
_POINTWISE_CANDIDATE_KEYS = {
    "point_clouds",
    "cloud_normal",
    "cloud_colors",
    "coordinates_for_voxel",
    "point_scale",
    "point_class",
    "point_semantic_weight",
    "seg",
    "graspness_label",
    "objectness_label",
    "segmentation_label",
}


def collate_scale_sampling_batch(batch):

    from robo_orchard_lab.dataset.collates import (
        collate_batch_dict
    )

    fixed_items = []

    candidate_items = {
        key: [] for key in _POINTWISE_CANDIDATE_KEYS
    }

    for item in batch:

        fixed_item = {}

        for key, value in item.items():

            if key in _POINTWISE_CANDIDATE_KEYS:

                if isinstance(value, torch.Tensor):
                    tensor = value

                else:
                    array = np.asarray(value)

                    tensor = torch.from_numpy(
                        np.ascontiguousarray(array)
                    )

                candidate_items[key].append(tensor)

            else:
                fixed_item[key] = value

        fixed_items.append(fixed_item)

    collated = collate_batch_dict(fixed_items)

    for key, values in candidate_items.items():

        if values:
            collated[key] = values

    return collated

class EconomicGraspNet1BDataset(Dataset):
    def __init__(self, data_cfgs):
        self.root = data_cfgs.data_root
        self.camera = data_cfgs.camera

        self.split = data_cfgs.split
        self.voxel_size = data_cfgs.voxel_size
        self.num_sample_points = data_cfgs.num_sample_points
        self.remove_outlier = data_cfgs.remove_outlier
        self.remove_invisible = data_cfgs.remove_invisible
        self.use_new_graspness = data_cfgs.use_new_graspness

        self.augment = data_cfgs.augment
        self.load_label = data_cfgs.load_label
        self.collision_labels = {}

        if self.split == "train":
            self.sceneIds = list(range(100))
        elif self.split == "test":
            self.sceneIds = list(range(100, 190))
        elif self.split == "test_seen":
            self.sceneIds = list(range(100, 130))
        elif self.split == "test_similar":
            self.sceneIds = list(range(130, 160))
        elif self.split == "test_novel":
            self.sceneIds = list(range(160, 190))
        elif self.split == "mini":
            self.sceneIds = [0]

        self.sceneIds = [
            "scene_{}".format(str(x).zfill(4)) for x in self.sceneIds
        ]

        semantic_path = os.path.join(self.root, "semantic_prior.json")
        with open(semantic_path, "r") as f:
            self.semantic_prior = json.load(f)
        self.id_to_name = {
            1: "cracker_box",
            2: "sugar_box",
            3: "tomato_soup_can",
            4: "mustard_bottle",
            5: "potted_meat_can",
            6: "banana",
            7: "bowl",
            8: "mug",
            9: "power_drill",
            10: "scissors",
            11: "chips_can",
            12: "strawberry",
            13: "apple",
            14: "lemon",
            15: "peach",
            16: "pear",
            17: "orange",
            18: "plum",
            19: "knife",
            20: "phillips_screwdriver",
            21: "flat_screwdriver",
            22: "racquetball",
            23: "b_cups",
            24: "d_cups",
            25: "a_toy_airplane",
            26: "c_toy_airplane",
            27: "d_toy_airplane",
            28: "f_toy_airplane",
            29: "h_toy_airplane",
            30: "i_toy_airplane",
            31: "j_toy_airplane",
            32: "k_toy_airplane",
            33: "padlock",
            34: "dragon",
            35: "sum37_secret_repair",
            36: "jvr_cleansing_foam",
            37: "dabao_wash_soup",
            38: "nzskincare_mouth_rinse",
            39: "dabao_sod",
            40: "soap_box",
            41: "kispa_cleanser",
            42: "darlie_toothpaste",
            43: "nivea_men_oil_control",
            44: "baoke_marker",
            45: "hosjam",
            46: "pitcher_cap",
            47: "dish",
            48: "white_mouse",
            49: "camel",
            50: "deer",
            51: "zebra",
            52: "large_elephant",
            53: "rhinocero",
            54: "small_elephant",
            55: "monkey",
            56: "giraffe",
            57: "gorilla",
            58: "weiquan",
            59: "darlie_box",
            60: "soap",
            61: "black_mouse",
            62: "dabao_facewash",
            63: "pantene",
            64: "head_shoulders_supreme",
            65: "thera_med",
            66: "dove",
            67: "head_shoulders_care",
            68: "lion",
            69: "coconut_juice_box",
            70: "hippo",
            71: "tape",
            72: "rubiks_cube",
            73: "peeler_cover",
            74: "peeler",
            75: "ice_cube_mould",
            76: "bar_clamp",
            77: "climbing_hold",
            78: "endstop_holder",
            79: "gearbox",
            80: "mount1",
            81: "mount2",
            82: "nozzle",
            83: "part1",
            84: "part3",
            85: "pawn",
            86: "pipe_connector",
            87: "turbine_housing",
            88: "vase"
        }

        self.colorpath = []
        self.normalpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.graspnesspath = []
        self.grasp_labels = {}

        for x in self.sceneIds:
            for img_num in range(256):
                self.colorpath.append(
                    os.path.join(
                        self.root,
                        "scenes",
                        x,
                        self.camera,
                        "rgb",
                        str(img_num).zfill(4) + ".png",
                    )
                )
                self.normalpath.append(
                    os.path.join(
                        self.root,
                        "scenes",
                        x,
                        self.camera,
                        "normal",
                        str(img_num).zfill(4) + ".npy",
                    )
                )
                self.depthpath.append(
                    os.path.join(
                        self.root,
                        "scenes",
                        x,
                        self.camera,
                        "depth",
                        str(img_num).zfill(4) + ".png",
                    )
                )
                self.labelpath.append(
                    os.path.join(
                        self.root,
                        "scenes",
                        x,
                        self.camera,
                        "label",
                        str(img_num).zfill(4) + ".png",
                    )
                )
                self.metapath.append(
                    os.path.join(
                        self.root,
                        "scenes",
                        x,
                        self.camera,
                        "meta",
                        str(img_num).zfill(4) + ".mat",
                    )
                )
                if self.use_new_graspness is False:
                    self.graspnesspath.append(
                        os.path.join(
                            self.root,
                            "graspness",
                            x,
                            self.camera,
                            str(img_num).zfill(4) + ".npy",
                        )
                    )
                elif self.use_new_graspness is True:
                    self.graspnesspath.append(
                        os.path.join(
                            self.root,
                            "graspness_finegrasp",
                            x,
                            self.camera,
                            str(img_num).zfill(4) + ".npy",
                        )
                    )
                # strip is for removing the space at the beginning and the end
                self.scenename.append(x.strip())
                self.frameid.append(img_num)

            if self.load_label:
                self.grasp_labels[x.strip()] = os.path.join(
                    self.root,
                    "economic_grasp_label_300views",
                    x + "_labels.npz",
                )

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list,normals=None):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
            point_clouds = self.transform_point_cloud(
                point_clouds, flip_mat, "3x3"
            )
            if normals is not None:
                normals_vec = normals*2.0-1.0
                normals_vec = self.transform_point_cloud(normals_vec, flip_mat, "3x3")
                normals = (normals_vec+1.0)/2.0
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(
                    flip_mat, object_poses_list[i]
                ).astype(np.float32)

        return point_clouds, object_poses_list, normals

    def __getitem__(self, index):
        if self.load_label:
            data = self.get_data_label(index)
        else:
            data = self.get_data(index)
        return data

    def transform_point_cloud(self, cloud, transform, format="4x4"):
        """Transform points to new coordinates with transformation matrix.

        Args:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or
                rotation + translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Return:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
        """
        if not (format == "3x3" or format == "4x4" or format == "3x4"):
            raise ValueError(
                "Unknown transformation format, only support '3x3' or '4x4' "
                "or '3x4'."
            )
        if format == "3x3":
            cloud_transformed = np.dot(transform, cloud.T).T
        elif format == "4x4" or format == "3x4":
            ones = np.ones(cloud.shape[0])[:, np.newaxis]
            cloud_ = np.concatenate([cloud, ones], axis=1)
            cloud_transformed = torch.mm(
                torch.from_numpy(transform), torch.from_numpy(cloud_).T
            ).T.numpy()
            cloud_transformed = cloud_transformed[:, :3]
        return cloud_transformed

    def get_workspace_mask(
        self, cloud, seg, trans=None, organized=True, outlier=0.0
    ):
        """Keep points in workspace as input.

        Args:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and
                workspace is greater than outlier, the point will be removed

        Return:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
        """
        if organized:
            h, w, _ = cloud.shape
            cloud = cloud.reshape([h * w, 3])
            seg = seg.reshape(h * w)
        if trans is not None:
            cloud = self.transform_point_cloud(cloud, trans)
        foreground = cloud[seg > 0]
        xmin, ymin, zmin = foreground.min(axis=0)
        xmax, ymax, zmax = foreground.max(axis=0)
        mask_x = (cloud[:, 0] > xmin - outlier) & (
            cloud[:, 0] < xmax + outlier
        )
        mask_y = (cloud[:, 1] > ymin - outlier) & (
            cloud[:, 1] < ymax + outlier
        )
        mask_z = (cloud[:, 2] > zmin - outlier) & (
            cloud[:, 2] < zmax + outlier
        )
        workspace_mask = mask_x & mask_y & mask_z
        if organized:
            workspace_mask = workspace_mask.reshape([h, w])

        return workspace_mask
    def _build_point_scale_metadata(self, cloud_masked, seg_masked, obj_idxs):

        point_scale = np.zeros(len(seg_masked), dtype=np.float32)
        point_class = np.zeros(len(seg_masked), dtype=np.int64)
        point_semantic_weight = np.ones(len(seg_masked), dtype=np.float32)

        # 背景权重保持原来的 0.3
        point_semantic_weight[seg_masked == 0] = 0.3

        lambda_sem = 0.03

        for instance_id in np.unique(seg_masked):

            if instance_id == 0:
                continue

            instance_mask = seg_masked == instance_id
            pts = cloud_masked[instance_mask]

            if pts.shape[0] == 0:
                continue

            # 计算物体最小边长
            bbox = pts.max(axis=0) - pts.min(axis=0)
            min_edge = max(float(np.min(bbox)), 0.02)

            point_scale[instance_mask] = min_edge

            # 保留原来的尺度类别
            if min_edge < 0.04:
                point_class[instance_mask] = 1
            elif min_edge < 0.07:
                point_class[instance_mask] = 2
            else:
                point_class[instance_mask] = 3

            # 实例 ID 对应 meta 中的类别 ID
            category_index = int(instance_id) - 1

            if 0 <= category_index < len(obj_idxs):
                category_id = int(obj_idxs[category_index])
                class_name = self.id_to_name.get(category_id, None)
            else:
                class_name = None

            sem_score = (
                self.semantic_prior.get(class_name, 0.5)
                if class_name is not None else 0.5
            )

            sem_weight = 1.0 + lambda_sem * (sem_score - 0.5)

            point_semantic_weight[instance_mask] = sem_weight

        return point_scale, point_class, point_semantic_weight


    def get_data(self, index, return_raw_cloud=False):
        color = (
            np.array(Image.open(self.colorpath[index]), dtype=np.float32)
            / 255.0
        )
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        normal = np.load(self.normalpath[index]) / 255.0
        

        # camera in
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        obj_idxs = meta["cls_indexes"].flatten().astype(np.int32)
        intrinsic = meta["intrinsic_matrix"]
        factor_depth = meta["factor_depth"]

        # generate cloud
        cloud = depth_to_range_image(
            depth, intrinsic, depth_scale=factor_depth
        )

        # get valid points
        depth_mask = depth > 0
        # they are not the outliers, just the points far away from the objects
        if self.remove_outlier:
            camera_poses = np.load(
                os.path.join(
                    self.root, "scenes", scene, self.camera, "camera_poses.npy"
                )
            )
            align_mat = np.load(
                os.path.join(
                    self.root,
                    "scenes",
                    scene,
                    self.camera,
                    "cam0_wrt_table.npy",
                )
            )
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = self.get_workspace_mask(
                torch.from_numpy(cloud),
                torch.from_numpy(seg),
                trans=trans,
                organized=True,
                outlier=0.02,
            )
            mask = depth_mask & workspace_mask
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        normal_masked = normal[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        if return_raw_cloud:
            return cloud_masked, color_masked
        point_scale, point_class, point_semantic_weight = self._build_point_scale_metadata(
            cloud_masked, seg_masked, obj_idxs
        )
        ret_dict = {}
        ret_dict["cloud_masked"] = [cloud_masked]
        ret_dict["color_masked"] = [color_masked]
        ret_dict["point_clouds"] = cloud_masked.astype(np.float32)
        ret_dict["point_scale"] = point_scale
        ret_dict["point_class"] = point_class
        ret_dict["point_semantic_weight"] = point_semantic_weight

        ret_dict["cloud_normal"] = normal_masked.astype(np.float32)
        ret_dict["cloud_colors"] = color_masked.astype(np.float32)
        ret_dict["coordinates_for_voxel"] = (
            cloud_masked.astype(np.float32) / self.voxel_size
        )
        ret_dict["seg"] = seg_masked.astype(np.int64)
        ret_dict["data_idx"] = index
        ret_dict["scene_name"] = self.scene_list()[index]
        return ret_dict


    def get_data_label(self, index):
        color = (
            np.array(Image.open(self.colorpath[index]), dtype=np.float32)
            / 255.0
        )
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        normal = np.load(self.normalpath[index]) / 255.0
        graspness = np.load(
            self.graspnesspath[index]
        )  # already remove outliers

        obj_idxs = meta["cls_indexes"].flatten().astype(np.int32)
        poses = meta["poses"]
        intrinsic = meta["intrinsic_matrix"]
        factor_depth = meta["factor_depth"]

        # generate cloud
        # depth is in millimeters (mm), the transformed cloud is in meters (m).
        cloud = depth_to_range_image(
            depth, intrinsic, depth_scale=factor_depth
        )

        # get valid points
        depth_mask = depth > 0
        # they are not the outliers, just the points far away from the objects
        if self.remove_outlier:
            camera_poses = np.load(
                os.path.join(
                    self.root, "scenes", scene, self.camera, "camera_poses.npy"
                )
            )
            align_mat = np.load(
                os.path.join(
                    self.root,
                    "scenes",
                    scene,
                    self.camera,
                    "cam0_wrt_table.npy",
                )
            )
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = self.get_workspace_mask(
                torch.from_numpy(cloud),
                torch.from_numpy(seg),
                trans=trans,
                organized=True,
                outlier=0.02,
            )
            mask = depth_mask & workspace_mask
        else:
            mask = depth_mask
        normal_masked = normal[mask]
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        point_scale, point_class, point_semantic_weight = self._build_point_scale_metadata(
            cloud_masked, seg_masked, obj_idxs
        )
        objectness_label= seg_masked.copy()
        segmentation_label = objectness_label.copy()
        objectness_label[objectness_label > 1] = 1

        graspness = np.asarray(graspness, dtype=np.float32)

        if graspness.shape == mask.shape:
            graspness_candidate = graspness[mask]

        elif graspness.size == mask.size:
            graspness_candidate = graspness.reshape(-1)[mask.reshape(-1)]

        elif graspness.size == cloud_masked.shape[0]:
            graspness_candidate = graspness.reshape(-1)

        else:
            raise ValueError(
                f"graspness length {graspness.size} "
                f"does not match candidate points {cloud_masked.shape[0]}"
            )

        object_poses_list = []
        grasp_points_list = []
        grasp_rotations_list = []
        grasp_depth_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        view_graspness_list = []
        top_view_index_list = []

        # load labels
        grasp_labels = np.load(self.grasp_labels[scene])

        points = grasp_labels["points"]
        rotations = grasp_labels["rotations"].astype(np.int32)
        depth = grasp_labels["depth"].astype(np.int32)
        scores = grasp_labels["scores"].astype(np.float32) / 10.0
        widths = grasp_labels["widths"].astype(np.float32) / 1000.0
        topview = grasp_labels["topview"].astype(np.int32)
        view_graspness = grasp_labels["vgraspness"].astype(np.float32)
        pointid = grasp_labels["pointid"]
        for i, _obj_idx in enumerate(obj_idxs):
            object_poses_list.append(poses[:, :, i])
            grasp_points_list.append(points[pointid == i])
            grasp_rotations_list.append(rotations[pointid == i])
            grasp_depth_list.append(depth[pointid == i])
            grasp_scores_list.append(scores[pointid == i])
            grasp_widths_list.append(widths[pointid == i])
            view_graspness_list.append(view_graspness[pointid == i])
            top_view_index_list.append(topview[pointid == i])

        if self.augment:
            cloud_sampled, object_poses_list, normal_masked = self.augment_data(
                cloud_masked, object_poses_list, normal_masked
            )

        ret_dict = {}
        ret_dict["point_clouds"] = cloud_masked.astype(np.float32)

        ret_dict["point_scale"] = point_scale
        ret_dict["point_class"] = point_class
        ret_dict["point_semantic_weight"] = point_semantic_weight

        ret_dict["cloud_normal"] = normal_masked.astype(np.float32)
        ret_dict["cloud_colors"] = color_masked.astype(np.float32)

        ret_dict["coordinates_for_voxel"] = (
            cloud_masked.astype(np.float32) / self.voxel_size
        )

        ret_dict["graspness_label"] = graspness_candidate.astype(np.float32)

        # [scene_points, 1 (graspness)]
        ret_dict["objectness_label"] = objectness_label.astype(np.int64)

        # [scene_points, 1 (objectness)]
        ret_dict["segmentation_label"] = segmentation_label.astype(np.int64)

        # [scene_points, 1 (objectness)]
        ret_dict["object_poses_list"] = object_poses_list

        # list has a length of objects amount,
        # each has size [3, 4] (pose matrix)
        ret_dict["grasp_points_list"] = grasp_points_list

        # list has a length of objects amount,
        # each has size [object_points, 3 (coordinate)]
        ret_dict["grasp_rotations_list"] = grasp_rotations_list

        # list has a length of objects amount,
        # each has size [object_points, 60 (view)]
        ret_dict["grasp_depth_list"] = grasp_depth_list

        # list has a length of objects amount,
        # each has size [object_points, 60 (view)]
        ret_dict["grasp_widths_list"] = grasp_widths_list

        # list has a length of objects amount,
        # each has size [object_points, 60 (view)]
        ret_dict["grasp_scores_list"] = grasp_scores_list

        # list has a length of objects amount,
        # each has size [object_points, 60 (view)]
        ret_dict["view_graspness_list"] = view_graspness_list

        # list has a length of objects amount,
        # each has size [object_points, 300 (view graspness)]
        ret_dict["top_view_index_list"] = top_view_index_list

        # list has a length of objects amount,
        # each has size [object_points, 60 (top 60 views index)]
        ret_dict["data_idx"] = index

        ret_dict["scene_name"] = self.scene_list()[index]

        return ret_dict
