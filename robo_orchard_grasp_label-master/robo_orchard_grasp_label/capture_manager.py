# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import json
import os
import random

import cv2
import numpy as np
import torch
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from PIL import Image

from robo_orchard_grasp_label.camera_pose_manager import CameraPoseManager
from robo_orchard_grasp_label.sim_cfg import D435I_CFG, SceneCfg
from robo_orchard_grasp_label.sim_utils import (
    add_random_position_noise,
    add_random_rotation_noise,
)


class ViewCaptureManager:
    """Manager for capturing views of objects in a simulation environment.

    Args:
        simulation_app: The simulation application instance.
        obj_params(list[ObjectParam]):
            A list of object parameters containing names and IDs.
        num_objects(int): The number of objects to capture in the scene.
        asset_dir(str): The directory containing the object assets.
        output_dir(str): The directory where captured data will be saved.
        table_asset_path(str): The path to the table asset.
        t_cam_to_table_path(str): The camera to table transformation file path.
    """

    def __init__(
        self,
        simulation_app,
        obj_params,
        num_objects,
        output_dir,
        asset_dir,
        table_asset_path,
        t_cam_to_table_path,
    ):
        self.simulation_app = simulation_app
        self.sim = SimulationContext(
            SimulationCfg(dt=0.0083, render_interval=1)
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.asset_dir = os.path.abspath(asset_dir)
        self.num_objects = num_objects
        self.table_asset_path = os.path.abspath(table_asset_path)
        self.factor_depth = 1000.0
        self.num_views = 256
        self.camera_manager = CameraPoseManager(t_cam_to_table_path)
        self.obj_params = obj_params
        self.obj_to_id = self.get_obj_to_id(obj_params)

    def get_obj_to_id(self, obj_params):
        obj_to_id = {}
        for item in obj_params:
            obj_to_id[item.name.lower()] = item.id
        return obj_to_id

    def setup_camera_view(self):
        t_camere_to_graspnet_table = (
            self.camera_manager.get_graspnet_cam_pose()
        )  # (256, 4, 4)
        t_camera_to_world = self.camera_manager.get_isaac_cam_pose(
            t_camere_to_graspnet_table
        )
        t_camera_to_world_vector = CameraPoseManager.trans_matrix_to_pose(
            t_camera_to_world
        )  # (256, 7)
        self.t_camera_to_world_vector = t_camera_to_world_vector
        self.t_camera_to_world_mat = t_camera_to_world

    def update_camera_view(self, cam_idx):
        camera = self.scene["cameras"]
        pos = torch.tensor(
            self.t_camera_to_world_vector[cam_idx][:3]
        ).unsqueeze(0)
        rot = torch.tensor(
            self.t_camera_to_world_vector[cam_idx][3:][[3, 0, 1, 2]]
        ).unsqueeze(0)  # [x, y, z, wq] -> [w, x, y, z]
        camera.set_world_poses(pos, rot)
        camera.update(self.sim.get_physics_dt())
        self.sim.step()
        if int(cam_idx) % 50 == 0:
            print(f"Camera view {cam_idx} updated.")

    def select_object(self, num_objects, random_picking=False, obj_list=None):
        all_objects = []
        for obj in self.obj_params:
            folder_path = os.path.join(self.asset_dir, obj.name)
            if not os.path.exists(folder_path):
                print(f"Folder {folder_path} does not exist, skipping...")
                continue
            for file in os.listdir(folder_path):
                if file.endswith("_converted.usd"):
                    all_objects.append(
                        {
                            "id": obj.id,
                            "class": obj.name,
                            # add 'object_' to avoid starting with a number
                            "prefix": "object_" + obj.name,
                            "path": os.path.join(folder_path, file),
                        }
                    )
        if random_picking:
            selected_objects = random.sample(all_objects, min(num_objects, 10))
        else:
            assert obj_list is not None
            selected_list = random.sample(
                obj_list, min(len(obj_list), self.num_objects)
            )
            selected_objects = [
                obj for obj in all_objects if obj["class"] in selected_list
            ]
        print("Selected objects:")
        for obj in selected_objects:
            print(obj["class"])
        return selected_objects

    def update_object_cfg(self, selected_objects, env_ns, usd_cfg):
        objects_cfg = {}
        random.shuffle(selected_objects)
        for _, obj in enumerate(selected_objects):
            prim_path = f"{env_ns}/{obj['prefix']}"
            init_state_cfg = RigidObjectCfg.InitialStateCfg(
                pos=add_random_position_noise(
                    [0, 0, 0],
                    x_range=(0.2, 0.5),
                    y_range=(-0.2, 0.2),
                    z_range=(0.1, 0.5),
                ),
                rot=add_random_rotation_noise(
                    [1, 0, 0, 0],
                    delta_range=(0, 180),
                ),
            )
            spawn_cfg = usd_cfg.replace(
                usd_path=obj["path"],
                semantic_tags=[("class", obj["class"])],
            )
            objects_cfg[obj["class"]] = RigidObjectCfg(
                prim_path=prim_path, init_state=init_state_cfg, spawn=spawn_cfg
            )
        return objects_cfg

    def setup_scene(self):
        scene_cfg = SceneCfg(
            num_envs=1,
            env_spacing=2.5,
        )
        scene_cfg.cameras = D435I_CFG.replace(
            prim_path="{ENV_REGEX_NS}/static_camera"
        )

        table_cfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.4, 0.0, -0.02]),
            spawn=UsdFileCfg(
                usd_path=self.table_asset_path,
                scale=(1.5, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=0.01,
                    max_linear_velocity=0.01,
                    max_depenetration_velocity=1.0,
                    disable_gravity=True,
                ),
                mass_props=MassPropertiesCfg(mass=100.0),
                semantic_tags=[("class", "desk")],
            ),
        )
        scene_cfg.table = table_cfg

        self.scene = InteractiveScene(scene_cfg)

        usd_cfg = UsdFileCfg(
            usd_path=None,
            scale=(1, 1, 1),
            collision_props=CollisionPropertiesCfg(),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                max_angular_velocity=100.0,
                max_linear_velocity=1.0,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(
                mass=1.0,
            ),
        )

        selected_objects = self.select_object(
            num_objects=self.num_objects,
            random_picking=True,
        )

        objects_cfg = self.update_object_cfg(
            selected_objects=selected_objects,
            env_ns="{ENV_REGEX_NS}",
            usd_cfg=usd_cfg,
        )

        for obj_name, obj_cfg in objects_cfg.items():
            obj_cfg.prim_path = obj_cfg.prim_path.format(
                ENV_REGEX_NS=self.scene.env_regex_ns
            )
            rigid_obj = RigidObject(obj_cfg)
            self.scene._rigid_objects[obj_name] = rigid_obj

    def capture_observation(self, cam_idx):
        cam = self.scene["cameras"]
        rgb_image = (
            cam.data.output["rgb"].cpu().numpy().squeeze()
        )  # unit8, 0-255
        depth_image = (
            cam.data.output["distance_to_image_plane"].cpu().numpy().squeeze()
        )  # float32, in meters
        seg_image = (
            cam.data.output["semantic_segmentation"].cpu().numpy().squeeze()
        )
        seg_info = cam.data.info[0]["semantic_segmentation"]
        for idx, obj_class in seg_info["idToLabels"].items():
            obj_class = obj_class["class"].lower()
            if obj_class in [
                "BACKGROUND".lower(),
                "UNLABELLED".lower(),
                "desk",
            ]:
                seg_image[seg_image == int(idx)] = 0
            elif obj_class in [i.lower() for i in self.obj_to_id]:
                seg_image[seg_image == int(idx)] = (
                    self.obj_to_id[obj_class] + 1
                )
            else:
                raise ValueError(
                    f"Object class {obj_class} not found in obj2id mapping."
                )
        if len(np.unique(seg_image)) == 1:
            raise ValueError("No object in the scene.")

        color_dir = os.path.join(self.output_dir, "rgb")
        depth_dir = os.path.join(self.output_dir, "depth")
        label_dir = os.path.join(self.output_dir, "label")
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            os.path.join(color_dir, f"{str(cam_idx).zfill(4)}.png"), bgr_image
        )
        depth_image = (
            (depth_image * self.factor_depth).astype(np.uint16).squeeze()
        )  # uint16, in millimeters
        depth_image = Image.fromarray(depth_image)
        depth_image.save(
            os.path.join(depth_dir, f"{str(cam_idx).zfill(4)}.png"),
            format="PNG",
        )
        cv2.imwrite(
            os.path.join(label_dir, f"{str(cam_idx).zfill(4)}.png"), seg_image
        )

        if int(cam_idx) % 50 == 0:
            print(f"View_{cam_idx} observation captured.")

    def _dump_object_pose(self):
        pose_dir = os.path.join(self.output_dir, "poses")
        os.makedirs(pose_dir, exist_ok=True)
        for i in range(self.num_views):
            objects = self.scene.rigid_objects
            t_obejct_to_camera = {}
            for _, obj in objects.items():
                _, obj_name = obj.cfg.spawn.semantic_tags[0]
                pos = (obj.data.root_pos_w).cpu().numpy().squeeze()
                quat = (
                    (obj.data.root_quat_w).cpu().numpy().squeeze()
                )  # [w, x, y, z]
                t_object_to_world = np.concatenate([pos, quat], axis=0)
                t_object_to_world_mat = CameraPoseManager.trans_pose_to_matrix(
                    t_object_to_world, scalar_first=True
                )  # object2world
                t_obejct_to_camera[obj_name] = np.dot(
                    np.linalg.inv(self.t_camera_to_world_mat[i]),
                    t_object_to_world_mat,
                )  # object2camere
            np.savez(
                os.path.join(pose_dir, f"{str(i).zfill(4)}.npz"),
                **t_obejct_to_camera,
            )

    def _dump_meta(self):
        cam = self.scene["cameras"]
        meta = {}
        intrinsic_matrix = cam.data.intrinsic_matrices
        intrinsic_matrix_np = intrinsic_matrix.cpu().numpy().squeeze()
        meta["intrinsic_matrix"] = intrinsic_matrix_np
        meta["factor_depth"] = self.factor_depth
        np.savez(os.path.join(self.output_dir, "meta.npz"), **meta)

    def _dump_object_name(self):
        objects = self.scene.rigid_objects
        output = os.path.join(self.output_dir, "object_name_list.json")
        obj_count = 0
        obj_name_list = []
        for obj in objects.keys():
            obj_name = obj.split("/")[-1]
            obj_name_list.append(obj_name)
            obj_count += 1
        assert obj_count == self.num_objects
        with open(output, "w") as f:
            json.dump(obj_name_list, f)

    def _dump_camera_pose(self):
        np.save(
            os.path.join(self.output_dir, "T_cam_to_world.npy"),
            self.t_camera_to_world_mat,
        )

    def get_observation(self):
        for cam_idx in range(self.num_views):
            self.update_camera_view(cam_idx)
            for _ in range(5):
                self.sim.step()
            self.capture_observation(cam_idx)

    def dump_data(self):
        self._dump_object_pose()
        self._dump_meta()
        self._dump_object_name()
        self._dump_camera_pose()
        print("Data dumped successfully.")

    def setup(self):
        self.sim.set_camera_view([1, 1, 1], [0.0, 0.0, 0.0])
        self.setup_scene()
        self.setup_camera_view()
        self.sim.reset()
        print("scene keys:  ", self.scene.keys())
        print("Isaac sim Setup complete...")

    def run_simulation(self):
        sim_dt = self.sim.get_physics_dt()
        count = 0
        warmup_steps = 500

        while self.simulation_app.is_running():
            if count == warmup_steps:
                self.get_observation()
                self.dump_data()

            # perform step
            self.sim.step()
            # update the scene to reflect the physics changes
            self.scene.update(sim_dt)

            count += 1
            if count > 600:
                break

        print("Isaac sim exited...")
