# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import numpy as np
from scipy.spatial.transform import Rotation as R


class CameraPoseManager:
    def __init__(self, t_cam_to_table_path, num_views=256):
        t_cam_to_table = np.load(t_cam_to_table_path)
        self.cam0_wrt_table = t_cam_to_table["T_cam0_to_table"]
        self.camera_poses_cam0 = t_cam_to_table["T_cam_to_cam0"]
        self.num_views = num_views

    @staticmethod
    def trans_matrix_to_pose(matrix):
        """Convert a 4x4 transformation matrix to a 7D pose vector.

        Args:
            matrix (np.ndarray): transformation matrices of shape (N, 4, 4).

        Returns:
            poses (np.ndarray):
                pose vector of shape (N, 7) in [x, y, z, qx, qy, qz, qw] order.
        """
        if matrix.ndim == 2:
            matrix = matrix.reshape(1, 4, 4)
        N = matrix.shape[0]
        poses = np.zeros((N, 7))

        for i in range(N):
            # Extract the translation part
            translation = matrix[i, :3, 3]

            # Extract the rotation part and convert to quaternion
            rotation_matrix = matrix[i, :3, :3]
            quaternion = R.from_matrix(rotation_matrix).as_quat()

            # Store in the result
            poses[i, :3] = translation
            poses[i, 3:] = quaternion

        return poses

    @staticmethod
    def trans_pose_to_matrix(pose_vector, scalar_first):
        """Convert a 7D pose vector to a 4x4 transformation matrix.

        Args:
            pose_vector (np.ndarray): A (N, 7) shaped array.
            scalar_first (bool): If True, the quaternion is [qw, qx, qy, qy].

        Returns:
            np.ndarray: A (N, 4, 4) shaped array of transformation matrices.
        """
        if pose_vector.ndim == 1:
            pose_vector = pose_vector.reshape(1, -1)
        N = pose_vector.shape[0]
        transform_matrices = np.zeros((N, 4, 4))

        for i in range(N):
            # Extract translation and quaternion parts
            translation = pose_vector[i, :3]
            quaternion = pose_vector[i, 3:]

            # Create a 3x3 rotation matrix from the quaternion
            rotation = R.from_quat(
                quaternion, scalar_first=scalar_first
            ).as_matrix()

            # Construct the 4x4 transformation matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation
            transform_matrix[:3, 3] = translation

            transform_matrices[i] = transform_matrix

        return transform_matrices.squeeze()

    def get_graspnet_cam_pose(self):
        """Get the camera poses in the GraspNet table coordinate system."""
        poses_wrt_table = []
        for ann_id in range(self.num_views):
            camera_pose_cam = self.camera_poses_cam0[ann_id]
            trans = np.dot(self.cam0_wrt_table, camera_pose_cam)
            poses_wrt_table.append(trans)
        return np.array(poses_wrt_table)

    @staticmethod
    def get_isaac_cam_pose(matrices):
        """Adjust the camera poses to match the IsaacSim coordinate system."""
        first_pose = matrices[0]
        rotation_center = first_pose[:3, 3].copy()

        theta = np.radians(-90)
        R_z = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        for i in range(len(matrices)):
            matrices[i][:3, 3] -= rotation_center
            matrices[i] = np.matmul(R_z, matrices[i])
            matrices[i][:3, 3] += rotation_center

        # add an x-axis offset to the view wrt the graspnet table coord
        matrices[:, 0, 3] += 0.5
        # add an y-axis offset to the view wrt the graspnet table coord
        matrices[:, 1, 3] += 0.05
        return matrices
