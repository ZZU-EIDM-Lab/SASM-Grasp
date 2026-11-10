# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE
import os

import numpy as np


def load_mesh_as_pointcloud(path, voxel_size, max_points=None) -> np.ndarray:
    """Load mesh model and downsample it.

    Args:
        path(str): path to model.
        voxel_size(float): voxel size for downsampling.
        max_points(int): max points for model.

    Return:
        cloud_np(numpy.ndarray): point cloud of model.
    """
    import open3d as o3d

    try:
        cloud = o3d.io.read_point_cloud(path)
        if not cloud.has_points():
            raise ValueError("Point cloud has no points.")
    except Exception:
        mesh = o3d.io.read_triangle_mesh(path)
        cloud = o3d.geometry.PointCloud()
        cloud.points = mesh.vertices
    cloud = cloud.voxel_down_sample(voxel_size)
    cloud_np = np.asarray(cloud.points).astype(np.float32)
    if max_points is not None and len(cloud_np) > max_points:
        cloud_np = cloud_np[
            np.random.choice(len(cloud_np), max_points, replace=False)
        ]
    return cloud_np


def load_dexnet_model(data_path):
    """Load dexnet model from data_path.

    Args:
        data_path (str): path to load .obj & .sdf files
    Returns:
        obj (GraspableObject3D): dexnet model
    """
    from robo_orchard_grasp_label.utils.dexnet.grasping.graspable_object import (  # noqa: E501
        GraspableObject3D,
    )
    from robo_orchard_grasp_label.utils.dexnet.grasping.meshpy.obj_file import (  # noqa: E501
        ObjFile,
    )
    from robo_orchard_grasp_label.utils.dexnet.grasping.meshpy.sdf_file import (  # noqa: E501
        SdfFile,
    )

    assert os.path.exists("{}.obj".format(data_path)) and os.path.exists(
        "{}.sdf".format(data_path)
    ), "Dexnet model files not found at {}".format(data_path)
    of = ObjFile("{}.obj".format(data_path))
    sf = SdfFile("{}.sdf".format(data_path))
    mesh = of.read()
    sdf = sf.read()
    obj = GraspableObject3D(sdf, mesh)
    return obj


def analyze_width(points_in_gripper, hole_size=0.018, loose_factor=0.004):
    """Estimate the gripper width based on points located inside the gripper.

    Args:
        points_in_gripper (np.ndarray):
            1D array of points along the grasp axis.
        hole_size (float):
            Minimum space in the center to be considered a "hole".
        loose_factor (float):
            Additional looseness margin added to width.

    Returns:
        float: Required gripper width.
    """
    points_in_gripper = points_in_gripper.astype(np.float32)

    left_points = -np.sort(
        -points_in_gripper[points_in_gripper <= 0]
    )  # from 0 to -∞
    right_points = np.sort(
        points_in_gripper[points_in_gripper >= 0]
    )  # from 0 to +∞

    # If only one side has points, fall back to symmetric width estimate
    if len(left_points) == 0 or len(right_points) == 0:
        max_half = np.max(np.abs(points_in_gripper))
        return float(2 * max_half + loose_factor)

    # Max compressed width (outermost contact points on both sides)
    max_half = max(left_points[-1], right_points[-1])
    max_compress_width = 2 * max_half + loose_factor

    # Find all valid "gaps" between adjacent points that are ≥ hole_size
    def find_gaps(arr):
        next_arr = np.append(arr[1:], max_half)
        gap = next_arr - arr
        return arr[gap >= hole_size], next_arr[gap >= hole_size]

    left_starts, left_ends = find_gaps(left_points)
    right_starts, right_ends = find_gaps(right_points)

    if len(left_starts) == 0 or len(right_starts) == 0:
        return float(max_compress_width)

    # Check all start/end pair intersections between left and right gaps
    hole_start = np.maximum(left_starts[:, None], right_starts[None, :])
    hole_end = np.minimum(left_ends[:, None], right_ends[None, :])
    hole_widths = hole_end - hole_start

    valid = hole_widths >= hole_size
    if np.any(valid):
        i, j = np.argwhere(valid)[0]
        hole_center = hole_start[i, j]
        return float(2 * hole_center + loose_factor)

    return float(max_compress_width)


def transform_point_cloud(cloud, transform, format="4x4"):
    """Applies a transformation matrix to a 3D point cloud.

    Args:
        cloud: [np.ndarray, (N,3), np.float32]
            point cloud to be transformed
        transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
            transformation matrix
        format: [str]
            transformation format, can be "3x3", "4x4" or "3x4"

    Returns:
        np.ndarray: Transformed point cloud of shape (N, 3), dtype float32.

    Raises:
        ValueError: If `format` is not one of the supported formats.
    """
    if format not in {"3x3", "3x4", "4x4"}:
        raise ValueError(
            "Unsupported transformation format. \
                Must be '3x3', '3x4', or '4x4'."
        )

    if format == "3x3":
        return (transform @ cloud.T).T.astype(np.float32)

    # For 3x4 or 4x4, use homogeneous coordinates
    ones = np.ones((cloud.shape[0], 1), dtype=np.float32)
    cloud_homogeneous = np.hstack((cloud, ones))
    transformed = (transform @ cloud_homogeneous.T).T
    return transformed[:, :3].astype(np.float32)


def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """Computes a workspace mask based on a segmented point cloud.

    Args:
        cloud (np.ndarray):
            Scene point cloud of shape (H, W, 3), dtype float32 if organized;
            (H*W, 3) if unorganized.
        seg (np.ndarray):
            Segmentation mask of shape (H, W) or (H*W,), dtype uint8.
        trans (np.ndarray, optional):
            Transformation matrix of shape (4, 4), dtype float32.
        organized (bool, optional):
            Whether the point cloud is organized. Defaults to True.
        outlier (float, optional):
            Outlier margin to expand the workspace bounding box.
            Defaults to 0.0.

    Returns:
        np.ndarray: Workspace mask of shape (H, W) if organized,
            else (H*W,), dtype bool.
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape(-1, 3)
        seg = seg.reshape(-1)

    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)

    # Use foreground (non-zero seg) points to define workspace bounds
    foreground = cloud[seg > 0]
    if foreground.size == 0:
        # Handle edge case: no foreground points
        workspace_mask = np.zeros(cloud.shape[0], dtype=bool)
    else:
        bounds_min = foreground.min(axis=0) - outlier
        bounds_max = foreground.max(axis=0) + outlier

        workspace_mask = np.all(
            (cloud > bounds_min) & (cloud < bounds_max), axis=1
        )

    if organized:
        workspace_mask = workspace_mask.reshape(h, w)

    return workspace_mask


def get_grasp_score(grasp, obj, fc_list, force_closure_quality_config):
    """Evaluate grasp score (force closure) across friction coefficients.

    Args:
        grasp: A Grasp3D object (from Dex-Net or similar)
        obj: A GraspableObject3D object (Dex-Net compatible)
        fc_list: List[float], friction coefficients to test
            (e.g., [0.2, 0.3, 0.4, ...])
        force_closure_quality_config: Dict[float, GraspQualityConfig],
            Mapping from friction coefficient to quality config.

    Returns:
        float: The maximum friction coefficient (rounded).
               Returns -1 if no force closure is ever achieved.
    """
    from robo_orchard_grasp_label.utils.dexnet.grasping.quality import (
        PointGraspMetrics3D,
    )

    prev_closure = False
    for i, mu in enumerate(fc_list):
        mu = round(mu, 2)

        if mu not in force_closure_quality_config:
            raise ValueError(
                f"Missing quality config for friction coefficient {mu}"
            )

        is_closure = PointGraspMetrics3D.grasp_quality(
            grasp, obj, force_closure_quality_config[mu]
        )

        # First entry and not a force closure → fail
        if i == 0 and not is_closure:
            return -1.0

        # Force closure breaks after previously satisfied,
        # use previous coefficient
        if prev_closure and not is_closure:
            return round(fc_list[i - 1], 2)

        # Last entry and still satisfies → return this
        if is_closure and mu == round(fc_list[-1], 2):
            return mu

        prev_closure = is_closure

    return -1.0


def batch_viewpoint_params_to_matrix(
    batch_towards: np.ndarray, batch_angle: np.ndarray
) -> np.ndarray:
    """Convert towards vectors and in-plane angles to rotation matrices.

    Args:
        batch_towards (np.ndarray):
            (N, 3) unit direction vectors ("approach" vectors).
        batch_angle (np.ndarray): (N,) in-plane rotation angles (radians).

    Returns:
        np.ndarray: (N, 3, 3) rotation matrices.
    """
    batch_towards = np.atleast_2d(batch_towards).astype(np.float32)
    batch_angle = np.atleast_1d(batch_angle).astype(np.float32)

    N = batch_towards.shape[0]
    axis_x = batch_towards

    # Construct temporary Y axis (orthogonal in XY plane)
    axis_y = np.stack([-axis_x[:, 1], axis_x[:, 0], np.zeros(N)], axis=1)

    # Handle degenerate case (e.g. when towards is aligned with z-axis)
    degenerate_mask = np.linalg.norm(axis_y, axis=1) < 1e-6
    axis_y[degenerate_mask] = np.array([0, 1, 0], dtype=np.float32)

    axis_x = axis_x / np.linalg.norm(axis_x, axis=1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)

    # Local basis: [x, y, z] as columns
    R2 = np.stack([axis_x, axis_y, axis_z], axis=-1)  # shape: (N, 3, 3)

    # R1: rotation about X-axis (in-plane rotation)
    cos = np.cos(batch_angle)
    sin = np.sin(batch_angle)
    ones = np.ones(N, dtype=np.float32)
    zeros = np.zeros(N, dtype=np.float32)

    R1 = np.stack(
        [
            np.stack([ones, zeros, zeros], axis=1),
            np.stack([zeros, cos, -sin], axis=1),
            np.stack([zeros, sin, cos], axis=1),
        ],
        axis=1,
    )  # shape: (N, 3, 3)

    # Final rotation: R = R2 @ R1
    R = np.matmul(R2, R1)
    return R.astype(np.float32).squeeze()


def generate_views(
    N: int,
    phi: float = (np.sqrt(5) - 1) / 2,
    center: np.ndarray = None,
    R: float = 1.0,
) -> np.ndarray:
    """Generates viewpoints on a sphere using Fibonacci lattice.

    Args:
        N (int): Number of viewpoints to generate.
        phi (float): Golden ratio angle increment (default ≈ 0.618).
        center (np.ndarray): Center of the sphere, shape (3,).
        R (float): Radius of the sphere.

    Returns:
        np.ndarray: Viewpoints on the sphere, shape (N, 3), dtype float32.
    """
    if center is None:
        center = np.zeros(3, dtype=np.float32)
    idxs = np.arange(N, dtype=np.float32)
    z = (2 * idxs + 1) / N - 1  # Uniform sampling on Z-axis
    radius = np.sqrt(1.0 - z**2)
    theta = 2 * np.pi * idxs * phi

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    views = np.stack([x, y, z], axis=1)
    views = (views * R + center).astype(np.float32)
    return views


def matrix_to_dexnet_params(matrix: np.ndarray):
    """Converts a rotation matrix to Dex-Net binormal and angle parameters.

    Args:
        matrix (np.ndarray): Rotation matrix of shape (3, 3).

    Returns:
        Tuple[np.ndarray, float]:
            - binormal (np.ndarray):
                Vector of shape (3,), Dex-Net binormal axis.
            - angle (float): Grasp angle in radians.
    """
    approach = matrix[:, 0]  # First column: approach direction
    binormal = matrix[:, 1]  # Second column: binormal direction

    axis_y = binormal / np.linalg.norm(binormal)
    axis_x = np.array([axis_y[1], -axis_y[0], 0.0])

    # Handle degenerate case where axis_x is zero
    #  (e.g., binormal is along Z-axis)
    if np.allclose(axis_x, 0):
        axis_x = np.array([1.0, 0.0, 0.0])
    else:
        axis_x /= np.linalg.norm(axis_x)

    axis_z = np.cross(axis_x, axis_y)

    # Reconstruct orthonormal frame
    R = np.column_stack((axis_x, axis_y, axis_z))

    # Project approach into the new frame
    approach_local = R.T @ approach
    cos_theta = approach_local[0]
    sin_theta = -approach_local[2]

    angle = np.arccos(
        np.clip(cos_theta, -1.0, 1.0)
    )  # Clip for numerical safety
    if sin_theta < 0:
        angle = 2 * np.pi - angle

    return binormal.astype(np.float32), float(angle)


def create_table_points(
    lx: float,
    ly: float,
    lz: float,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    grid_size: float = 0.01,
) -> np.ndarray:
    """Creates a regular grid of 3D points within a cuboid.

    Args:
        lx (float): Length of the cuboid along the x-axis.
        ly (float): Length of the cuboid along the y-axis.
        lz (float): Length of the cuboid along the z-axis.
        dx (float, optional): Offset along x-axis. Default is 0.0.
        dy (float, optional): Offset along y-axis. Default is 0.0.
        dz (float, optional): Offset along z-axis. Default is 0.0.
        grid_size (float, optional):
            Distance between adjacent points in the grid.

    Returns:
        np.ndarray: Array of 3D points with shape (N, 3), dtype float64.
    """
    x = np.linspace(0, lx, int(np.ceil(lx / grid_size)))
    y = np.linspace(0, ly, int(np.ceil(ly / grid_size)))
    z = np.linspace(0, lz, int(np.ceil(lz / grid_size)))

    # Create meshgrid in XY-plane with Z extrusion
    xx, yy, zz = np.meshgrid(x, y, z, indexing="xy")

    # Apply translation offsets
    xx += dx
    yy += dy
    zz += dz

    # Stack and reshape to (N, 3)
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return points.astype(np.float32)


def create_point_cloud_from_depth_image(
    depth: np.ndarray, camera, organized: bool = True
) -> np.ndarray:
    """Creates a 3D point cloud from a depth image using camera intrinsics.

    Args:
        depth (np.ndarray): Depth image of shape (H, W),
            dtype float32 or uint16.
        camera: An object with attributes:
            width, height, fx, fy, cx, cy, scale.
        organized (bool, optional):
            Whether to keep the point cloud in (H, W, 3) format.

    Returns:
        np.ndarray: Point cloud of shape (H, W, 3) if organized else (H*W, 3).
    """
    assert depth.shape == (
        camera.height,
        camera.width,
    ), "Depth shape does not match camera resolution."

    # Generate image coordinate grid
    xmap, ymap = np.meshgrid(np.arange(camera.width), np.arange(camera.height))

    # Convert depth to meters if scale is applied
    z = depth.astype(np.float32) / camera.scale
    x = (xmap - camera.cx) * z / camera.fx
    y = (ymap - camera.cy) * z / camera.fy

    cloud = np.stack((x, y, z), axis=-1)  # Shape: (H, W, 3)
    if not organized:
        cloud = cloud.reshape(-1, 3)

    return cloud.astype(np.float32)
