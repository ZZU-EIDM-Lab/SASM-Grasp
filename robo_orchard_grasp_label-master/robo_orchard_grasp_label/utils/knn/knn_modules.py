# Project RoboOrchard
#
# Copyright (c) 2024–2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Non-Commercial Open Source License Version 1.0.
# To view a copy of this license, visit:
#   https://github.com/HorizonRobotics/robo_orchard_grasp_label/blob/master/LICENSE

import os
import torch


def load_knn_pytorch():
    from pathlib import Path
    from torch.utils.cpp_extension import load
    from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

    root = Path(__file__).resolve().parent
    extension = CppExtension
    extra_cflags, extra_cuda_cflags = None, None
    sources = [root / "src/vision.cpp", root / "src/cpu/knn_cpu.cpp"]

    if torch.cuda.is_available() and CUDA_HOME is not None:
        sources = [
            root / "src/vision.cpp",
            root / "src/cpu/knn_cpu.cpp",
            root / "src/cuda/knn.cu",
        ]
        extension = CUDAExtension
        extra_cflags = ["-DWITH_CUDA"]
        extra_cuda_cflags = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    func = load(
        f"knn_pytorch",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
    )
    return func


try:
    import knn_pytorch
except ImportError:
    knn_pytorch = load_knn_pytorch()


def knn(ref, query, k=1):
    """Compute k nearest neighbors for each query point."""
    device = ref.device
    ref = ref.float().to(device)
    query = query.float().to(device)
    inds = torch.empty(query.shape[0], k, query.shape[2]).long().to(device)
    knn_pytorch.knn(ref, query, inds)
    return inds
