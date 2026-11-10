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

import argparse
import json
import logging
import os
import sys

import numpy as np
from graspnetAPI import GraspNetEval
# # 获取当前文件所在目录，然后向上找到项目根目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 假设项目根目录是 /home/zhangsai/Downloads/Finegrasp
# project_root = os.path.abspath(os.path.join(current_dir, "../../"))  # 根据实际结构调整
# sys.path.append(project_root)
# sys.path.append(os.path.join(project_root, "robo_orchard_lab"))
from robo_orchard_lab.dataset.graspnet1b.metrics import (
    GraspNetEvalScale,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from train import load_config  # noqa: E402

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_finegrasp_minkunet.py",
    )
    parser.add_argument("--data_root", type=str, default="/home/zhangsai/Downloads/EconomicGrasp/dataset/6dofdata")
    parser.add_argument("--workspace_root", type=str, default="/home/zhangsai/Downloads/Finegrasp/robo_orchard_lab/projects/finegrasp_graspnet1b/workspace0005sllmone-film2kinect-17")
    parser.add_argument("--kwargs", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    get_trainer_cfg = config.get_trainer_cfg
    trainer_cfg = get_trainer_cfg()

    # load config from param
    trainer_cfg = trainer_cfg.model_copy(update=vars(args))
    if args.kwargs is not None:
        kwargs = json.loads(args.kwargs)
        trainer_cfg = trainer_cfg.model_copy(update=kwargs)

    build_dataset = config.build_dataset
    build_model = config.build_model
    build_metric = config.build_metric
    (
        train_data_cfgs,
        val_data_cfgs,
        train_dataloader,
        val_dataloader,
    ) = build_dataset(trainer_cfg)
    model_cfgs, model = build_model()

    results = {}

    eval_save_dir = os.path.join(
        args.workspace_root,
        "eval_results",
        model_cfgs.model_name,
        train_data_cfgs.camera,
    )
    metric_cfgs, metric = build_metric(model_cfgs, trainer_cfg, eval_save_dir)
    test_mode_list = trainer_cfg.test_mode

    for test_mode in test_mode_list:
        if test_mode == "test":
            logger.info("=" * 50 + f"BEGIN EVAL {test_mode}" + "=" * 50)
            ge = GraspNetEval(
                root=trainer_cfg.data_root,
                camera=metric_cfgs.camera,
                split=test_mode,
            )
            res, ap = ge.eval_novel(
                dump_folder=eval_save_dir, proc=metric_cfgs.num_test_proc
            )
            results[test_mode] = res
            np.save(
                os.path.join(
                    eval_save_dir,
                    f"ap_{metric_cfgs.camera}_{test_mode}_{ap}.npy",
                ),
                res,
            )
            print(f"seen testing, AP 0.8={np.mean(res[:, :, :, 3])}, AP 0.4={np.mean(res[:, :, :, 1])}")

        elif test_mode == "test_scale":
            # scales = ["medium","large"]
            scales = ["small","medium","large"]
            # scales = ["large"]
            ge_scale = GraspNetEvalScale(
                root=metric_cfgs.data_root,
                camera=metric_cfgs.camera,
                split="test",
            )
            for scale in scales:
                logger.info(
                    "=" * 50
                    + f"BEGIN EVAL {test_mode} SCALE {scale}"
                    + "=" * 50
                )
                res_scale, ap_scale = ge_scale.eval_novel3(
                    dump_folder=eval_save_dir,
                    scale=scale,
                    proc=metric_cfgs.num_test_proc,
                )
                results[f"{test_mode}_{scale}"] = res_scale
                np.save(
                    os.path.join(
                        eval_save_dir,
                        f"ap_{metric_cfgs.camera}_{test_mode}_{scale}_{ap_scale}.npy",
                    ),
                    res_scale,
                )
