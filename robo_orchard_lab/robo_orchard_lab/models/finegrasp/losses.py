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

import torch
import torch.nn as nn
import torchvision


class ObjectnessBCELoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, end_points):
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        objectness_score = end_points["objectness_score"]
        objectness_label = end_points["objectness_label"]
        loss = criterion(objectness_score, objectness_label.float())
        end_points["objectness_loss"] = loss * self.loss_weight
        return end_points


class ObjectnessFocalLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, end_points):
        objectness_score = end_points["objectness_score"]
        objectness_label = end_points["objectness_label"]
        loss = torchvision.ops.sigmoid_focal_loss(
            objectness_score, objectness_label.float(), reduction="mean"
        )
        end_points["objectness_loss"] = loss * self.loss_weight
        return end_points


class ObjectnessLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, end_points):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        objectness_score = end_points["objectness_score"]
        objectness_label = end_points["objectness_label"]
        loss = criterion(objectness_score, objectness_label)
        end_points["objectness_loss"] = loss * self.loss_weight

        objectness_pred = torch.argmax(objectness_score, 1)
        end_points["stage1_objectness_acc"] = (
            (objectness_pred == objectness_label.long()).float().mean()
        )
        end_points["stage1_objectness_prec"] = (
            (objectness_pred == objectness_label.long())[objectness_pred == 1]
            .float()
            .mean()
        )
        end_points["stage1_objectness_recall"] = (
            (objectness_pred == objectness_label.long())[objectness_label == 1]
            .float()
            .mean()
        )
        return end_points


class GraspnessLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, end_points):
        criterion = nn.SmoothL1Loss(reduction="none")
        graspness_score = end_points["graspness_score"].squeeze(1)
        graspness_label = end_points["graspness_label"].squeeze(-1)
        loss_mask = end_points["objectness_label"].bool()
        loss = criterion(graspness_score, graspness_label)
        loss = loss[loss_mask]
        loss = loss.mean()

        graspness_score_c = graspness_score.detach().clone()[loss_mask]
        graspness_label_c = graspness_label.detach().clone()[loss_mask]
        graspness_score_c = torch.clamp(graspness_score_c, 0.0, 0.99)
        graspness_label_c = torch.clamp(graspness_label_c, 0.0, 0.99)
        rank_error = (
            torch.abs(
                torch.trunc(graspness_score_c * 20)
                - torch.trunc(graspness_label_c * 20)
            )
            / 20.0
        ).mean()
        end_points["stage1_graspness_acc_rank_error"] = rank_error

        end_points["graspness_loss"] = loss * self.loss_weight
        return end_points


class ViewLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, end_points):
        criterion = nn.SmoothL1Loss(reduction="mean")
        view_score = end_points["view_score"]
        view_label = end_points["batch_grasp_view_graspness"]
        loss = criterion(view_score, view_label)
        end_points["view_loss"] = loss * self.loss_weight
        return end_points


class AngleLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, end_points):
        criterion = nn.CrossEntropyLoss(reduction="none")
        grasp_angle_pred = end_points["grasp_angle_pred"]
        grasp_angle_label = end_points["batch_grasp_rotations"].long()
        valid_mask = end_points["batch_valid_mask"]
        loss = criterion(grasp_angle_pred, grasp_angle_label)
        if torch.sum(valid_mask) == 0:
            loss = 0 * torch.sum(loss)
        else:
            loss = loss[valid_mask].mean()
        end_points["angle_loss"] = loss * self.loss_weight
        end_points["D: Angle Acc"] = (
            (torch.argmax(grasp_angle_pred, 1) == grasp_angle_label)[
                valid_mask
            ]
            .float()
            .mean()
        )

        return end_points


class DepthLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, end_points):
        criterion = nn.CrossEntropyLoss(reduction="none")
        grasp_depth_pred = end_points["grasp_depth_pred"]
        grasp_depth_label = end_points["batch_grasp_depth"].long()
        valid_mask = end_points["batch_valid_mask"]
        loss = criterion(grasp_depth_pred, grasp_depth_label)
        if torch.sum(valid_mask) == 0:
            loss = 0 * torch.sum(loss)
        else:
            loss = loss[valid_mask].mean()
        end_points["depth_loss"] = loss * self.loss_weight
        end_points["D: Depth Acc"] = (
            (torch.argmax(grasp_depth_pred, 1) == grasp_depth_label)[
                valid_mask
            ]
            .float()
            .mean()
        )
        return end_points


class WidthLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, end_points):
        criterion = nn.SmoothL1Loss(reduction="none")
        grasp_width_pred = end_points["grasp_width_pred"]
        grasp_width_label = end_points["batch_grasp_width"] * 10
        loss = criterion(grasp_width_pred.squeeze(1), grasp_width_label)
        valid_mask = end_points["batch_valid_mask"]
        if torch.sum(valid_mask) == 0:
            loss = 0 * torch.sum(loss)
        else:
            loss = loss[valid_mask].mean()
        end_points["width_loss"] = loss * self.loss_weight
        return end_points


class ScoreClsLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight


    def forward(self, end_points):
        criterion = nn.CrossEntropyLoss(reduction="none")
        grasp_score_pred = end_points["grasp_score_pred"]
        grasp_score_label = (end_points["batch_grasp_score"] * 10 / 2).long()
        valid_mask = end_points["batch_valid_mask"]
        loss = criterion(grasp_score_pred.squeeze(1), grasp_score_label)

        if "grasp_confidence" in end_points:
            ######################smu#################
            # confidence = end_points["grasp_confidence"].squeeze(1)
            # confidence = torch.clamp(confidence, 1e-4, 1.0)
            # weight_loss = confidence * loss
            # confidence_reg = -0.1 * torch.log(confidence)
            # loss = weight_loss + confidence_reg
            ######################smu#################
            confidence = end_points["grasp_confidence"].squeeze(1)
            confidence = torch.clamp(confidence, 1e-4, 1.0)
            weight_ce = (1.0 + confidence) * loss
            target_confidence = torch.exp(-loss.detach())
            confidence_consistency = nn.functional.mse_loss(confidence, target_confidence , reduction="none")
            weight_ce = weight_ce[valid_mask]
            confidence_consistency = confidence_consistency[valid_mask]
            loss = weight_ce.mean() + 0.1 * confidence_consistency.mean()
        # print("confidencemean", confidence.mean().item())
        # print("confidencemin", confidence.min().item())
        # print("confidencemax", confidence.max().item())
        else:
            if torch.sum(valid_mask) == 0:
                loss = 0 * torch.sum(loss)
            else:
                loss = loss[valid_mask].mean()
        end_points["score_loss"] = loss * self.loss_weight
        end_points["D: Score Acc"] = (
            (torch.argmax(grasp_score_pred, 1) == grasp_score_label)[
                valid_mask
            ]
            .float()
            .mean()
        )
        return end_points


class ScoreRegLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, end_points):
        criterion = nn.SmoothL1Loss(reduction="mean")
        grasp_score_pred = end_points["grasp_score_pred"]
        grasp_score_label = end_points["batch_grasp_score"]
        loss = criterion(grasp_score_pred, grasp_score_label)

        end_points["score_loss"] = loss * self.loss_weight
        return end_points


class ScoreClsFocalLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=0.5, gamma=2) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, end_points):
        criterion = nn.CrossEntropyLoss(reduction="none")
        grasp_score_pred = end_points["grasp_score_pred"]
        grasp_score_label = (end_points["batch_grasp_score"] * 10 / 2).long()
        valid_mask = end_points["batch_valid_mask"]
        ce_loss = criterion(grasp_score_pred.squeeze(1), grasp_score_label)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if torch.sum(valid_mask) == 0:
            loss = 0 * torch.sum(loss)
        else:
            loss = loss[valid_mask].mean()
        end_points["score_Loss"] = loss * self.loss_weight
        end_points["D: Score Acc"] = (
            (torch.argmax(grasp_score_pred, 1) == grasp_score_label)[
                valid_mask
            ]
            .float()
            .mean()
        )
        return end_points
    
class ScaleAuxLoss(nn.Module):
    """Auxiliary scale regression loss for use_scale_modulation.
    
    强制 backbone 输出特征保留尺度信息，防止 FiLM 调制模块被模型绕过。
    仅在 use_scale_modulation=True 时生效；否则返回 0，不影响 baseline。
    """

    def __init__(self, loss_weight=0.1) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, end_points):
        # 如果没启用 scale modulation，则跳过（保持 baseline 兼容）
        if "scale_pred" not in end_points:
            # 用一个已存在的 tensor 占位，避免破坏训练日志
            ref = end_points["objectness_score"]
            end_points["scale_aux_loss"] = (
                ref.sum() * 0.0
            )
            return end_points

        pred = end_points["scale_pred"]            # [B, N]
        target = end_points["scale_target"]        # [B, N]
        mask = end_points["scale_fg_mask"]         # [B, N]

        if mask.sum() < 1:
            loss = pred.sum() * 0.0
        else:
            diff = (pred - target).abs()
            loss = (diff * mask).sum() / (mask.sum() + 1e-6)

        end_points["scale_aux_loss"] = loss * self.loss_weight
        end_points["D: Scale Aux MAE"] = loss.detach()
        return end_points
