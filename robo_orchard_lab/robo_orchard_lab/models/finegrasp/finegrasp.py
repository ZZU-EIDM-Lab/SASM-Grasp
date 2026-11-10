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
import math
import MinkowskiEngine as ME  # noqa: N817
import numpy as np
import torch
import torch.nn as nn
from pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
import torch.nn.functional as F    
from robo_orchard_lab.models.finegrasp.backbone import MinkUNet
from robo_orchard_lab.models.finegrasp.head import (
    ApproachNet,
    CylinderGroup,
    GraspableNet,
    LocalInteraction,
)
from robo_orchard_lab.models.finegrasp.utils import process_grasp_labels
from robo_orchard_lab.models.mixin import (
    ClassType_co,
    ModelMixin,
    TorchModuleCfg,
    TorchModuleCfgType_co,
)
from robo_orchard_lab.utils.build import (
    DelayInitDictType,
)

######################add  ScalePriorMLP######################
class ScalePriorMLP(nn.Module):
    def __init__(self,  hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(1,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)
    def forward(self, scale):
        x = 1.0/scale.clamp(min=0.02)
        x = F.gelu(self.fc1(x.unsqueeze(-1)))
        return F.softplus(self.fc2(x)).squeeze(-1)
######################add  ScalePriorMLP######################
class GroupTransformerFusion(nn.Module):
    def __init__(self, feature_dim, num_groups):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=8, dim_feedforward=feature_dim * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x: [B, G, D, N]
        B, G, D, N = x.shape  # noqa: N806

        x = x.permute(1, 3, 0, 2)  # [G, N, B, D]
        x = x.reshape(G, N * B, D)  # [G, N*B, D]

        out = self.transformer(x)  # [G, N*B, D]

        attn_weights = torch.softmax(self.attn(out), dim=0)  # [G, N*B, 1]
        out = (out * attn_weights).sum(dim=0)  # [N*B, D]
        out = out.reshape(N, B, D).permute(1, 2, 0)  # [B, D, N]

        return out

# ============================================================
# Scale Modulation 子模块（仅在 use_scale_modulation=True 时使用）
# ============================================================
class ScaleSinusoidalEncoder(nn.Module):
    """NeRF-style 频域编码，比裸 MLP 对小尺度变化更敏感。"""

    def __init__(self, num_freqs: int = 6, out_dim: int = 128):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs) * math.pi
        self.register_buffer("freqs", freqs)
        in_dim = 2 * num_freqs + 1
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        # x: [B, N, 1] in [0, 1]
        s = x * self.freqs.view(1, 1, -1)
        enc = torch.cat([x, torch.sin(s), torch.cos(s)], dim=-1)
        return self.proj(enc)
    
class ScaleFiLM(nn.Module):
    """Size-emphasized Single-Head FiLM.
    
    设计原则：
    1. 单头避免 Multi-Head 的泛化问题（AP_similar 下降）
    2. Size-conditioned amplification：small 物体调制幅度 ×3，
       large 物体调制幅度 ×0（identity），medium ×1
    3. 零初始化最终层，step-0 等价 baseline
    """

    def __init__(self, feat_dim: int, embed_dim: int = 128,
                 small_boost: float = 3.0):
        super().__init__()
        self.encoder = ScaleSinusoidalEncoder(num_freqs=6, out_dim=embed_dim)
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.gamma_head = nn.Linear(embed_dim, feat_dim)
        self.beta_head = nn.Linear(embed_dim, feat_dim)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
        self.small_boost = small_boost  # ← 关键：small 物体调制幅度放大

    @staticmethod
    def _size_amplifier(size_m, small_boost):
        """
        Size-dependent amplification:
          size ≤ 0.04m  → amp = small_boost (e.g., 3.0)  小物体强调制
          size ≥ 0.10m  → amp = 0.0                       大物体零调制
          中间          → 线性插值
        """
        amp = (0.10 - size_m) / (0.10 - 0.04)
        amp = torch.clamp(amp, 0.0, 1.0)

        small_region = (amp >= 0.8).float()
        amp = amp * (1.0 - small_region) + amp * small_boost * small_region
        return amp

    def forward(self, feat, scale_norm, fg_mask, point_scale_m):
        """
        feat:          [B, C, N]
        scale_norm:    [B, N]
        fg_mask:       [B, N]
        point_scale_m: [B, N]
        """
        x = scale_norm.unsqueeze(-1)
        emb = self.encoder(x)
        emb = self.trunk(emb)
        gamma = torch.tanh(self.gamma_head(emb)).permute(0, 2, 1)
        beta = self.beta_head(emb).permute(0, 2, 1)

        # Size-dependent amplification
        amp = self._size_amplifier(point_scale_m, self.small_boost)
        amp = (amp * fg_mask).unsqueeze(1)  # [B, 1, N]

        gamma = gamma * amp
        beta = beta * amp

        return feat * (1.0 + gamma) + beta
    

class ScaleAuxHead(nn.Module):
    """辅助 scale 回归头，强迫特征中保留尺度信息（防止调制被绕过）。"""

    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(feat_dim, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        return self.head(feat).squeeze(1)  # [B, N]

# ============================================================
# 主模型
# ============================================================
class FineGrasp(ModelMixin):  # noqa: N801
    cfg: "FineGraspConfig"  # for type hint

    def __init__(self, cfg: "FineGraspConfig"):
        super().__init__(cfg)
        self.cfg = cfg
        self.seed_feature_dim = cfg.seed_feat_dim
        self.num_depth = cfg.num_depth
        self.num_angle = cfg.num_angle
        self.num_seed_points = cfg.num_seed_points
        self.num_view = cfg.num_view
        self.voxel_size = cfg.voxel_size
        self.cylinder_radius = cfg.cylinder_radius
        self.cylinder_groups = cfg.cylinder_groups
        self.use_normal = cfg.use_normal
        self.loss = cfg.loss
        ############################add########################
        if cfg.use_scale_sampling:
            self.scale_prior_mlp = ScalePriorMLP(hidden_dim=32)
        ############################add########################

        if self.use_normal is True:
            in_channels = 6  # xyz + normal
        else:
            in_channels = 3

        # Backbone
        self.backbone = MinkUNet(
            in_channels=in_channels, out_channels=self.seed_feature_dim, D=3
        )

        # Objectness and graspness
        self.graspable = GraspableNet(seed_feature_dim=self.seed_feature_dim)

        # View Selection
        self.view = ApproachNet(
            self.num_view,
            seed_feature_dim=self.seed_feature_dim,
        )

        # Cylinder
        self.cy_groups = torch.nn.ModuleList(
            [
                CylinderGroup(
                    nsample=16,
                    cylinder_radius=self.cylinder_radius * cylinder_group,
                    seed_feature_dim=self.seed_feature_dim,
                )
                for cylinder_group in self.cylinder_groups
            ]
        )

        self.fuse_multi_scale = GroupTransformerFusion(
            256, len(cfg.cylinder_groups)
        )

        # Depth and Score searching
        self.grasp_head = LocalInteraction(
            num_angle=self.num_angle, num_depth=self.num_depth
        )

        self.graspness_threshold = cfg.graspness_threshold
        if self.cfg.use_scale_modulation:
            self.scale_film_seed = ScaleFiLM(
        feat_dim=self.seed_feature_dim, embed_dim=128,
        small_boost=2.0,   # Stage-1 用 2.0
    )
            self.scale_film_group = ScaleFiLM(
        feat_dim=256, embed_dim=128,
        small_boost=3.0,   # Stage-2 用 3.0（针对 Bug 2 强化）
    )
            self.scale_aux_head = ScaleAuxHead(feat_dim=self.seed_feature_dim)

        if self.cfg.use_uncertainty:
            self.confidence_head = nn.Sequential(nn.Conv1d(256, 64, 1), nn.ReLU(inplace=True), nn.Conv1d(64,1,1), nn.Sigmoid())
        if self.cfg.use_semantic:
            self.semantic_embed = nn.Embedding(10, 16)
            self.semantic_mlp = nn.Sequential(nn.Linear(16, self.num_angle * self.num_depth),nn.ReLU())
    def _scale_guided_sampling(self, end_points):

        if not self.cfg.use_scale_sampling:
            return end_points

        device = next(self.parameters()).device
        num_sample_points = 30000

        point_keys = [
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
        ]

        point_clouds = end_points["point_clouds"]

        if isinstance(point_clouds, torch.Tensor):
            batch_size = point_clouds.shape[0]
        else:
            batch_size = len(point_clouds)

        sampled = {key: [] for key in point_keys}

        for b in range(batch_size):

            # 当前场景的完整候选点
            xyz = torch.as_tensor(
                point_clouds[b], dtype=torch.float32, device=device
            )

            scale = torch.as_tensor(
                end_points["point_scale"][b],
                dtype=torch.float32,
                device=device,
            )

            semantic_weight = torch.as_tensor(
                end_points["point_semantic_weight"][b],
                dtype=torch.float32,
                device=device,
            )

            N = xyz.shape[0]

            if N == 0:
                raise ValueError("Empty point cloud encountered.")

            # 1. 学习尺度先验
            rho_scale = self.scale_prior_mlp(scale)

            # 2. 融合尺度与语义先验
            rho = rho_scale * semantic_weight

            # 背景权重保持 0.3
            rho = torch.where(
                scale > 0,
                rho,
                torch.full_like(rho, 0.3),
            )

            # 3. 归一化采样概率
            prob = rho.clamp_min(1e-8)
            prob = prob / prob.sum()

            # 4. Gumbel-TopK：无放回采样
            if N >= num_sample_points:

                u = torch.rand_like(prob).clamp(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(u))

                logits = torch.log(prob.clamp_min(1e-8))
                noisy_logits = logits + gumbel

                idx = torch.topk(
                    noisy_logits,
                    k=num_sample_points,
                    sorted=False,
                ).indices

                soft_prob = torch.softmax(noisy_logits, dim=0)

            # 5. 候选点不足：全部保留，再有放回补足
            else:

                extra_idx = torch.multinomial(
                    prob,
                    num_sample_points - N,
                    replacement=True,
                )

                idx = torch.cat([
                    torch.arange(N, device=device),
                    extra_idx,
                ])

                soft_prob = prob

            # 6. 构造 ST 尺度属性，保留近似梯度
            hard_scale = scale[idx]

            soft_scale = torch.sum(soft_prob * scale)

            scale_st = hard_scale + (
                soft_scale - soft_scale.detach()
            )

            # 7. 同步采样所有点级属性
            for key in point_keys:

                if key not in end_points:
                    continue

                if key == "point_scale":
                    selected = scale_st

                elif key == "point_clouds":
                    selected = xyz[idx]

                elif key == "coordinates_for_voxel":
                    coords = torch.as_tensor(
                        end_points[key][b], dtype=torch.float32
                    )
                    selected = coords[idx.cpu()].cpu()

                else:
                    value = torch.as_tensor(
                        end_points[key][b], device=device
                    )
                    selected = value[idx]

                sampled[key].append(selected)

        # 8. 恢复 FineGrasp 原有输入格式
        for key in point_keys:

            if not sampled[key]:
                continue

            if key == "coordinates_for_voxel":
                end_points[key] = sampled[key]
            else:
                end_points[key] = torch.stack(
                    sampled[key], dim=0
                )

        return end_points

    def forward(self, end_points):

        # ===== Scale-aware differentiable sampling =====
        end_points = self._scale_guided_sampling(end_points)

        is_training = self.training

        seed_xyz = end_points[
            "point_clouds"
        ]


        if self.use_normal is True:
            seed_normal = end_points[
                "cloud_normal"
            ]  # use all sampled point cloud, [B, point_num (20000)， 3]
            seed_point_input = torch.cat([seed_xyz, seed_normal], dim=-1)
            seed_init_feature = seed_point_input.cpu()
        else:
            seed_point_input = seed_xyz
            seed_init_feature = np.ones_like(seed_point_input.cpu()).astype(
                np.float32
            )

        B, point_num, _ = seed_point_input.shape  # noqa: N806

        # Generate input to meet the Minkowski Engine
        coordinates_batch, features_batch = ME.utils.sparse_collate(
            [coord for coord in end_points["coordinates_for_voxel"]],
            [feat for feat in seed_init_feature],
        )

        (
            coordinates_batch,
            features_batch,
            _,
            end_points["quantize2original"],
        ) = ME.utils.sparse_quantize(
            coordinates_batch,
            features_batch,
            return_index=True,
            return_inverse=True,
        )

        coordinates_batch = coordinates_batch.cuda()
        features_batch = features_batch.cuda()

        # [points of the whole scenes after quantize, 3(coors) + 1(index)]
        end_points["coors"] = coordinates_batch

        # [points of the whole scenes after quantize, 3 (input feature dim)]
        end_points["feats"] = features_batch
        mink_input = ME.SparseTensor(
            features_batch, coordinates=coordinates_batch
        )

        # Minkowski Backbone
        seed_features = self.backbone(mink_input).F
        seed_features = (
            seed_features[end_points["quantize2original"]]
            .view(B, point_num, -1)
            .transpose(1, 2)
        )
        # [B (batch size), 512 (feature dim), 20000 (points in a scene)]

        ##########################add##################
        # ===== Scale-aware modulation =====
        if self.cfg.use_scale_modulation:
            # print("5555555555555555599999999999999999999")
            point_scale = end_points["point_scale"].to(
                seed_features.device
                )  # [B, N]
            fg_mask = (point_scale > 0).float()  # [B, 1, N]
                        # 反尺度：小物体得到更大值
            safe_scale = point_scale.clamp(min=0.02,max=0.30)
            inv_scale = 1.0 / safe_scale  # 最大≈50（对应 0.02m）
            # 固定归一化到 [0,1]，跨 batch 一致；50 对应数据集最小尺度
            scale_norm = (inv_scale / 50.0).clamp(0.0, 1.0)
            scale_norm = scale_norm * fg_mask  # 背景置 0

            # 第一阶段调制
            seed_features = self.scale_film_seed(
                seed_features, scale_norm, fg_mask, point_scale
            )

            # 辅助 scale 监督（仅训练）
            if is_training:
                scale_pred = self.scale_aux_head(seed_features)
                end_points["scale_pred"] = scale_pred
                end_points["scale_target"] = scale_norm.detach()
                end_points["scale_fg_mask"] = fg_mask

            # 暂存供第二阶段（FPS 后）使用
            end_points["_scale_norm_full"] = scale_norm.detach()
            end_points["_fg_mask_full"] = fg_mask
            end_points["_scale_meter_full"] = point_scale
        # ===== End of Stage-1 =====
            
            ###################sm###############
            # 取对数使尺度变化更平滑（可选）
            # scale_input = torch.log(point_scale + 1e-6).unsqueeze(-1)  # [B, N, 1]   ###sm结果
            # scale_feat = self.scale_mlp(scale_input)  # [B, N, C]
            # scale_feat = scale_feat.permute(0, 2, 1)  # [B, C, N]
            # scale_feat = scale_feat * fg_mask          # 背景点特征置 0 → 调制因子 = 1
            # seed_features = seed_features * (1 + scale_feat)    ###sm结果
            ###################sm###############

            ###################sm2###############
            # scale_input = 1.0 / (point_scale + 1e-6)# 可选：限制范围，防止爆炸<br/>
            # scale_input = torch.clamp(scale_input, max=50.0)
            # scale_input = torch.sqrt(scale_input)
            # scale_input = scale_input.unsqueeze(-1)
            # scale_feat = self.scale_mlp(scale_input)  # [B, N, C]
            # scale_feat = scale_feat.permute(0, 2, 1)  # [B, C, N]
            # scale_feat = scale_feat * fg_mask          # 背景点特征置 0 → 调制因子 = 1
            # seed_features = seed_features * (1 + 0.6*scale_feat)
            ###################sm2###############

            ###################sm3###############
            # scale_input = 1.0 / (point_scale + 1e-6)# 可选：限制范围，防止爆炸<br/>
            # scale_input = torch.clamp(scale_input, max=50.0)
            # scale_input = torch.sqrt(scale_input)
            # scale_input = scale_input.unsqueeze(-1)
            # scale_feat = self.scale_mlp(scale_input)  # [B, N, C]
            # scale_feat = scale_feat.permute(0, 2, 1)  # [B, C, N]
            # scale_feat = torch.relu(scale_feat)          #sm3
            # scale_feat = scale_feat * fg_mask          # 背景点特征置 0 → 调制因子 = 1
            # small_mask = (point_scale < 0.04).float().unsqueeze(1)
            # seed_features = seed_features * (1 + 0.6*scale_feat*small_mask)
            ###################sm3###############

                    
        # Generate the masks of the objectness and the graspness
        end_points = self.graspable(seed_features, end_points)
        seed_features_flipped = seed_features.transpose(1, 2)
        objectness_score = end_points["objectness_score"]
        # [B (batch size), 2 (object classification), 20000 (points per scene)]
        graspness_score = end_points["graspness_score"].squeeze(
            1
        )  # [B (batch size), 20000 (points in a scene)]
        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = objectness_pred == 1
        graspness_mask = graspness_score > self.graspness_threshold
        graspable_mask = objectness_mask & graspness_mask

        # Generate the downsample point (1024 per scene)
        # using the furthest point sampling
        seed_features_graspable = []
        seed_xyz_graspable = []
        seed_normal_graspable = []
        scale_graspable_list = []
        fg_mask_graspable_list = []
        meter_graspable_list = []
        graspable_num_batch = 0.0
        for i in range(B):
            cur_mask = graspable_mask[i]
            if cur_mask.sum() < 1:
                print("No graspable points in the scene")
                continue

            graspable_num_batch += cur_mask.sum()
            cur_feat = seed_features_flipped[i][cur_mask]
            cur_seed_xyz = seed_xyz[i][cur_mask]

            cur_seed_xyz = cur_seed_xyz.unsqueeze(0)
            fps_idxs = furthest_point_sample(
                cur_seed_xyz, self.num_seed_points
            )
            cur_seed_xyz_flipped = cur_seed_xyz.transpose(1, 2).contiguous()
            cur_seed_xyz = (
                gather_operation(cur_seed_xyz_flipped, fps_idxs)
                .transpose(1, 2)
                .squeeze(0)
                .contiguous()
            )
            if self.use_normal is True:
                cur_seed_normal = seed_normal[i][cur_mask]
                cur_seed_normal_flipped = (
                    cur_seed_normal.unsqueeze(0).transpose(1, 2).contiguous()
                )
                cur_seed_normal = (
                    gather_operation(cur_seed_normal_flipped.float(), fps_idxs)
                    .squeeze(0)
                    .contiguous()
                )
            cur_feat_flipped = (
                cur_feat.unsqueeze(0).transpose(1, 2).contiguous()
            )
            cur_feat = (
                gather_operation(cur_feat_flipped.float(), fps_idxs)
                .squeeze(0)
                .contiguous()
            )

            # ===== 同步采样 scale / fg_mask =====
            if self.cfg.use_scale_modulation:
                cur_scale = end_points["_scale_norm_full"][i][cur_mask]
                cur_fg = end_points["_fg_mask_full"][i][cur_mask]
                cur_meter = end_points["_scale_meter_full"][i][cur_mask]
                cur_scale_in = cur_scale.view(1, 1, -1).contiguous().float()
                cur_fg_in = cur_fg.view(1, 1, -1).contiguous().float()
                cur_meter_in = cur_meter.view(1, 1, -1).contiguous().float()
                cur_scale_sampled = (
                    gather_operation(cur_scale_in, fps_idxs)
                    .squeeze(0).squeeze(0).contiguous()
                )
                cur_fg_sampled = (
                    gather_operation(cur_fg_in, fps_idxs)
                    .squeeze(0).squeeze(0).contiguous()
                )
                cur_meter_sampled = (
                    gather_operation(cur_meter_in, fps_idxs)
                    .squeeze(0).squeeze(0).contiguous()
                )
                scale_graspable_list.append(cur_scale_sampled)
                fg_mask_graspable_list.append(cur_fg_sampled)
                meter_graspable_list.append(cur_meter_sampled)


            seed_features_graspable.append(cur_feat)
            seed_xyz_graspable.append(cur_seed_xyz)
            if self.use_normal is True:
                seed_normal_graspable.append(cur_seed_normal)
        seed_xyz_graspable = torch.stack(seed_xyz_graspable, 0)
        if self.use_normal is True:
            seed_normal_graspable = torch.stack(seed_normal_graspable, 0)
        # [B (batch size), 512 (feature dim), 1024 (points after sample)]
        seed_features_graspable = torch.stack(seed_features_graspable)
        end_points["xyz_graspable"] = seed_xyz_graspable
        end_points["normal_graspable"] = seed_normal_graspable
        end_points["D: Graspable Points"] = graspable_num_batch / B

        if self.cfg.use_scale_modulation:
            end_points["scale_graspable"] = torch.stack(
                scale_graspable_list, 0
            )  # [B, 1024]
            end_points["fg_mask_graspable"] = torch.stack(
                fg_mask_graspable_list, 0
            )
            end_points["meter_graspable"] = torch.stack(
                meter_graspable_list, 0
            )  #

        # Select the view for each point
        end_points, res_feat = self.view(
            seed_features_graspable, end_points, is_training
        )
        # [B (batch size), 512 (feature dim), 1024 (points after sample)]
        seed_features_graspable = seed_features_graspable + res_feat
        # [B (batch size), 512 (feature dim), 1024 (points after sample)]

        # Generate the labels
        if is_training:
            # generate the scene-level grasp labels from the object-level
            # grasp label and the object poses
            # map the scene sampled points to the labeled object points
            # (note that the labeled object points and the sampled points
            # may not 100% match due to the sampling and argumentation)
            grasp_top_views_rot, end_points = process_grasp_labels(
                end_points, self.cfg
            )
        else:
            grasp_top_views_rot = end_points["grasp_top_view_rot"]

        # Cylinder grouping
        group_features = []
        for cy_group in self.cy_groups:
            group_features.append(
                cy_group(
                    seed_xyz_graspable.contiguous(),
                    seed_features_graspable.contiguous(),
                    grasp_top_views_rot,
                )
            )

        group_features = torch.stack(group_features, dim=1)
        group_features = self.fuse_multi_scale(group_features)
        # ===== Scale Modulation Stage-2（group-level）=====
        if self.cfg.use_scale_modulation:
            group_features = self.scale_film_group(
                group_features,
                end_points["scale_graspable"],
                end_points["fg_mask_graspable"],
                end_points["meter_graspable"],
            )
        # Width and score predicting
        end_points = self.grasp_head(group_features, end_points)
        ##########################add##################
        if self.cfg.use_uncertainty:
            confidence = self.confidence_head(group_features)
            end_points["grasp_confidence"] = confidence

        ##########################add##################

        if self.cfg.use_uncertainty and self.cfg.use_semantic:
            point_class = end_points["point_class"].to(group_features.device)

            sem_feat = self.semantic_embed(point_class)  # [B, N, 16]
            sem_bias = self.semantic_mlp(sem_feat)       # [B, N, A*D]
            sem_bias = sem_bias.permute(0, 2, 1)         # [B, A*D, N]

            end_points["grasp_logvar"] = end_points["grasp_logvar"] - sem_bias
        ##########################add##################
    
        if is_training:
            for loss_func in self.loss:
                end_points = loss_func(end_points)

        return end_points


MODULE_TPYE = TorchModuleCfgType_co | DelayInitDictType  # noqa: E501


class FineGraspConfig(TorchModuleCfg[FineGrasp]):
    class_type: ClassType_co[FineGrasp] = FineGrasp

    # Model hyperparameters
    model_name: str = "FineGrasp"
    seed_feat_dim: int = 512
    graspness_threshold: float = 0.1
    grasp_max_width: float = 0.1
    num_depth: int = 4
    num_view: int = 300
    num_angle: int = 12
    num_seed_points: int = 1024
    voxel_size: float = 0.005
    cylinder_radius: float = 0.05
    cylinder_groups: list = [0.25, 0.5, 0.75, 1.0]
    use_normal: bool = False

    loss: MODULE_TPYE | None = None
        # ===== Ablation switches =====
    use_scale_sampling: bool = False
    use_scale_modulation: bool = False
    use_uncertainty: bool = False
    use_semantic: bool = False
