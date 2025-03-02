# 通道到高度映射：
# 输入：(B, 128, Dy, Dx)。
# 使用一个 2D 卷积层将通道数映射为 (B, hidden_dim * Dz, Dy, Dx)，其中 hidden_dim 可设为 32 或 64。
# 重塑为 (B, Dy, Dx, Dz, hidden_dim)。
# Z 轴 Transformer：
# 对每个 (X, Y) 位置的 Z 轴特征 (Dz, hidden_dim) 应用小型 Transformer。
# 添加 Z 轴位置编码，建模高度层间的关系。
# 使用 2-3 层 Transformer，控制计算成本。
# 占用预测：
# 通过线性层将 (B, Dy, Dx, Dz, hidden_dim) 映射为 (B, Dy, Dx, Dz, num_classes)。
# 根据需要应用 softmax 或其他激活函数。

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS, build_loss
import numpy as np
from ..losses.semkitti_loss import sem_scal_loss, geo_scal_loss
from ..losses.lovasz_softmax import lovasz_softmax

nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])

# 简化的 Transformer 编码层，仅用于 Z 轴
class ZTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src
    
@HEADS.register_module()
class BEVZTransformerHead(BaseModule):
    def __init__(self,
                 out_dim=None,         # 输出维度不继承
                 in_dim=128,        # 输入通道数，与 img_bev_encoder_neck 输出匹配
                 hidden_dim=64,     # 隐藏维度，控制计算量
                 Dz=16,             # Z 轴分辨率
                 num_classes=18,    # 类别数
                 num_layers=2,      # Z 轴 Transformer 层数
                 nhead=2,           # 注意力头数
                 dim_feedforward=128,  # 前馈网络维度
                 dropout=0.1,       # Dropout 率
                 use_mask=True,     # 是否使用掩码
                 class_balance=False,
                 use_predicter=True,
                 loss_occ=None):
        super(BEVZTransformerHead, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.Dz = Dz
        self.num_classes = num_classes
        self.use_mask = use_mask

        # 通道到高度映射
        self.channel_to_height = nn.Conv2d(
            in_channels=in_dim,
            out_channels=hidden_dim * Dz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        # Z 轴位置编码
        self.z_pos_embed = nn.Parameter(torch.zeros(1, Dz, hidden_dim))
        nn.init.trunc_normal_(self.z_pos_embed, std=0.02)

        # Z 轴 Transformer
        z_layer = ZTransformerLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        #self.z_transformer = nn.Sequential(*[z_layer for _ in range(num_layers)])
        self.z_transformer = nn.Sequential(*[
        ZTransformerLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ) for _ in range(num_layers)
])

        # 预测层
        self.predictor = nn.Linear(hidden_dim, num_classes)

        # 损失相关
        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            if loss_occ is not None:
                loss_occ['class_weight'] = class_weights
        self.loss_occ = build_loss(loss_occ)
    
    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx), 输入为 img_bev_encoder_neck 的输出
        Returns:
            occ_pred: (B, Dx, Dy, Dz, num_classes)
        """
        B, C, Dy, Dx = img_feats.shape

        # 通道到高度映射
        x = self.channel_to_height(img_feats)  # (B, hidden_dim*Dz, Dy, Dx)
        x = x.reshape(B, self.hidden_dim, self.Dz, Dy, Dx)  # 确保通道顺序正确
        x = x.permute(0, 3, 4, 2, 1)  # (B, Dy, Dx, Dz, hidden_dim)

        # 添加 Z 轴位置编码
        x = x.reshape(-1, self.Dz, self.hidden_dim)  # (B*Dy*Dx, Dz, hidden_dim)
        x = x + self.z_pos_embed  # (B, Dy, Dx, Dz, hidden_dim)

        # Z 轴 Transformer 处理
        x = self.z_transformer(x)  # (B*Dy*Dx, Dz, hidden_dim)
        x = x.view(B, Dy, Dx, self.Dz, self.hidden_dim)  # (B, Dy, Dx, Dz, hidden_dim)
        x = x.permute(0, 2, 1, 3, 4)  # (B, Dx, Dy, Dz, hidden_dim)

        # 预测占用
        occ_pred = self.predictor(x)  # (B, Dx, Dy, Dz, num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
            dict: losses
        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                mask_camera,
                avg_factor=num_total_samples
            )
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                avg_factor=num_total_samples
            )

        loss['loss_occ'] = loss_occ
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
        Returns:
            List[(Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)
        return list(occ_res)
    
    def get_occ_gpu(self, occ_pred, img_metas=None):
        """GPU版本的占用预测结果获取，用于加速推理"""
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1).cpu().numpy().astype(np.uint8)  # 转为NumPy数组
        return list(occ_res)