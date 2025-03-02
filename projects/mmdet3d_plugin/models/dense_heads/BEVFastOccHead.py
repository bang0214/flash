import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS, build_loss
import numpy as np
from ..losses.semkitti_loss import sem_scal_loss, geo_scal_loss
from ..losses.lovasz_softmax import lovasz_softmax

nusc_class_frequencies = np.array([
    944004, 1897170, 152386, 2391677, 16957802, 724139, 189027,
    2074468, 413451, 2384460, 5916653, 175883646, 4275424,
    51393615, 61411620, 105975596, 116424404, 1892500630
])

class HeightEncoder(nn.Module):
    """简单高效的高度编码器"""
    def __init__(self, height_dim, feat_dim):
        super(HeightEncoder, self).__init__()
        self.height_dim = height_dim
        self.feat_dim = feat_dim
        # 一种简单的高度位置编码
        self.height_encoding = nn.Parameter(
            torch.zeros(1, height_dim, feat_dim//2))
        nn.init.normal_(self.height_encoding, std=0.02)
        
        # 将位置编码映射到特征空间
        self.height_proj = nn.Sequential(
            nn.Linear(feat_dim//2, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: BEV特征 (B, C, H, W)
        Returns:
            体素特征 (B, C, D, H, W)
        """
        B, C, H, W = x.shape
        
        # 沿高度维度重复BEV特征
        x_3d = x.unsqueeze(2).repeat(1, 1, self.height_dim, 1, 1)  # (B, C, D, H, W)
        
        # 高度编码
        height_feat = self.height_proj(self.height_encoding)  # (1, D, C)
        height_feat = height_feat.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (1, C, D, 1, 1)
        
        # 融合高度信息
        x_3d = x_3d + height_feat
        
        return x_3d

@HEADS.register_module()
class BEVFastOccHead(BaseModule):
    """基于FastOcc思想的简化版占用预测头"""
    def __init__(self,
                 in_dim=128,
                 hidden_dim=128,
                 out_dim=None,  # 为兼容性保留
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 class_balance=True,
                 loss_occ=None):
        super(BEVFastOccHead, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.Dz = Dz
        self.num_classes = num_classes
        self.use_mask = use_mask
        
        # BEV特征编码模块
        self.bev_encoder = nn.Sequential(
            ConvModule(in_dim, hidden_dim, 3, padding=1, 
                      conv_cfg=dict(type='Conv2d')),
            ConvModule(hidden_dim, hidden_dim, 3, padding=1, 
                      conv_cfg=dict(type='Conv2d'))
        )
        
        # 高度扩展模块
        self.height_encoder = HeightEncoder(Dz, hidden_dim)
        
        # 3D融合模块 - 轻量级设计
        self.fusion_conv3d = nn.Sequential(
            ConvModule(hidden_dim, hidden_dim, 
                      kernel_size=3, padding=1,
                      conv_cfg=dict(type='Conv3d')),
            ConvModule(hidden_dim, hidden_dim//2, 
                      kernel_size=1, padding=0,
                      conv_cfg=dict(type='Conv3d'))
        )
        
        # 分类头
        self.classifier = nn.Conv3d(hidden_dim//2, num_classes, kernel_size=1)
        
        # 损失函数设置
        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            if loss_occ is not None and hasattr(loss_occ, 'class_weight'):
                loss_occ['class_weight'] = class_weights
        
        self.loss_occ = build_loss(loss_occ)
    
    def forward(self, img_feats):
        """
        Args:
            img_feats: BEV特征 (B, C, Dy, Dx)
        Returns:
            occ_pred: 占用预测 (B, Dx, Dy, Dz, num_classes)
        """
        # 1. BEV特征编码
        bev_feat = self.bev_encoder(img_feats)  # (B, hidden_dim, Dy, Dx)
        
        # 2. 特征扩展到3D空间
        voxel_feat = self.height_encoder(bev_feat)  # (B, hidden_dim, Dz, Dy, Dx)
        
        # 3. 3D卷积融合
        voxel_feat = self.fusion_conv3d(voxel_feat)  # (B, hidden_dim//2, Dz, Dy, Dx)
        
        # 4. 分类预测
        preds = self.classifier(voxel_feat)  # (B, num_classes, Dz, Dy, Dx)
        
        # 5. 调整维度顺序以匹配期望的输出格式
        occ_pred = preds.permute(0, 4, 3, 2, 1)  # (B, Dx, Dy, Dz, num_classes)
        
        return occ_pred
    
    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """计算损失"""
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
        
        # 只使用CrossEntropyLoss
        loss['loss_occ'] = loss_occ
        
        return loss
    
    def get_occ(self, occ_pred, img_metas=None):
        """CPU版本的占用预测结果获取"""
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)
        return list(occ_res)
    
    def get_occ_gpu(self, occ_pred, img_metas=None):
        """GPU版本的占用预测结果获取，用于加速推理"""
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1).cpu().numpy().astype(np.uint8)  # 转为NumPy数组
        return list(occ_res)