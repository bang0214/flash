import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
import torch.nn.functional as F
from einops import rearrange
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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
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


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(encoder_layer.norm1.normalized_shape[0])

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        output = self.norm(output)
        return output


@HEADS.register_module()
class TransformerBEVOccHead(BaseModule):
    def __init__(self,
                 in_dim=128,
                 out_dim=128,
                 hidden_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 num_layers=3,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 class_balance=False,
                 loss_occ=None,
                 use_predicter=True,
                 use_position_embed=True):
        super(TransformerBEVOccHead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.Dz = Dz
        self.num_classes = num_classes
        self.use_mask = use_mask

        # Convert 2D BEV features to proper shape for Transformer
        self.init_conv = nn.Sequential(
            ConvModule(
                in_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv2d')
            ),
            nn.Flatten(2),  # Flatten spatial dimensions
        )
        
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # Projection for final prediction
        self.predicter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, num_classes * Dz),
        )

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            if loss_occ is not None and hasattr(loss_occ, 'class_weight'):
                loss_occ['class_weight'] = class_weights

        self.loss_occ = build_loss(loss_occ)

        self.use_position_embed = use_position_embed
        
        # 添加位置编码
        if use_position_embed:
            # 可学习的位置编码
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 200*200, hidden_dim))  # 假设最大空间尺寸为200x200
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
        """
        B, C, Dy, Dx = img_feats.shape
        
        # 使用卷积处理输入特征
        x = self.init_conv(img_feats)  # (B, hidden_dim, Dy, Dx)
        x = x.permute(0, 2, 1)  # (B, Dy*Dx, hidden_dim)
        
        # 添加位置编码
        if self.use_position_embed:
            # 裁剪位置编码以匹配当前特征图大小
            pos_embed = self.pos_embed[:, :Dy*Dx, :]
            x = x + pos_embed

        # 应用Transformer编码器
        x = self.transformer_encoder(x)  # (B, Dy*Dx, hidden_dim)
        
        # 使用预测器生成每个位置的类别预测
        x = self.predicter(x)  # (B, Dy*Dx, num_classes*Dz)
        
        # 重新调整形状以匹配期望的输出
        x = x.reshape(B, Dx, Dy, self.Dz, self.num_classes)
        
        return x

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
        voxel_semantics = voxel_semantics.long()    # (B, Dx, Dy, Dz)
        
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
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
        
        # Add auxiliary losses for better training
        preds_permute = occ_pred.permute(0, 4, 1, 2, 3).contiguous()    # (B, n_cls, Dx, Dy, Dz)
        loss['loss_voxel_sem_scal'] = sem_scal_loss(preds_permute, voxel_semantics)
        loss['loss_voxel_geo_scal'] = geo_scal_loss(preds_permute, voxel_semantics, non_empty_idx=17)
        loss['loss_voxel_lovasz'] = lovasz_softmax(torch.softmax(preds_permute, dim=1), voxel_semantics)
        
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)

    def get_occ_gpu(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1).int()      # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module()
class TransformerBEVOccHead3D(BaseModule):
    """A more sophisticated 3D transformer-based occupancy head with 3D attention."""
    def __init__(self,
                 in_dim=128,
                 hidden_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 num_layers=3,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.1,
                 class_balance=False,
                 use_position_embed=True,
                 loss_occ=None,
                 use_predicter=True):
        super(TransformerBEVOccHead3D, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.Dz = Dz
        self.num_classes = num_classes
        self.use_mask = use_mask
        self.use_position_embed = use_position_embed
        
        # Initial 3D convolution projection
        self.init_conv = ConvModule(
            in_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        
        # Positional embeddings
        if use_position_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, Dz, hidden_dim))
            
        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # Final prediction head
        self.predicter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, num_classes)
        )

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            if loss_occ is not None and 'class_weight' in loss_occ:
                loss_occ['class_weight'] = class_weights

        self.loss_occ = build_loss(loss_occ)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize positional embeddings."""
        if self.use_position_embed:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
        """
        B, C, Dy, Dx = img_feats.shape
        
        # Project BEV features
        x = self.init_conv(img_feats)  # (B, hidden_dim, Dy, Dx)
        
        # Process each spatial position with transformer
        outputs = []
        for i in range(Dx):
            for j in range(Dy):
                # Initialize a query embedding for each z level
                query = torch.zeros(B, self.Dz, self.hidden_dim, device=x.device)
                
                # Add BEV feature as context
                ctx_feature = x[:, :, j, i].unsqueeze(1).expand(-1, self.Dz, -1)  # (B, Dz, hidden_dim)
                query = query + ctx_feature
                
                # Add positional embedding
                if self.use_position_embed:
                    query = query + self.pos_embed
                
                # Apply transformer
                query = self.transformer_encoder(query)  # (B, Dz, hidden_dim)
                
                # Get predictions
                pred = self.predicter(query)  # (B, Dz, num_classes)
                outputs.append(pred)
        
        # Combine all spatial positions
        outputs = torch.stack(outputs, dim=1)  # (B, Dx*Dy, Dz, num_classes)
        outputs = outputs.reshape(B, Dx, Dy, self.Dz, self.num_classes)
        
        return outputs

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """Loss function same as TransformerBEVOccHead."""
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
        
        # Add auxiliary losses
        preds_permute = occ_pred.permute(0, 4, 1, 2, 3).contiguous()
        loss['loss_voxel_sem_scal'] = sem_scal_loss(preds_permute, voxel_semantics)
        loss['loss_voxel_geo_scal'] = geo_scal_loss(preds_permute, voxel_semantics, non_empty_idx=17)
        loss['loss_voxel_lovasz'] = lovasz_softmax(torch.softmax(preds_permute, dim=1), voxel_semantics)
        
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """Same as TransformerBEVOccHead."""
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)
        return list(occ_res)

    def get_occ_gpu(self, occ_pred, img_metas=None):
        """Same as TransformerBEVOccHead."""
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1).int()
        return list(occ_res)
