import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS
from mmcv.runner import auto_fp16, force_fp32


@NECKS.register_module()
class TemporalBEVFusion(nn.Module):
    """时序BEV特征融合模块，增强对运动目标的感知能力"""
    
    def __init__(self,
                 in_channels=128,
                 out_channels=None,
                 queue_length=3,
                 fusion_method='gru',  # 'attn', '3d_conv', 'gru'
                 with_skip=True,
                 norm_cfg=dict(type='BN'),
                 ):
        super(TemporalBEVFusion, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.queue_length = queue_length
        self.fusion_method = fusion_method
        self.with_skip = with_skip
        
        # 初始化特征队列
        self.feature_queue = []  # 改用Python列表而非tensor
        
        # 特征投影
        if self.in_channels != self.out_channels:
            self.projection = nn.Conv2d(
                self.in_channels, self.out_channels, kernel_size=1)
        
        # GRU融合
        if self.fusion_method == 'gru':
            self.gru = nn.GRU(
                self.out_channels, self.out_channels, 
                num_layers=1, batch_first=True)
            
            self.gru_norm = nn.LayerNorm(self.out_channels)
            
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
    @auto_fp16()
    def forward(self, x, is_training=True):
        """前向传播，保证历史特征完全分离"""
        batch_size, channels, h, w = x.shape
        curr_feat = x
        
        # 特征投影
        if hasattr(self, 'projection'):
            curr_feat = self.projection(curr_feat)
        
        # 获取增强特征
        enhanced_feat = self._fuse_temporal_features(curr_feat)
        
        # 更新队列 (仅在训练时)
        if is_training:
            # 确保特征完全分离
            if len(self.feature_queue) >= self.queue_length:
                self.feature_queue.pop(0)
            # 明确使用clone和detach来切断梯度流
            self.feature_queue.append(curr_feat.clone().detach())
        
        # 残差连接
        if self.with_skip:
            enhanced_feat = enhanced_feat + curr_feat
            
        # 输出层
        output = self.output_layer(enhanced_feat)
        
        return output
    
    def _fuse_temporal_features(self, curr_feat):
        """融合时序特征，确保不出现双重反向传播"""
        batch_size, channels, h, w = curr_feat.shape
        
        # GRU模式 - 简化实现
        if self.fusion_method == 'gru' and len(self.feature_queue) > 0:
            # 使用无梯度上下文处理历史特征
            with torch.no_grad():
                # 处理所有历史特征
                history_feats_tensor = []
                for feat in self.feature_queue:
                    # 确保特征已分离
                    history_feats_tensor.append(feat.detach())
                
                if history_feats_tensor:
                    # 准备历史特征
                    history_feats = torch.stack(history_feats_tensor, dim=1)  # [B, T, C, H, W]
                    b, t, c, h, w = history_feats.shape
                    history_flat = history_feats.reshape(b, t, c, -1).permute(0, 3, 1, 2)  # [B, H*W, T, C]
                    history_flat = history_flat.reshape(-1, t, c)  # [B*H*W, T, C]
                    
                    # 获取历史状态
                    _, h_state = self.gru(history_flat)  # h_state: [1, B*H*W, C]
                else:
                    # 如果没有历史特征，则返回当前特征
                    return curr_feat
                
            # 单独处理当前帧
            curr_flat = curr_feat.reshape(batch_size, channels, -1).permute(0, 2, 1)  # [B, H*W, C]
            curr_flat = curr_flat.reshape(-1, 1, channels)  # [B*H*W, 1, C]
            
            # 使用历史状态初始化当前帧处理
            _, h_n = self.gru(curr_flat, h_state.detach())  # h_n: [1, B*H*W, C]
            
            # 整形回原始形状
            gru_out = h_n.squeeze(0)  # [B*H*W, C]
            enhanced_feat = gru_out.reshape(batch_size, h*w, channels).permute(0, 2, 1)  # [B, C, H*W]
            enhanced_feat = enhanced_feat.reshape(batch_size, channels, h, w)  # [B, C, H, W]
            
            return enhanced_feat
        else:
            # 没有历史数据，返回当前特征
            return curr_feat
    
    def train(self, mode=True):
        """在训练模式切换时清空队列"""
        if mode and hasattr(self, 'feature_queue'):
            self.feature_queue = []
        return super(TemporalBEVFusion, self).train(mode)