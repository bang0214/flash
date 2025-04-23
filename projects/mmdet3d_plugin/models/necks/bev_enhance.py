import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS
from mmcv.cnn import ConvModule, build_norm_layer


@NECKS.register_module()
class BEVMultiScaleEnhance(nn.Module):
    """多尺度BEV特征增强模块，提升BEV特征的表达能力"""
    
    def __init__(self, 
                 in_channels=128,
                 out_channels=None,
                 norm_cfg=dict(type='BN'),
                 use_dcn=True,
                 use_attention=True):
        super(BEVMultiScaleEnhance, self).__init__()
        
        out_channels = out_channels or in_channels
        self.use_attention = use_attention
        
        # 多尺度卷积分支
        self.branch_3x3 = ConvModule(
            in_channels, in_channels // 2, 3, padding=1, 
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
            
        self.branch_5x5 = nn.Sequential(
            ConvModule(in_channels, in_channels // 4, 1, 
                      norm_cfg=norm_cfg, act_cfg=dict(type='ReLU')),
            ConvModule(in_channels // 4, in_channels // 2, 5, padding=2, 
                      norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
        )
        
        # 可变形卷积分支(可选)
        if use_dcn:
            from mmcv.ops import DeformConv2d
            self.dcn = DeformConv2d(
                in_channels, in_channels // 2, 3, padding=1)
            self.dcn_bn = build_norm_layer(norm_cfg, in_channels // 2)[1]
            self.dcn_act = nn.ReLU(inplace=True)
        else:
            self.branch_1x1 = ConvModule(
                in_channels, in_channels // 2, 1,
                norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
        
        # 多尺度特征融合
        fusion_channels = in_channels // 2 * 3
        self.fusion = ConvModule(
            fusion_channels, out_channels, 1, 
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
            
        # 通道注意力机制(可选)
        if use_attention:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 8, out_channels, 1),
                nn.Sigmoid()
            )
            
    def forward(self, x):
        """前向传播函数
        
        Args:
            x (Tensor): 输入BEV特征，形状为 [B, C, H, W]
            
        Returns:
            Tensor: 增强后的BEV特征，形状为 [B, C, H, W]
        """
        feat_3x3 = self.branch_3x3(x)
        feat_5x5 = self.branch_5x5(x)
        
        if hasattr(self, 'dcn'):
            # DCN分支需要初始化offset
            batch, _, height, width = x.shape
            offset = torch.zeros(
                batch, 18, height, width, device=x.device)
            dcn_feat = self.dcn_act(self.dcn_bn(self.dcn(x, offset)))
        else:
            dcn_feat = self.branch_1x1(x)
        
        # 拼接多尺度特征
        fused_feat = torch.cat([feat_3x3, feat_5x5, dcn_feat], dim=1)
        output = self.fusion(fused_feat)
        
        # 应用通道注意力
        if self.use_attention:
            attention = self.channel_attention(output)
            output = output * attention
            
        return output


@NECKS.register_module()
class SimpleBEVEnhance(nn.Module):
    """简化的BEV特征增强模块，使用残差连接保留信息"""
    
    def __init__(self, 
                 in_channels=128,
                 norm_cfg=dict(type='BN')):
        super(SimpleBEVEnhance, self).__init__()
        
        # 两个简单分支
        self.conv1 = ConvModule(
            in_channels, in_channels, 3, padding=1, 
            norm_cfg=norm_cfg, act_cfg=dict(type='ReLU'))
            
        self.conv2 = ConvModule(
            in_channels, in_channels, 3, padding=1,
            norm_cfg=norm_cfg, act_cfg=None)  # 注意这里不使用激活函数
            
    def forward(self, x):
        """前向传播函数
        
        Args:
            x (Tensor): 输入BEV特征
            
        Returns:
            Tensor: 增强后的BEV特征
        """
        identity = x  # 保存原始输入
        
        # 简单处理
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 残差连接 - 关键部分
        out = out + identity
        
        # 最后的激活
        out = F.relu(out)
        
        return out