import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS

@NECKS.register_module()
class DualPathAttention(nn.Module):
    """双路径注意力模块：结合空间和通道注意力以增强BEV特征表示
    
    Args:
        in_channels (int): 输入通道数
        reduction (int): 通道注意力维度缩减系数
        spatial_kernel (int): 空间注意力卷积核大小
    """
    
    def __init__(self, 
                 in_channels=128, 
                 reduction=8, 
                 spatial_kernel=7,
                 use_scale=True):
        super(DualPathAttention, self).__init__()
        self.in_channels = in_channels
        self.use_scale = use_scale
        
        # 通道注意力路径（SE模块改进版）
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        
        # 空间注意力路径（CBAM空间模块改进版）
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False),
            nn.BatchNorm2d(1)
        )
        
        # 注意力融合
        if use_scale:
            self.scale_params = nn.Parameter(torch.zeros(1))
        
        # 初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        """前向传播函数
        
        Args:
            x (Tensor): 输入特征，形状为 [B, C, H, W]
            
        Returns:
            Tensor: 增强后的特征，形状为 [B, C, H, W]
        """
        identity = x
        
        # 通道注意力
        avg_out = self.channel_mlp(self.channel_avg_pool(x))
        max_out = self.channel_mlp(self.channel_max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        
        # 先应用通道注意力
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_in))
        
        # 应用空间注意力
        x = x * spatial_att
        
        # 残差连接
        if self.use_scale:
            att_scaling = torch.sigmoid(self.scale_params)
            output = identity + att_scaling * (x - identity)
        else:
            output = x
        
        return output
    

@NECKS.register_module()
class LiteHybridAttention(nn.Module):
    """轻量级混合注意力模块，结合通道和空间注意力以提高计算效率
    
    Args:
        in_channels (int): 输入通道数
        reduction (int): 通道注意力降维比率
    """
    
    def __init__(self, 
                 in_channels=128,
                 reduction=8,
                 use_spatial=True,
                 use_ffn=False):
        super(LiteHybridAttention, self).__init__()
        self.in_channels = in_channels
        self.use_spatial = use_spatial
        self.use_ffn = use_ffn
        
        # 通道注意力 - 高效实现
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力 - 轻量级实现
        if use_spatial:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            
        # 可选的前馈网络
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
            )
            self.ffn_scale = nn.Parameter(torch.ones(1) * 0.5)
        
        # 可学习的残差权重
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播函数
        
        Args:
            x (Tensor): 输入特征，形状为 [B, C, H, W]
            
        Returns:
            Tensor: 增强后的特征，形状为 [B, C, H, W]
        """
        identity = x
        
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        if self.use_spatial:
            # 聚合通道信息
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            spatial_in = torch.cat([avg_out, max_out], dim=1)
            
            # 应用空间注意力
            sa = self.spatial_attention(spatial_in)
            x = x * sa
            
        # 应用可选的前馈网络
        if self.use_ffn:
            x = x + self.ffn_scale * self.ffn(x)
            
        # 应用残差连接与自适应重要性缩放
        out = identity + self.gamma * (x - identity)
        
        return out