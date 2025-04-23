import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS


@NECKS.register_module()
class EnhancedBEVAttention(nn.Module):
    """增强型BEV注意力模块，结合自注意力、非局部操作和多尺度特征处理
    
    Args:
        in_channels (int): 输入通道数
        reduction (int): 降维比例
        num_heads (int): 自注意力头数
        dropout (float): Dropout比例
        norm_cfg (dict): 归一化层配置
    """
    
    def __init__(self, 
                 in_channels=128, 
                 hidden_dim=256,
                 reduction=4, 
                 num_heads=8,
                 dropout=0.1,
                 use_non_local=True,
                 use_multi_scale=True,
                 norm_cfg=dict(type='LN')):
        super(EnhancedBEVAttention, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_non_local = use_non_local
        self.use_multi_scale = use_multi_scale
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # 自注意力模块
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # 非局部模块(Non-local)
        if use_non_local:
            self.non_local = NonLocalBlock(hidden_dim, hidden_dim // 2)
        
        # 多尺度特征处理
        if use_multi_scale:
            self.multi_scale = MultiScaleBranch(hidden_dim)
        
        # 前馈网络(FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 输出投影
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
        # 可学习的残差系数
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x):
        """前向传播函数
        
        Args:
            x (Tensor): 输入特征，形状为 [B, C, H, W]
            
        Returns:
            Tensor: 增强后的特征，形状为 [B, C, H, W]
        """
        identity = x
        batch, c, h, w = x.shape
        
        # 输入投影
        feat = self.input_proj(x)  # [B, hidden_dim, H, W]
        
        # 多尺度处理
        if self.use_multi_scale:
            feat = self.multi_scale(feat)  # [B, hidden_dim, H, W]
        
        # 非局部处理
        if self.use_non_local:
            feat = self.non_local(feat)  # [B, hidden_dim, H, W]
            
        # 准备自注意力输入
        feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, HW, hidden_dim]
        
        # 自注意力层
        attn_out = self.self_attn(
            feat_flat, feat_flat, feat_flat,
            need_weights=False)[0]
        
        feat_flat = feat_flat + self.dropout1(attn_out)
        feat_flat = self.norm1(feat_flat)
        
        # FFN
        ffn_out = self.ffn(feat_flat)
        feat_flat = feat_flat + ffn_out
        feat_flat = self.norm2(feat_flat)
        
        # 恢复空间维度
        feat = feat_flat.permute(0, 2, 1).view(batch, self.hidden_dim, h, w)
        
        # 输出投影
        out = self.output_proj(feat)
        
        # 缩放残差
        return identity + self.res_scale * out


class NonLocalBlock(nn.Module):
    """非局部模块，捕获全局上下文
    
    Args:
        in_channels (int): 输入通道数
        inter_channels (int): 中间通道数
    """
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels // 2
        
        # 定义变换矩阵
        self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
        
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1),
            nn.BatchNorm2d(self.in_channels)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.W[0].weight, std=0.01)
        nn.init.constant_(self.W[0].bias, 0)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # [B, HW, C]
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # [B, HW, C]
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [B, C, HW]
        
        f = torch.matmul(theta_x, phi_x)  # [B, HW, HW]
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)  # [B, HW, C]
        y = y.permute(0, 2, 1).contiguous()  # [B, C, HW]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # [B, C, H, W]
        
        W_y = self.W(y)
        z = W_y + x
        
        return z


class MultiScaleBranch(nn.Module):
    """多尺度特征处理分支，处理不同尺度的特征
    
    Args:
        channels (int): 特征通道数
    """
    def __init__(self, channels):
        super(MultiScaleBranch, self).__init__()
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=1, padding=2),
                nn.Conv2d(channels, channels // 4, kernel_size=1),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=9, stride=1, padding=4),
                nn.Conv2d(channels, channels // 4, kernel_size=1),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.AvgPool2d(kernel_size=17, stride=1, padding=8),
                nn.Conv2d(channels, channels // 4, kernel_size=1),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(channels, channels // 4, kernel_size=1),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True)
            )
        ])
        
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branches_out = []
        for branch in self.branches:
            branches_out.append(branch(x))
        
        concat_feat = torch.cat(branches_out, dim=1)
        out = self.fuse(concat_feat)
        
        return out