_base_ = ['./flashocc-r50-M0.py']
# 确保系统知道要加载插件
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# 修改模型配置
model = dict(
    occ_head=dict(
        type='BEVZTransformerHead',
        in_dim=128,           # 与 img_bev_encoder_neck 输出匹配
        hidden_dim=32,        # 隐藏维度
        Dz=16,
        use_mask=True,
        num_classes=18,
        num_layers=2,         # Z 轴 Transformer 层数
        nhead=2,              # 注意力头数
        dim_feedforward=64,   # 前馈网络维度
        dropout=0.1,
        class_balance=False,  # 与 BEVOCCHead2D 保持一致
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        )
    )
)

# 优化器配置
optimizer = dict(
    type='AdamW',
    lr=2e-4,  # Transformer通常需要较小的学习率
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'occ_head.z_transformer': dict(lr_mult=0.1),  # Transformer部分使用较小学习率
        }
    )
)

# 学习率调整 - 确保与基础配置兼容
lr_config = dict(
    _delete_=True,  # 删除基础配置中的lr_config
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, ]
)
runner = dict(max_epochs=24)

# 修改数据配置，可以根据需要调整batch size
data = dict(
    samples_per_gpu=4,  # 与 BEVOCCHead2D 保持一致
    workers_per_gpu=4,
)
