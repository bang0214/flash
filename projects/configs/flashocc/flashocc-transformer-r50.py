_base_ = ['./flashocc-r50-M0.py']
# 确保系统知道要加载插件
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# 修改模型配置
model = dict(
    occ_head=dict(
        type='TransformerBEVOccHead',  # 使用新的Transformer头部
        in_dim=128,
        hidden_dim=128,
        Dz=16,
        use_mask=True,
        num_classes=18,
        num_layers=2,  # Transformer层数
        nhead=4,       # 多头注意力的头数
        dim_feedforward=1024,
        dropout=0.1,
        class_balance=True,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
    )
)

# 优化器配置
optimizer = dict(
    type='AdamW',
    lr=2e-4,  # Transformer通常需要较小的学习率
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'occ_head.transformer_encoder': dict(lr_mult=0.1),  # Transformer部分使用较小学习率
        }
    )
)

# 学习率调整
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22],
)
runner = dict(max_epochs=24)

# 修改数据配置，可以根据需要调整batch size
data = dict(
    samples_per_gpu=1,  # 由于Transformer内存占用更高，可能需要减小batch size
    workers_per_gpu=2,
)
