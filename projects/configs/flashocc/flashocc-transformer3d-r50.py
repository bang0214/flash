_base_ = ['./flashocc-r50-M0.py']

# 修改模型配置
model = dict(
    occ_head=dict(
        type='TransformerBEVOccHead3D',  # 使用3D Transformer头部
        in_dim=128,
        hidden_dim=256,
        Dz=16,
        use_mask=True,
        num_classes=18,
        num_layers=3,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        use_position_embed=True,  # 使用位置编码
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
    lr=1e-4,  # 使用更小的学习率
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'occ_head': dict(lr_mult=1.0),
            'occ_head.transformer_encoder': dict(lr_mult=0.1),
            'occ_head.pos_embed': dict(lr_mult=1.0),
        }
    )
)

# 学习率调整
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=1e-6,
)
runner = dict(max_epochs=24)

# 修改数据配置
data = dict(
    samples_per_gpu=1,  # 由于内存占用更高，需要减小batch size
    workers_per_gpu=1,
)
