_base_ = ['./flashocc/flashocc-r50-M0.py']
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# 修改模型配置
model = dict(
    occ_head=dict(
        _delete_=True,  # 删除基础配置中的参数
        type='BEVFastOccHead',
        in_dim=128,
        hidden_dim=128,
        Dz=16,
        num_classes=18,
        use_mask=True,
        class_balance=False,  # 与BEVOCCHead2D保持一致
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
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'occ_head.fusion_conv3d': dict(lr_mult=1.0),
        }
    )
)

# 学习率调整 - 余弦退火策略
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=1e-6
)
runner = dict(max_epochs=24)

# 数据配置
data = dict(
    samples_per_gpu=16,  
    workers_per_gpu=16,
)

# 启用混合精度训练以提升速度
fp16 = dict(loss_scale=512.)
workflow = [('train', 1), ('val', 1)]