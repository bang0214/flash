_base_ = ['./flashocc/flashocc-r50-M0.py']
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# 修改模型配置
model = dict(
    occ_head=dict(
        _delete_=True,  # 删除基础配置中的参数
        type='BEVFastOccHead',
        in_dim=128,
        hidden_dim=96,
        Dz=16,
        num_classes=18,
        use_mask=True,
        class_balance=False,
        dropout=0.2,  # 与BEVOCCHead2D保持一致
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
    weight_decay=0.07,
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
runner = dict(max_epochs=20)
checkpoint_config = dict(interval=runner['max_epochs'] + 1)
# 数据配置
data = dict(
    samples_per_gpu=15,  
    workers_per_gpu=15,
)

# 启用混合精度训练以提升速度
fp16 = dict(loss_scale=512.)
workflow = [('train', 1), ('val', 1)]

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560, # 这个值来自您的 flashocc-r50-M0.py
        decay=0.9990,       # 这个值来自您的 ema.py
        resume=None,        # 根据需要设置
        num_last_epochs_to_save=3 # 明确指定只保存最后3个epoch的EMA
    ),
    # 如果 flashocc-r50-M0.py 中还有其他 custom_hooks，您可能需要将它们也复制过来
    # 或者只修改 MEGVIIEMAHook 的部分。
    # 最简单的方式是，如果 _base_ 中的 custom_hooks 只有 MEGVIIEMAHook，
    # 这里的定义会完全覆盖它。
]






