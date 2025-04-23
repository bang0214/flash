_base_ = ['../flashocc-r50-M0.py']
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

numC_Trans = 64

# 修改模型配置
model = dict(
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=128,
    ),
    # 在img_bev_encoder_neck之后添加时序融合模块
    temporal_fusion_module=dict(
        type='TemporalBEVFusion',
        in_channels=128,
        out_channels=128,
        queue_length=3,  # 存储前3帧
        fusion_method='gru',  # 可选:'attn', '3d_conv', 'gru'
        with_skip=True
    ),
    occ_head=dict(
        _delete_=True,  # 删除基础配置中的参数
        type='BEVFastOccHead',
        in_dim=128,
        hidden_dim=128,
        Dz=16,
        num_classes=18,
        use_mask=True,
        class_balance=False,
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
            'temporal_fusion_module': dict(lr_mult=1.0),
        }
    )
)

# 学习率调整
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, ]
)
runner = dict(max_epochs=24)

# 数据配置
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

# 启用混合精度训练以提升速度
fp16 = dict(loss_scale=512.)