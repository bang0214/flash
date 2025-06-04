_base_ = ['./flashocc/flashocc-r50-M0.py']
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

numC_Trans = 64

# 修改模型配置
model = dict(
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=128,
        bev_enhance_module=dict(
            type='SimpleBEVEnhance',
            in_channels=128
        )
    ),
    # 添加轻量级注意力模块
    bev_attention_module=dict(
        type='LiteHybridAttention',
        in_channels=128,
        reduction=8,
        use_spatial=True,  # 启用空间注意力
        use_ffn=False      # 禁用前馈网络以节省内存
    ),
    occ_head=dict(
        _delete_=True,
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


 # 启用混合精度训练以提升速度
fp16 = dict(loss_scale=512.)
workflow = [('train', 1), ('val', 1)]