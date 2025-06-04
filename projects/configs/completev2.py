_base_ = ['./flashocc/flashocc-r50-M0.py']
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

numC_Trans = 64

# 修改模型配置 - 降低复杂度以减少过拟合
model = dict(
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=96,  # 从128降低到96，减少模型容量
        bev_enhance_module=dict(
            type='SimpleBEVEnhance',
            in_channels=96  # 相应调整
        )
    ),
    # 轻量级注意力模块 - 增加正则化
    bev_attention_module=dict(
        type='LiteHybridAttention',
        in_channels=96,  # 相应调整
        reduction=16,  # 从8增加到16，进一步减少参数数量
        use_spatial=True,
        use_ffn=False,
        dropout=0.1  # 添加dropout正则化
    ),
    occ_head=dict(
        _delete_=True,
        type='BEVFastOccHead',
        in_dim=96,  # 相应调整
        hidden_dim=96,  # 从128降低到96
        Dz=16,
        num_classes=18,
        use_mask=True,
        class_balance=True,  # 启用类别平衡以提高小类别性能
        dropout=0.1,  # 添加dropout
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        )
    )
)

# 优化器配置 - 调整weight_decay
optimizer = dict(
    type='AdamW',
    lr=1e-4,  # 降低学习率到基础配置一致的1e-4
    weight_decay=0.05,  # 调整weight_decay，0.1可能过高，0.05是个平衡值
    paramwise_cfg=dict(
        custom_keys={
            'occ_head.fusion_conv3d': dict(lr_mult=1.0),
        }
    )
)

# 学习率调整 - 保持余弦退火但增加warmup时间
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,  # 增加warmup时间，从500到1000
    warmup_ratio=0.001,
    min_lr=1e-6
)

# 减少训练epochs以避免过拟合
runner = dict(max_epochs=18)  # 从24减少到18，因为第15个epoch后开始过拟合

# 添加EMA以提高模型稳定性（来自基础配置）
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

# 启用混合精度训练以提升速度
fp16 = dict(loss_scale=512.)

# 设置工作流程，确保每个epoch都进行验证
workflow = [('train', 1), ('val', 1)]

# 添加早期停止设置
evaluation = dict(
    interval=1,  # 每个epoch评估一次
    start=8,     # 从第8个epoch开始评估
    pipeline=test_pipeline,
    save_best='mIoU',  # 保存最佳mIoU模型
    rule='greater'     # mIoU越大越好
)

# 检查点配置
checkpoint_config = dict(interval=1, max_keep_ckpts=3)  # 只保留最好的3个检查点

# 加载预训练模型（继承自基础配置）
load_from = "ckpts/bevdet-r50-cbgs.pth"