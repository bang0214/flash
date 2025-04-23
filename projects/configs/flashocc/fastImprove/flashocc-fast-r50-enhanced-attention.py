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

# 优化器配置 - 更简单的优化器设置
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
)

# 数据配置 - 原批量大小应该没问题
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

# 启用混合精度训练，提高内存效率
fp16 = dict(loss_scale=512.)