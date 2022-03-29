norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='/root/autodl-tmp/pretrain/mit_b5.pth',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=1.0,
                class_weight=[
                    0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1, 1, 1,
                    0.95, 1.05, 1.5, 2, 1, 1.05
                ])
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CelebADataset'
data_root = 'data/CelebA/'
img_norm_cfg = dict(
    mean=[132.52878295898438, 106.67704703776042, 92.83904142252604],
    std=[77.05962840493828, 69.7348132715338, 68.29496534260919],
    to_rgb=True)
img_scale = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=16,
        contrast_range=(0.75, 1.25),
        saturation_range=(0.75, 1.25),
        hue_delta=9),
    dict(
        type='Normalize',
        mean=[132.52878295898438, 106.67704703776042, 92.83904142252604],
        std=[77.05962840493828, 69.7348132715338, 68.29496534260919],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[
                    132.52878295898438, 106.67704703776042, 92.83904142252604
                ],
                std=[77.05962840493828, 69.7348132715338, 68.29496534260919],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CelebADataset',
        data_root='data/CelebA/',
        img_dir='image/complete_train',
        ann_dir='mask/complete_train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomFlip', prob=0),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=16,
                contrast_range=(0.75, 1.25),
                saturation_range=(0.75, 1.25),
                hue_delta=9),
            dict(
                type='Normalize',
                mean=[
                    132.52878295898438, 106.67704703776042, 92.83904142252604
                ],
                std=[77.05962840493828, 69.7348132715338, 68.29496534260919],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CelebADataset',
        data_root='data/CelebA/',
        img_dir='image/val',
        ann_dir='mask/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[
                            132.52878295898438, 106.67704703776042,
                            92.83904142252604
                        ],
                        std=[
                            77.05962840493828, 69.7348132715338,
                            68.29496534260919
                        ],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CelebADataset',
        data_root='data/CelebA/',
        img_dir='image/val',
        ann_dir='mask/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[
                            132.52878295898438, 106.67704703776042,
                            92.83904142252604
                        ],
                        std=[
                            77.05962840493828, 69.7348132715338,
                            68.29496534260919
                        ],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
work_dir = '/root/autodl-tmp/final_segformer_b5'
gpu_ids = [0]
auto_resume = False
