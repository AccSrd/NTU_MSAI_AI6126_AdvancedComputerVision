norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    decode_head=[
        dict(
            type='FPNHead',
            in_channels=[256, 256, 256, 256],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=-1,
            num_classes=19,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[
                    0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1, 1, 1,
                    0.95, 1.05, 1.5, 2, 1, 1.05
                ])),
        dict(
            type='PointHead',
            in_channels=[256],
            in_index=[0],
            channels=256,
            num_fcs=3,
            coarse_pred_each_layer=True,
            dropout_ratio=-1,
            num_classes=19,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[
                    0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1, 1, 1,
                    0.95, 1.05, 1.5, 2, 1, 1.05
                ]))
    ],
    train_cfg=dict(
        num_points=2048, oversample_ratio=3, importance_sample_ratio=0.75),
    test_cfg=dict(
        mode='whole',
        subdivision_steps=2,
        subdivision_num_points=8196,
        scale_factor=2))
dataset_type = 'CelebADataset'
data_root = 'data/CelebA/'
img_norm_cfg = dict(
    mean=[132.461776171875, 106.68389479166666, 92.87505299479167],
    std=[77.23084617170329, 69.89919500532964, 68.40051107601049],
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
        mean=[132.461776171875, 106.68389479166666, 92.87505299479167],
        std=[77.23084617170329, 69.89919500532964, 68.40051107601049],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[132.59228645833332, 106.38666015625, 92.44869856770833],
                std=[76.45734671058177, 69.33311870522137, 68.2494277844337],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CelebADataset',
        data_root='data/CelebA/',
        img_dir='image/large_train',
        ann_dir='mask/large_train',
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
                mean=[132.461776171875, 106.68389479166666, 92.87505299479167],
                std=[77.23084617170329, 69.89919500532964, 68.40051107601049],
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
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[
                            132.59228645833332, 106.38666015625,
                            92.44869856770833
                        ],
                        std=[
                            76.45734671058177, 69.33311870522137,
                            68.2494277844337
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
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[
                            132.59228645833332, 106.38666015625,
                            92.44869856770833
                        ],
                        std=[
                            76.45734671058177, 69.33311870522137,
                            68.2494277844337
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
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=0.0001,
    by_epoch=False,
    warmup='linear',
    warmup_iters=200)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True)
work_dir = '/root/autodl-tmp/point_r101_10k'
gpu_ids = [0]
auto_resume = False
