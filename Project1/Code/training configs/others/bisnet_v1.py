norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='BiSeNetV1',
        in_channels=3,
        context_channels=(512, 1024, 2048),
        spatial_channels=(256, 256, 256, 512),
        out_indices=(0, 1, 2),
        out_channels=1024,
        backbone_cfg=dict(
            type='ResNet',
            in_channels=3,
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 1, 1),
            strides=(1, 2, 2, 2),
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet50_v1c')),
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        init_cfg=None),
    decode_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=0,
        channels=1024,
        num_convs=1,
        concat_input=False,
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
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    loss_name='loss_ce',
                    loss_weight=1.0,
                    class_weight=[
                        0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1,
                        1, 1, 0.95, 1.05, 1.5, 2, 1, 1.05
                    ])
            ]),
        dict(
            type='FCNHead',
            in_channels=512,
            channels=256,
            num_convs=1,
            num_classes=19,
            in_index=2,
            norm_cfg=dict(type='BN', requires_grad=True),
            concat_input=False,
            align_corners=False,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    loss_name='loss_ce',
                    loss_weight=1.0,
                    class_weight=[
                        0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1,
                        1, 1, 0.95, 1.05, 1.5, 2, 1, 1.05
                    ])
            ])
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
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
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=0.0001,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1000)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
work_dir = 'D:\celebA_results\bisnetv1'
gpu_ids = [0]
auto_resume = False
