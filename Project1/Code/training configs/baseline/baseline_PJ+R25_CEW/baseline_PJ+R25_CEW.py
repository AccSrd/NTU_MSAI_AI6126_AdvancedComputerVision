norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
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
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
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
                loss_weight=0.4,
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
    mean=[132.461776171875, 106.68389479166666, 92.87505299479167],
    std=[77.23084617170329, 69.89919500532964, 68.40051107601049],
    to_rgb=True)
img_scale = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0),
    dict(type='RandomRotate', prob=0.5, degree=25, pad_val=0, seg_pad_val=0),
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
        img_dir='image/train',
        ann_dir='mask/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomFlip', prob=0),
            dict(
                type='RandomRotate',
                prob=0.5,
                degree=25,
                pad_val=0,
                seg_pad_val=0),
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
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=20000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
work_dir = '/root/autodl-tmp/base_partJ_rotate_25'
gpu_ids = [0]
auto_resume = False
