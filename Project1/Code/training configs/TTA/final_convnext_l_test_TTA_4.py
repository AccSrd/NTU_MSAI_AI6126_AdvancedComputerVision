norm_cfg = dict(type='BN', requires_grad=True)
custom_imports = dict(imports='mmcls.models', allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-xlarge_3rdparty_in21k_20220301-08aa5ddc.pth'
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='xlarge',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-xlarge_3rdparty_in21k_20220301-08aa5ddc.pth',
            prefix='backbone.')),
    decode_head=dict(
        type='UPerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=[
                    0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1, 1, 1,
                    0.95, 1.05, 1.5, 2, 1, 1.05
                ]),
            dict(
                 type='LovaszLoss',
                 loss_type='multi_class',
                 reduction='none',
                 loss_weight=2.0,
                 loss_name='loss_lovasz')]),
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
                use_sigmoid=False,
                loss_weight=0.4,
                class_weight=[
                    0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1, 1, 1,
                    0.95, 1.05, 1.5, 2, 1, 1.05
                ]),
            dict(
                 type='LovaszLoss',
                 loss_type='multi_class',
                 reduction='none',
                 loss_weight=0.8,
                 loss_name='loss_lovasz')]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CelebADataset'
data_root = 'data/CelebA/'
img_norm_cfg = dict(
    mean=[132.52878295898438, 106.67704703776042, 92.83904142252604],
    std=[77.05962840493828, 69.7348132715338, 68.29496534260919],
    to_rgb=True)
img_scale = (512, 512)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=25, pad_val=0, seg_pad_val=0),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(1, 1.2)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
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
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[132.52878295898438, 106.67704703776042, 92.83904142252604],
                std=[77.05962840493828, 69.7348132715338, 68.29496534260919],
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
        img_dir='image/complete_train',
        ann_dir='mask/complete_train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomRotate', prob=0.5, degree=25, pad_val=0, seg_pad_val=0),
            dict(type='Resize', img_scale=(512, 512), ratio_range=(1, 1.2)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
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
                img_scale=crop_size,
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[132.52878295898438, 106.67704703776042, 92.83904142252604],
                        std=[77.05962840493828, 69.7348132715338, 68.29496534260919],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CelebADataset',
        data_root='data/CelebA/',
        img_dir='image/test',
        ann_dir='mask/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[132.52878295898438, 106.67704703776042, 92.83904142252604],
                        std=[77.05962840493828, 69.7348132715338, 68.29496534260919],
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
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=8e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=12))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
fp16 = dict()
work_dir = '/root/autodl-tmp/final_convnext_l'
gpu_ids = [0]
auto_resume = False
