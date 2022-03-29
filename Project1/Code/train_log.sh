#!/bin/sh

# Baseline: resnet50 DeeplabV3+ 20000iter 0.5RandomFlip
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_baseline_config.py --work-dir /root/autodl-tmp/baseline --seed 210
# python ./tools/test.py /root/autodl-tmp/baseline/my_baseline_config.py /root/autodl-tmp/baseline/latest.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/baseline/results_random"
# python evaluate.py

# 本次流程中第二次，也就是没有random的输出是有random flip的
# python ./tools/test.py /root/autodl-tmp/baseline/my_baseline_config.py /root/autodl-tmp/baseline/latest.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/baseline/results"
# python evaluate.py

# Baseline_noflip: resnet50 DeeplabV3+ 20000iter
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_baseline_noflip_config.py --work-dir /root/autodl-tmp/baseline_noflip --seed 210

# Baseline: resnet50 DeeplabV3+ 20000iter Focal:Dice 1:2
#         dict(type='FocalLoss', loss_name='loss_focal',loss_weight=1.0, class_weight=[0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1, 1, 1, 0.95, 1.05, 1.5, 2, 1, 1.05]),
#         dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)]),
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_baseline_noflip_config.py --work-dir /root/autodl-tmp/baseline_F1D2 --seed 210

# Baseline: resnet50 DeeplabV3+ 20000iter CE:FL 1:1
#         dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1, 1, 1, 0.95, 1.05, 1.5, 2, 1, 1.05]),
#         dict(type='FocalLoss', loss_name='loss_focal', loss_weight=1.0, gamma=3., alpha=0.25)]),
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_baseline_noflip_config.py --work-dir /root/autodl-tmp/baseline_C1F1 --seed 210

# Baseline: resnet50 DeeplabV3+ 20000iter CE:FL 1:2
#        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1, 1, 1, 0.95, 1.05, 1.5, 2.5, 1, 1.05]),
#        dict(type='FocalLoss', loss_name='loss_focal', loss_weight=2.0, gamma=3., alpha=0.25)]
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_baseline_noflip_config.py --work-dir /root/autodl-tmp/baseline_C1F2 --seed 210

# Baseline: resnet50 DeeplabV3+ 20000iter CE:FL 1:10
#        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1, 1, 1, 0.95, 1.05, 1.5, 8, 1, 1.05]),
#        dict(type='FocalLoss', loss_name='loss_focal', loss_weight=10.0, gamma=3., alpha=0.25)]),
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_baseline_noflip_config.py --work-dir /root/autodl-tmp/baseline_C1F10 --seed 210 && shutdown

# Baseline: resnet50 DeeplabV3+ 20000iter CE:FL 1:0
#        dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.95, 0.95, 0.95, 1, 1, 1, 1.05, 1.05, 1.05, 1.05, 1, 1, 1, 0.95, 1.05, 1.5, 2, 1, 1.05])
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_baseline_noflip_config.py --work-dir /root/autodl-tmp/baseline_C1F0 --seed 210

# Baseline: largeset resnet50 DeeplabV3+ 40000iter CE:FL 1:0 Crop dict(type='Resize', img_scale=(620, 512), ratio_range=(0.8, 1.2)),
    # dict(type='RandomCrop', crop_size=img_scale, cat_max_ratio=0.75),
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_baseline_noflip_config.py --work-dir /root/autodl-tmp/baseline_Crop_620-512 --seed 210

# Baseline: oriset resnet50 DeeplabV3+ 40000iter CE:FL 1:0 Crop dict(type='Resize', img_scale=(620, 512), ratio_range=(0.8, 1.2)),
    # dict(type='RandomCrop', crop_size=img_scale, cat_max_ratio=0.75),
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_baseline_noflip_config.py --work-dir /root/autodl-tmp/baseline_Crop_620-512_oriset --seed 210

# ce_w 1, 0.4, 
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_baseline_noflip_config.py --work-dir /root/autodl-tmp/baseline_cew_oriset --seed 210


# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_CFD.py --work-dir /root/autodl-tmp/base_partJ_CFD --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_CF.py --work-dir /root/autodl-tmp/base_partJ_CF --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_C1L0.5.py --work-dir /root/autodl-tmp/base_partJ_C1L0.5 --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_C1L1.py --work-dir /root/autodl-tmp/base_partJ_C1L1 --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_C1L2.py --work-dir /root/autodl-tmp/base_partJ_C1L2 --seed 210 && shutdown

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_CWhigh.py --work-dir /root/autodl-tmp/base_partJ_CWhigh --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ.py --work-dir /root/autodl-tmp/base_partJ --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_noJ.py --work-dir /root/autodl-tmp/base_noJ --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_noW.py --work-dir /root/autodl-tmp/base_partJ_noW --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rC.py --work-dir /root/autodl-tmp/base_partJ_rC --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_large.py --work-dir /root/autodl-tmp/base_partJ_large --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rotate.py --work-dir /root/autodl-tmp/base_partJ_rotate --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rotate_25.py --work-dir /root/autodl-tmp/base_partJ_rotate_25_new --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rotate_25_large.py --work-dir /root/autodl-tmp/base_partJ_rotate_25_large --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rotate_25_large_40k.py --work-dir /root/autodl-tmp/base_partJ_rotate_25_large_40 --seed 210

# [N] python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rotate_25_40k.py --work-dir /root/autodl-tmp/base_partJ_rotate_25_40 --seed 210

# [N] python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rotate_25_large_40k_crop.py --work-dir /root/autodl-tmp/base_partJ_rotate_25_large_40_rc --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rotate_35.py --work-dir /root/autodl-tmp/base_partJ_rotate_35 --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_longer.py --work-dir /root/autodl-tmp/base_partJ_40k --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_r101-v3+.py --work-dir /root/autodl-tmp/r101_deeplab --seed 42

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rotate_25_cutout.py --work-dir /root/autodl-tmp/base_partJ_rotate_25_cutout --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rotate_25_0.5.py --work-dir /root/autodl-tmp/base_partJ_rotate_25_0.5 --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rotate_25_0.3.py --work-dir /root/autodl-tmp/base_partJ_rotate_25_0.3 --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_partJ_rotate_25_cutout_0.3.py --work-dir /root/autodl-tmp/base_partJ_rotate_25_cutout_0.3 --seed 210


# ===== ResNeSt =====

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_rs101-v3+.py --work-dir /root/autodl-tmp/rs101_deeplab --seed 42


# ===== PointRend
# r101
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_pointrend_r101.py --work-dir /root/autodl-tmp/point_r101_10k --seed 42




# ===== Swin =====
# swin - b - 5k test
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_swin_config.py --work-dir /root/autodl-tmp/swin_b_5k --seed 210

# swin - b - 5k test with weight
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_swin_config.py --work-dir /root/autodl-tmp/swin_b_5k_weight --seed 210

# swin - b - 5k test with weight ,laege
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_swin_config.py --work-dir /root/autodl-tmp/swin_b_5k_w_large --seed 210 && shutdown

# swin - b - 160k - large dataset - C1F1 - batch16
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_swin_config.py --work-dir /root/autodl-tmp/swin_b_160k_c1f1_b16 --seed 210

# swin - l - 160k - large - CE_W, PJ
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_swin_l_mmpara.py --work-dir /root/autodl-tmp/swin_l_160k_cw_pj_mmpara --seed 210


# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_swin_small.py --work-dir /root/autodl-tmp/swin_b_160k_c1f1_b16_small --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_swin_crop.py --work-dir /root/autodl-tmp/swin_b_160k_c1f1_b16_crop --seed 210





# ===== SegFormer ======
# python tools/model_converters/mit2mmseg.py /root/autodl-tmp/pretrain/mit_b5_ori.pth /root/autodl-tmp/pretrain/mit_b5.pth

# segformer - 160k - C1F1 - batch 4
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_segformer_config.py --work-dir /root/autodl-tmp/segformer_b_160k_c1f1_b4 --seed 210

# segformer - 160k - C1F1 - batch 16 [Not Finish!]
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_segformer_config.py --work-dir /root/autodl-tmp/segformer_b_160k_c1f1_b16 --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_segformer_160k_cew_pJ_b4.py --work-dir /root/autodl-tmp/segformer_160k_cew_pJ_b4 --seed 210



# ===== ResNet 101 + DeepLabV3+ - 80k - batch 16
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_baseline_noflip_config.py --work-dir /root/autodl-tmp/r101_deeplab_80k_b16 --seed 210


# ===== Convnext - 160k - batch8 
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/configs/my_convnext_config.py --work-dir /root/autodl-tmp/convext_160k_b8 --seed 210

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/my_convext_160k_pJ_b8.py --work-dir /root/autodl-tmp/convext_160k_pJ_b8 --seed 210


# ===== K-net ====
# python /root/mmsegmentation/tools/train.py /root/autodl-tmp/knet/my_knet.py --work-dir /root/autodl-tmp/knet --seed 210



# Final Train: ===============
# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/final/final_swin_b.py --work-dir /root/autodl-tmp/final_swin_b --seed 42

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/final/final_swin_l.py --work-dir /root/autodl-tmp/final_swin_l --seed 42

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/final/final_segformer_b5.py --work-dir /root/autodl-tmp/final_segformer_b5 --seed 42

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/final/final_convnext.py --work-dir /root/autodl-tmp/final_convnext --seed 42

# python /root/mmsegmentation/tools/train.py /root/mmsegmentation/final/final_convnext_new.py --work-dir /root/autodl-tmp/final_convnext_new --seed 42

python /root/mmsegmentation/tools/train.py /root/mmsegmentation/final/final_convnext_l.py --work-dir /root/autodl-tmp/final_convnext_l --resume-from /root/autodl-tmp/final_convnext_l/iter_144000.pth --seed 42


