# baseline
# python /root/mmediting/tools/train.py /root/mmediting/configs/baseline.py --work-dir /root/autodl-tmp/baseline --seed 42

# baseline - 20 block; batch 16; 300k
# python /root/mmediting/tools/train.py /root/mmediting/configs/MSR_20_b16_300k.py --work-dir /root/autodl-tmp/MSR_20_b16_300k --seed 42

# python /root/mmediting/tools/train.py /root/mmediting/configs/MSR_20_b8_1m.py --work-dir /root/autodl-tmp/MSR_20_b8_1m --seed 42

# python /root/mmediting/tools/train.py /root/mmediting/configs/MSR_20_b8_d01_1m.py --work-dir /root/autodl-tmp/MSR_20_b8_d01_1m --seed 42

# python /root/mmediting/tools/train.py /root/mmediting/configs/MSR_20_b8_d07_1m.py --work-dir /root/autodl-tmp/MSR_20_b8_d07_1m --seed 42

# python /root/mmediting/tools/train.py /root/mmediting/configs/MSR_20_b8_cutb_1m.py --work-dir /root/autodl-tmp/MSR_20_b8_cutb_1m --seed 42

# python /root/mmediting/tools/train.py /root/mmediting/configs/MSR_20_b8_d01_1m.py --work-dir /root/autodl-tmp/MSR_20_b8_d01_1m --seed 42

# python /root/mmediting/tools/train.py /root/mmediting/configs/EDSR_.py --work-dir /root/autodl-tmp/EDSR_ --seed 42

#python /root/mmediting/tools/train.py /root/mmediting/configs/MyEDSR.py --work-dir /root/autodl-tmp/MyEDSR --seed 42

# python tools/test.py /root/mmediting/configs/MyEDSR.py /root/autodl-tmp/MyEDSR/iter_5000.pth --save-path /root/autodl-tmp/MyEDSR/result

# python tools/test.py /root/mmediting/configs/MSR_20_b8_1m.py /root/autodl-tmp/MSR_20_b8_1m/iter_1000000.pth --save-path /root/autodl-tmp/MSR_20_b8_1m/result

python /root/mmediting/tools/train.py /root/mmediting/configs/FinalConfig.py --work-dir /root/autodl-tmp/Final --seed 42