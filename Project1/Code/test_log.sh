# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_n.py /root/autodl-tmp/final_convnext_new/iter_104000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/104"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_56000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_56"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_TTA_4.py /root/autodl-tmp/final_convnext_new/iter_104000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/104_TTA"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_n.py /root/autodl-tmp/final_convnext_new/iter_112000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/112"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_TTA_4.py /root/autodl-tmp/final_convnext_new/iter_112000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/112_TTA"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_n.py /root/autodl-tmp/final_convnext_new/iter_120000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/120"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_TTA_4.py /root/autodl-tmp/final_convnext_new/iter_120000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/120_TTA"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_n.py /root/autodl-tmp/final_convnext_new/iter_128000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/128"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_TTA_4.py /root/autodl-tmp/final_convnext_new/iter_128000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/128_TTA"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_n.py /root/autodl-tmp/final_convnext_new/iter_144000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/144"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_TTA_4.py /root/autodl-tmp/final_convnext_new/iter_144000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/144_TTA"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_n.py /root/autodl-tmp/final_convnext_new/iter_152000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/152"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_TTA_4.py /root/autodl-tmp/final_convnext_new/iter_152000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/152_TTA"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_n.py /root/autodl-tmp/final_convnext_new/iter_160000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/160"

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_TTA_4.py /root/autodl-tmp/final_convnext_new/iter_160000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/result/160_TTA"


# mkdir /root/autodl-tmp/final_convnext_l/result/l_56_TTA
# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test_TTA.py /root/autodl-tmp/final_convnext_l/iter_56000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_56_TTA"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_56_TTA.tar /root/autodl-tmp/final_convnext_l/result/l_56_TTA

# mkdir /root/autodl-tmp/final_convnext_l/result/l_48
# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_48000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_48"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_48.tar /root/autodl-tmp/final_convnext_l/result/l_48

# mkdir /root/autodl-tmp/final_convnext_l/result/l_40
# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_40000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_40"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_40.tar /root/autodl-tmp/final_convnext_l/result/l_40

# mkdir /root/autodl-tmp/final_convnext_l/result/l_32
# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_32000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_32"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_32.tar /root/autodl-tmp/final_convnext_l/result/l_32

# mkdir /root/autodl-tmp/final_convnext_l/result/l_24
# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_24000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_24"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_24.tar /root/autodl-tmp/final_convnext_l/result/l_24

# mkdir /root/autodl-tmp/final_convnext_l/result/l_16
# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_16000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_16"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_16.tar /root/autodl-tmp/final_convnext_l/result/l_16

# mkdir /root/autodl-tmp/final_convnext_l/result/l_8
# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_8000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_8"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_8.tar /root/autodl-tmp/final_convnext_l/result/l_8

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_64000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_64"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_64.tar /root/autodl-tmp/final_convnext_l/result/l_64

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_72000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_72"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_72.tar /root/autodl-tmp/final_convnext_l/result/l_72

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_80000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_80"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_80.tar /root/autodl-tmp/final_convnext_l/result/l_80

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_88000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_88"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_88.tar /root/autodl-tmp/final_convnext_l/result/l_88

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_96000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_96"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_96.tar /root/autodl-tmp/final_convnext_l/result/l_96

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_104000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_104"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_104.tar /root/autodl-tmp/final_convnext_l/result/l_104

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_112000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_112"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_112.tar /root/autodl-tmp/final_convnext_l/result/l_112

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_120000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_120"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_120.tar /root/autodl-tmp/final_convnext_l/result/l_120

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_128000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_128"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_128.tar /root/autodl-tmp/final_convnext_l/result/l_128

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_136000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_136"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_136.tar /root/autodl-tmp/final_convnext_l/result/l_136

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_144000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_144"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_144.tar /root/autodl-tmp/final_convnext_l/result/l_144

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_152000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_152"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_152.tar /root/autodl-tmp/final_convnext_l/result/l_152

# python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test.py /root/autodl-tmp/final_convnext_l/iter_160000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_160"
# tar -cvf /root/autodl-tmp/final_convnext_l/result/l_160.tar /root/autodl-tmp/final_convnext_l/result/l_160

python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test_TTA.py /root/autodl-tmp/final_convnext_l/iter_72000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_72_TTA"
tar -cvf /root/autodl-tmp/final_convnext_l/result/l_72_TTA.tar /root/autodl-tmp/final_convnext_l/result/l_72_TTA

python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test_TTA_2.py /root/autodl-tmp/final_convnext_l/iter_72000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_72_TTA2"
tar -cvf /root/autodl-tmp/final_convnext_l/result/l_72_TTA2.tar /root/autodl-tmp/final_convnext_l/result/l_72_TTA2

python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test_TTA_3.py /root/autodl-tmp/final_convnext_l/iter_72000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_72_TTA3"
tar -cvf /root/autodl-tmp/final_convnext_l/result/l_72_TTA3.tar /root/autodl-tmp/final_convnext_l/result/l_72_TTA3

python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test_TTA_4.py /root/autodl-tmp/final_convnext_l/iter_72000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_72_TTA4"
tar -cvf /root/autodl-tmp/final_convnext_l/result/l_72_TTA4.tar /root/autodl-tmp/final_convnext_l/result/l_72_TTA4

python /root/mmsegmentation/tools/test.py /root/mmsegmentation/final/final_convnext_l_test_TTA_5.py /root/autodl-tmp/final_convnext_l/iter_72000.pth --format-only --eval-options "imgfile_prefix=/root/autodl-tmp/final_convnext_l/result/l_72_TTA5"
tar -cvf /root/autodl-tmp/final_convnext_l/result/l_72_TTA5.tar /root/autodl-tmp/final_convnext_l/result/l_72_TTA5

