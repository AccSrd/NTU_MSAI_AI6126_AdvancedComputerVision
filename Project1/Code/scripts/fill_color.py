"""
@Author:    Li Hantao
@Date:      2022.3.18
@Description:
    This script is to fill the corresponding color in PALETTE 
    to the mask.png output by ./tools/test.py -- format only command.
    The fill color has no effect on the final Miou result.
@Para:
    PALETTE -> The color in RGB which corresponding to the 19 labels.
    image_root -> The path of the input images.
                  The name of them should be 0.png, 1.png, ...., image_num-1.png
    save_root -> The saving path of the output color images.
    image_num -> The number of images in the input images.
"""

from PIL import Image
import numpy as np

PALETTE = [(0, 0, 0), 
           (204, 0, 0), 
           (76, 153, 0), 
           (204, 204, 0), 
           (51, 51, 255), 
           (204, 0, 204), 
           (0, 255, 255), 
           (255, 204, 204), 
           (102, 51, 0), 
           (255, 0, 0), 
           (102, 204, 0), 
           (255, 255, 0), 
           (0, 0, 153), 
           (0, 0, 204), 
           (255, 51, 153), 
           (0, 204, 204), 
           (0, 51, 0), 
           (255, 153, 51), 
           (0, 204, 0)]
num_classes = 19
colors = np.array([1 for i in range(num_classes)])
colors = colors[:, None] * PALETTE
colors = colors.astype("uint8")

image_num = 1000
image_root = 'xxx'
save_root = 'xxx'

for i in range(image_num):
    image_path = image_root + str(i) + '.png'
    color_image_path = save_root + str(i) + '.png'
    img = Image.open(image_path)
    img.putpalette(colors)
    img.save(color_image_path)
    if (i+1) % 100 == 0: print(f'== Finished for {i+1} images ==')