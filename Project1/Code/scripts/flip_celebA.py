"""
@Author:    Li Hantao
@Date:      2022.3.10
@Description:
    This script is to flip the image and their label without 
    changing the L/R eye or brow or ears' direction.
    Which means, for the mask, we first flip the mask, then recover
    the label which has L/R attributes.
@Para:
    PALETTE -> The color in RGB which corresponding to the 19 labels.
    image_root -> The path of the input images.
                  The name of them should be 0.png, 1.png, ...., image_num-1.png
    save_root -> The saving path of the output color images.
    image_num -> The number of images in the input images.
"""

from PIL import Image
import numpy as np

def flip_celebA(img):
    """
    Flip the mask image except the label which has L/R attribute.
    """
    im_f = np.fliplr(img)
    im_f.flags.writeable = True
    for r in range(len(im_f)):
        for c in range(len(im_f[0])):
            if im_f[r][c] == 4: # 'L/R eye'
                im_f[r][c] = 5
            elif im_f[r][c] == 5:
                im_f[r][c] = 4
            elif im_f[r][c] == 6: # 'L/R brow'
                im_f[r][c] = 7
            elif im_f[r][c] == 7:
                im_f[r][c] = 6   
            elif im_f[r][c] == 8: # 'L/R ear'
                im_f[r][c] = 9
            elif im_f[r][c] == 9:
                im_f[r][c] = 8
    return im_f

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
    image_path = image_root + str(i) + '.jpg'
    save_path = image_root + str(i + 10000) + '.jpg'
    flip_save_path = save_root + str(i + 10000 + image_num) + '.jpg'
    img = Image.open(image_path)
    img.save(save_path)

    # for .jpg images
    # flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # for .png masks
    flip_img = np.array(img)
    flip_img = flip_celebA(flip_img)
    flip_img = Image.fromarray(flip_img).convert("P")
    flip_img.putpalette(colors)

    flip_img.save(flip_save_path)