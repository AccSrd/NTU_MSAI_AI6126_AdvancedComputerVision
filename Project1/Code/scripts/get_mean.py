"""
@Author:    Li Hantao
@Date:      2022.3.1
@Description:
    This script is to calculate the mean and std of images in the path.
@Para:
    image_root -> The path of the images.
"""

import numpy as np
import cv2
import os

imgs_path = 'xxx'

img_h, img_w = 32, 32
means, stdevs = [], []
img_list = []
imgs_path_list = os.listdir(imgs_path)
print(f'There are {len(imgs_path_list)} images in this path.')

for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
 
imgs = np.concatenate(img_list, axis=3)

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
 
# BGR -> RGB
means.reverse()
stdevs.reverse()
 
print("normMean = ", means)
print("normStd = ", stdevs)