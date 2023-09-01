# @FileName：FackScribble.py
# @Description：
# @Author：dyh
# @Time：2023/3/20 11:13
# @Website：www.xxx.com
# @Version：V1.0
import sys
import time

import cv2
import numpy as np
from PIL import Image
from controlnet_aux.util import HWC3

from utils.util import resize_image

# def pad64(x):
#     return int(np.ceil(float(x) / 64.0) * 64 - x)
#
# def safer_memory(x):
#     # Fix many MAC/AMD problems
#     return np.ascontiguousarray(x.copy()).copy()
# def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
#     if skip_hwc3:
#         img = input_image
#     else:
#         img = HWC3(input_image)
#     H_raw, W_raw, _ = img.shape
#     k = float(resolution) / float(min(H_raw, W_raw))
#     interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
#     H_target = int(np.round(float(H_raw) * k))
#     W_target = int(np.round(float(W_raw) * k))
#     img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
#     H_pad, W_pad = pad64(H_target), pad64(W_target)
#     img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')
#
#     def remove_pad(x):
#         return safer_memory(x[:H_target, :W_target])
#
#     return safer_memory(img_padded), remove_pad


low_threshold = 100
high_threshold = 200

image = cv2.imread('../../../img/control/20230728-154401.jpg')
s = time.time()
image = HWC3(image)
image = resize_image(image, 512)
# image, remove_pad = resize_image_with_pad(image, 512)
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)
spend = time.time() - s
print(spend)
image.save('control_img.png')
