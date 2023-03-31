# @FileName：Demo27.py
# @Description：
# @Author：dyh
# @Time：2023/3/9 13:57
# @Website：www.xxx.com
# @Version：V1.0
import base64
import io
import os

import cv2
import einops as einops
import numpy
import numpy as np
import torch
from PIL import Image, ImageChops

from utils.util import resize_image, HWC3


# canny
# im = cv2.imread('./img/xiangao.png')
# canny = cv2.Canny(im, 100, 70)
# cv2.imwrite('./img/xiangao-wireframe.png', canny)

#
def base64_to_img(img_base64):
    image = Image.open(io.BytesIO(base64.b64decode(img_base64))).convert('RGB')
    return image


def img_array_to_base64(img):
    buff = io.BytesIO()
    Image.fromarray(img).convert('RGB').save(buff, format="JPEG")
    img_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_base64


im = Image.open('./img/1679638479361.jpg').convert("RGB")
im = im.resize((im.size[0] // 4, im.size[1] // 4))
im.save('./img/1679638479361_small.jpg')
