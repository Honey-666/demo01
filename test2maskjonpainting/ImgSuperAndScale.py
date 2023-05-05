# @FileName：ImgSuperAndScale.py
# @Description：
# @Author：dyh
# @Time：2023/3/17 11:47
# @Website：www.xxx.com
# @Version：V1.0
import math
import os
import sys

import cv2
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer


def img_super(input_img, scale):
    print('img_super scale=', scale)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    bg_upsampler = RealESRGANer(
        scale=4,  # 这里的值不能修改不然放大一倍或两倍图片会有问题
        model_path='weights/RealESRGAN_x4plus.pth',
        dni_weight=None,
        model=model,
        tile=400,
        half=True,
        tile_pad=10,
        pre_pad=0)
    # my_logger.info('task only need super resolution..............')
    restored_img, _ = bg_upsampler.enhance(input_img, outscale=scale)
    return restored_img


args = sys.argv
filename = args[1]
img = cv2.imread(filename)
file_name = args[1].split('.')[0]
print(file_name)
super_img = img_super(img, 2)
cv2.imwrite('super_img.png', super_img)
