# @FileName：RealESRGANTest.py
# @Description：
# @Author：dyh
# @Time：2023/7/12 15:08
# @Website：www.xxx.com
# @Version：V1.0
import os

import cv2
import matplotlib.pyplot as plt


def load_upscale_model():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    upscale = RealESRGANer(
        scale=4,  # 这里的值不能修改不然放大一倍或两倍图片会有问题
        model_path='C:\\work\\pythonProject\\aidazuo\\models\\ESRGAN\\RealESRGAN_x4plus.pth',
        model=model,
        dni_weight=None,
        tile=400,
        half=True,
        tile_pad=10,
        pre_pad=0)

    return upscale


filename_lst = os.listdir('../img/watermark-result')
bg_upsampler = load_upscale_model()
for filename in filename_lst:
    filepath = '../img/watermark-result/' + filename
    print(filepath)
    input_img = cv2.imread(filepath)

    restored_img, _ = bg_upsampler.enhance(input_img, outscale=2)

    print(restored_img.shape)
    name = filename.split('.')[0]
    cv2.imwrite('../img/watermark-result-upscale/2/' + name + '.jpg', restored_img)


