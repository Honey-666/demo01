# @FileName：FackScribble.py
# @Description：
# @Author：dyh
# @Time：2023/3/20 11:13
# @Website：www.xxx.com
# @Version：V1.0
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from controlnet_aux import ZoeDetector
from controlnet_aux.zoe import ZoeDepthNK, get_config
from controlnet_aux.midas import MidasDetector


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def img_midas(img, pretrained_model_or_path, detect_resolution, image_resolution):
    midas_depth = MidasDetector.from_pretrained(pretrained_model_or_path).to('cuda')
    image = midas_depth(img, detect_resolution=detect_resolution, image_resolution=image_resolution)
    return image


def img_zoe(img, pretrained_model_or_path, detect_resolution, image_resolution):
    conf = get_config('zoedepth_nk', "infer")
    prefix = 'local::'
    pretrained_resource = os.path.join(pretrained_model_or_path, 'ZoeD_M12_NK.pt')
    conf['pretrained_resource'] = prefix + pretrained_resource
    model = ZoeDepthNK.build_from_config(conf)
    model_path = os.path.join(pretrained_model_or_path, 'zoed_nk.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    model.eval()

    zoe_depth = ZoeDetector(model).to('cuda')
    image = zoe_depth(img, detect_resolution=detect_resolution, image_resolution=image_resolution)
    return image


base_model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts'
image = Image.open('../../../img/control/room.png')
image2 = Image.open('../../../img/control/room.png')

s = time.time()
handle_img1 = img_midas(base_model_path, image, 512)
handle_img2 = img_zoe(base_model_path, image, 512)
handle_img1.show()
handle_img2.show()
print(time.time() - s)
