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
from transformers import pipeline
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


im = cv2.imread('oss_img.png')
im = resize_image(im, 512)
im.save('oss_img2.png')
# image = Image.open('../test_img/' + sys.argv[1])

# depth_estimator = pipeline('depth-estimation', model='Intel/dpt-hybrid-midas')
# depth_estimator = pipeline('depth-estimation', model='Intel/dpt-large')
# depth_estimator = pipeline('depth-estimation')


image = Image.open('../../img/control/house.png')
s = time.time()
depth = MidasDetector(model_path='C:/Users/bbw/.cache/huggingface/hub/models--lllyasviel--ControlNet/snapshots/e78a8c4a5052a238198043ee5c0cb44e22abb9f7/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt')
# depth = MidasDetector.from_pretrained('lllyasviel/ControlNet')

image = depth(input_image=image)
print(time.time() - s)
image.save('depth_scribble_out.png')


# s = time.time()
# image = depth_estimator(image)['depth']
# image = np.array(image)
# image = image[:, :, None]
# image = np.concatenate([image, image, image], axis=2)
# image = Image.fromarray(image)
# spend = time.time() - s
# print(spend)
