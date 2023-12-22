# @FileName：WaterMarkTest.py
# @Description：
# @Author：dyh
# @Time：2023/9/21 15:06
# @Website：www.xxx.com
# @Version：V1.0
import json
import os
import random
import time
from datetime import datetime

import cv2
import numpy
from PIL import Image
from imwatermark import WatermarkDecoder, WatermarkEncoder


def get_encoder(crop_img, filename):
    bgr = cv2.cvtColor(numpy.asarray(crop_img), cv2.COLOR_RGB2BGR)
    wm = '{"ServiceProvider": "大设AIGC", "Time": "2023-10-09 11:55:09.525", "ContentID": "000000005099746_11_1"}'
    s = time.time()
    encoder = WatermarkEncoder()
    encoder.loadModel()
    encoder.set_watermark('bytes', wm.encode('utf-8'))
    block = 4
    scales = [0, 80, 0]
    # bgr_encoded = encoder.encode(bgr, 'rivaGan')
    bgr_encoded = encoder.encode(bgr, 'dwtDctSvd', block=block, scales=scales)
    print(f'consumting time = {time.time() - s} , length = {encoder.get_length()}')
    name = filename.split('.')[0]
    cv2.imwrite('../test-img-result/' + name + '.jpeg', bgr_encoded, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return bgr_encoded


def get_decoder(bgr_encoded, length):
    decoder = WatermarkDecoder('bytes', length)
    watermark = decoder.decode(bgr_encoded, 'dwtDctSvd')
    return watermark.decode('utf-8')


img_dir = '../img/watermark/'
filename_lst = os.listdir(img_dir)

for filename in filename_lst:
    filepath = img_dir + filename
    print(filepath)
    img = Image.open(filepath)
    encoder_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)

    print(encoder_img.shape[:2])
    mark = get_decoder(encoder_img, 832)
    if '{"ServiceProvider": "大设AIGC", "Time": "2023-10-0911:55:09.525", "ContentID": "000000005099746_11_1"}' != mark:
        print('error')
    print(mark)
