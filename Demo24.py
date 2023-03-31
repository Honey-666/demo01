# @FileName：Demo24.py
# @Description：
# @Author：dyh
# @Time：2023/2/3 10:39
# @Website：www.xxx.com
# @Version：V1.0
import datetime
import io
import json
import os.path
import unittest

import cv2
import numpy as np
import requests
from PIL import Image


class ParamClass:
    def __init__(self, c, a=1, b=2, ):
        self.a = a
        self.b = b
        self.c = c

    def printParam(self):
        print('a=', self.a, ' b=', self.b, ' c=', self.c)


param = ParamClass(a=7, c=8)
param.printParam()

num = None
if num:
    print('不为空')

ori_img_url = 'https://ai-pin.oss-cn-hangzhou-internal.aliyuncs.com/ai-pin/326-8f4ac7ed-75f9-11ed-9680-00163e1fcedf.png'

str_json = {"result_id": 1,
            "oss_url": "https://ai-pin.oss-cn-hangzhou-internal.aliyuncs.com/ai-pin/326-8f4ac7ed-75f9-11ed-9680-00163e1fcedf.png"}

obj_json = json.loads(json.dumps(str_json))
print(obj_json["result_id"])
print(obj_json["oss_url"])

img = cv2.imread('img/61eb5ed3-a2c8-11ed-9f31-52540068821e.jpg', cv2.IMREAD_COLOR)
print(img)


def get_oss_img(url):
    # wei: change to inner network address
    print("Downloading: ", url)
    response = requests.get(url, timeout=(3, 20))
    response.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(response.content)))


print('--------------------------------------------------------------------')
img2 = get_oss_img(
    'https://apin.bigurl.ink/ai-pin/7d5443d5-a6c8-11ed-9754-52540068821e.png?Expires=1676702290&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=OXdDTFORaBBv%2F16hDNdBDVG3V1Q%3D&x-oss-process=imageimage%2Fformat%2Cjpg')
pli_restored_img = Image.fromarray(np.uint8(img2))
pli_restored_img.save("74c25ca8-a6c2-11ed-828c-52540068821e.jpg")
print(pli_restored_img)
print(np.asarray(pli_restored_img))
img2 = cv2.cvtColor(np.asarray(pli_restored_img), cv2.COLOR_BGR2RGB)
print(img2)
# success, encoded_image = cv2.imencode(".jpg", img2)
#
# with open('test2.jpg', 'wb') as f:
#     f.write(encoded_image.tobytes())



