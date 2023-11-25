# @FileName：LamdaTest.py
# @Description：
# @Author：dyh
# @Time：2023/4/19 15:17
# @Website：www.xxx.com
# @Version：V1.0
import decimal
import os.path
import random
import time
import uuid

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from datetime import datetime, timedelta
import json

str = '000000005099746_11_1'

# hex_num = hex(9999999)
# print(hex_num)

print(int('0xfffffff', 16))


print(type(5099746))

model_id = '../models/Stable-Diffusion/stable-diffusion-xl-1.0-inpainting-0.1/'
print('-xl-' in model_id)
