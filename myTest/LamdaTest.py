# @FileName：LamdaTest.py
# @Description：
# @Author：dyh
# @Time：2023/4/19 15:17
# @Website：www.xxx.com
# @Version：V1.0
import decimal

import cv2
import numpy as np
import torch
from PIL import Image

# img_w, img_h = image.size
# my_logger.info('img size：%d ,%d', img_w, img_h)
# if img_w > w or img_h > h:
#     ratio_w = img_w / w
#     ratio_h = img_h / h
#     ratio = max(ratio_w, ratio_h)
#     img_w = int(img_w / ratio)
#     img_h = int(img_h / ratio)
#     my_logger.info('img handle size：%d ,%d', img_w, img_h)
#     image = image.resize((img_w, img_h))

# im = cv2.imread('./768.jpg')
# h, w, c = im.shape
# im = cv2.resize(im, (768, 1024), interpolation=cv2.INTER_LINEAR)
# cv2.imwrite('768x1024.jpg', im)
output = cv2.imread('../img/control/boy.png')
outscale = 2
w_input = 512
h_input = 512
output = cv2.resize(output, (int(w_input * outscale), int(h_input * outscale)), interpolation=cv2.INTER_LANCZOS4)
cv2.imshow('demo', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
