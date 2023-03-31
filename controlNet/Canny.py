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

low_threshold = 100
high_threshold = 200

image = cv2.imread('../test_img/' + sys.argv[1])
s = time.time()
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)
spend = time.time() - s
print(spend)
image.save('canny_scribble_out.png')
