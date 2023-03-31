# @FileName：FackScribble.py
# @Description：
# @Author：dyh
# @Time：2023/3/20 11:13
# @Website：www.xxx.com
# @Version：V1.0
import sys
import time

import numpy as np
from PIL import Image
from transformers import pipeline

depth_estimator = pipeline('depth-estimation')

image = Image.open('../test_img/' + sys.argv[1])
s = time.time()
image = depth_estimator(image)['depth']
image = np.array(image)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)
spend = time.time() - s
print(spend)
image.save('depth_scribble_out.png')