# @FileName：ShufflePreprocess.py
# @Description：
# @Author：dyh
# @Time：2023/8/22 10:14
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy
import numpy as np
from PIL import Image
from controlnet_aux import ContentShuffleDetector

img = Image.open('C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\input.jpg')
process = ContentShuffleDetector()
control_image = process(img)
cv2.imshow('demo', cv2.cvtColor(numpy.array(control_image), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
# control_image.show()
