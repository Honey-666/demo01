# @FileName：LineartPreprocess.py
# @Description：
# @Author：dyh
# @Time：2023/8/22 10:38
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy
from PIL import Image
from controlnet_aux import LineartDetector

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts'
# coarse_model = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts\\sk_model2.pth'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\lineart.png'
preprocess = LineartDetector.from_pretrained(model_path)

img = Image.open(img_path)
img = preprocess(img)

np_img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
cv2.imshow('demo', np_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

