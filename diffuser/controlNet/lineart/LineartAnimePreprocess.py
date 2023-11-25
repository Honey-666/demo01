# @FileName：LineartPreprocess.py
# @Description：
# @Author：dyh
# @Time：2023/8/22 10:38
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy
from PIL import Image
from controlnet_aux import LineartAnimeDetector
model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts'
# coarse_model = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts\\sk_model2.pth'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\lineart.png'
preprocess = LineartAnimeDetector.from_pretrained(model_path)

img = Image.open(img_path)
img = preprocess(img)

img.show()

