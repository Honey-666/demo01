# @FileName：FackScribble.py
# @Description：
# @Author：dyh
# @Time：2023/3/20 11:13
# @Website：www.xxx.com
# @Version：V1.0
import sys
import time

from PIL import Image
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose.body import Body

s = time.time()
image = Image.open('./to_img_0.png')
# model_body = Body('../../models/annotator/ckpts/body_pose_model.pth')
# openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
# openpose = OpenposeDetector(model_body)
openpose = OpenposeDetector.from_pretrained(
    'C:\\Users\\bbw\\.cache\\huggingface\\hub\\models--lllyasviel--Annotators\\snapshots\\b23789dd40e7bb6984141590032e21817928d914')
image = openpose(image, include_hand=True)
spend = time.time() - s
print(spend)
image.save('pose_scribble_out.png')
