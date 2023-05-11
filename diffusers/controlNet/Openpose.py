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
image = Image.open('../test_img/' + sys.argv[1])
model_body = Body('../../models/annotator/ckpts/body_pose_model.pth')
# openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
openpose = OpenposeDetector(model_body)
image = openpose(image)
spend = time.time() - s
print(spend)
image.save('pose_scribble_out.png')
