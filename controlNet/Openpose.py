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

s = time.time()
image = Image.open('../test_img/' + sys.argv[1])
openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
image = openpose(image)
spend = time.time() - s
print(spend)
image.save('pose_scribble_out.png')