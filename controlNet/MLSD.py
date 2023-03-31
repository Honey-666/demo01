# @FileName：FackScribble.py
# @Description：
# @Author：dyh
# @Time：2023/3/20 11:13
# @Website：www.xxx.com
# @Version：V1.0
import sys
import time

from PIL import Image
from controlnet_aux import MLSDdetector

mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

image = Image.open('../test_img/' + sys.argv[1])
s = time.time()
image = mlsd(image)
spend = time.time() - s
print(spend)
image.save('mlsd_scribble_out.png')
