# @FileName：FackScribble.py
# @Description：
# @Author：dyh
# @Time：2023/3/20 11:13
# @Website：www.xxx.com
# @Version：V1.0
import sys
import time

from PIL import Image
from controlnet_aux import HEDdetector
from controlnet_aux.hed import Network
from fastapi import Body

network_model = Network('../../models/annotator/ckpts/network-bsds500.pth')
hed = HEDdetector(network_model)
image = Image.open('../img/bag.png')
s = time.time()
image = hed(image, scribble=True)
spend = time.time() - s
print(spend)
image.save('scribble_scribble_out.png')

