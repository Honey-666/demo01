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
from controlnet_aux.hed import ControlNetHED_Apache2

# from controlnet_aux.hed import Network

# netNetwork = ControlNetHED_Apache2()
# hed = HEDdetector(netNetwork)
# network_model = Network('../../models/annotator/ckpts/network-bsds500.pth')
hed = HEDdetector.from_pretrained("C:\\Users\\bbw\\.cache\\huggingface\\hub\\models--lllyasviel--Annotators\\snapshots\\9a7d84251d487d11c4834466779de6b0d2c44486")
image = Image.open('../../../img/control/bag.png')
s = time.time()
image = hed(image, scribble=True)
spend = time.time() - s
print(spend)
image.save('scribble_scribble_out.png')

