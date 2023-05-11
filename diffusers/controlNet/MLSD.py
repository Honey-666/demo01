# @FileName：FackScribble.py
# @Description：
# @Author：dyh
# @Time：2023/3/20 11:13
# @Website：www.xxx.com
# @Version：V1.0
import sys
import time

import torch
from PIL import Image
from controlnet_aux import MLSDdetector
from controlnet_aux.mlsd import MobileV2_MLSD_Large

mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
# model_path = '../../models/annotator/ckpts/mlsd_large_512_fp32.pth'
# model = MobileV2_MLSD_Large()
# model.load_state_dict(torch.load(model_path), strict=True)
# mlsd = MLSDdetector(model)

# image = Image.open('../test_img/' + sys.argv[1])
image = Image.open('../img/a.jpg')
s = time.time()
image = mlsd(image)
spend = time.time() - s
print(spend)
image.save('mlsd_scribble_out.png')
