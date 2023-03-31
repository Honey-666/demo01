# @FileName：ImgWhiteTest.py
# @Description：
# @Author：dyh
# @Time：2023/3/29 14:21
# @Website：www.xxx.com
# @Version：V1.0
import re

from PIL import Image
models = ['../models/sd-controlnet-scribble','../models/sd-controlnet-canny']
s = re.sub('.*/', '', str(models))
print(s)
