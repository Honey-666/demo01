# @FileName：PidLinetTest.py
# @Description：
# @Author：dyh
# @Time：2023/10/25 16:25
# @Website：www.xxx.com
# @Version：V1.0
from PIL import Image
from controlnet_aux import PidiNetDetector

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\org_sketch.png'
pidNet = PidiNetDetector.from_pretrained(model_path)

img = Image.open(img_path)
handle_img = pidNet(img, 1024, 1024, apply_filter=True)
handle_img.show()



