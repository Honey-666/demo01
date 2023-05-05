# @FileName：midasTest.py
# @Description：
# @Author：dyh
# @Time：2023/4/10 10:24
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

from test.api import MiDaSInference
from test.dpt_depth import MidasDetector

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)
model_type = "DPT_Hybrid"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
img = cv2.imread('dog.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()
cv2.imshow('demo',output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ----------------------------------------------------------
# apply_midas = MidasDetector()
# img = cv2.imread('dog.jpg')
# detected_map, _ = apply_midas(img)
# print(detected_map)