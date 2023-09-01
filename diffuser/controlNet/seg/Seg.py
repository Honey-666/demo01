# @FileName：FackScribble.py
# @Description：
# @Author：dyh
# @Time：2023/3/20 11:13
# @Website：www.xxx.com
# @Version：V1.0
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from controlnet_utils import ade_palette

image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

# image = Image.open('../test_img/' + sys.argv[1]).convert("RGB")
image = Image.open('../img/20230407102039.png').convert("RGB")
s = time.time()
pixel_values = image_processor(image, return_tensors="pt").pixel_values

with torch.no_grad():
    outputs = image_segmentor(pixel_values)

seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3

palette = np.array(ade_palette())

for label, color in enumerate(palette):
    color_seg[seg == label, :] = color

color_seg = color_seg.astype(np.uint8)
color_seg = cv2.resize(color_seg, (1984, 1280), interpolation=cv2.INTER_NEAREST)
image = Image.fromarray(color_seg)
spend = time.time() - s
print(spend)

image.save('seg_scribble_out_1.png')
