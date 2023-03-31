# @FileName：ControlNet.py
# @Description：
# @Author：dyh
# @Time：2023/3/17 14:44
# @Website：www.xxx.com
# @Version：V1.0
import io
import json
import sys
import time
from time import sleep

import cv2
import numpy

import torch
import inspect
from python_bigbigwork_util import DefaultNacos, LoggingUtil
from PIL import Image, ImageChops
from stable_diffusion_multi_controlnet import StableDiffusionMultiControlNetPipeline, ControlNetProcessor
from diffusers import (ControlNetModel)

my_logger = LoggingUtil.get_logging()

pipe = StableDiffusionMultiControlNetPipeline.from_pretrained(
    "../models/stable-diffusion-v1-5", safety_checker=None, torch_dtype=torch.float16
).to("cuda")
canny = ControlNetModel.from_pretrained("../models/sd-controlnet-canny", torch_dtype=torch.float16).to(
    "cuda")
scribble = ControlNetModel.from_pretrained("../models/sd-controlnet-scribble",
                                           torch_dtype=torch.float16).to("cuda")
hed = ControlNetModel.from_pretrained("../models/sd-controlnet-hed", torch_dtype=torch.float16).to("cuda")
depth = ControlNetModel.from_pretrained("../models/sd-controlnet-depth", torch_dtype=torch.float16).to(
    "cuda")
mlsd = ControlNetModel.from_pretrained("../models/sd-controlnet-mlsd", torch_dtype=torch.float16).to(
    "cuda")
normal = ControlNetModel.from_pretrained("../models/sd-controlnet-normal", torch_dtype=torch.float16).to(
    "cuda")
openpose = ControlNetModel.from_pretrained("../models/sd-controlnet-openpose",
                                           torch_dtype=torch.float16).to("cuda")
seg = ControlNetModel.from_pretrained("../models/sd-controlnet-seg", torch_dtype=torch.float16).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

keys = set(inspect.signature(StableDiffusionMultiControlNetPipeline.__call__).parameters.keys())


# pli类型的图片转 cv2(其中numpy.array(img) 将一个pli转numpy)
def pil_to_cv2(im):
    return cv2.cvtColor(numpy.array(im), cv2.COLOR_RGB2BGR)


def picture_additional_color(pose_image: Image, pose_image2: Image):
    # for control_version in range(1, 9):
    #     contro_model = None
    #     filename_suffix = ''
    #     if control_version == 1:
    #         contro_model = canny
    #         filename_suffix = 'canny'
    #         # 如果是白底黑线要做反转，canny只能处理黑底白线的图片
    #         # 判断平均像素值是否大于128，如果是，则认为是白色，否则认为是黑色
    #         mean_pixel_value = cv2.mean(pil_to_cv2(pose_image))[0]
    #         if mean_pixel_value > 128:
    #             print('白色线框图，对颜色做反转~~~~~')
    #             pose_image = ImageChops.invert(pose_image)
    #
    #     elif control_version == 2:
    #         contro_model = scribble
    #         filename_suffix = 'scribble'
    #     elif control_version == 3:
    #         contro_model = hed
    #         filename_suffix = 'hed'
    #     elif control_version == 4:
    #         contro_model = depth
    #         filename_suffix = 'depth'
    #     elif control_version == 5:
    #         contro_model = mlsd
    #         filename_suffix = 'mlsd'
    #     elif control_version == 6:
    #         contro_model = normal
    #         filename_suffix = 'normal'
    #     elif control_version == 7:
    #         contro_model = openpose
    #         filename_suffix = 'openpose'
    #     elif control_version == 8:
    #         contro_model = seg
    #         filename_suffix = 'seg'

    pipe.set_progress_bar_config(disable=None)

    prompt_en = 'There is a beautiful house on the grassland'
    n_prompt_en = 'Noise, messy lines, too much color difference, cold tones'
    s = time.time()
    image = pipe(prompt=prompt_en,
                 negative_prompt=n_prompt_en,
                 # processors=[ControlNetProcessor(seg, pose_image), ControlNetProcessor(canny, pose_image2)],
                 processors=[ControlNetProcessor(depth, pose_image), ControlNetProcessor(scribble, pose_image2)],
                 generator=torch.Generator(device="cpu").manual_seed(0),
                 num_inference_steps=30,
                 width=512,
                 height=512).images[0]

    spend = time.time() - s
    print('generate img', spend)
    image.save('multi_control_house.png')


if __name__ == '__main__':
    rs = sys.argv
    img = Image.open("./wireframe/" + rs[1]).convert("RGB")
    img2 = Image.open("./wireframe/" + rs[2]).convert("RGB")
    picture_additional_color(img, img2)
