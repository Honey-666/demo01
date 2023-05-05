# @FileName：ControlNet.py
# @Description：
# @Author：dyh
# @Time：2023/3/17 14:44
# @Website：www.xxx.com
# @Version：V1.0
import sys
import time

import cv2
import numpy

import torch
import inspect

from diffusers import ControlNetModel, DPMSolverMultistepScheduler
from python_bigbigwork_util import LoggingUtil
from PIL import Image
from stable_diffusion_multi_controlnet import StableDiffusionMultiControlNetPipeline, ControlNetProcessor


def get_keys():
    return set(inspect.signature(StableDiffusionMultiControlNetPipeline.__call__).parameters.keys())


my_logger = LoggingUtil.get_logging()
model = "../models/sd_chaji_cream_xinzs_mix"

dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(model, subfolder="scheduler")

pipe = StableDiffusionMultiControlNetPipeline.from_pretrained(
    model, torch_dtype=torch.float16, scheduler=dpm_scheduler
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()
mlsd = ControlNetModel.from_pretrained("../models/control-chaji-mlsd", torch_dtype=torch.float16).to(
    "cuda")


def picture_additional_color(pose_image: Image, pose_image2: Image = None):
    js = {
        "processors": [
            ControlNetProcessor(mlsd, pose_image)
        ],
        "guidance_scale": 18.5,
        "prompt": "Living room, Interior design,Comfortable seating, single chair, peaceful and relaxing atmosphere, side table, TV, marble fireplace",
        "negative_prompt": "low resolution, unclear, distorted, formless, disgusting, tacky, ugly, oversaturated, abrupt, out of focus, out of frame, poorly drawn, broken",
        "num_images_per_prompt": 1,
        "num_inference_steps": 20,
        "height": 512,
        "width": 512,
        "generator": torch.Generator(device="cuda").manual_seed(6510994357051747)
    }

    s = time.time()
    image = pipe(**js).images[0]

    spend = time.time() - s
    print('generate img', spend)
    image.save('chaji_control_house.png')


if __name__ == '__main__':
    rs = sys.argv
    img = Image.open("./wireframe/" + rs[1]).convert("RGB")
    # img2 = Image.open("./wireframe/" + rs[2]).convert("RGB")
    picture_additional_color(img)
