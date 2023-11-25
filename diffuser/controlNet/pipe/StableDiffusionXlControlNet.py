# @FileName：StableDiffusionXlControlNet.py
# @Description：
# @Author：dyh
# @Time：2023/8/9 14:44
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionXLImg2ImgPipeline, ControlNetModel, \
    UniPCMultistepScheduler, EulerAncestralDiscreteScheduler, DPMSolverSDEScheduler, DPMSolverMultistepScheduler

from compel import Compel, ReturnedEmbeddingsType

sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
controlnet_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\sdxl-controlnet-seg'
# img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\images_1.png'
img_path = 'C:\\Users\\bbw\\Desktop\\image.png'

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, device_map='auto')

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(sd_xl_path,
                                                           controlnet=controlnet,
                                                           torch_dtype=torch.float16,
                                                           safety_checker=None).to('cuda')

pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

prompt = 'two elephants are walking in a zoo enclosure'
negative_prompt = ''
controlnet_conditioning_scale = 1.0

image = Image.open(img_path)

img = pipe(prompt=prompt,
           negative_prompt=negative_prompt,
           controlnet_conditioning_scale=controlnet_conditioning_scale,
           num_inference_steps=30,
           image=image).images[0]
img.show()
