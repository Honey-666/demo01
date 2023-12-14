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
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\DucHaiten-AIart-SDXL_V2.0'

pipe = StableDiffusionXLPipeline.from_pretrained(sd_xl_path,
                                                 torch_dtype=torch.float16,
                                                 safety_checker=None,
                                                 variant='fp16').to('cuda')


pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)


prompt = 'beautiful girl'
negative_prompt = 'lowres, cropped, worst quality, low quality, normal quality, artifacts, signature, watermark, username, blurry, more than one bridge, bad architecture'

guidance_scale = 7.5
img = pipe(prompt=prompt,
           negative_prompt=negative_prompt,
           guidance_scale=guidance_scale,
           num_inference_steps=35).images[0]
img.show()
