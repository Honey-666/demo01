# @FileName：StableDiffusionXlControlNet.py
# @Description：
# @Author：dyh
# @Time：2023/8/9 14:44
# @Website：www.xxx.com
# @Version：V1.0
import time

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline, AutoencoderKL

sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\animagine-xl-3.0'
vae_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\VAE\\sdxl-vae-fp16-fix'
# sd_xl_path = '../models/Stable-diffusion/sd_xl_base_1.0'
vae = AutoencoderKL.from_pretrained(vae_xl_path, torch_dtype=torch.bfloat16)
pipe = StableDiffusionXLPipeline.from_pretrained(sd_xl_path,
                                                 torch_dtype=torch.bfloat16,
                                                 safety_checker=None,
                                                 vae=vae,
                                                 variant='fp16').to('cuda')

# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

prompt_lst = [
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "picture of elf,an extremely delicate and beautiful face,in the forest",
    "(((cute anime rabbit))),bunny,chinese new year,year of the rabbit,sharp focus,perfect composition,volumetric lighting,dynamic",
    "A cyberpunk city,by Qibaishi,ink wash painting,perfect composition",
    "The streets of a beautiful Mediterranean town with many tourists,masterfully composed",
    "Water color, Switzerland, skiing, glass house, morning, cableway",
    "Blue sea, beach with few coconut trees, less people and water houses on the beach",
    "Cyberpunk Oriental Myth future world Miracle forest Sunshine Colorful Clouds Wizard of Oz Flying Birds Green Light",
    "Girl, stunning face, cartoon style, dynamic lighting, intricate details",
    "A group of adult women are walking, petals fall from the sky, 1080p, dense forest, sun through the shade of the tree, a child is running, big vision, photography"
]

for prompt in prompt_lst:
    # prompt = 'beautiful girl'
    negative_prompt = 'lowres, cropped, worst quality, low quality, normal quality, artifacts, signature, watermark, username, blurry, more than one bridge, bad architecture'

    guidance_scale = 7.5
    s = time.time()
    img = pipe(prompt=prompt,
               negative_prompt=negative_prompt,
               guidance_scale=guidance_scale,
               num_inference_steps=30).images[0]
    print(f'consuming time = {time.time() - s}')
    print(img)
