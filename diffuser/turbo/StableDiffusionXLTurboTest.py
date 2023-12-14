# @FileName：StableDiffusionXLTurboTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/30 15:38
# @Website：www.xxx.com
# @Version：V1.0
import time

import torch
from diffusers import AutoPipelineForText2Image, AutoencoderKL, StableDiffusionXLPipeline

model_path = '../models/Stable-diffusion/sdxl-turbo'
vae_path = '../models/VAE/sdxl-vae-fp16-fix'
# model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sdxl-turbo'
# vae_path = 'C:\\work\\pythonProject\\aidazuo\\models\\VAE\\sdxl-vae-fp16-fix'

vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)

pipeline = StableDiffusionXLPipeline.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     vae=vae,
                                                     safety_checker=None,
                                                     variant="fp16").to('cuda')

# pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

prompt = 'A cinematic shot of a baby racoon wearing an intricate italian priest robe.'
s = time.time()
images = pipeline(prompt=prompt,
                  guidance_scale=0.0,
                  num_inference_steps=1,
                  num_images_per_prompt=2,
                  width=512,
                  height=512).images
print(time.time() - s)
print(images)
# image.show()
