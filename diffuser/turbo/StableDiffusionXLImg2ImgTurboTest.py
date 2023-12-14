# @FileName：StableDiffusionXLTurboTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/30 15:38
# @Website：www.xxx.com
# @Version：V1.0
import time

import torch
from PIL import Image
from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline

model_path = '../models/Stable-diffusion/sdxl-turbo'
vae_path = '../models/VAE/sdxl-vae-fp16-fix'
img_path = './vermeer.png'

vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)

pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_path,
                                                            torch_dtype=torch.float16,
                                                            vae=vae,
                                                            safety_checker=None,
                                                            variant="fp16").to('cuda')

img = Image.open(img_path).convert("RGB").resize((512, 512))
prompt = 'A beautiful girl'
s = time.time()
images = pipeline(
    image=img,
    prompt=prompt,
    guidance_scale=0.0,
    num_images_per_prompt=4,
    num_inference_steps=2,
    strength=0.5).images
print(time.time() - s)
# image.show()
print(images)
