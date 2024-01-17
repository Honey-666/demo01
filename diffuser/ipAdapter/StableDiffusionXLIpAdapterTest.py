# @FileName：StableDiffusionXLIpAdapterTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/25 14:00
# @Website：www.xxx.com
# @Version：V1.0
import os

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from transformers import CLIPVisionModelWithProjection

# model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
model_path = '../models/Stable-diffusion/sd_xl_base_1.0'
# vae_path = 'C:\\work\\pythonProject\\aidazuo\\models\\VAE\\sdxl-vae-fp16-fix'
vae_path = '../models/VAE/sdxl-vae-fp16-fix'
ip_adapter_path = '../models/IP-Adapter'

ip_img_path = './vermeer.png'

vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)

image_encoder = CLIPVisionModelWithProjection.from_pretrained(ip_adapter_path,
                                                              subfolder='models/image_encoder',
                                                              torch_dtype=torch.float16).to('cuda')

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_path,
    safety_checker=None,
    variant="fp16",
    torch_dtype=torch.float16,
    image_encoder=image_encoder,
    vae=vae).to("cuda")

pipe.load_ip_adapter(ip_adapter_path, subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.bin")

# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

ip_adapter_img = Image.open(ip_img_path)

images = pipe(
    prompt='a coffee machine',
    ip_adapter_image=ip_adapter_img,
    negative_prompt="",
    num_inference_steps=30,
    num_images_per_prompt=1,
    width=1024,
    height=1024
).images

pipe.unload_ip_adapter()

for img in images:
    # img.show()
    print(img)