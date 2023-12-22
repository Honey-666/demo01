# @FileName：LCMPipelineTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/7 14:36
# @Website：www.xxx.com
# @Version：V1.0
import time

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler, AutoencoderKL

model_path = '../models/Stable-diffusion/sd_xl_base_1.0'
lcm_path = '../models/Stable-diffusion/lcm-sdxl'
vae_path = '../models/VAE/sdxl-vae-fp16-fix'
ip_adapter_path = '../models/IP-Adapter'

ip_img_path = './'

unet = UNet2DConditionModel.from_pretrained(
    lcm_path,
    torch_dtype=torch.float16,
    variant="fp16")

vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_path,
    unet=unet,
    torch_dtype=torch.float16,
    vae=vae
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# pipe.load_ip_adapter(ip_adapter_path, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
num_inference_steps = 8
# generator = torch.manual_seed(0)

ip_adapter_image = Image.open(ip_img_path)

s = time.time()
images = pipe(prompt=prompt,
              num_inference_steps=num_inference_steps,
              guidance_scale=8.0,
              # generator=generator,
              # ip_adapter_image=ip_adapter_image,
              width=1024,
              height=1024).images
print(f'consuming time = {time.time() - s}')
images[0].show()
# print(images)
