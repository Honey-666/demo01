# @FileName：StableDiffusionXlReferenceLCMTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/25 18:07
# @Website：www.xxx.com
# @Version：V1.0
import time

import torch
from diffusers import UNet2DConditionModel, LCMScheduler

from stable_diffusion_xl_reference import StableDiffusionXLPipeline

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
lcm_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\lcm-sdxl'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\20231125-153039.jpg'

unet = UNet2DConditionModel.from_pretrained(
    lcm_path,
    torch_dtype=torch.float16,
    variant="fp16")

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_path,
    unet=unet,
    torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
num_inference_steps = 8
# generator = torch.manual_seed(0)

s = time.time()
images = pipe(prompt=prompt,
              ref_image=None,
              num_inference_steps=num_inference_steps,
              guidance_scale=8.0,
              width=1024,
              height=1024).images
print(f'consuming time = {time.time() - s}')
images[0].show()
# print(images)
