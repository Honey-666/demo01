# @FileName：LCMPipelineTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/7 14:36
# @Website：www.xxx.com
# @Version：V1.0
import time

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler

model_path = '../models/Stable-diffusion/sd_xl_base_1.0'
lcm_path = '../models/Stable-diffusion/lcm-sdxl'

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
    # prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
    num_inference_steps = 8
    # generator = torch.manual_seed(0)

    s = time.time()
    images = pipe(prompt=prompt,
                  num_inference_steps=num_inference_steps,
                  guidance_scale=8.0,
                  # generator=generator,
                  width=1024,
                  height=1024).images
    print(f'consuming time = {time.time() - s}')
    # images[0].show()
    print(images)
