# @FileName：LCMXLLoraPipelineTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/7 14:36
# @Website：www.xxx.com
# @Version：V1.0
import time

import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
lora_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Lora\\lcm-lora-sdxl'
pipe = StableDiffusionXLPipeline.from_pretrained(model_path,
                                                 torch_dtype=torch.float16,
                                                 variant="fp16").to('cuda')
pipe.load_lora_weights(lora_path)
# pipe.fuse_lora()
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

    # prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
    num_inference_steps = 8
    s = time.time()
    images = pipe(prompt=prompt,
                  num_inference_steps=num_inference_steps,
                  guidance_scale=0.0,
                  width=1024,
                  height=1024).images
    print(f'consuming time = {time.time() - s}')
    # pipe.unfuse_lora()
    # images[0].show()
    print(images[0])
