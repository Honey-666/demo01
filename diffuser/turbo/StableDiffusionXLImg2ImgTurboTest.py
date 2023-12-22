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

img = Image.open(img_path).convert("RGB").resize((1024, 1024))

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
    # prompt = 'A beautiful girl'
    s = time.time()
    images = pipeline(
        image=img,
        prompt=prompt,
        guidance_scale=0.0,
        num_images_per_prompt=1,
        num_inference_steps=2,
        strength=0.5).images
    print(time.time() - s)
    # image.show()
    print(images)
