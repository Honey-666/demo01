import time

import torch
from diffusers import AutoPipelineForText2Image, AutoencoderKL, StableDiffusionXLPipeline, \
    StableDiffusionXLImg2ImgPipeline
from onediff.infer_compiler import oneflow_compile

# from onediff.infer_compiler import oneflow_compile

model_path = '../models/Stable-diffusion/sdxl-turbo'
vae_path = '../models/VAE/sdxl-vae-fp16-fix'
# model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sdxl-turbo'
# vae_path = 'C:\\work\\pythonProject\\aidazuo\\models\\VAE\\sdxl-vae-fp16-fix'

vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)

pipeline = StableDiffusionXLPipeline.from_pretrained(model_path,
                                                     torch_dtype=torch.float16,
                                                     vae=vae,
                                                     safety_checker=None,
                                                     variant="fp16",
                                                     add_watermarker=False).to('cuda')

# img_pipe = StableDiffusionXLImg2ImgPipeline(**pipeline.components).to('cuda')

print("unet is compiled to oneflow.")
pipeline.unet = oneflow_compile(pipeline.unet)
print("vae is compiled to oneflow.")
pipeline.vae = oneflow_compile(pipeline.vae)

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

dict_size = {"512": 512, "384": 576, "576": 384}
w = 512
h = 512
for i, prompt in enumerate(prompt_lst):
    # for k, v in dict_size.items():
    # w += 8
    # h += 8
    s = time.time()
    images = pipeline(prompt=prompt,
                      guidance_scale=0.0,
                      num_inference_steps=2,
                      num_images_per_prompt=2,
                      width=w,
                      height=h).images
    print('text2img', time.time() - s)
    print(images)

# img = images[0].resize((1024, 1024))
#
# s = time.time()
# images = img_pipe(
#     image=img,
#     prompt=prompt,
#     guidance_scale=0.0,
#     num_images_per_prompt=1,
#     num_inference_steps=2,
#     strength=0.5).images
# print('img2img', time.time() - s)
#
# print(images)

# prompt = 'A cinematic shot of a baby racoon wearing an intricate italian priest robe.'

# image.show()
