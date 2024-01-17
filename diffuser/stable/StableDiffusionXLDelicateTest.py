# @FileName：StableDiffusionXLDelicateTest.py
# @Description：
# @Author：dyh
# @Time：2023/12/22 16:19
# @Website：www.xxx.com
# @Version：V1.0
# @FileName：StableDiffusionXlControlNet.py
# @Description：
# @Author：dyh
# @Time：2023/8/9 14:44
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import gc

sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
sd_xl_refiner_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_refiner_1.0'

pipe = StableDiffusionXLPipeline.from_pretrained(sd_xl_path,
                                                 torch_dtype=torch.float16,
                                                 safety_checker=None,
                                                 variant='fp16').to('cuda')

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

prompt = 'beautiful girl'
negative_prompt = 'lowres, cropped, worst quality, low quality, normal quality, artifacts, signature, watermark, username, blurry, more than one bridge, bad architecture'

guidance_scale = 7.5
img = pipe(prompt=prompt,
           negative_prompt=negative_prompt,
           guidance_scale=guidance_scale,
           output_type="latent",
           # denoising_end=0.8,
           num_inference_steps=35,
           width=768,
           height=768).images[0]

print(img)

pipe = None
del pipe
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

image_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(sd_xl_refiner_path,
                                                              torch_dtype=torch.float16,
                                                              safety_checker=None,
                                                              variant='fp16').to('cuda')
image_pipe.scheduler = DPMSolverMultistepScheduler.from_config(image_pipe.scheduler.config, use_karras_sigmas=True)

img = image_pipe(
    image=img,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale,
    # denoising_start=0.8,
    original_size=(512, 512),
    target_size=(1536, 1536),
    num_inference_steps=35).images[0]

img.show()
