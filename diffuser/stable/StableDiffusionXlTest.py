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
from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'

pipe = StableDiffusionXLPipeline.from_pretrained(sd_xl_path,
                                                 torch_dtype=torch.float16,
                                                 device_map='auto',
                                                 safety_checker=None)

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

prompt = 'beautiful girl'
negative_prompt = 'lowres, cropped, worst quality, low quality, normal quality, artifacts, signature, watermark, username, blurry, more than one bridge, bad architecture'
output_type = "latent"
denoising_end = 0.8
guidance_scale = 7.5
img = pipe(prompt=prompt,
           negative_prompt=negative_prompt,
           guidance_scale=guidance_scale,
           num_inference_steps=35,
           denoising_end=denoising_end).images[0]
img.show()
