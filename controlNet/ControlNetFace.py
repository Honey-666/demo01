# @FileName：ControlNetFace.py
# @Description：
# @Author：dyh
# @Time：2023/4/13 10:27
# @Website：www.xxx.com
# @Version：V1.0
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

image = load_image(
    "https://huggingface.co/CrucibleAI/ControlNetMediaPipeFace/resolve/main/samples_laion_face_dataset/family_annotation.png"
)

# Stable Diffusion 2.1-base:
# controlnet = ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", torch_dtype=torch.float16, variant="fp16")
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
# )
# OR
# Stable Diffusion 1.5:
controlnet = ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", subfolder="diffusion_sd15")
pipe = StableDiffusionControlNetPipeline.from_pretrained("../models/stable-diffusion-v1-5", controlnet=controlnet)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
pipe.enable_xformers_memory_efficient_attention()

image = pipe("a happy family at a dentist advertisement", image=image, num_inference_steps=30).images[0]
image.save('./images.png')
