# @FileName：T2IAdapterRecolorTest.py
# @Description：
# @Author：dyh
# @Time：2023/9/6 14:24
# @Website：www.xxx.com
# @Version：V1.0
import os

import torch
from PIL import Image
from diffusers import T2IAdapter, AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLAdapterPipeline

from controlnet_aux import ZoeDetector, CannyDetector
from controlnet_aux.zoe import ZoeDepthNK, get_config
from controlnet_aux.midas import MidasDetector

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts'
t2i_depth_recolor_path = 'C:\\work\\pythonProject\\aidazuo\\models\\T2I\\t2i-adapter-canny-sdxl-1.0'
sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
vae_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\VAE\\sdxl-vae-fp16-fix'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\org_canny.webp'

dpm_a = EulerAncestralDiscreteScheduler.from_pretrained(sd_xl_path, subfolder="scheduler")

adapter = T2IAdapter.from_pretrained(t2i_depth_recolor_path,
                                     torch_dtype=torch.float16,
                                     variant="fp16",
                                     adapter_type="full_adapter_xl").to('cuda')

vae = AutoencoderKL.from_pretrained(vae_xl_path, torch_dtype=torch.float16)

pipe = StableDiffusionXLAdapterPipeline.from_pretrained(sd_xl_path,
                                                        vae=vae,
                                                        adapter=adapter,
                                                        scheduler=dpm_a,
                                                        torch_dtype=torch.float16,
                                                        variant="fp16").to('cuda')
pipe.enable_xformers_memory_efficient_attention()

im = Image.open(img_path)
canny_detector = CannyDetector()
pre_img = canny_detector(im, detect_resolution=384, image_resolution=1024)
pre_img.show()

prompt = "Mystical fairy in real, magic, 4k picture, high quality"
negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

rs_img = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=pre_img,
    num_inference_steps=30,
    adapter_conditioning_scale=0.8,
    guidance_scale=7.5,
).images[0]
rs_img.show()
