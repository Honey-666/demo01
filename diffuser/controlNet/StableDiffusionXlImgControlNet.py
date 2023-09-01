# @FileName：StableDiffusionXlImgControlNet.py
# @Description：
# @Author：dyh
# @Time：2023/8/9 14:44
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy as np
import torch
from PIL import Image
from pipeline_controlnet_sd_xl_img2img import StableDiffusionXLControlNetImg2ImgPipeline
from diffusers import ControlNetModel


sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
controlnet_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\controlnet-canny-sdxl-1.0'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\000000009.png'

controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", torch_dtype=torch.float16).to('cuda')

pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(sd_xl_path,
                                                                  controlnet=controlnet,
                                                                  torch_dtype=torch.float16,
                                                                  safety_checker=None).to('cuda')

pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_model_cpu_offload()

prompt = 'A robot, 4k photo, (red horse: 1.5) aerial view, a futuristic research complex in a bright foggy jungle, hard lighting'
negative_prompt = 'low quality, bad quality, sketches'
controlnet_conditioning_scale = 0.5

source_img = Image.open(img_path)
image = np.array(source_img)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

img = pipe(prompt=prompt,
           negative_prompt=negative_prompt,
           controlnet_conditioning_scale=controlnet_conditioning_scale,
           image=source_img,
           strength=0.99,
           num_inference_steps=50,
           control_image=canny_image).images[0]
img.show()
