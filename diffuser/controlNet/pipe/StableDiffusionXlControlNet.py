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
from diffusers import StableDiffusionXLControlNetPipeline,StableDiffusionXLImg2ImgPipeline, ControlNetModel

from compel import Compel, ReturnedEmbeddingsType

sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
controlnet_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\controlnet-canny-sdxl-1.0'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\hf-logo.png'

controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", torch_dtype=torch.float16, device_map='auto')

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(sd_xl_path,
                                                           controlnet=controlnet,
                                                           torch_dtype=torch.float16,
                                                           device_map='auto',
                                                           safety_checker=None)

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

prompt = 'aerial view, a futuristic research complex in a bright foggy jungle, hard lighting'
negative_prompt = 'low quality, bad quality, sketches'
controlnet_conditioning_scale = 0.5

image = Image.open(img_path)
image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

img = pipe(prompt=prompt,
           negative_prompt=negative_prompt,
           controlnet_conditioning_scale=controlnet_conditioning_scale,
           num_inference_steps=30,
           image=canny_image).images[0]
img.show()
