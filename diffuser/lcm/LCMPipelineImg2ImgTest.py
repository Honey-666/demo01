# @FileName：LCMPipelineTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/7 14:36
# @Website：www.xxx.com
# @Version：V1.0
import time

import torch
from PIL import Image
from diffusers import LatentConsistencyModelImg2ImgPipeline

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\LCM_Dreamshaper_v7'
pipe = LatentConsistencyModelImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto')
pipe.enable_model_cpu_offload()

prompt = "best, 8k, girl"
num_inference_steps = 4

image = Image.open('C:\\Users\\bbw\\Desktop\\to_img_0.png').resize((1536, 1536))

s = time.time()
images = pipe(image=image, prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0).images
print(f'consuming time = {time.time() - s}')
images[0].show()
