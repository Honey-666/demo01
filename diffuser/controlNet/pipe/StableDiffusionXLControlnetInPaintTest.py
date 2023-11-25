# @FileName：StableDiffusionXLControlnetInPaintTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/8 15:14
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, AutoencoderKL, \
    DPMSolverMultistepScheduler

sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
vae_path = 'C:\\work\\pythonProject\\aidazuo\\models\\VAE\\sdxl-vae-fp16-fix'
controlnet_path = 'C:\\work\\pythonProject\\aidazuo\models\\ControlNet\\controlnet-canny-sdxl-1.0'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\59c96b0e08d84497a6f4b0c3e07e4dde.jpeg'
mask_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\base64-img.jpg'
canny_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\7fd3bab494d7431c98c86f8686c7ff66.jpg'

controlnet = ControlNetModel.from_pretrained(controlnet_path,
                                             torch_dtype=torch.float16,
                                             variant='fp16').to('cuda')

vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(sd_xl_path,
                                                                  controlnet=controlnet,
                                                                  torch_dtype=torch.float16,
                                                                  vae=vae,
                                                                  safety_checker=None).to('cuda')

pipe.enable_model_cpu_offload()
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

source_img = Image.open(img_path).resize((1024, 1024))
mask_img = Image.open(mask_path).convert('RGB')
canny_img = Image.open(canny_path)
image = np.array(canny_img)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image).resize((1024, 1024))

image = pipe(prompt='best,4k',
             negative_prompt='',
             guidance_scale=7.5,
             num_inference_steps=30,
             image=source_img,
             mask_image=mask_img,
             control_image=canny_image,
             controlnet_conditioning_scale=1.0
             ).images[0]

image.show()
