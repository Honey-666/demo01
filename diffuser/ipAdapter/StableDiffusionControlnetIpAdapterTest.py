# @FileName：StableDiffusionControlnetIpAdapterTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/25 14:00
# @Website：www.xxx.com
# @Version：V1.0
import os

import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, DPMSolverMultistepScheduler


def load_embeddings(embeddings, pipeline):
    for emb in embeddings:
        if emb.endswith('.pt'):
            token = emb.lower().split('.')[0]
            print('load embedding: ', emb)
            pipeline.load_textual_inversion(os.path.join('C:\\work\\pythonProject\\aidazuo\\models\\Embedding', emb),
                                            token,
                                            torch_dtype=torch.float16)


model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\realisticVision_v30VAE'
canny_model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\control-v11p-sd15-canny'
depth_model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\control-v11f1p-sd15-depth'
vae_path = 'C:\\work\\pythonProject\\aidazuo\\models\\VAE\\vae-ft-mse-840000-ema-pruned'
ip_adapter_path = 'C:\\work\\pythonProject\\aidazuo\\models\\IP-Adapter'

canny_img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\tmpsljqx8s_.png'
depth_img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\tmp3jc1p7_9.png'
ip_img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\20231125-153039.jpg'

canny_control = ControlNetModel.from_pretrained(canny_model_path, variant="fp16", torch_dtype=torch.float16).to('cuda')
depth_control = ControlNetModel.from_pretrained(depth_model_path, variant="fp16", torch_dtype=torch.float16).to('cuda')
vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_path,
    controlnet=[canny_control, depth_control],
    safety_checker=None,
    variant="fp16",
    torch_dtype=torch.float16,
    vae=vae).to("cuda")

pipe.load_ip_adapter(ip_adapter_path, subfolder="models", weight_name="ip-adapter-plus_sd15.bin")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

load_embeddings(["EasyNegative.pt", "badhandv4.pt"], pipe)

canny_img = Image.open(canny_img_path)
depth_img = Image.open(depth_img_path)
ip_adapter_img = Image.open(ip_img_path)

images = pipe(
    prompt='a coffee machine',
    image=[canny_img, depth_img],
    ip_adapter_image=ip_adapter_img,
    negative_prompt="easynegative,badhandv4",
    num_inference_steps=30,
    controlnet_conditioning_scale=[1.0, 1.0],
    num_images_per_prompt=1,
    width=512,
    height=768
).images

for img in images:
    img.show()
