# @FileName：ControlNetImg2ImgPipline.py
# @Description：
# @Author：dyh
# @Time：2023/3/31 10:45
# @Website：www.xxx.com
# @Version：V1.0
import torch
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler

from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline

input_image = Image.open('input_image_vermeer.png').convert("RGB")

controlnet = ControlNetModel.from_pretrained("../models/sd-controlnet-canny", torch_dtype=torch.float16)

pipe_controlnet = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "../models/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)

pipe_controlnet.scheduler = UniPCMultistepScheduler.from_config(pipe_controlnet.scheduler.config)
pipe_controlnet.enable_xformers_memory_efficient_attention()
pipe_controlnet.enable_model_cpu_offload()

# using image with edges for our canny controlnet
control_image = Image.open('vermeer_canny_edged.png')

js = {
    "height": 512,
    "width": 512,
    "num_images_per_prompt": 4,
    "num_inference_steps": 30,
    "prompt": "an android robot, cyberpank, digitl art masterpiece",
    "image": [input_image] * 4,
    "controlnet_conditioning_image": control_image,
    "generator": [torch.Generator(device='cuda').manual_seed(12345),
                  torch.Generator(device='cuda').manual_seed(7894389),
                  torch.Generator(device='cuda').manual_seed(30154578),
                  torch.Generator(device='cuda').manual_seed(8258277)]
}
result_img = pipe_controlnet(**js).images
print(result_img)
for (i, img) in enumerate(result_img):
    img.save('contronet_img2img' + str(i) + '.png')
