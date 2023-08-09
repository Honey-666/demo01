import sys
import time

import torch
from controlnet_aux import OpenposeDetector
from PIL import Image
from controlnet_aux.open_pose import Body, Hand, Face
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler,StableDiffusionControlNetImg2ImgPipeline
)

image = Image.open('../../img/control/pose.png')


model_body = Body('C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts\\body_pose_model.pth')
model_hand = Hand('C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts\\hand_pose_model.pth')
model_face = Face('C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts\\facenet.pth')
processor = OpenposeDetector(model_body, model_hand)
for _ in range(5):
    s = time.time()
    control_image = processor(image, hand_and_face=True)
    print(time.time() - s)
# control_image.save("openpose_control.png")
#
# controlnet = ControlNetModel.from_pretrained("../models/control-v11p-sd15-openpose", variant="fp16",torch_dtype=torch.float16).to(
#     "cuda")
# dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained("../models/stable-diffusion-v1-5", subfolder="scheduler")
# pipe = StableDiffusionControlNetPipeline.from_pretrained(
#     "../models/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
#     scheduler=dpm_scheduler
# ).to("cuda")
# StableDiffusionControlNetPipeline.from_pretrained()
#
# pipe.enable_xformers_memory_efficient_attention()
# generator = torch.Generator(device='cuda').manual_seed(1430804514)
# prompt = "a beautiful girl"
# image = pipe(prompt, num_inference_steps=30, guidance_scale=10.0, width=512, height=768, generator=generator,
#              image=control_image).images[0]
# print(time.time() - s)
# image.save('openpose_image_out.png')
