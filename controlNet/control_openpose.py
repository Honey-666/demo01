import sys
import time

import torch
from controlnet_aux import OpenposeDetector
from PIL import Image
from controlnet_aux.open_pose import Body, Hand, Face
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler
)

image = Image.open('1.png')

s = time.time()
model_body = Body('../models/annotator/ckpts/body_pose_model.pth')
model_hand = Hand('../models/annotator/ckpts/hand_pose_model.pth')
model_face = Face('../models/annotator/ckpts/facenet.pth')
processor = OpenposeDetector(model_body, model_hand)

control_image = processor(image, hand_and_face=True)
print(time.time() - s)
control_image.save("openpose_control.png")

controlnet = ControlNetModel.from_pretrained("../models/control-v11p-sd15-openpose", torch_dtype=torch.float16).to(
    "cuda")
dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained("../models/stable-diffusion-v1-5", subfolder="scheduler")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "../models/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, scheduler=dpm_scheduler
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()
pipe.load_textual_inversion('../models/stable-diffusion-v1-5/embeddings/ng_deepnegative_v1_75t.pt')
generator = torch.Generator(device='cuda').manual_seed(1430804514)
prompt = "a beautiful girl"
image = pipe(prompt, num_inference_steps=30, guidance_scale=10.0, width=512, height=768, generator=generator,
             image=control_image).images[0]
image.save('openpose_image_out.png')
