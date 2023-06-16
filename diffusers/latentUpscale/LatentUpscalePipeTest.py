# @FileName：LatentUpscalePipeTest.py
# @Description：
# @Author：dyh
# @Time：2023/5/11 12:27
# @Website：www.xxx.com
# @Version：V1.0
import os

from PIL import Image

from diffusers import StableDiffusionLatentUpscalePipeline, DPMSolverMultistepScheduler
import torch

model_id = "/data/sd-x2-latent-upscaler"
upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id,
                                                                torch_dtype=torch.float16,
                                                                revision="fp16").to("cuda")

files = os.listdir('./shinei')

for f in files:
    print(f)
    filename = f.split('.')[0]
    image = Image.open('./shinei/' + f)

    prompt = ""

    upscaled_image = upscaler(
        prompt=prompt,
        image=image,
        num_inference_steps=30
    ).images[0]

    upscaled_image.save("./shinei/" + filename + "_upscale.jpg")
