# @FileName：LatentUpscalePipeTest.py
# @Description：
# @Author：dyh
# @Time：2023/5/11 12:27
# @Website：www.xxx.com
# @Version：V1.0
import os

from PIL import Image

from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler, \
    UniPCMultistepScheduler
import torch

v15_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\stable-diffusion-v1-5'
model_id = "C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd-x2-latent-upscaler"

pipeline = StableDiffusionPipeline.from_pretrained(v15_path, torch_dtype=torch.float16).to("cuda")

upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id,
                                                                torch_dtype=torch.float16,
                                                                variant='fp16').to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
# upscaler.scheduler = UniPCMultistepScheduler.from_config(upscaler.scheduler.config)

prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"
generator = torch.manual_seed(33)

low_res_latents = pipeline(prompt, generator=generator, output_type="latent",num_inference_steps=20).images

with torch.no_grad():
    image = pipeline.decode_latents(low_res_latents)
image = pipeline.numpy_to_pil(image)[0]

print(image)

upscaled_image = upscaler(
    prompt=prompt,
    image=low_res_latents,
    num_inference_steps=20,
    guidance_scale=0,
    generator=generator,
).images[0]

upscaled_image.show()
