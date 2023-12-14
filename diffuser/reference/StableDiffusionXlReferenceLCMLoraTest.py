# @FileName：StableDiffusionXlReferenceLCMTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/25 18:07
# @Website：www.xxx.com
# @Version：V1.0
import time

import torch
from PIL import Image
from diffusers import UNet2DConditionModel, LCMScheduler

from stable_diffusion_xl_reference import StableDiffusionXLReferencePipeline

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
adapter_id = 'C:\\work\\pythonProject\\aidazuo\\models\\Lora\\lcm-lora-sdxl'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\vermeer.png'

pipe = StableDiffusionXLReferencePipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# load and fuse lcm lora
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

prompt = "a beautiful girl"
num_inference_steps = 8
ref_img = Image.open(img_path)
# generator = torch.manual_seed(0)

s = time.time()
images = pipe(prompt=prompt,
              ref_image=ref_img,
              num_inference_steps=num_inference_steps,
              reference_attn=True,
              reference_adain=False,
              style_fidelity=0.8,
              attention_auto_machine_weight=1.0,
              guidance_scale=0).images
print(f'consuming time = {time.time() - s}')
images[0].show()

# print(images)
