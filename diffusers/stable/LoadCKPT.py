# @FileName：LoadCKPT.py
# @Description：
# @Author：dyh
# @Time：2023/6/19 16:39
# @Website：www.xxx.com
# @Version：V1.0
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

pipeline = StableDiffusionPipeline.from_ckpt(
    "./xsarchitectural_v11.ckpt"
)
# pipeline = StableDiffusionPipeline.from_ckpt('./v1-5-pruned-emaonly.ckpt')

prompt = "masterpiece, illustration, ultra-detailed, cityscape, san francisco, golden gate bridge, california, bay area, in the snow, beautiful detailed starry sky"
negative_prompt = "lowres, cropped, worst quality, low quality, normal quality, artifacts, signature, watermark, username, blurry, more than one bridge, bad architecture"

images = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=512,
    height=512,
    num_inference_steps=25,
    num_images_per_prompt=4,
    generator=torch.manual_seed(0),
).images[0]
images.save('test.png')
