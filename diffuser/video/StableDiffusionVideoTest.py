# @FileName：StableDiffusionVideoTest.py
# @Description：
# @Author：dyh
# @Time：2023/12/12 14:49
# @Website：www.xxx.com
# @Version：V1.0
import os
import time

import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from onediff.infer_compiler import oneflow_compile
from onediff.infer_compiler.utils import set_boolean_env_var


def compile_model(model, attention_fp16_score_accum_max_m=-1):
    # set_boolean_env_var('ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION',
    #                     False)
    # The absolute element values of K in the attention layer of SVD is too large.
    # The unfused attention (without SDPA) and MHA with half accumulation would both overflow.
    # But disabling all half accumulations in MHA would slow down the inference,
    # especially for 40xx series cards.
    # So here by partially disabling the half accumulation in MHA partially,
    # we can get a good balance.
    #
    # On RTX 4090:
    # | ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_ACCUMULATION | MAX_M | Output | Duration |
    # | --------------------------------------------------- | ------| ------ | -------- |
    # | False                                               | -1    | OK     | 32.251s  |
    # | True                                                | -1    | NaN    | 29.089s  |
    # | True                                                | 0     | OK     | 30.947s  |
    # | True                                                | 2304  | OK     | 30.820s  |
    set_boolean_env_var(
        'ONEFLOW_ATTENTION_ALLOW_HALF_PRECISION_SCORE_ACCUMULATION_MAX_M',
        attention_fp16_score_accum_max_m)

    model.image_encoder = oneflow_compile(model.image_encoder)
    model.unet = oneflow_compile(model.unet)
    model.vae.decoder = oneflow_compile(model.vae.decoder)
    model.vae.encoder = oneflow_compile(model.vae.encoder)
    return model


# model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\stable-video-diffusion-img2vid-xt'
model_path = '../models/Stable-diffusion/stable-video-diffusion-img2vid-xt'
# img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\rocket.png'
# img_path = './rocket.png'

pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    variant="fp16").to("cuda")

pipe = compile_model(pipe)
generator = [torch.Generator(device='cpu').manual_seed(torch.Generator(device='cpu').seed()) for _ in range(1)]
# Load the conditioning image

files = os.listdir('./image-video')
for f in files:
    img_path = './image-video/' + f
    print(img_path)
    image = Image.open(img_path)
    image = image.resize((1024, 576))

    # generator = torch.manual_seed(42)

    s = time.time()
    frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]
    print(f'consuming time = {time.time() - s}')

# export_to_video(frames, "generated.mp4", fps=7)
