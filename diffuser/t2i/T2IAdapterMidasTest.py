# @FileName：T2IAdapterRecolorTest.py
# @Description：
# @Author：dyh
# @Time：2023/9/6 14:24
# @Website：www.xxx.com
# @Version：V1.0
import os

import torch
from PIL import Image
from diffusers import T2IAdapter, AutoencoderKL, EulerAncestralDiscreteScheduler, StableDiffusionXLAdapterPipeline

from controlnet_aux import ZoeDetector
from controlnet_aux.zoe import ZoeDepthNK, get_config
from controlnet_aux.midas import MidasDetector

def img_midas(img, pretrained_model_or_path, detect_resolution, image_resolution):
    midas_depth = MidasDetector.from_pretrained(pretrained_model_or_path).to('cuda')
    image = midas_depth(img, detect_resolution=detect_resolution, image_resolution=image_resolution)
    return image


def img_zoe(img, pretrained_model_or_path, detect_resolution, image_resolution):
    conf = get_config('zoedepth_nk', "infer")
    prefix = 'local::'
    pretrained_resource = os.path.join(pretrained_model_or_path, 'ZoeD_M12_NK.pt')
    conf['pretrained_resource'] = prefix + pretrained_resource
    model = ZoeDepthNK.build_from_config(conf)
    model_path = os.path.join(pretrained_model_or_path, 'zoed_nk.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    model.eval()

    zoe_depth = ZoeDetector(model).to('cuda')
    image = zoe_depth(img, detect_resolution=detect_resolution, image_resolution=image_resolution)
    return image


model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts'
t2i_depth_recolor_path = 'C:\\work\\pythonProject\\aidazuo\\models\\T2I\\t2i-adapter-depth-midas-sdxl-1.0'
sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
vae_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\VAE\\sdxl-vae-fp16-fix'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\org_mid.jpg'

dpm_a = EulerAncestralDiscreteScheduler.from_pretrained(sd_xl_path, subfolder="scheduler")

adapter = T2IAdapter.from_pretrained(t2i_depth_recolor_path,
                                     torch_dtype=torch.float16,
                                     variant="fp16",
                                     adapter_type="full_adapter_xl").to('cuda')

vae = AutoencoderKL.from_pretrained(vae_xl_path, torch_dtype=torch.float16)

pipe = StableDiffusionXLAdapterPipeline.from_pretrained(sd_xl_path,
                                                        vae=vae,
                                                        adapter=adapter,
                                                        scheduler=dpm_a,
                                                        torch_dtype=torch.float16,
                                                        variant="fp16").to('cuda')
pipe.enable_xformers_memory_efficient_attention()

im = Image.open(img_path)
pre_img = img_midas(im, model_path, detect_resolution=512, image_resolution=1024)
pre_img.show()

prompt = "A photo of a room, 4k photo, highly detailed"
negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"

rs_img = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=pre_img,
    num_inference_steps=30,
    adapter_conditioning_scale=1,
    guidance_scale=7.5,
).images[0]
rs_img.show()
