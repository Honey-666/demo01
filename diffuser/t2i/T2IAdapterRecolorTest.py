# @FileName：T2IAdapterRecolorTest.py
# @Description：
# @Author：dyh
# @Time：2023/9/6 14:24
# @Website：www.xxx.com
# @Version：V1.0
import torch
from PIL import Image
from controlnet_aux import LineartDetector
from diffusers import T2IAdapter, AutoencoderKL, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

from diffuser.t2i.pipeline_stable_diffusion_xl_adapter import StableDiffusionXLAdapterPipeline

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts'
t2i_depth_recolor_path = 'C:\\work\\pythonProject\\aidazuo\\models\\T2I\\t2i-adapter-lineart-sdxl-1.0'
sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
vae_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\VAE\\sdxl-vae-fp16-fix'

preprocess = LineartDetector.from_pretrained(model_path)

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

im = Image.open('./org_lin.jpg')
pre_img = preprocess(im, detect_resolution=384, image_resolution=1024)
pre_img.show()

prompt = "Ice dragon roar, 4k photo"
negative_prompt = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"

rs_img = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=pre_img,
    num_inference_steps=30,
    adapter_conditioning_scale=0.8,
    guidance_scale=7.5,
).images[0]
rs_img.show()

img_lst = []
for i, control in enumerate(control_model_lst):
    if 'lineart' == control:
        lineart_img = img_lineart(Image.open(img_path[i]), lineart_model_path)
        handle_img = detectmap_proc(numpy.array(lineart_img), 'lineart', 'Crop and Resize', height, width)
    else:
        handle_img = Image.open(img_path[i])


    img_lst.append(handle_img)

js['image'] = img_lst