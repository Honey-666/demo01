# @FileName：StableDiffusionXlInpaintTest.py
# @Description：
# @Author：dyh
# @Time：2023/10/9 17:28
# @Website：www.xxx.com
# @Version：V1.0
import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL

t2i_depth_recolor_path = 'C:\\work\\pythonProject\\aidazuo\\models\\T2I\\t2i-adapter-depth-midas-sdxl-1.0'
sd_xl_base_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
sd_xl_inpaint_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\stable-diffusion-xl-1.0-inpainting-0.1'
sd_xl_refiner_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_refiner_1.0'
vae_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\VAE\\sdxl-vae-fp16-fix'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\20230711-155847.jpg'
mask_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\20230711-155958.jpg'

# vae = AutoencoderKL.from_pretrained(vae_xl_path, torch_dtype=torch.float16)
# pipe = StableDiffusionXLInpaintPipeline.from_pretrained(sd_xl_base_path,
#                                                         torch_dtype=torch.float16,
#                                                         variant="fp16",
#                                                         vae=vae).to('cuda')
#
# init_img = Image.open(img_path).resize((512, 512))
# mask_img = Image.open(mask_path).resize((512, 512))
#
# js = {
#     "guidance_scale": 7.5,
#     "num_images_per_prompt": 1,
#     "num_inference_steps": 30,
#     "prompt": "(A table made of marble:1.5)",
#     "negative_prompt": "worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, (extra fingers), (mutated hands), poorly drawn hands, poorly drawn face, mutation, deformed,dehydrated, bad anatomy, bad proportions",
#     "image": init_img,
#     "mask_image": mask_img,
#     "output_type": "latent",
#     "denoising_end": 0.8
# }
#
# images = pipe(**js).images
#
# import gc
#
# gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()
# pipe = None
# del pipe
#
# refiner = StableDiffusionXLInpaintPipeline.from_pretrained(sd_xl_refiner_path,
#                                                            vae=vae,
#                                                            torch_dtype=torch.float16,
#                                                            variant="fp16").to('cuda')
# js2 = {
#     "guidance_scale": 7.5,
#     "num_images_per_prompt": 1,
#     "num_inference_steps": 30,
#     "prompt": "(A table made of marble:1.5)",
#     "negative_prompt": "worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, (extra fingers), (mutated hands), poorly drawn hands, poorly drawn face, mutation, deformed,dehydrated, bad anatomy, bad proportions",
#     "image": images,
#     "mask_image": mask_img,
#     "denoising_start": 0.8
# }
#
# image = refiner(**js2).images[0]
# image.show()
import torch
from PIL import Image

base = StableDiffusionXLInpaintPipeline.from_pretrained(
    sd_xl_base_path,
    torch_dtype=torch.float16,
    variant="fp16")
base.to("cuda")

refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
    sd_xl_refiner_path,
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    variant="fp16")
refiner.to("cuda")

init_image = Image.open(img_path).resize((1024, 1024)).convert("RGB")
mask_image = Image.open(mask_path).resize((1024, 1024)).convert("L")

n_steps = 15
strength = 0.8
prompt = "best quality"

use_latents = True

if use_latents:
    base_output = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=strength,
        image=init_image,
        mask_image=mask_image,
        output_type="latent",
    ).images

else:
    base_output = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=strength,
        image=init_image,
        mask_image=mask_image,
    ).images[0]

images = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=strength,
    image=base_output,
    mask_image=mask_image).images

for i, img in enumerate(images):
    img.show()
