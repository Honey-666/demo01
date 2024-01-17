# @FileName：StableDiffusionXLIpAdapterTest.py
# @Description：
# @Author：dyh
# @Time：2023/11/25 14:00
# @Website：www.xxx.com
# @Version：V1.0

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPVisionModelWithProjection

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\realisticVision_v30VAE'
ip_adapter_path = 'C:\\work\\pythonProject\\aidazuo\\models\\IP-Adapter'
ip_img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\vermeer.png'
lora_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Lora\\figma.safetensors'

# image_encoder = CLIPVisionModelWithProjection.from_pretrained(ip_adapter_path,
#                                                               subfolder='models/image_encoder',
#                                                               torch_dtype=torch.float16).to('cuda')

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    safety_checker=None,
    variant="fp16",
    torch_dtype=torch.float16,
    # image_encoder=image_encoder
).to("cuda")

# pipe.load_ip_adapter(ip_adapter_path, subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
# pipe.set_ip_adapter_scale(0.6)
print('before', pipe.get_active_adapters())

pipe.load_lora_weights(lora_path, adapter_name='figma')
pipe.set_adapters("figma", adapter_weights=0.7)
print('before', pipe.get_active_adapters())
pipe.disable_lora()
print('after', pipe.get_active_adapters())
# ip_adapter_img = Image.open(ip_img_path)

images = pipe(
    prompt='A professional portrait photo of a beautiful Russian model\'s face taken at a pizza restaurant at noon, featuring ultra fine skin, facial focus, high resolution, Nikon Z9, film shooting, super details, excellent high fidelity photography, rich details, 8k',
    # ip_adapter_image=ip_adapter_img,
    negative_prompt="",
    num_inference_steps=30,
    num_images_per_prompt=1,
    width=512,
    height=512,
    generator=torch.Generator(device="cuda").manual_seed(0)
).images

pipe.unload_lora_weights()
print('after', pipe.get_active_adapters())
# pipe.unload_ip_adapter()

for img in images:
    print(img)
    img.show()
