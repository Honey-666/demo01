# @FileName：ControlNetInPaintTest.py
# @Description：
# @Author：dyh
# @Time：2023/7/6 13:44
# @Website：www.xxx.com
# @Version：V1.0
# !pip install transformers accelerate
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, \
    StableDiffusionControlNetPipeline, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from diffusers.pipelines.controlnet import MultiControlNetModel
import numpy as np
import torch
from PIL import Image

# init_image = Image.open('../../img/control/boy.png')
# init_image = init_image.resize((512, 512))
#
generator = torch.Generator(device="cuda").manual_seed(1)


#
# mask_image = Image.open('../../img/control/boy_mask.png')
# mask_image = mask_image.resize((512, 512))


def make_inpaint_condition(image, image_mask):
    # astype(np.float32) / 255.0:   uint8数据转float32
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    #
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    print(image)
    image = torch.from_numpy(image)
    print(image)
    return image


# control_image = make_inpaint_condition(init_image, mask_image)
canny_img = Image.open('C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\20231030-145313.png').convert("RGB")
pose_img = Image.open('C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\20231030-145327.png').convert("RGB")
mask_img = Image.open('C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\20231030-145323.png').convert("RGB")
img = Image.open('C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\20231030-145318.png').convert("RGB")

mask_img = mask_img.resize(canny_img.size)
img = img.resize(canny_img.size)

control_image = [canny_img, pose_img]

controlnet1 = ControlNetModel.from_pretrained(
    "C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\control-v11p-sd15-canny", torch_dtype=torch.float16,
    variant="fp16"
).to('cuda')

controlnet2 = ControlNetModel.from_pretrained(
    "C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\control-v11p-sd15-openpose", torch_dtype=torch.float16,
    variant="fp16"
).to('cuda')

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\realisticVision_v30VAE",
    controlnet=[controlnet1, controlnet2],
    torch_dtype=torch.float16
).to('cuda')

# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# generate image
image = pipe(
    "a man",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=generator,
    strength=0.8,
    image=[img],
    mask_image=[mask_img],
    control_image=control_image,
).images[0]
image.show()

