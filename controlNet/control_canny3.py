from diffusers.utils import load_image
from PIL import Image, ImageChops
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline
)

pose_image = Image.open('women.png').resize((512, 512))
# pose_image = ImageChops.invert(pose_image)
controlnet = ControlNetModel.from_pretrained("../models/sd-controlnet-canny")
# controlnet = ControlNetModel.from_pretrained("../models/sd-controlnet-scribble")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "../models/stable-diffusion-v1-5", low_cpu_mem_usage=False, device_map=None, safety_checker=None,
    controlnet=controlnet
).to("cuda")

pipe.safety_checker = lambda images, clip_input: (images, False)

pipe.set_progress_bar_config(disable=None)

# prompt = '1gril,masterpiece,graden'
prompt = 'girl'
n_prompt = 'ungainly,bab,blurry,mutations'
output = pipe(prompt, pose_image, 512, 512, 50, 8, n_prompt)

image = output.images[0]
image.save('generated_women.png')
