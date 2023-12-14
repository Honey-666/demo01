# @FileName：StableDiffusionVideoTest.py
# @Description：
# @Author：dyh
# @Time：2023/12/12 14:49
# @Website：www.xxx.com
# @Version：V1.0
import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\stable-video-diffusion-img2vid-xt'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\rocket.png'

pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    variant="fp16").to("cuda")

# pipe.enable_model_cpu_offload()

# Load the conditioning image
image = Image.open(img_path)
image = image.resize((1024, 576))

generator = [torch.Generator(device='cpu').manual_seed(torch.Generator(device='cpu').seed()) for _ in range(1)]
# generator = torch.manual_seed(42)
frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

export_to_video(frames, "generated.mp4", fps=7)
