# @FileName：StableDiffusionXlControlNet.py
# @Description：
# @Author：dyh
# @Time：2023/8/9 14:44
# @Website：www.xxx.com
# @Version：V1.0
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

sd_xl_path = ''
controlnet_path = ''
controlnet = ControlNetModel.from_pretrained(controlnet_path,
                                             variant="fp16",
                                             torch_dtype=torch.float16,
                                             device_map='auto')
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(sd_xl_path,
                                                           controlnet=controlnet,
                                                           torch_dtype=torch.float16,
                                                           device_map='auto')
