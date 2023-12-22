# @FileName：ControlNetSavePretrained.py
# @Description：
# @Author：dyh
# @Time：2023/12/18 15:21
# @Website：www.xxx.com
# @Version：V1.0
import torch
from diffusers import ControlNetModel
from diffusers.pipelines.controlnet import MultiControlNetModel

openpose_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\controlnet-openpose-sdxl-1.0'
dump_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\controlnet-canny-sdxl-1.0-fp16'

openpose_model = ControlNetModel.from_pretrained(openpose_path, torch_dtype=torch.float16).to('cuda')
openpose_model.save_pretrained(dump_path, safe_serialization=False)
