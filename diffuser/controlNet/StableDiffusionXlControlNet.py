# @FileName：StableDiffusionXlControlNet.py
# @Description：
# @Author：dyh
# @Time：2023/8/9 14:44
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline, ControlNetModel

from compel import Compel, ReturnedEmbeddingsType


def long_prompt_handle(pipe, prompt, negative_prompt):
    compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    truncate_long_prompts=False, device='cuda',
                    requires_pooled=[False, True])

    pp_embeds, pp_pooled = compel(prompt)
    np_embeds, np_pooled = compel(negative_prompt)

    [pp_embeds, np_embeds] = compel.pad_conditioning_tensors_to_same_length([pp_embeds, np_embeds])
    return pp_embeds, np_embeds


sd_xl_path = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
controlnet_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\controlnet-canny-sdxl-1.0'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\hf-logo.png'

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(sd_xl_path,
                                                           controlnet=controlnet,
                                                           torch_dtype=torch.float16,
                                                           safety_checker=None)

pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

prompt = 'aerial view, a futuristic research complex in a bright foggy jungle, hard lighting'
negative_prompt = 'low quality, bad quality, sketches'
controlnet_conditioning_scale = 0.5

prompt_embeds, negative_prompt_embeds = long_prompt_handle(pipe, prompt, negative_prompt)

image = Image.open(img_path)
image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

img = pipe(prompt_embeds=prompt_embeds,
           negative_prompt_embeds=negative_prompt_embeds,
           controlnet_conditioning_scale=controlnet_conditioning_scale,
           num_inference_steps=30,
           image=canny_image).images[0]
img.show()


def load_controlnet_model(model_keys: list = None):
    controlnet_model_dict = {}
    def m(x, v):return ControlNetModel.from_pretrained(x, variant=v, torch_dtype=torch.float16, device_map='auto')

    model = {
        'scribble': '../models/ControlNet/control-v11p-sd15-scribble',
        'canny': '../models/ControlNet/control-v11p-sd15-canny',
        'depth': '../models/ControlNet/control-v11f1p-sd15-depth',
        'seg': '../models/ControlNet/control-v11p-sd15-seg',
        'mlsd': '../models/ControlNet/control-v11p-sd15-mlsd',
        'openpose': '../models/ControlNet/control-v11p-sd15-openpose',
        'tile': '../models/ControlNet/control-v11f1e-sd15-tile'}

    if not model_keys:  # 文生图需要添加openpose
        model_keys = model.keys()

    print(f'init control mode: {",".join(model_keys)}')
    for k in model_keys:
        variant = None if k == 'tile' else "fp16"
        model_id = model[k]
        if os.path.exists(model_id):
            controlnet_model_dict[k] = m(model_id, variant)
        else:
            print(f'model {model_id} not exist')


    return controlnet_model_dict