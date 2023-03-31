# @FileName：MaskInPaintingService.py
# @Description：
# @Author：dyh
# @Time：2023/2/18 15:57
# @Website：www.xxx.com
# @Version：V1.0
import base64
import io
import json
import math
import sys
from typing import Optional

import cv2
import requests
import uvicorn
from PIL import ImageChops, Image
from fastapi import FastAPI, Body
from pydantic import BaseModel
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
import numpy as np


class Item(BaseModel):
    img_url: str
    prompt: Optional[str] = ''
    negative_prompt: Optional[str] = ''


def base64_to_img(img_base64):
    image = Image.open(io.BytesIO(base64.b64decode(img_base64))).convert('RGB')
    return image


def img_array_to_base64(img):
    buff = io.BytesIO()
    Image.fromarray(img).convert('RGB').save(buff, format="JPEG")
    img_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_base64


def get_img(url):
    # wei: change to inner network address
    print("Downloading: ", url)
    response = requests.get(url, timeout=(3, 20))
    return Image.open(io.BytesIO(response.content))


def get_mask(url: str, img_url: str, prompt: str, negative_prompt: str):
    body = {
        "img_url": img_url,
        "prompt": prompt,
        "negative_prompt": negative_prompt
    }
    response = requests.post(url, json=body, timeout=(3, 20))
    response.encoding = 'utf-8'
    return response.text


def img_super(input_img, scale):
    print('img_super scale=', scale)
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4,
                            act_type='prelu')
    bg_upsampler = RealESRGANer(
        scale=4,  # 这里的值不能修改不然放大一倍或两倍图片会有问题
        model_path=['weights/realesr-general-x4v3.pth', 'weights/realesr-general-wdn-x4v3.pth'],
        dni_weight=[0.5, 0.5],
        model=model,
        tile=400,
        half=True,
        tile_pad=10,
        pre_pad=0)
    # my_logger.info('task only need super resolution..............')
    restored_img, _ = bg_upsampler.enhance(input_img, outscale=scale)
    return restored_img


def img_painting(url: str, mask_base64: str, img_base64: str, prompt: str, n_prompt: str):
    body = {
        "mask_base64": mask_base64,
        "img_base64": img_base64,
        "prompt": prompt,
        "negative_prompt": n_prompt,
        "return_oss": False
    }
    response = requests.post(url, json=body, timeout=(3, 20))
    response.encoding = 'utf-8'
    return response.text


# 获取扩大后的裁剪区域的两个像素点
def super_corp_region(size: int, source_big: int, crop_width: int, left: int, right: int):
    # 首先减去截去图片的宽度 然后取一半
    half = (size - crop_width) // 2
    # 然后用一半减去左边距，如果大于0表示头像靠近左边，要截取256 * 256的话就要在右边多取点像素
    left_diff = half - left
    if left_diff >= 0:  # 大于等于0证明头像偏右或居中
        # 这里证明右边不止要加一般的像素，还要把左边多出来的也加上去
        right_add_focus = half + left_diff
        new_left = 0
        new_right = right + right_add_focus
    else:  # 小于0证明左边可以加上一半的像素，判断右边
        # 同样的方法去判断右边距是否能够加一半
        right_diff = right + half
        if right_diff <= source_big:  # 右边可以完整的加上一半像素
            new_left = math.fabs(left_diff)
            new_right = right_diff
        else:
            left_add_focus = right_diff - source_big
            new_left = math.fabs(left_diff) - left_add_focus
            new_right = source_big
    return int(new_left), int(new_right)


class MaskInPainting:
    def run(self, img_url: str = None, prompt: str = '', negative_prompt: str = ''):
        file_name = img_url.split('/')[-1].split('.')[0]
        source_img = get_img(img_url)
        # TODO： 首先调用mask的接口获取base64
        mask_response = get_mask("http://127.0.0.1:9901/mask", img_url, "face", "")
        dict_json = json.loads(mask_response)
        img_base64 = dict_json['img_base64']
        mask_img = base64_to_img(img_base64)
        # 对两个图片进行累加
        diff = ImageChops.add(mask_img, mask_img)
        # 获取只有白色区域的坐标
        bbox = diff.getbbox()
        print('bbox=', bbox)
        left = bbox[0]
        upper = bbox[1]
        right = bbox[2]
        lower = bbox[3]
        crop_width = right - left
        crop_height = lower - upper
        source_width, source_height = mask_img.size
        scale = 2
        # 512 x 512图片 如果白色区域很小，就扩大
        if source_width == 512 and crop_width < 256 and source_height == 512 and crop_height < 256 and (
                crop_width * crop_height) < 256 * 256:  # 裁剪256 * 256
            nf, nr = super_corp_region(256, source_width, crop_width, left, right)
            nt, nb = super_corp_region(256, source_height, crop_height, upper, lower)
            bbox = (nf, nt, nr, nb)
            scale = 4
        elif source_width == 512 and source_height == 512:
            bbox = (0, 0, 512, 512)
            scale = 2

        # 1024 x 1024图片 如果白色区域很小，就扩大
        if source_width == 1024 and crop_width < 512 and source_height == 1024 and crop_height < 512 and (
                crop_width * crop_height) < 512 * 512:  # 裁剪256 * 256
            nf, nr = super_corp_region(512, source_width, crop_width, left, right)
            nt, nb = super_corp_region(512, source_height, crop_height, upper, lower)
            bbox = (nf, nt, nr, nb)
            scale = 2
        elif source_width == 1024 and source_height == 1024:
            bbox = (0, 0, 1024, 1024)
            scale = 1
        # 通过坐标裁剪
        print("big_bbox=", bbox)
        new_img = mask_img.crop(bbox)
        source_new_img = source_img.crop(bbox)

        # 对图片进行放大两倍
        numpy_super_img = img_super(np.asarray(new_img), scale)
        source_super_img = img_super(np.asarray(source_new_img), scale)
        # cv2.imwrite(file_name + 'crop_source.jpg', cv2.cvtColor(source_super_img, cv2.COLOR_RGB2BGR))
        # 调用inpainting
        painting_response = img_painting('http://127.0.0.1:5000/inpaint',
                                         mask_base64=img_array_to_base64(numpy_super_img),
                                         img_base64=img_array_to_base64(source_super_img), prompt=prompt,
                                         n_prompt=negative_prompt)
        paint_json = json.loads(painting_response)
        paint_base64 = paint_json['img_base64']
        painting_img = base64_to_img(paint_base64)
        # painting_img.save(file_name + '_inPain.jpg')
        # 获取图片大小
        w, h = painting_img.size
        painting_img = painting_img.resize((math.ceil(w / scale), math.ceil(h / scale)))
        # 把原图像素的位置给替换掉
        source_img.paste(painting_img, bbox)
        source_img.save('mask_paint_result/' + file_name + '.jpg')


app = FastAPI()


@app.post("/process")
async def extract(item: Item):
    print('img_url=', item.img_url, ' ,prompt=', item.prompt, ' ,negative_prompt=', item.negative_prompt)
    MaskInPainting().run(img_url=item.img_url, prompt=item.prompt, negative_prompt=item.negative_prompt)


if __name__ == "__main__":
    # param = sys.argv[1]
    # print(param)
    # MaskInPainting().run(img_url=param)
    uvicorn.run(app='MaskInPaintingService:app', host="0.0.0.0", port=9905, reload=False)
