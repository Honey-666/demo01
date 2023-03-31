# @FileName：ImgSuperService.py
# @Description：
# @Author：dyh
# @Time：2023/2/18 18:21
# @Website：www.xxx.com
# @Version：V1.0
import base64
from io import BytesIO

import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer


def base64_to_img(img_base64):
    image = Image.open(BytesIO(base64.b64decode(img_base64))).convert('RGB')
    return image


def img_array_to_base64(img):
    buff = BytesIO()
    Image.fromarray(img).convert('RGB').save(buff, format="JPEG")
    img_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_base64


def img_super(input_img):
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
    restored_img, _ = bg_upsampler.enhance(input_img, outscale=2)
    print(restored_img)
    return restored_img


app = FastAPI()


@app.post("/img/super")
def extract(img_base64: str):
    pli_img = base64_to_img(img_base64)
    print(pli_img)
    array_img = np.asfarray(pli_img)
    print(array_img)
    super_img = img_super(array_img)
    print(super_img)
    return img_array_to_base64(super_img)


if __name__ == "__main__":
    uvicorn.run(app='ImgSuperService:app', host="0.0.0.0", port=9902, reload=False)
