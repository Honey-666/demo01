# @FileName：Text2MaskService.py
# @Description：
# @Author：dyh
# @Time：2023/2/10 16:13
# @Website：www.xxx.com
# @Version：V1.0
# Author: Therefore Games
# https://github.com/ThereforeGames/txt2img2img
import datetime
import io
import sys

import torch
import cv2
import requests
import os.path

from clipseg import CLIPDensePredT
from PIL import ImageChops, Image
from torchvision import transforms
import numpy
from python_bigbigwork_util import LoggingUtil

# Keep the console clear - configure werkzeug (flask's WSGI web app) not to log the detail of every incoming request
my_logger = LoggingUtil.get_logging('text2_mask_service')


def get_img(url):
    # wei: change to inner network address
    print("Downloading: ", url)
    response = requests.get(url, timeout=(3, 20))
    return Image.open(io.BytesIO(response.content))


# 下载模型
def download_file(filename, url):
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        # 如果下载失败抛出异常
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)


def load_dfile(model):
    model.eval()
    model_dir = "./weights"
    os.makedirs(model_dir, exist_ok=True)
    d64_file = f"{model_dir}/rd64-uni.pth"
    d16_file = f"{model_dir}/rd16-uni.pth"

    # Download model weights if we don't have them yet
    if not os.path.exists(d64_file):
        print("Downloading clipseg model weights...")
        download_file(d64_file,
                      "https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd64-uni.pth")
        download_file(d16_file,
                      "https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd16-uni.pth")
    # Mirror:
    # https://github.com/timojl/clipseg/raw/master/weights/rd64-uni.pth
    # https://github.com/timojl/clipseg/raw/master/weights/rd16-uni.pth

    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load(d64_file, map_location=torch.device('cuda')), strict=False)


# load model
print('init model start..............', str(datetime.datetime.now()))
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
# load path
load_dfile(model)
print('init model end..............', str(datetime.datetime.now()))


class Mask:
    def run(self, image: Image, mask_prompt: str, negative_mask_prompt: str, mask_precision: int, mask_padding: int,
            brush_mask_mode: int, filename: str):

        # pli类型的图片转 cv2(其中numpy.array(img) 将一个pli转numpy)
        def pil_to_cv2(img):
            return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)

        # 将numpy转为PLI类型的图片
        def gray_to_pil(img):
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 对图片进行裁剪
        def center_crop(img, new_width, new_height):
            width, height = img.size  # Get dimensions

            left = (width - new_width) / 2
            top = (height - new_height) / 2
            right = (width + new_width) / 2
            bottom = (height + new_height) / 2

            # Crop the center of the image
            return (img.crop((left, top, right, bottom)))

        # 传入两张图片进行组合，返回组合后的新图片
        def overlay_mask_part(img_a, img_b, mode):
            if mode == 0:
                # 将两张图片的每个像素取出做对比，哪个像素颜色深决定该像素用哪个，组合出一张新的图片
                img_a = ImageChops.darker(img_a, img_b)
            else:
                # 与darker相反，取像素值浅的哪个组合出一张新的图片
                img_a = ImageChops.lighter(img_a, img_b)
            return img_a

        def process_mask_parts(these_preds, these_prompt_parts, mode, final_img=None):
            for i in range(these_prompt_parts):
                # filename = f"mask_{mode}_{i}.png"
                # plt.imsave(filename, torch.sigmoid(these_preds[i][0]))
                #
                # # TODO: Figure out how to convert the plot above to numpy instead of re-loading image
                # img = cv2.imread(filename)
                # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # --------------上面哪种方式会往本地保存图片，多个线程访问这张图的话会出现mask获取不对的情况，
                tensor_to_numpy = torch.sigmoid(these_preds[i][0])
                # tensor 转换成 Image
                trans_pil = transforms.ToPILImage()
                image_pil = trans_pil(tensor_to_numpy)
                gray_image = cv2.cvtColor(pil_to_cv2(image_pil), cv2.COLOR_BGR2GRAY)
                # ---------------到这里是替换上面注释掉往本地写图片再去读取的情况

                (thresh, bw_image) = cv2.threshold(gray_image, mask_precision, 255, cv2.THRESH_BINARY)

                if mode == 0:
                    bw_image = numpy.invert(bw_image)

                # overlay mask parts
                bw_image = gray_to_pil(bw_image)
                if i > 0 or final_img is not None:
                    bw_image = overlay_mask_part(bw_image, final_img, mode)

                final_img = bw_image

            return final_img

        # -----------------------------------------get mask-----------------
        def get_mask(oss_img):
            delimiter_string = "|"
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((512, 512)),
            ])
            img = transform(oss_img).unsqueeze(0)
            # 根据传入的描述以 "|" 分割成数组
            prompts = mask_prompt.split(delimiter_string)
            # 描述数组的长度
            prompt_parts = len(prompts)
            # 根据传入的排除描述以 "|" 分割成数组
            negative_prompts = negative_mask_prompt.split(delimiter_string)
            # 排除描述数组的长度
            negative_prompt_parts = len(negative_prompts)

            # predict
            with torch.no_grad():
                preds = model(img.repeat(prompt_parts, 1, 1, 1), prompts)[0]
                negative_preds = model(img.repeat(negative_prompt_parts, 1, 1, 1), negative_prompts)[0]

            if brush_mask_mode == 1 and p.image_mask is not None:
                final_img = p.image_mask.convert("RGBA")
            else:
                # 判断走这里
                final_img = None

            # process masking
            final_img = process_mask_parts(preds, prompt_parts, 1, final_img)

            if negative_mask_prompt:
                final_img = process_mask_parts(negative_preds, negative_prompt_parts, 0, final_img)
            # Increase mask size with padding
            if mask_padding > 0:
                aspect_ratio = oss_img.width / oss_img.height
                new_width = oss_img.width + mask_padding * 2
                new_height = round(new_width / aspect_ratio)
                final_img = final_img.resize((new_width, new_height))
                final_img = center_crop(final_img, oss_img.width, oss_img.height)

            return final_img

        try:
            # get_mask_img
            image_mask = get_mask(image).resize((image.width, image.height))
            image_mask = image_mask.convert('RGB')
            # 变为numpy类型，做中值滤波平滑边缘
            cv2_img = pil_to_cv2(image_mask)
            median_blur_img = cv2.medianBlur(cv2_img, 51)
            cv2.imwrite('./dress-edit/' + filename + '_mask.jpg', median_blur_img)
            my_logger.info("handle success..........")
        except Exception as e:
            my_logger.exception('handle exception:{}', e)


files = os.listdir('./dress-edit')
for f in files:
    im = Image.open('./dress-edit/' + f).convert("RGB")
    im = im.resize((im.size[0] // 4, im.size[1] // 4))
    file_name = f.split('.')[0]
    en_prompt = 'dress'
    en_negative_prompt = ''
    Mask().run(image=im, mask_prompt=en_prompt, negative_mask_prompt=en_negative_prompt,
               mask_precision=100,
               mask_padding=0, brush_mask_mode=0, filename=file_name)
