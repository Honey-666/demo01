# @FileName：ImgTest.py
# @Description：
# @Author：dyh
# @Time：2023/6/25 16:38
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy as np

from PIL import Image, ImageChops
from controlnet_aux.util import HWC3

from diffuser.controlNet.lvminthin import lvmin_thin, nake_nms


def img_show(img):
    cv2.imshow('demo', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def fill_white_area(background_img_path, overlay_img_path, source_img_path, output_img_path):
#     # 打开背景图片和叠加图片
#     background_img = Image.open(background_img_path)
#     overlay_img = Image.open(overlay_img_path)
#     source_img = Image.open(source_img_path)
#
#     # 确保两张图片的大小一致
#     if background_img.size != overlay_img.size or background_img.size != overlay_img.size:
#         overlay_img = overlay_img.resize(background_img.size, Image.LANCZOS)
#         source_img = source_img.resize(background_img.size, Image.LANCZOS)
#
#     # 获取两张图片的像素数据
#     background_data = background_img.load()
#     overlay_data = overlay_img.load()
#     source_data = source_img.load()
#
#     # 遍历图片像素，将白色区域填充
#     for y in range(background_img.size[1]):
#         for x in range(background_img.size[0]):
#             # 如果背景图片的像素为白色，则用叠加图片的像素替换
#             if background_data[x, y] == (0, 0, 0) or background_data[x, y] == (0, 0, 0, 0):
#                 source_data[x, y] = overlay_data[x, y]
#
#     # 保存结果图片
#     source_img.save(output_img_path)
#
#
# # 使用示例
# source_img_path = "../img/test/01b14459-579a-11ee-aef7-525400aa6c77.jpeg"  # 彩色背景图片路径，其中有一块区域是白色
# mask_img_path = "../img/test/invert_mask.png"  # 彩色背景图片路径，其中有一块区域是白色
# overlay_img_path = "../img/test/20230711-155847.jpg"  # 另一张彩色图片路径
# output_img_path = "../img/test/output.jpg"  # 输出图片路径
#
# fill_white_area(mask_img_path, overlay_img_path, source_img_path, output_img_path)

# ----------------------------------------------------
# from torchvision import transforms
# import matplotlib.pyplot as plt
#
# trans_pil = transforms.ToPILImage()
#
# pli_img = Image.open(source_img_path)
# pli_img.show()
# transform = transforms.Compose([transforms.ToTensor()])
# tensor_img = transform(pli_img)
#
# # 显示PyTorch张量表示的图像
# plt.imshow(tensor_img.permute(1, 2, 0))  # 调整维度顺序以适应Matplotlib的显示
# plt.axis('off')  # 关闭坐标轴
# plt.show()
#
# resize_transform = transforms.Resize((1536, 2048))
# resized_img_tensor = resize_transform(tensor_img)
# new_img = trans_pil(resized_img_tensor)
# new_img.show()


def fill_white_area(background_img_path, overlay_img_path, source_img_path):
    # 打开背景图片和叠加图片
    background_img = Image.open(background_img_path).convert('RGB')
    overlay_img = Image.open(overlay_img_path).convert('RGB')
    source_img = Image.open(source_img_path).convert('RGB')

    # 确保两张图片的大小一致
    if background_img.size != overlay_img.size or background_img.size != overlay_img.size:
        overlay_img = overlay_img.resize(background_img.size, Image.LANCZOS)
        source_img = source_img.resize(background_img.size, Image.LANCZOS)

    # 获取两张图片的像素数据
    background_data = background_img.load()
    overlay_data = overlay_img.load()
    source_data = source_img.load()

    # 遍历图片像素，将白色区域填充
    for y in range(background_img.size[1]):
        for x in range(background_img.size[0]):
            # 如果背景图片的像素为白色，则用叠加图片的像素替换
            if background_data[x, y] == (0, 0, 0) or background_data[x, y] == (0, 0, 0, 0):
                source_data[x, y] = overlay_data[x, y]

    return source_img


# mask_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\vermeer-mask.jpg'
# img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\vermeer.png'
# rs_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\vermeer2.jpg'
# source_img = fill_white_area(mask_path, img_path, rs_path)
# source_img.show()

img_path = 'C:\\Users\\bbw\\Desktop\\base64-img.jpg'

im = Image.open(img_path)
invert_im = ImageChops.invert(im).convert("RGB")
invert_im.save('C:\\Users\\bbw\\Desktop\\invert-img.jpg')
