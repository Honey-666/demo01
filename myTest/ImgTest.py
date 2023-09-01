# @FileName：ImgTest.py
# @Description：
# @Author：dyh
# @Time：2023/6/25 16:38
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy as np

from PIL import Image
from controlnet_aux.util import HWC3

from diffuser.controlNet.lvminthin import lvmin_thin, nake_nms


def img_show(img):
    cv2.imshow('demo', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fill_white_area2(background_img_path, overlay_img_path, output_img_path):
    # 读取背景图片和叠加图片
    background_img = cv2.imread(background_img_path)
    overlay_img = cv2.imread(overlay_img_path)

    # 确保两张图片的大小一致
    overlay_img = cv2.resize(overlay_img, (background_img.shape[1], background_img.shape[0]))

    # 将背景图片转换为灰度图像
    gray_background = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
    # 通过阈值化将白色区域分割出来
    _, white_area_mask = cv2.threshold(gray_background, 240, 255, cv2.THRESH_BINARY)
    # 反转白色区域的掩码，将白色区域变为黑色，其余区域变为白色
    white_area_mask_inv = cv2.bitwise_not(white_area_mask)
    # 将白色区域的掩码应用到叠加图片上
    img_show(white_area_mask)
    overlay_img_white_area = cv2.bitwise_and(overlay_img, overlay_img, mask=white_area_mask)
    # 将背景图片中的白色区域替换为叠加图片中对应的区域
    background_img_white_area = cv2.bitwise_and(background_img, background_img, mask=white_area_mask_inv)
    img_show(white_area_mask_inv)
    img_show(background_img_white_area)
    img_show(overlay_img_white_area)
    result_img = cv2.add(background_img_white_area, overlay_img_white_area)
    # 保存结果图片
    cv2.imwrite(output_img_path, result_img)


def fill_white_area(background_img_path, overlay_img_path, source_img_path, output_img_path):
    # 打开背景图片和叠加图片
    background_img = Image.open(background_img_path)
    overlay_img = Image.open(overlay_img_path)
    source_img = Image.open(source_img_path)

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
            if background_data[x, y] == (255, 255, 255) or background_data[x, y] == (255, 255, 255, 255):
                source_data[x, y] = overlay_data[x, y]

    # 保存结果图片
    source_img.save(output_img_path)


# 使用示例
source_img_path = "../img/test/20230724-result.jpg"  # 彩色背景图片路径，其中有一块区域是白色
mask_img_path = "../img/test/20230724-110413.jpg"  # 彩色背景图片路径，其中有一块区域是白色
overlay_img_path = "../img/test/20230724-110418.jpg"  # 另一张彩色图片路径
output_img_path = "../img/test/output.jpg"  # 输出图片路径
output_img_path2 = "../img/test/output2.jpg"  # 输出图片路径

# fill_white_area(mask_img_path, overlay_img_path, source_img_path, output_img_path)
# fill_white_area2(mask_img_path, overlay_img_path, output_img_path2)
from torchvision import transforms
import matplotlib.pyplot as plt

trans_pil = transforms.ToPILImage()

pli_img = Image.open(source_img_path)
pli_img.show()
transform = transforms.Compose([transforms.ToTensor()])
tensor_img = transform(pli_img)

# 显示PyTorch张量表示的图像
plt.imshow(tensor_img.permute(1, 2, 0))  # 调整维度顺序以适应Matplotlib的显示
plt.axis('off')  # 关闭坐标轴
plt.show()

resize_transform = transforms.Resize((1536, 2048))
resized_img_tensor = resize_transform(tensor_img)
new_img = trans_pil(resized_img_tensor)
new_img.show()
