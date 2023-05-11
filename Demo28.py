# @FileName：Demo28.py
# @Description：
# @Author：dyh
# @Time：2023/3/27 14:23
# @Website：www.xxx.com
# @Version：V1.0
import os.path

import cv2
import numpy
from PIL import Image


# files = os.listdir('./img/skirt_mask')
# for f in files:
#     # Load image
#     img = cv2.imread('./img/skirt_mask/' + f)
#
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Threshold to get white regions
#     thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
#     # thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
#
#     # Find contours
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Get largest contour
#     largest_contour = max(contours, key=cv2.contourArea)
#
#     # Get bounding rectangle of largest contour
#     x, y, w, h = cv2.boundingRect(largest_contour)
#
#     constant = 0
#     img2 = cv2.imread('./img/dress-edit/' + f.replace("_mask", ''))
#     # img2 = Image.open('./img/dress-edit/' + f.replace("_mask", ''))
#     # img2 = img2.resize((img2.size[0] // 4, img2.size[1] // 4))
#     # img2 = cv2.cvtColor(numpy.asarray(img2), cv2.COLOR_RGB2BGR)
#     # Crop image to bounding rectangle
#     crop_img = img2[y:(y + constant) + (h + constant), x:(x + constant) + (w + constant)]
#
#     # Save cropped image
#     cv2.imwrite('./img/result/' + f.replace("_mask", ''), crop_img)

def resize768(path: str):
    img = cv2.imread(path)
    height, width, _ = img.shape
    if width < height:
        img = cv2.copyMakeBorder(img, 0, 0, (height - width) // 2, (height - width) // 2, cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])
    else:
        img = cv2.copyMakeBorder(img, (width - height) // 2, (width - height) // 2, 0, 0, cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])
    img = cv2.resize(img, (768, 768))
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


def resize768_pal(path: str):
    pli_img = Image.open(path).convert('RGBA')
    new_image = Image.new("RGBA", pli_img.size, "WHITE")
    new_image.paste(pli_img, mask=pli_img)
    new_image = new_image.convert("RGB")
    img = cv2.cvtColor(numpy.asarray(new_image), cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape
    if width < height:
        img = cv2.copyMakeBorder(img, 0, 0, (height - width) // 2, (height - width) // 2, cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])
    else:
        img = cv2.copyMakeBorder(img, (width - height) // 2, (width - height) // 2, 0, 0, cv2.BORDER_CONSTANT,
                                 value=[255, 255, 255])
    img = cv2.resize(img, (768, 768))
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


def is_white(img: Image):
    # 获取图像的所有像素值
    pixels = img.load()

    # 遍历所有像素并检查它们是否是白色
    is_all_white = True
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixels[i, j] != (255, 255, 255):
                is_all_white = False
                break
        if not is_all_white:
            break
    return is_all_white


# files = os.listdir('./img/result')
# for f in files:
#     print(f)
#     result_path = './img/result/' + f
#     result_img = resize768(result_path)
#
#     source_path = './img/dress-edit/' + f.replace("_result", "")
#     source_img = resize768_pal(source_path)
#     # 不是全部白色图，进行剪切，重拼
#     is_white = is_white(result_img)
#     if not is_white:
#         bbox = (0, result_img.size[1] // 2, result_img.size[0], result_img.size[1])
#         result_section_img = result_img.crop(bbox)
#         source_img.paste(result_section_img, bbox)
#
#     source_img.save('./img/768/' + f.replace("_result", "_768"))


img = Image.open('img/control/bag.png')
mode = img.mode
print(mode)
