# @FileName：Demo29.py
# @Description：
# @Author：dyh
# @Time：2023/3/27 16:21
# @Website：www.xxx.com
# @Version：V1.0
import os

import cv2
from PIL import Image, ImageChops


# Load image
# img = cv2.imread('./img/skirt_mask/1679638479386_mask.jpg')
#
# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Threshold to get white regions
# thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
# # thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
#
# # Find contours
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Get largest contour
# largest_contour = max(contours, key=cv2.contourArea)
#
# # Get bounding rectangle of largest contour
# x, y, w, h = cv2.boundingRect(largest_contour)
# print(x, y, h, w)


def make_transparent(image_path):
    img = Image.open(image_path)
    img = ImageChops.invert(img)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    img.save('./img/transparent/' + image_path.split('/')[-1].split('.')[0] + '.png', "PNG")


def combine_images(background_path, foreground_path, output_path):
    background = Image.open(background_path)
    foreground = Image.open(foreground_path).convert("RGBA")
    background.alpha_composite(foreground)
    background.save(output_path)


# combine_images('./img/dress-edit/1679638479386.jpg', 'test_crop.png', 'imgAddCV.png')

files = os.listdir('./img/skirt_mask')
for f in files:
    print(f)
    make_transparent('./img/skirt_mask/' + f)
    source_img = Image.open("./img/dress-edit/" + f.replace('_mask', ''))
    transparent_img = Image.open("./img/transparent/" + f.split('.')[0] + '.png')

    # 将第二张图片叠加到第一张图片上，并取透明部分为主
    result = Image.alpha_composite(source_img, transparent_img)
    new_image = Image.new("RGBA", result.size, "WHITE")
    new_image.paste(result, mask=result)
    new_image = new_image.convert("RGB")
    # 保存结果图片
    new_image.save('./img/result/' + f.replace('_mask', '_result'))
