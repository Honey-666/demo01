# @FileName：PathTest.py
# @Description：
# @Author：dyh
# @Time：2023/6/19 15:23
# @Website：www.xxx.com
# @Version：V1.0
import decimal
import math
from decimal import Decimal
import json


def cropWH(size: int) -> int:
    mod = size % 8
    if mod == 0:
        return size
    return size + (8 - mod)


def calcWAndH(w: int, h: int, strength: float, flag_delicate: bool, flag_txt: bool, flag_short: bool,
              imgLongSide: float) -> json:
    global img2imgH, img2imgW
    param = {}

    if flag_short:  # 短边和512计算比例
        targetLen = h if w > h else w
        txtRatio = targetLen / 512.0
    else:  # 长边和768计算比例
        targetLen = w if w > h else h
        txtRatio = targetLen / 768.0

    upscale = txtRatio
    text2imgH = cropWH(math.ceil(h / txtRatio))
    text2imgW = cropWH(math.ceil(w / txtRatio))

    if flag_delicate:
        text2imgMaxLen = text2imgW if text2imgW > text2imgH else text2imgH
        imgRatio = imgLongSide / text2imgMaxLen
        img2imgH = cropWH(math.ceil(text2imgH * imgRatio))
        img2imgW = cropWH(math.ceil(text2imgW * imgRatio))

        maxLen = w if w > h else h
        upscale = maxLen / (img2imgW if img2imgW > img2imgH else img2imgH)
        upscale = float(Decimal(upscale).quantize(Decimal('0.00'), rounding=decimal.ROUND_UP))

    if flag_txt and not flag_delicate:  # 文生图非精绘
        param['text2img'] = {'height': text2imgH, 'width': text2imgW}
        param['upscale'] = upscale
    if flag_txt and flag_delicate:  # 文生图精绘
        param['text2img'] = {'height': text2imgH, 'width': text2imgW}
        param['img2img2'] = {'height': img2imgH, 'width': img2imgW, 'strength': 0.6}
        param['upscale'] = upscale
    if not flag_txt and not flag_delicate:  # 图生图非精绘
        param['img2img1'] = {'height': text2imgH, 'width': text2imgW, 'strength': strength}
        param['upscale'] = upscale
    if not flag_txt and flag_delicate:  # 图生图精绘
        param['img2img1'] = {'height': text2imgH, 'width': text2imgW, 'strength': strength}
        param['img2img2'] = {'height': img2imgH, 'width': img2imgW, 'strength': 0.6}
        param['upscale'] = upscale

    return param


def getPlanParam(plan: int, width: int, height: int, strength: float) -> json:
    js = {}
    if plan == 0:
        js['text2img'] = {'height': height, "width": width}
    elif plan == 1:
        js.update(calcWAndH(width, height, strength, False, True, True, 1024.0))
    elif plan == 2:
        js.update(calcWAndH(width, height, strength, False, True, False, 1024.0))
    elif plan == 3:
        js.update(calcWAndH(width, height, strength, True, True, True, 1024.0))
    elif plan == 4:
        js.update(calcWAndH(width, height, strength, True, True, False, 1024.0))
    elif plan == 5:
        js['img2img1'] = {'height': height, "width": width, "strength": strength}
    elif plan == 6:
        js.update(calcWAndH(width, height, strength, False, False, False, 1024.0))
    elif plan == 7:
        js.update(calcWAndH(width, height, strength, True, False, False, 1024.0))
    elif plan == 8:
        js.update(calcWAndH(width, height, strength, True, True, False, 1536.0))
    elif plan == 9:
        js.update(calcWAndH(width, height, strength, True, False, False, 1536.0))

    return js


print(getPlanParam(7, 2339, 2339, 0.8))
