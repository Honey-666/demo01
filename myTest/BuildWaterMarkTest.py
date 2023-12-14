# -*- coding: utf-8 -*-
# @FileName：WaterMarkTest.py
# @Description：
# @Author：dyh
# @Time：2023/9/21 15:06
# @Website：www.xxx.com
# @Version：V1.0
import json
import os
import random
import time
from datetime import datetime

import cv2
import numpy
import numpy as np
from PIL import Image
from blind_watermark import WaterMark
from blind_watermark import WaterMark
from blind_watermark import att
from blind_watermark.recover import estimate_crop_parameters, recover_crop
import cv2
import os


def get_encoder(crop_img):
    bgr = cv2.cvtColor(numpy.asarray(crop_img), cv2.COLOR_RGB2BGR)
    s = time.time()
    bwm1 = WaterMark(password_img=1, password_wm=1)
    bwm1.read_img(img=bgr)
    wm = '{"ServiceProvider": "大设AIGC", "Time": "2023-10-09 11:55:09.525", "ContentID": "000000005099746_11_1"}'
    bwm1.read_wm(wm, mode='str')
    embed_img = bwm1.embed()

    len_wm = len(bwm1.wm_bit)
    print(f'consumting time = {time.time() - s} , length = {len_wm}')
    # pli_img = Image.fromarray(np.uint8(cv2.cvtColor(embed_img, cv2.COLOR_BGR2RGB)))
    # return pli_img
    return embed_img


def get_decoder(bgr_encoded, length):
    # bgr = cv2.cvtColor(numpy.asarray(bgr_encoded), cv2.COLOR_RGB2BGR)
    bwm1 = WaterMark(password_img=1, password_wm=1)
    # wm_extract = bwm1.extract(filename='./embedded.png', wm_shape=length, mode='str')
    wm_extract = bwm1.extract(embed_img=bgr_encoded, wm_shape=length, mode='str')
    return wm_extract


img_path = './bigbigai-1696902406038.jpg'
img = Image.open(img_path).convert('RGB')
w, h = img.size
for i in range(100):
    encoder_img = get_encoder(img)
    # x1 = 0
    # x2 = 512
    # y1 = 0
    # y2 = 512
    # encoder_img = encoder_img[y1:y2, x1:x2]  # [upper: lower, left: right]
    #
    # img_recovered = np.zeros((h, w, 3))
    #
    # img_recovered[y1:y2, x1:x2, :] = cv2.resize(encoder_img, dsize=(x2 - x1, y2 - y1))

    small_img = cv2.resize(encoder_img, (random.randint(384, 2048), random.randint(384, 2048)))
    img_recovered = cv2.resize(encoder_img, encoder_img.shape[:2][::-1])
    print(small_img.shape)
    mark = get_decoder(img_recovered, 839)
    if '{"ServiceProvider": "大设AIGC", "Time": "2023-10-09 11:55:09.525", "ContentID": "000000005099746_11_1"}' != mark:
        print('error')
    print(mark)

# os.chdir(os.path.dirname(__file__))
#
# bwm = WaterMark(password_img=1, password_wm=1)
# bwm.read_img('./bigbigai-1696902406038.jpg')
# wm = '@guofei9987 开源万岁！'
# bwm.read_wm(wm, mode='str')
# bwm.embed('./embedded.png')
#
# len_wm = len(bwm.wm_bit)  # 解水印需要用到长度
# print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))
#
# ori_img_shape = cv2.imread('./bigbigai-1696902406038.jpg').shape[:2]  # 抗攻击有时需要知道原图的shape
# h, w = ori_img_shape
#
# # %% 解水印
# bwm1 = WaterMark(password_img=1, password_wm=1)
# wm_extract = bwm1.extract('./embedded.png', wm_shape=len_wm, mode='str')
# print("不攻击的提取结果：", wm_extract)
#
# assert wm == wm_extract, '提取水印和原水印不一致'

# # %%截屏攻击1 = 裁剪攻击 + 缩放攻击 + 知道攻击参数（之后按照参数还原）
#
# loc_r = ((0.1, 0.1), (0.5, 0.5))
# scale = 0.7
#
# x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])
#
# # 截屏攻击
# att.cut_att3(input_filename='./embedded.png', output_file_name='./截屏攻击1.png',
#              loc=(x1, y1, x2, y2), scale=scale)
#
# recover_crop(template_file='./截屏攻击1.png', output_file_name='./截屏攻击1_还原.png',
#              loc=(x1, y1, x2, y2), image_o_shape=ori_img_shape)
#
# bwm1 = WaterMark(password_wm=1, password_img=1)
# wm_extract = bwm1.extract('./截屏攻击1_还原.png', wm_shape=len_wm, mode='str')
# print("截屏攻击，知道攻击参数。提取结果：", wm_extract)
# assert wm == wm_extract, '提取水印和原水印不一致'
#
# # %% 截屏攻击2 = 剪切攻击 + 缩放攻击 + 不知道攻击参数（因此需要 estimate_crop_parameters 来推测攻击参数）
# loc_r = ((0.1, 0.1), (0.7, 0.6))
# scale = 0.7
#
# x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])
#
# print(f'Crop attack\'s real parameters: x1={x1},y1={y1},x2={x2},y2={y2}')
# att.cut_att3(input_filename='./embedded.png', output_file_name='./截屏攻击2.png',
#              loc=(x1, y1, x2, y2), scale=scale)
#
# # estimate crop attack parameters:
# (x1, y1, x2, y2), image_o_shape, score, scale_infer = estimate_crop_parameters(original_file='./embedded.png',
#                                                                                template_file='./截屏攻击2.png',
#                                                                                scale=(0.5, 2), search_num=200)
#
# print(f'Crop att estimate parameters: x1={x1},y1={y1},x2={x2},y2={y2}, scale_infer = {scale_infer}. score={score}')
#
# # recover from attack:
# recover_crop(template_file='./截屏攻击2.png', output_file_name='./截屏攻击2_还原.png',
#              loc=(x1, y1, x2, y2), image_o_shape=image_o_shape)
#
# bwm1 = WaterMark(password_wm=1, password_img=1)
# wm_extract = bwm1.extract('./截屏攻击2_还原.png', wm_shape=len_wm, mode='str')
# print("截屏攻击，不知道攻击参数。提取结果：", wm_extract)
# assert wm == wm_extract, '提取水印和原水印不一致'
#
# # %%裁剪攻击1 = 裁剪 + 不做缩放 + 知道攻击参数
# loc_r = ((0.1, 0.2), (0.5, 0.5))
# x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])
#
# att.cut_att3(input_filename='./embedded.png', output_file_name='./随机裁剪攻击.png',
#              loc=(x1, y1, x2, y2), scale=None)
#
# # recover from attack:
# recover_crop(template_file='./随机裁剪攻击.png', output_file_name='./随机裁剪攻击_还原.png',
#              loc=(x1, y1, x2, y2), image_o_shape=image_o_shape)
#
# bwm1 = WaterMark(password_wm=1, password_img=1)
# wm_extract = bwm1.extract('./随机裁剪攻击_还原.png', wm_shape=len_wm, mode='str')
# print("裁剪攻击，知道攻击参数。提取结果：", wm_extract)
# assert wm == wm_extract, '提取水印和原水印不一致'
#
# # %% 裁剪攻击2 = 裁剪 + 不做缩放 + 不知道攻击参数
# loc_r = ((0.1, 0.1), (0.5, 0.4))
# x1, y1, x2, y2 = int(w * loc_r[0][0]), int(h * loc_r[0][1]), int(w * loc_r[1][0]), int(h * loc_r[1][1])
#
# att.cut_att3(input_filename='./embedded.png', output_file_name='./随机裁剪攻击2.png',
#              loc=(x1, y1, x2, y2), scale=None)
#
# print(f'Cut attack\'s real parameters: x1={x1},y1={y1},x2={x2},y2={y2}')
#
# # estimate crop attack parameters:
# (x1, y1, x2, y2), image_o_shape, score, scale_infer = estimate_crop_parameters(original_file='./embedded.png',
#                                                                                template_file='./随机裁剪攻击2.png',
#                                                                                scale=(1, 1), search_num=None)
#
# print(f'Cut attack\'s estimate parameters: x1={x1},y1={y1},x2={x2},y2={y2}. score={score}')
#
# # recover from attack:
# recover_crop(template_file='./随机裁剪攻击2.png', output_file_name='./随机裁剪攻击2_还原.png',
#              loc=(x1, y1, x2, y2), image_o_shape=image_o_shape)
#
# bwm1 = WaterMark(password_wm=1, password_img=1)
# wm_extract = bwm1.extract('./随机裁剪攻击2_还原.png', wm_shape=len_wm, mode='str')
# print("裁剪攻击，不知道攻击参数。提取结果：", wm_extract)
# assert wm == wm_extract, '提取水印和原水印不一致'
#
# # %%缩放攻击
# att.resize_att(input_filename='./embedded.png', output_file_name='./缩放攻击.png', out_shape=(400, 300))
# att.resize_att(input_filename='./缩放攻击.png', output_file_name='./缩放攻击_还原.png',
#                out_shape=ori_img_shape[::-1])
# # out_shape 是分辨率，需要颠倒一下
#
# bwm1 = WaterMark(password_wm=1, password_img=1)
# wm_extract = bwm1.extract('./缩放攻击_还原.png', wm_shape=len_wm, mode='str')
# print("缩放攻击后的提取结果：", wm_extract)
# assert np.all(wm == wm_extract), '提取水印和原水印不一致'
