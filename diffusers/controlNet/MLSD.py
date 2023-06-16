# @FileName：FackScribble.py
# @Description：
# @Author：dyh
# @Time：2023/3/20 11:13
# @Website：www.xxx.com
# @Version：V1.0
import sys
import time
from enum import Enum

import cv2
import numpy as np
import torch
from PIL import Image
from controlnet_aux import MLSDdetector
from controlnet_aux.mlsd import MobileV2_MLSD_Large, apply_mlsd
from controlnet_aux.util import HWC3
from einops import rearrange

from lvminthin import nake_nms, lvmin_thin
from utils.util import resize_image


def detectmap_proc(detected_map, module, resize_mode, h, w):
    if 'inpaint' in module:
        detected_map = detected_map.astype(np.float32)
    else:
        detected_map = HWC3(detected_map)

    def safe_numpy(x):
        # A very safe method to make sure that Apple/Mac works
        y = x

        # below is very boring but do not change these. If you change these Apple or Mac may fail.
        y = y.copy()
        y = np.ascontiguousarray(y)
        y = y.copy()
        return y

    # def get_pytorch_control(x):
    #     # A very safe method to make sure that Apple/Mac works
    #     y = x
    #
    #     # below is very boring but do not change these. If you change these Apple or Mac may fail.
    #     y = torch.from_numpy(y)
    #     y = y.float() / 255.0
    #     y = rearrange(y, 'h w c -> c h w')
    #     y = y.clone()
    #     y = y.to(devices.get_device_for("controlnet"))
    #     y = y.clone()
    #     return y

    def high_quality_resize(x, size):
        # Written by lvmin
        # Super high-quality control map up-scaling, considering binary, seg, and one-pixel edges

        inpaint_mask = None
        if x.ndim == 3 and x.shape[2] == 4:
            inpaint_mask = x[:, :, 3]
            x = x[:, :, 0:3]

        new_size_is_smaller = (size[0] * size[1]) < (x.shape[0] * x.shape[1])
        new_size_is_bigger = (size[0] * size[1]) > (x.shape[0] * x.shape[1])
        unique_color_count = np.unique(x.reshape(-1, x.shape[2]), axis=0).shape[0]
        is_one_pixel_edge = False
        is_binary = False
        if unique_color_count == 2:
            is_binary = np.min(x) < 16 and np.max(x) > 240
            if is_binary:
                xc = x
                xc = cv2.erode(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                xc = cv2.dilate(xc, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)
                one_pixel_edge_count = np.where(xc < x)[0].shape[0]
                all_edge_count = np.where(x > 127)[0].shape[0]
                is_one_pixel_edge = one_pixel_edge_count * 2 > all_edge_count

        if 2 < unique_color_count < 200:
            interpolation = cv2.INTER_NEAREST
        elif new_size_is_smaller:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC  # Must be CUBIC because we now use nms. NEVER CHANGE THIS

        y = cv2.resize(x, size, interpolation=interpolation)
        if inpaint_mask is not None:
            inpaint_mask = cv2.resize(inpaint_mask, size, interpolation=interpolation)

        if is_binary:
            y = np.mean(y.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)
            if is_one_pixel_edge:
                y = nake_nms(y)
                _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                y = lvmin_thin(y, prunings=new_size_is_bigger)
            else:
                _, y = cv2.threshold(y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            y = np.stack([y] * 3, axis=2)

        if inpaint_mask is not None:
            y[inpaint_mask > 127] = - 255

        return y

    if resize_mode == 'Just Resize':
        detected_map = high_quality_resize(detected_map, (w, h))
        detected_map = safe_numpy(detected_map)
        return detected_map

    old_h, old_w, _ = detected_map.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w

    safeint = lambda x: int(np.round(x))

    if resize_mode == 'Resize and Fill':
        k = min(k0, k1)
        borders = np.concatenate(
            [detected_map[0, :, 0:3], detected_map[-1, :, 0:3], detected_map[:, 0, 0:3], detected_map[:, -1, 0:3]],
            axis=0)
        high_quality_border_color = np.median(borders, axis=0).astype(detected_map.dtype)
        high_quality_background = np.tile(high_quality_border_color[None, None], [h, w, 1])
        detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (h - new_h) // 2)
        pad_w = max(0, (w - new_w) // 2)
        high_quality_background[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = detected_map
        detected_map = high_quality_background
        detected_map = safe_numpy(detected_map)
        return detected_map
    else:
        k = max(k0, k1)
        detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (new_h - h) // 2)
        pad_w = max(0, (new_w - w) // 2)
        detected_map = detected_map[pad_h:pad_h + h, pad_w:pad_w + w]
        detected_map = safe_numpy(detected_map)
        return detected_map


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


def resize_image_with_pad(input_image, resolution):
    img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad


mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')


def mlsd_pre(img, res=512, thr_a=0.1, thr_b=0.1, **kwargs):
    thr_v, thr_d = thr_a, thr_b
    img = resize_image(HWC3(img), res)
    image = apply_mlsd(img, thr_v, thr_d)
    image = detectmap_proc(image, 'mlsd', 'Crop and Resize', 352, 768)
    return image


# image = Image.open('../test_img/' + sys.argv[1])
image = cv2.imread('../../img/control/shinei.png')
s = time.time()
mlsd(image)
image = mlsd_pre(image)
spend = time.time() - s
print(spend)
cv2.imwrite('mlsd_scribble_out.png', image)
