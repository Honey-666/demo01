# @FileName：YoLoTest.py
# @Description：
# @Author：dyh
# @Time：2023/7/7 13:43
# @Website：www.xxx.com
# @Version：V1.0
import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


# 获取扩大后的裁剪区域的两个像素点
def padding_corp_region(size: int, source_big: int, crop_width: int, left: int, right: int):
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


def create_mask_from_bbox(bboxes: list[list[float]], shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def mask_to_pil(masks, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image
    """
    from torchvision.transforms.functional import to_pil_image

    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


def calc_max_face(bboxes):
    sort_dict = {}
    for i, bbox in enumerate(bboxes):
        x0 = bbox[0]
        x1 = bbox[2]
        y0 = bbox[1]
        y1 = bbox[3]
        area = ((x1 - x0) + (y1 - y0)) * 2  # 周长
        sort_dict[i] = area

    max_tuple = max(sorted(sort_dict.items(), key=lambda x: x[1], reverse=True))
    return max_tuple


image = Image.open('./bigbigai.com-1690539927794.jpg')
model_path = 'C:\\Users\\bbw\\.cache\\huggingface\\hub\\models--Bingsu--adetailer\\snapshots\\fdb6e26f5212c6a7184b359f62cc4b41fd731bb3\\face_yolov8s.pt'

model = YOLO(model_path)

pred = model(image, conf=0.5)
bboxes = pred[0].boxes.xyxy.cpu().numpy()

mask_idx = 0
if bboxes.size == 0:
    masks = None
else:
    bboxes = bboxes.tolist()
    mask_idx = calc_max_face(bboxes)[0]

    if pred[0].masks is None:
        masks = create_mask_from_bbox([bboxes[mask_idx]], image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

mask_img = masks[0]
bbox = bboxes[mask_idx]
print(bbox)
left = bbox[0]
upper = bbox[1]
right = bbox[2]
lower = bbox[3]

diff_width = right - left
diff_height = lower - upper
source_width, source_height = mask_img.size
face_w = face_h = None
if diff_width < 512 and diff_height < 512:
    nf, nr = padding_corp_region(diff_width + (32 * 2), source_width, diff_width, left, right)
    nt, nb = padding_corp_region(diff_height + (32 * 2), source_height, diff_height, upper, lower)
    print(nf, nt, nr, nb)
    bbox = (nf, nt, nr, nb)
    crop_face = mask_img.crop(bbox)
    crop_source_img = image.crop(bbox)
    mask_img.show()
    crop_face.show()
    crop_source_img.show()
    face_w, face_h = crop_face.size
    max_len = max(face_w, face_h)
    ratio = 512 / max_len
    super_width = int(face_w * ratio)
    super_height = int(face_h * ratio)
    # restored_img = cv2.resize(cv2.cvtColor(np.asarray(crop_face), cv2.COLOR_RGB2BGR), (super_width, super_height),
    #                           interpolation=cv2.INTER_LANCZOS4)
    # super_mask_face = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
    # restored_img2 = cv2.resize(cv2.cvtColor(np.asarray(crop_source_img), cv2.COLOR_RGB2BGR), (super_width, super_height),
    #                           interpolation=cv2.INTER_LANCZOS4)
    # super_source_face = Image.fromarray(cv2.cvtColor(restored_img2, cv2.COLOR_BGR2RGB))
    super_mask_face = crop_face.resize((super_width, super_height), resample=Image.LANCZOS)
    super_source_face = crop_source_img.resize((super_width, super_height), resample=Image.LANCZOS)
    super_mask_face.show()
    super_source_face.show()

# preview = pred[0].plot()
# preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
# preview = Image.fromarray(preview)
# preview.show()
# for mask in masks:
#     mask.show()
