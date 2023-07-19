# @FileName：YoLoTest.py
# @Description：
# @Author：dyh
# @Time：2023/7/7 13:43
# @Website：www.xxx.com
# @Version：V1.0
import time

import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO


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


image = Image.open('./yolo_test3.jpg')
model_path = 'C:\\Users\\bbw\\.cache\\huggingface\\hub\\models--Bingsu--adetailer\\snapshots\\fdb6e26f5212c6a7184b359f62cc4b41fd731bb3\\face_yolov8s.pt'

model = YOLO(model_path)

for _ in range(5):
    s = time.time()
    pred = model(image, conf=0.3)
    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    print('spend time=', time.time() - s)

    if bboxes.size == 0:
        masks = None
    else:
        bboxes = bboxes.tolist()
        mask_idx = calc_max_face(bboxes)[0]

        if pred[0].masks is None:
            masks = create_mask_from_bbox([bboxes[mask_idx]], image.size)
        else:
            masks = mask_to_pil(pred[0].masks.data, image.size)
# preview = pred[0].plot()
# preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
# preview = Image.fromarray(preview)
# preview.show()
# for mask in masks:
#     mask.show()
