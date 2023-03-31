# @FileName：Demo26.py
# @Description：
# @Author：dyh
# @Time：2023/2/18 15:02
# @Website：www.xxx.com
# @Version：V1.0

# import cv2
#
# img = cv2.imread('./img/mask_file.jpg')
# print(img)
# result = cv2.bilateralFilter(img, 101, 101, 101)
# cv2.imwrite('./img/bilateralFilter.jpg', result)
# --------------------------------------------------------
import torch
from torch import tensor

A = torch.Tensor([[1, 2, 3], [1, 2, 3]])  # 二维
B = torch.sigmoid(A)
print(B)
i = tensor([[-7.3620, -7.3418, -7.3157, ..., -7.3211, -7.3372, -7.3591],
            [-7.3273, -7.3287, -7.2999, ..., -7.2988, -7.3246, -7.3373],
            [-7.2595, -7.2491, -7.2221, ..., -7.2364, -7.2596, -7.2504],
            ...,
            [-7.2944, -7.2846, -7.2622, ..., -7.2592, -7.2727, -7.2776],
            [-7.3861, -7.3911, -7.3581, ..., -7.3633, -7.3806, -7.3782],
            [-7.4806, -7.4862, -7.4631, ..., -7.4932, -7.4904, -7.5079]])

