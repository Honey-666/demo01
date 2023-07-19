# @FileName：ESRGANTest.py
# @Description：
# @Author：dyh
# @Time：2023/7/12 14:29
# @Website：www.xxx.com
# @Version：V1.0
import os.path as osp
import glob
import time

import cv2
import numpy as np
import torch
from PIL import Image

import RRDBNet_arch as arch

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ESRGAN\\RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

# idx = 0
# for path in glob.glob(test_img_folder):
#     s = time.time()
#     idx += 1
#     base = osp.splitext(osp.basename(path))[0]
#     print(idx, base)
#     # read images
#     img = cv2.imread(path, cv2.IMREAD_COLOR)
#     img = img * 1.0 / 255
#     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#     img_LR = img.unsqueeze(0)
#     img_LR = img_LR.to(device)
#
#     with torch.no_grad():
#         output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#     output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#     output = (output * 255.0).round()
#     print(type(output))
#     output = Image.fromarray(np.uint8(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)))
#     print(time.time() - s)
#     output.show()
# cv2.imwrite('results/{:s}_rlt.png'.format(base), output)

img = cv2.imread('./LR/bigbigai.com-1689242786563.jpg')
shape = img.shape
img = cv2.resize(img, (shape[1] // 4, shape[0] // 4))
cv2.imwrite('./LR/192x192.png', img)
for i in range(5):
    s = time.time()
    # read images
    img = cv2.imread('./LR/192x192.png', cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    output = Image.fromarray(np.uint8(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)))
    print(time.time() - s)
