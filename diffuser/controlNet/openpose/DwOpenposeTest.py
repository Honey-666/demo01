# @FileName：DwOpenposeTest.py
# @Description：
# @Author：dyh
# @Time：2023/9/1 14:21
# @Website：www.xxx.com
# @Version：V1.0

import cv2
import numpy as np
import torch
from PIL import Image
from controlnet_aux import DWposeDetector
from controlnet_aux.dwpose import util

# from diffuser.controlNet.openpose import util
from diffuser.controlNet.openpose.wholebody import Wholebody  # DW Pose
from typing import List


# def load_dw_model():
#     onnx_det = 'C:\\Software\\stable-diffusion-webui\\extensions\\sd-webui-controlnet\\annotator\\downloads\\openpose\\yolox_l.onnx'
#     onnx_pose = 'C:\\Software\\stable-diffusion-webui\\extensions\\sd-webui-controlnet\\annotator\\downloads\\openpose\\dw-ll_ucoco_384.onnx'
#     dw_pose_estimation = Wholebody(onnx_det, onnx_pose)
#
#     return dw_pose_estimation
#
#
# def detect_poses_dw(oriImg) -> List[PoseResult]:
#     """
#     Detect poses in the given image using DW Pose:
#     https://github.com/IDEA-Research/DWPose
#
#     Args:
#         oriImg (numpy.ndarray): The input image for openpose detection.
#
#     Returns:
#         List[PoseResult]: A list of PoseResult objects containing the detected poses.
#     """
#
#     dw_pose_estimation = load_dw_model()
#
#     with torch.no_grad():
#         keypoints_info = dw_pose_estimation(oriImg.copy())
#         return Wholebody.format_result(keypoints_info)
#
#
# def draw_poses(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True):
#     """
#     Draw the detected poses on an empty canvas.
#
#     Args:
#         poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
#         H (int): The height of the canvas.
#         W (int): The width of the canvas.
#         draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
#         draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
#         draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.
#
#     Returns:
#         numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
#     """
#     canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
#
#     for pose in poses:
#         if draw_body:
#             canvas = util.draw_bodypose(canvas, pose.body.keypoints)
#
#         if draw_hand:
#             canvas = util.draw_handpose(canvas, pose.left_hand)
#             canvas = util.draw_handpose(canvas, pose.right_hand)
#
#         if draw_face:
#             canvas = util.draw_facepose(canvas, pose.face)
#
#     return canvas
#
#
# img = cv2.imread('C:\\Users\\bbw\\Desktop\\8eb316ba8f2642ada2c0a39206b21992.jpg')
# # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# # img = cv2.resize(img, (512, 512))
# H, W, _ = img.shape
# poses = detect_poses_dw(img)
# pose_img = draw_poses(poses, H, W, draw_body=True, draw_hand=True, draw_face=False)
# pil_img = Image.fromarray(pose_img)
# pil_img.show()
# cv2.imshow('dw_pose', pose_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# -------------------------------------------------------------------
# def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
#     from controlnet_aux.dwpose.util import draw_bodypose, draw_handpose, draw_facepose
#     bodies = pose['bodies']
#     faces = pose['faces']
#     hands = pose['hands']
#     candidate = bodies['candidate']
#     subset = bodies['subset']
#
#     canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
#     if draw_body:
#         canvas = draw_bodypose(canvas, candidate, subset)
#     if draw_hand:
#         canvas = draw_handpose(canvas, hands)
#     if draw_face:
#         canvas = draw_facepose(canvas, faces)
#
#     return canvas
def draw_facepose(canvas, all_lmks):
    print("face detect overwrite")
    return canvas


util.draw_facepose = draw_facepose

onnx_det = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts\\yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
onnx_pose = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts\\dw-ll_ucoco_384.pth'
det_config = './yolox_config/yolox_l_8xb8-300e_coco.py'
pose_config = './dwpose_config/dwpose-l_384x288.py'
img_path = 'C:\\Users\\bbw\\Desktop\\8eb316ba8f2642ada2c0a39206b21992.jpg'
dw_pose = DWposeDetector(det_config=det_config, det_ckpt=onnx_det, pose_config=pose_config, pose_ckpt=onnx_pose,
                         device='cuda')
im = Image.open(img_path)
img = dw_pose(im)
img.show()
