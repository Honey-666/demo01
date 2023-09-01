# @FileName：DwOpenposeTest.py
# @Description：
# @Author：dyh
# @Time：2023/9/1 14:21
# @Website：www.xxx.com
# @Version：V1.0
from typing import List

import torch
from controlnet_aux.open_pose import PoseResult


def detect_poses_dw(self, oriImg) -> List[PoseResult]:
    """
    Detect poses in the given image using DW Pose:
    https://github.com/IDEA-Research/DWPose

    Args:
        oriImg (numpy.ndarray): The input image for pose detection.

    Returns:
        List[PoseResult]: A list of PoseResult objects containing the detected poses.
    """
    from .wholebody import Wholebody  # DW Pose

    self.load_dw_model()

    with torch.no_grad():
        keypoints_info = self.dw_pose_estimation(oriImg.copy())
        return Wholebody.format_result(keypoints_info)