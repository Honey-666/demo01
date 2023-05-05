# @FileName：ImgWhiteTest.py
# @Description：
# @Author：dyh
# @Time：2023/3/29 14:21
# @Website：www.xxx.com
# @Version：V1.0
import cv2
import numpy as np


def cv_show(im):
    cv2.imshow('demo', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取两张图片
img1 = cv2.imread('../img/seg5.png')
img2 = cv2.imread('../img/seg6.png')
# h, w, c = img2.shape
# img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
# img1 = cv2.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2), interpolation=cv2.INTER_AREA)
# img2 = cv2.resize(img2, (img2.shape[1] // 2, img2.shape[0] // 2), interpolation=cv2.INTER_AREA)
cv_show(img1)
cv_show(img2)
# 检查图片大小是否相同
if img1.shape != img2.shape:
    print("Error: images are not the same size.")
    exit()

# 计算两张图片的差异
diff = cv2.absdiff(img1, img2)
cv_show(diff)
# 将差异图像转换为灰度图像
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
# 对灰度图像进行二值化处理
threshold = 30
ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
cv_show(thresh)
# 开运算，先腐蚀在膨胀
kernel = np.ones((3, 3), np.uint8)
mor = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
cv_show(mor)
# 膨胀
kernel2 = np.ones((5, 5), np.uint8)
dilate = cv2.dilate(mor, kernel2, iterations=5)
cv_show(dilate)
# 腐蚀
kernel3 = np.ones((8, 8), np.uint8)
erode = cv2.erode(dilate, kernel3, iterations=3)
cv_show(erode)

# 中值滤波处理平滑
media = cv2.medianBlur(erode, 7)
cv_show(media)
# 将二值化图像取反，变成白底黑色图像
thresh_inv = cv2.bitwise_not(thresh)
# 将两张图像按照二值化图像的掩膜进行融合
result = cv2.bitwise_or(img1, img2, mask=thresh)
# 将融合后的图像按照二值化取反的掩膜变为白底黑色图像
result[thresh_inv == 255] = [255, 255, 255]
# 将结果图像转换为灰度图像
result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# 显示结果图像
cv_show(result_gray)
