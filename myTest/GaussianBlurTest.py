# @FileName：GaussianBlurTest.py
# @Description：
# @Author：dyh
# @Time：2023/7/26 18:23
# @Website：www.xxx.com
# @Version：V1.0
from PIL import ImageFilter, Image

img = Image.open('../img/test/20230724-110413.jpg')
img = img.filter(ImageFilter.GaussianBlur(8))
img.show()
