# @FileName：Demo25.py
# @Description：
# @Author：dyh
# @Time：2023/2/11 15:04
# @Website：www.xxx.com
# @Version：V1.0
import math

from PIL import Image, ImageChops


def super_corp_region(size: int, source_big: int, crop_width: int, left: int, right: int):
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


im = Image.open('./img/tmpj6b60u7r.PNG')
diff = ImageChops.add(im, im)
bbox = diff.getbbox()
print('bbox=', bbox)
left_focus = bbox[0]
upper_focus = bbox[1]
right_focus = bbox[2]
lower_focus = bbox[3]
crop_width = right_focus - left_focus
crop_height = lower_focus - upper_focus
source_width, source_height = im.size

nf, nr = super_corp_region(256, source_width, crop_width, left_focus, right_focus)
nt, nb = super_corp_region(256, source_height, crop_height, upper_focus, lower_focus)
print(nf, nt, nr, nb)

# new_img = im.crop((nf, nt, nr, nb))
# new_img.show('112233.jpg')


