# @Time : 2021/7/17 13:23
# @Author : dyh
# @Version：V 0.1
# @File : Demo2.py
# @desc :

# lst = ['abcd', 786, 2.23, 'runoob', 70.2]
# tinylist = [123, 'runoob']
# print(lst[0])
# print(lst + tinylist)
# print(lst * 2)

a = [1, 2, 3, 4, 5, 6]
a[0] = 9
print(a)
# 包头不包尾
print(a[2:5])

a[2:5] = []
print(a)
# 截取索引为2到5的步长为2
print(a[2:5:2])
