# @FileName：Demo9.py
# @Description：
# @Author：dyh
# @Time：2022/9/20 18:00
# @Website：www.xxx.com
# @Version：V0.1
a = 10
b = 20
lst = [1, 2, 3, 4, 5]

if a in lst:
    print("1 - 变量 a 在给定的列表中 list 中")
else:
    print("1 - 变量 a 不在给定的列表中 list 中")

if b not in lst:
    print("2 - 变量 b 不在给定的列表中 list 中")
else:
    print("2 - 变量 b 在给定的列表中 list 中")

# 修改变量 a 的值
a = 2
if a in lst:
    print("3 - 变量 a 在给定的列表中 list 中")
else:
    print("3 - 变量 a 不在给定的列表中 list 中")