# @Time : 2021/12/18 15:44
# @Author : dyh
# @Version：V 0.1
# @File : Demo4.py
# @desc : 元组

tuple = ('abcd', 786, 2.23, 'runoob', 70.2)
tinytuple = (123, 'runoob')

print(tuple)
print(tuple[0])
print(tuple[1:3])
print(tuple[2:])
print(tinytuple * 2)
print(tuple + tinytuple)
# 元组不可改变
# tuple[0] = 11

tup1 = ()
# 如果元组中只有一个元素，那么这个元素后面要加逗号
tup2 = (1,)
print(tup1)
print(tup2)
print(tup1 + tup2)
