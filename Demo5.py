# @Time : 2021/12/18 16:01
# @Author : dyh
# @Version：V 0.1
# @File : Demo5.py
# @desc : Set

"""
集合（set）是由一个或数个形态各异的大小整体组成的，构成集合的事物或对象称作元素或是成员。
基本功能是进行成员关系测试和删除重复元素。
可以使用大括号 { } 或者 set() 函数创建集合，注意：创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典
"""
sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}
print(sites)

if 'Runoob' in sites:
    print("Runoob在集合中")
else:
    print("Runoob不在集合中")

a = set('abcd')
b = set('defa')
print(a)
# a 和 b 的差集(输出b里面不包含a的字符)
print(a - b)
# a 和 b 的并集(a和b加起来一起的字符)
print(a | b)
# a 和 b 的交集(a和b里面相同的字符)
print(a & b)
# a 和 b 中不同时存在的元素(b里面不包含a的字符 和 a里面不包含b的)
print(a ^ b)
