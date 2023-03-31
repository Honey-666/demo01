# @Time : 2022/2/22 10:47
# @Author : dyh
# @Version：V 0.1
# @File : Demo7.py
# @desc :

'''tinydict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
print(len(tinydict))
print(str(tinydict))
print(type(tinydict))
t2 = tinydict.copy()
print(id(tinydict))
print(id(t2))
print(tinydict.fromkeys({1, 2, 3}))
print('Class1' in tinydict)'''

# s = '123456789'
# print(s)
# print(s[0:-1])
# print(s[0])
# print(s[2:5])
# print(s[2:])
# print(s[1:5:2])
# print(s * 2)
# print(s + ' hello world')
#
# print("hello\nword")
# print(r"hello\nword")

a = 21
b = 10
c = 0

c = a + b
print("1 - c 的值为：", c)

c += a
print("2 - c 的值为：", c)

c *= a
print("3 - c 的值为：", c)

c /= a
print("4 - c 的值为：", c, id(c), type(c))

c = 2
c %= a
print("5 - c 的值为：", c)

c **= a
print("6 - c 的值为：", c)

c //= a
print("7 - c 的值为：", c)

print(2 % 21)
