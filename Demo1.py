import keyword

s1 = 'asdxnoqebdkjac'
# len(s1) 获取这个字符串的长度
print(s1[0:len(s1):2])
print('hello\nrunoob')
# 在字符串前面添加一个 r，表示原始字符串，不会发生转义
print(r'hello\nrunoob')

var = keyword.kwlist
print(var)
