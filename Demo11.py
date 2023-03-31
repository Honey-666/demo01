# @FileName：Demo11.py
# @Description：
# @Author：dyh
# @Time：2022/9/23 16:12
# @Website：www.xxx.com
# @Version：V0.1

x = 1
print(f'{x + 1 = }')

print('qwe'.capitalize())

print('eee'.center(9) + '1')

print("werqwedqs".count('er', 0, 1))

print("我那位".encode("utf-8"))

print(bytes.decode(b'\xe6\x88\x91\xe9\x82\xa3\xe4\xbd\x8d', 'utf-8'))

print('sdfwefq'.endswith('fq'))

print('asdqw\tac'.expandtabs())

print('axclnqoihcoqiwhcqw'.find('wh', 0, 14))
# 报异常
# print('axclnqoihcoqiwhcqw'.index('wh', 0, 14))
print('asdqow12309sc12'.isalnum())
print('asdqow@12309sc12'.isalnum())

print("cwuehcu实现凑文本".isalpha())
print("cwuehcu实现凑文本123".isalpha())

print('12312412'.isdigit())
print('12312412asd'.isdigit())

print('qwuhdqc'.islower())
print('qwuhdqC'.islower())

print('QWUHDQC'.isupper())
print('QWUHDQc'.isupper())

print('12312312'.isnumeric())
print('12312312a'.isnumeric())

print('   '.isspace())
print('   1'.isspace())

# istitle() 方法检测字符串中所有的单词拼写首字母是否为大写，且其他字母为小写
print('Title'.istitle())
print('title'.istitle())

print('-'.join(('h', 'e', 'l', 'l', 'o')))

print('WER'.lower())

# 返回一个原字符串左对齐,并使用 fillchar 填充至长度 width 的新字符串，fillchar 默认为空格。
print('qwe'.ljust(6, 'r') + 'w')

# 截掉字符串左边的空格或指定字符。
print('   uuu'.lstrip())
print('@@@uuu'.lstrip('@'))

txt = "Google Runoob Taobao!"
x = "RG"  # 字符串中要替换的字符传
y = "WP"  # 替换成什么(要和上方字符长度一致)
z = "o"  # 设置删除的字符
mytable = txt.maketrans(x, y, z)
print(txt.translate(mytable))

print('123@qwe@456@iop'.split('@', 2))  # ['123', 'qwe', '456@iop']

print('You is my unique lover'.title())

# 自已生成一个对照表 translate方法会根据对照表返回信息(类似于map等)
table = str.maketrans('dyh', '哈哈哈')
print('dyh'.translate(table))

print('aaa'.zfill(60))

# 检查字符串是否只包含十进制字符，如果是返回 true，否则返回 false。
print('qwe'.isdecimal())
print('123'.isdecimal())
