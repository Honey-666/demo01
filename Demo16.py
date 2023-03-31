# @FileName：Demo16.py
# @Description：
# @Author：dyh
# @Time：2022/9/26 11:24
# @Website：www.xxx.com
# @Version：V1.0
languages = ["C", "C++", "Perl", "Python"]

for x in languages:
    print('languages is ' + x)

sites = ["Baidu", "Google", "Runoob", "Taobao"]
for site in sites:
    if site == "Runoob":
        print("菜鸟教程!")
        break
    print("循环数据 " + site)
else:
    print("没有循环数据!")
print("完成循环!")

for i in range(0, 10, 2):
    print(i)

a = ['Google', 'Baidu', 'Runoob', 'Taobao', 'QQ']

for i in range(len(a)):
    print(i, a[i])

n = 5
while n > 0:
    n -= 1
    if n == 2:
        break
    print(n)
print('循环结束。')

n = 5
while n > 0:
    n -= 1
    if n == 2:
        continue
    print(n)
print('循环结束。')


print("----------------------------------")
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, '等于', x, '*', n//x)
            break
    else:
        # 循环中没有找到元素
        print(n, ' 是质数')



while True:
    pass