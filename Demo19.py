# @FileName：Demo19.py
# @Description：
# @Author：dyh
# @Time：2022/9/26 16:17
# @Website：www.xxx.com
# @Version：V1.0

def hello():
    print('hello world!')


hello()


def maxNumber(a, b):
    if a > b:
        return a
    else:
        return b


big = maxNumber(8, 3)
print(big)


def area(w, h):
    return w * h


print(area(100, 100))


def printme(word):
    print(word)
    return


printme("我要调用用户自定义函数!")
printme("再次调用同一函数")

'''
python 函数的参数传递：

不可变类型：类似 C++ 的值传递，如整数、字符串、元组。如 fun(a)，传递的只是 a 的值，没有影响 a 对象本身。如果在 fun(a) 内部修改 a 的值，则是新生成一个 a 的对象。

可变类型：类似 C++ 的引用传递，如 列表，字典。如 fun(la)，则是将 la 真正的传过去，修改后 fun 外部的 la 也会受影响
'''


def change(a):
    print(id(a))
    a = 10
    print(id(a))


x = 1
print(id(x))
change(x)


def mutable_change(mylist):
    mylist.append([100, 200, 300])
    print("函数内取值: ", mylist)
    return


mylist = [10, 20, 30]
mutable_change(mylist)
print("函数外取值: ", mylist)

print("-----------------------")


def print_info(arg1, *var_tuple):
    "打印任何传入的参数"
    print("输出: ")
    print(arg1)
    print(var_tuple)
    for var in var_tuple:
        print(var)


print_info(200)
print_info(70, 80, 90, 100)

print("---------------------------------")


def print_info2(arg1, **var_dict):
    "打印任何传入的参数"
    print("输出: ")
    print(arg1)
    print(var_dict)


print_info2(10, a=1, b=2)


# 如果单独出现星号 *，则星号 * 后的参数必须用关键字传入：
def fun(a, b, *, c):
    print("a=" + str(a) + " b=" + str(b) + " c=" + str(c))


fun(1, 2, c=3)

arg_num = lambda arg1, arg2: arg1 + arg2

print(arg_num(10, 20))


def my_fun(n):
    return lambda a: a * n


my_doubler = my_fun(2)
my_tripler = my_fun(3)

print(my_doubler(11))
print(my_tripler(11))


def return_fun(arg1, arg2):
    return arg1 + arg2


sum2 = return_fun(10, 10)

print(sum2)
