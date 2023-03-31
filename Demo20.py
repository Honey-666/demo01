# @FileName：Demo20.py
# @Description：
# @Author：dyh
# @Time：2022/10/8 17:58
# @Website：www.xxx.com
# @Version：V1.0
# class MyClass:
#     i = 1
#
#     def f(self):
#         return "hello world"
#
#
# myClass = MyClass()
#
# print(myClass.i)
# print(myClass.f())

# --------------------------------------------------------
# class Complex:
#     def __init__(self, realpart, imagpart):
#         self.r = realpart
#         self.i = imagpart
#
#
# x = Complex(3.0, -4.5)
#
# print(x.i)
# print(x.r)


# ---------------------------------------------------------------
class Test:
    def prt(self):
        print(self)
        print(self.__class__)


t = Test()
t.prt()
