# @FileName：MethodOverload.py
# @Description：
# @Author：dyh
# @Time：2022/10/11 16:24
# @Website：www.xxx.com
# @Version：V1.0
import builtins


class Vector:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        return "Vector ({},{})".format(self.a, self.b)

    def __add__(self, other):
        return Vector(self.a + other.a, self.b + other.b)


v1 = Vector(2, 10)
v2 = Vector(5, -2)
print(v1 + v2)
print(dir(builtins))

if True:
    msg = 'this is message'
print(msg)

