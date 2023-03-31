# @FileName：MethodOverwrite.py
# @Description：
# @Author：dyh
# @Time：2022/10/11 13:54
# @Website：www.xxx.com
# @Version：V1.0

class Parent:
    def myMethod(self):
        print('调用父类的方法')


class Child(Parent):
    def myMethod(self):
        print('调用子类的方法')


c = Child()
c.myMethod()
# 使用子类调用父类被重写的方法
super(Child, c).myMethod()
