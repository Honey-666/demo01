# @FileName：ClassPrivateMethod.py
# @Description：
# @Author：dyh
# @Time：2022/10/11 15:57
# @Website：www.xxx.com
# @Version：V1.0

# class JustCounter:
#     __secretCount = 0
#     publicCount = 0
#
#     def count(self):
#         self.__secretCount += 1
#         self.publicCount += 1
#         print(self.__secretCount)
#
#
# counter = JustCounter()
# counter.count()
# counter.count()
# print(counter.publicCount)

class Site:
    def __init__(self, name, url):
        self.name = name
        self.__url = url

    def who(self):
        print('name  : ', self.name)
        print('url : ', self.__url)

    def __foo(self):
        print("这是私有方法")

    def foo(self):
        print("这是公共方法")
        self.__foo()


x = Site("Tom", "https://bigbigwork.com")
x.who()
x.foo()
