# @FileName：Demo17.py
# @Description：
# @Author：dyh
# @Time：2022/9/26 11:55
# @Website：www.xxx.com
# @Version：V1.0
# import sys
#
# lst = [1, 2, 3, 4]
# it = iter(lst)
# print(next(it))
#
# for i in it:
#     print(i)
#
# it2 = iter(lst)
# while True:
#     try:
#         print(next(it))
#     except StopIteration:
#         sys.exit()


class MyNumbers:
    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        x = self.a
        self.a += 1
        return x

myclass = MyNumbers()
myiter = iter(myclass)

print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))