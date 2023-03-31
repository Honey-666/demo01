# @FileName：Demo18.py
# @Description：
# @Author：dyh
# @Time：2022/9/26 14:39
# @Website：www.xxx.com
# @Version：V1.0
import sys


class MyNumber:
    def __iter__(self):
        self.a = 1
        return self

    def __next__(self):
        if self.a < 20:
            x = self.a
            self.a += 1
            return x
        else:
            raise StopIteration


number = MyNumber()
it = iter(number)

for i in it:
    print(i)


def fibonacci(n):
    a, b, counter = 0, 1, 0
    while True:
        if counter > n:
            return
        yield a
        a, b = b, a + b
        counter += 1


f = fibonacci(10)

while True:
    try:
        print(next(f), end=" ")
    except StopIteration:
        sys.exit()
