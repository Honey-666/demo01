# @FileName：Demo23.py
# @Description：
# @Author：dyh
# @Time：2022/10/12 17:01
# @Website：www.xxx.com
# @Version：V1.0
import datetime
from datetime import date
from timeit import Timer
import doctest

print(date.today())
print(datetime.date(2003, 12, 2))
print(date.today().strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B."))

print(Timer('t=a; a=b; b=t', 'a=1; b=2').timeit())
print(Timer('a,b = b,a', 'a=1; b=2').timeit())


def average(values):
    """Computes the arithmetic mean of a list of numbers.

        >>> print(average([20, 30, 70]))
        40.0
        """
    return sum(values) / len(values)


doctest.testmod()


def sum(a, b):
    return a + b;


import unittest


class myUnitTest(unittest.TestCase):
    def test_01(self):
        print(sum(1, 2))

    def test_02(self):
        print(sum(3, 4))
