# @FileName：Demo22.py
# @Description：
# @Author：dyh
# @Time：2022/10/10 17:34
# @Website：www.xxx.com
# @Version：V1.0
class People:
    name = ''
    age = 0
    __weight = 0

    def __init__(self, n, a, w):
        self.name = n
        self.age = a

    def speak(self):
        print("{} 说: 我 {} 岁。".format(self.name, self.age))


class Student(People):
    grade = ''

    def __init__(self, n, a, w, g):
        People.__init__(self, n, a, w)
        self.grade = g

    def speak(self):
        print("{} 说: 我 {} 岁，我在读 {} 年级".format(self.name, self.age, self.grade))


class Speaker:
    topic = ''
    name = ''

    def __init__(self, t, n):
        self.name = n
        self.topic = t

    def speak(self):
        print("我叫 {} ,我是一个演说家，我演讲的主题是 {}".format(self.name, self.topic))


class Sample(Speaker, Student):
    a = ''

    def __init__(self, n, a, w, g, t):
        Student.__init__(self, n, a, w, g)
        Speaker.__init__(self, t, n)


s = Sample('Tom', 10, 80, 4, "Python")
s.speak()
