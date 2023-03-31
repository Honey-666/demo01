# @FileName：Demo21.py
# @Description：
# @Author：dyh
# @Time：2022/10/8 18:27
# @Website：www.xxx.com
# @Version：V1.0
class People:
    name = ''
    age = 0
    # 定义私有属性 python使用 __属性名的方式来定义私有属性
    __weight = 0

    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.__weight = w

    def speak(self):
        print("{} 说: 我 {} 岁。".format(self.name, self.age))


# x = People('tom', 10, 30)
# x.speak()


class Student(People):
    grade = ''

    def __init__(self, n, a, w, g):
        self.grade = g
        People.__init__(self, n, a, w)

    def speak(self):
        print("{} 说: 我 {} 岁，我在读 {} 年级".format(self.name, self.age, self.grade))


stu = Student('Jerry', 18, 60, 6)
stu.speak()
