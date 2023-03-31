# @FileName：Demo 12.py
# @Description：
# @Author：dyh
# @Time：2022/9/23 18:26
# @Website：www.xxx.com
# @Version：V0.1
tup1 = (50)
tup2 = (50,)
print(type(tup1))
print(type(tup2))

emptyDict = dict()
print(emptyDict)
print(len(emptyDict))
print(type(emptyDict))

tinyDict = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}
print(tinyDict['Name'])

tinyDict['school'] = '菜鸟'
print(tinyDict)

tinyDict['school'] = '菜鸟教程'
print(tinyDict)

del tinyDict['school']
print(tinyDict)


tinyDict.clear()
print(tinyDict)