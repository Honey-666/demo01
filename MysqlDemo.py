# @FileName：MysqlDemo.py
# @Description：
# @Author：dyh
# @Time：2022/10/18 17:37
# @Website：www.xxx.com
# @Version：V1.0
import pymysql

db = pymysql.connect(host='localhost', port=13306, user='root', password='root', database='demo01')

cursor = db.cursor()
# 使用 execute()  方法执行 SQL 查询
createTableSql = '''
    CREATE TABLE IF NOT EXISTS employee (
         first_name  CHAR(20) NOT NULL,
         last_name  CHAR(20),
         age INT,  
         sex CHAR(1),
         income FLOAT)
'''

dropSql = 'DROP TABlE IF EXISTS employee'

insertSql = 'INSERT INTO employee(first_name,last_name,age,sex,income) VALUES ("Jery","Bul",18,1,4888.00)'
# cursor.execute(createTableSql)
cursor.execute(insertSql)
db.commit()
db.close()
