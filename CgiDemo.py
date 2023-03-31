# @FileName：CgiDemo.py
# @Description：
# @Author：dyh
# @Time：2022/10/18 15:41
# @Website：www.xxx.com
# @Version：V1.0
import cgi, sys, codecs

sys.stdout = codecs.getreader("utf-8")(sys.stdout.buffer)

# 创建 FieldStorage 的实例化
form = cgi.FieldStorage()
# 获取数据
name = form.getvalue("name")
url = form.getvalue("url")

print("Content-type:text/html")
print()
print("<html>")
print("<head>")
print("<meta charset=\"utf-8\">")
print("<title>菜鸟教程 CGI 测试实例</title>")
print("</head>")
print("<body>")
print("<h2>%s官网：%s</h2>" % (name, url))
print("</body>")
print("</html>")
