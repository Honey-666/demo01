# @FileName：SocketServerDemo.py
# @Description：
# @Author：dyh
# @Time：2022/10/27 12:25
# @Website：www.xxx.com
# @Version：V1.0
import socket

# 创建 socket 对象
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 获取本地主机名
hostname = socket.gethostname()
port = 9999
# 绑定端口号
serversocket.bind((hostname, port))
# 设置最大连接数，超过后排队
serversocket.listen(5)
while True:
    clientsocket, addr = serversocket.accept()
    print("连接地址: %s" % str(addr))

    msg = '欢迎访问菜鸟教程！' + "\r\n"
    clientsocket.send(msg.encode('utf-8'))
    clientsocket.close()
