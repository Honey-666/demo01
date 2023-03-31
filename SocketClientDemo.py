# @FileName：SocketClientDemo.py
# @Description：
# @Author：dyh
# @Time：2022/10/27 14:03
# @Website：www.xxx.com
# @Version：V1.0
import socket

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = socket.gethostname()
port = 9999

clientsocket.connect((host, port))

msg = clientsocket.recv(1024)
clientsocket.close()

print(msg.decode("utf-8"))
