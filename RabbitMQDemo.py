# @FileName：RabbitMQDemo.py
# @Description：
# @Author：dyh
# @Time：2023/1/20 15:25
# @Website：www.xxx.com
# @Version：V1.0
import pika
from pika.exchange_type import ExchangeType

# 生命资质(即用户名+密码)
user_info = pika.PlainCredentials("test", "qwe123")
# 创建链接
connection = pika.BlockingConnection(pika.ConnectionParameters('47.98.234.186', 5672, '/neptune.test', user_info))
# 获取一个通道
channel = connection.channel()
# 创建交换机
channel.exchange_declare('python.test.rabbit.mq.exchange', 'direct')
channel.queue_declare('python.test.rabbit.mq')
channel.queue_bind('python.test.rabbit.mq', 'python.test.rabbit.mq.exchange')

channel.basic_publish('', 'python.test.rabbit.mq', 'hello python queue'.encode('utf-8'))
