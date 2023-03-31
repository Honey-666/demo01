# @FileName：RabbitMQConsumerDemo.py
# @Description：
# @Author：dyh
# @Time：2023/1/20 16:40
# @Website：www.xxx.com
# @Version：V1.0
import pika


def callback(ch, method, properties, body):
    print('消息内容为:%s' % format(body), ch, method, properties)


user_info = pika.PlainCredentials('test', 'qwe123')
connection = pika.BlockingConnection(pika.ConnectionParameters('47.98.234.186', 5672, '/neptune.test', user_info))
channel = connection.channel()
# 推模式消费者将会处于堵塞状态，一直等待消息的到来
channel.basic_consume('python.test.rabbit.mq', callback, True)
channel.start_consuming()

# 拉模式 由消费者自主控制，什么时候去获取消息
# method_frame, header_frame, body = channel.basic_get('python.test.rabbit.mq', True)
# if method_frame:
#     print(method_frame, header_frame, body)  # 收到的全部数据  就要body
#     print('消息内容为:%s' % body)
# else:
#     print('没有收到消息')
