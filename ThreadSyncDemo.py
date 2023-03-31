# @FileName：ThreadSyncDemo.py
# @Description：
# @Author：dyh
# @Time：2022/10/29 15:22
# @Website：www.xxx.com
# @Version：V1.0
import threading
import time


class myThreadSync(threading.Thread):
    def __init__(self, threadId, name, delay):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.delay = delay

    def run(self):
        print("开启线程： " + self.name)
        # 获取锁，用于线程同步
        threadLock.acquire()
        print_time(self.name, self.delay, 3)
        # 释放锁，开启下一个线程
        threadLock.release()


def print_time(threadName, delay, counter):
    while counter:
        time.sleep(delay)
        print("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1


threadLock = threading.Lock()
threads = []
thread1 = myThreadSync(1, "Thread-1", 2)
thread2 = myThreadSync(2, "Thread-2", 1)

thread2.start()
thread1.start()

threads.append(thread1)
threads.append(thread2)

for t in threads:
    t.join()
print("退出主线程")
