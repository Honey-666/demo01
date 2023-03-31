# @FileName：ThreadAndQueueDemo.py
# @Description：
# @Author：dyh
# @Time：2022/10/29 16:57
# @Website：www.xxx.com
# @Version：V1.0
import threading
import time
import queue

exitFlag = 0


class MyThread(threading.Thread):
    def __init__(self, threadId, name, q):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.q = q

    def run(self):
        print("开启线程：" + self.name)
        process_data(self.name, self.q)
        print("退出线程：" + self.name)


def process_data(threadName, q):
    while not exitFlag:
        threadLock.acquire()
        if not workQueue.empty():
            data = q.get()
            print("%s processing %s" % (threadName, data))
            threadLock.release()
        else:
            threadLock.release()
        time.sleep(1)


threadLock = threading.Lock()
workQueue = queue.Queue(10)
threadList = ["Thread-1", "Thread-2", "Thread-3"]
nameList = ["One", "Two", "Three", "Four", "Five"]
threads = []
threadID = 1

# 创建新线程
for threadName in threadList:
    myThread = MyThread(threadID, threadName, workQueue)
    myThread.start()
    threads.append(myThread)
    threadID += 1

# 填充队列
threadLock.acquire()
for name in nameList:
    workQueue.put_nowait(name)
threadLock.release()
# 等待队列清空
while not workQueue.empty():
    pass
# 通知线程是时候退出
exitFlag = 1
# 等待所有线程完成
for t in threads:
    t.join()
print("执行完成退出主线程")
