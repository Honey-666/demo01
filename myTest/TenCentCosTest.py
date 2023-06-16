# @FileName：TenCentCosTest.py
# @Description：
# @Author：dyh
# @Time：2023/6/6 11:38
# @Website：www.xxx.com
# @Version：V1.0
# -*- coding=utf-8
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import os
import logging

# 正常情况日志级别使用 INFO，需要定位时可以修改为 DEBUG，此时 SDK 会打印和服务端的通信信息
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# 1. 设置用户属性, 包括 secret_id, secret_key, region等。Appid 已在 CosConfig 中移除，请在参数 Bucket 中带上 Appid。Bucket 由 BucketName-Appid 组成
secret_id = 'AKIDIbs7JSYYUMAyjQ1XyJYgjhYRF0gIpNn8'
secret_key = 'KYhkiaO7lJl7PcknldQApPtIcs3EKLIA'
region = None  # 通过 Endpoint 初始化不需要配置 region
token = None  # 如果使用永久密钥不需要填入 token，如果使用临时密钥需要填入，临时密钥生成和使用指引参见 https://cloud.tencent.com/document/product/436/14048
scheme = 'https'  # 指定使用 http/https 协议来访问 COS，默认为 https，可不填

# endpoint = 'train-model.aliyuncs.com'
endpoint = 'cos.ap-shanghai.myqcloud.com'
config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=token, Endpoint=endpoint,
                   Scheme=scheme)
client = CosS3Client(config)

#### 高级上传接口（推荐）
# 根据文件大小自动选择简单上传或分块上传，分块上传具备断点续传功能。
# response = client.upload_file(
#     Bucket='train-model-bin-1301885045',
#     LocalFilePath='./RealESRGAN_x4plus_anime_6B.pth',
#     Key='gan_x4plus_anime_6B-2.pth',
#     PartSize=1,
#     MAXThread=10,
#     EnableMD5=False
# )
# print(response['ETag'])

#### 文件流简单上传（不支持超过5G的文件，推荐使用下方高级上传接口）
# 强烈建议您以二进制模式(binary mode)打开文件,否则可能会导致错误
with open('test-upload.png', 'rb') as fp:
    response = client.put_object(
        Bucket='train-model-bin-1301885045',
        Body=fp,
        Key='test-upload.png',
        StorageClass='STANDARD',
        EnableMD5=False
    )
print(response)
print(response['ETag'])
