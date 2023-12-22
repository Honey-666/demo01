# @FileName：TosTest.py
# @Description：
# @Author：dyh
# @Time：2023/12/20 10:21
# @Website：www.xxx.com
# @Version：V1.0
import io

import requests
import tos


def get_oss_img(url):
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return io.BytesIO(response.content)


# 从环境变量获取 AK 和 SK 信息。
ak = 'AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI'
sk = 'TldNeFpqQTFaVGcwWmpKaU5ETXhNMkUxWW1Zek9XSXdNemMxWkdZMU1ETQ=='
endpoint = "tos-cn-beijing.volces.com"
region = "cn-beijing"
bucket = "turbotest"

url_lst = [
    "https://apin.bigurl.ink/ai-pin/d4a78972-9a66-11ee-a980-5d00e08b9823.jpeg?Expires=1703140651&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=pXL0MDeyJ5KSYIMxgdWrjTxAeeI%3D",
    "https://apin.bigurl.ink/ai-pin/d9434ba6-9a66-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=athsKYnqqOhXTNWckhU10lBcsYw%3D",
    "https://apin.bigurl.ink/ai-pin/e2452242-9a66-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=qRRfQTe2zXuY9ja0A9rKn5PU3BE%3D",
    "https://apin.bigurl.ink/ai-pin/e7a82e1e-9a66-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=TOGwfQxVaM8vSlzS%2F2c04Ci7YTg%3D",
    "https://apin.bigurl.ink/ai-pin/ed05cc68-9a66-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=vYMAW6InztjeNq8KibrHYERDrNo%3D",
    "https://apin.bigurl.ink/ai-pin/05217efa-9a67-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=XauXqDS0zjUtiLLnDh%2BY2pRHwEc%3D",
    "https://apin.bigurl.ink/ai-pin/0e4802e2-9a67-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=XjRgLzuhp1V%2BxSmrdafb8fnJehE%3D",
    "https://apin.bigurl.ink/ai-pin/16c2fd50-9a67-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=5lqA7ZtJosml25mGEZxxBHholEI%3D",
    "https://apin.bigurl.ink/ai-pin/1a4bf5a8-9a67-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=fS2ABhpg6tqeWPsEZM7wrXVfekQ%3D",
    "https://apin.bigurl.ink/ai-pin/1e64f7d4-9a67-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=EzC7L7%2Bz6TpCk4N4FONQSBLn1p4%3D",
    "https://apin.bigurl.ink/ai-pin/27f874e2-9a67-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=KPLC0d5O5mUONqyUgP7StI5%2BXJU%3D",
    "https://apin.bigurl.ink/ai-pin/878b7b78-9a6e-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=GZMhrdLgXLaGhMgmnE%2Bfn6FtDaM%3D",
    "https://apin.bigurl.ink/ai-pin/4443911a-9af6-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=p0A3XqEoKAS0VxMELT%2BUc%2BUXcvk%3D",
    "https://apin.bigurl.ink/ai-pin/8d162866-9d69-11ee-97be-525400aa6c77.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=1ibOZiPoMVXifUa%2FABLsG5dlBhw%3D",
    "https://apin.bigurl.ink/ai-pin/c449a045-9d69-11ee-a51d-525400aa6c77.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=ykgxAu2eDFIcFk8BJx2OzFJEjrU%3D",
    "https://apin.bigurl.ink/ai-pin/3e45a5f0-9d8a-11ee-a980-5d00e08b9823.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=%2BBeccWgHfetNVFOxZIjTQ76f%2FK4%3D",
    "https://apin.bigurl.ink/ai-pin/9c93ec66-9eed-11ee-9afc-b3f8789cdf27.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=eQnRdMirctbaA2WHzR5RWaHjlkw%3D",
    "https://apin.bigurl.ink/ai-pin/aafb0438-9eed-11ee-9afc-b3f8789cdf27.jpeg?Expires=1703140652&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=e1DpvPZZ2Az433TJcEte6kx%2B1OA%3D"]

try:
    # 创建TosClientV2对象
    client = tos.TosClientV2(ak, sk, endpoint, region)

    for url in url_lst:
        img_io_data = get_oss_img(url)
        key = url.split('?')[0].split('/')[-1]
        print(key)
        # 调用接口请求TOS服务，例如上传对象
        resp = client.put_object(bucket, key, content=img_io_data.getvalue())
        print('success, request id {}'.format(resp.request_id))
except tos.exceptions.TosClientError as e:
    # 操作失败，捕获客户端异常，一般情况为非法请求参数或网络异常
    print('fail with client error, message:{}, cause: {}'.format(e.message, e.cause))
except tos.exceptions.TosServerError as e:
    # 操作失败，捕获服务端异常，可从返回信息中获取详细错误信息
    print('fail with server error, code: {}'.format(e.code))
    # request id 可定位具体问题，强烈建议日志中保存
    print('error with request id: {}'.format(e.request_id))
    print('error with message: {}'.format(e.message))
    print('error with http code: {}'.format(e.status_code))
    print('error with ec: {}'.format(e.ec))
    print('error with request url: {}'.format(e.request_url))
except Exception as e:
    print('fail with unknown error: {}'.format(e))
