# @FileName：TosVolcTest.py
# @Description：
# @Author：dyh
# @Time：2023/12/20 10:50
# @Website：www.xxx.com
# @Version：V1.0
import base64
import io
import json
import time
import uuid
from datetime import datetime
from io import BytesIO

import requests
from PIL import Image
from volcengine.content_security.ContentSecurityService import ContentSecurityService


def img_to_base64(img):
    buff = BytesIO()
    img.convert('RGB').save(buff, format="JPEG")
    img_base64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_base64


def get_oss_img(url):
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


if __name__ == '__main__':
    riskDetector = ContentSecurityService()

    riskDetector.set_ak('AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI')
    riskDetector.set_sk('TldNeFpqQTFaVGcwWmpKaU5ETXhNMkUxWW1Zek9XSXdNemMxWkdZMU1ETQ==')
    params = dict()

    url_lst = [
        "https://turbotest.tos-cn-beijing.volces.com/d4a78972-9a66-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024648Z&X-Tos-Signature=fb34e9f44013c6a2058357907566239ceeaacc006cfeb0c4dd533f6e9b8fceed&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/d9434ba6-9a66-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=8bb7a02d0103ea46a939c89c3d1c01be17b604a94a122069fb2ba5d43b43d999&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/e2452242-9a66-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=81a39c350ccd8cc2df752cbf96ed9bb0c54be86f6170f6336af815d85255be5e&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/e7a82e1e-9a66-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=e2447178bd22ac4d29b3761214762c8d196c8143fdf7ca3c2c33ce8dd03abe93&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/ed05cc68-9a66-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=e0854175d549fce3c4612d72fec1ab265e9e99b5eda043a1ada7d15a563745c5&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/05217efa-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=2031f3e816b8b1d058fd9f9be1e18fdc599f4df00ea1398f53914cf1bbd98835&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/0e4802e2-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=d9f645074a09f93f9ad08d6dab0eddad9cfb2043cb1dc8d33b8b80772cf5df74&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/16c2fd50-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=1137c74246095f2c0b7730c6bcf6d843392ec8762ce165fa7f29f1a38a2c7c38&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/1a4bf5a8-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=2fa465841d00b11d60d6ad2bbca93b836bd17d3ef3ac6819a8f598ae7aa5a703&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/1e64f7d4-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=5d309a460c73040bab3c8da9c84d1aed75f6dd5ffd8c334d806885b5fe6d20f3&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/27f874e2-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=ea6ea02d93d27442ca36f69acecac45f90603a582deedd20c54178125aa3493b&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/878b7b78-9a6e-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=cc715d48fbdf78ce203845fe0f53f99e1c6543c9fc3341e412a4a99abf8499f1&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/4443911a-9af6-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=f18dd52e80487a34c496556bb7900d444185f4238a36e0ff856c17728085824f&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/8d162866-9d69-11ee-97be-525400aa6c77.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=b15895ac0a917e64e461df6b715b288ef94624383ce982866d4b14a8607c8b65&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/c449a045-9d69-11ee-a51d-525400aa6c77.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=d9c99441b85d22ea8e4956f6c2917b7926d1420be89cfe5aecd201d34f70c011&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/3e45a5f0-9d8a-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=8f83e2baed57d558608b3e49e4e59a0897789509b14017b34fb3e1f2f53abe3a&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/9c93ec66-9eed-11ee-9afc-b3f8789cdf27.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=b93343443bb7908c2d03004c677a150c0b2d776b6c3adb4dd3ca5e18a3e12325&X-Tos-SignedHeaders=host",
        "https://turbotest.tos-cn-beijing.volces.com/aafb0438-9eed-11ee-9afc-b3f8789cdf27.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231221%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231221T024649Z&X-Tos-Signature=ddfbbda235637fd6498a12e0b7a4a912be5d71e5eea8cfef72d0290b56f2dcfb&X-Tos-SignedHeaders=host"]

    # url_lst = [
    #     "https://turbotest.tos-cn-beijing.ivolces.com/d4a78972-9a66-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=14721abe5a6a220a447518cb81f193ad007e51e8428fc48e8341a80368127629&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/d9434ba6-9a66-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=97b8d783a503c99f9edc96f14a07e1842ed0e8bee9f82de29c1a222a353a6fd3&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/e2452242-9a66-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=776f593e649f2c76f959c485bb09489d19a6b136b8ffb768a1292feee1c9b389&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/e7a82e1e-9a66-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=11aa04a8028afc5b078206e3cf039069d784f2de93b8b7c4b147386a806e9111&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/ed05cc68-9a66-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=0530c58657a101a4421b525d904f8490c5c823cdef86f6ef149cefeb876095e0&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/05217efa-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=cef59858d0674b7178cc4aae8b5138bca5d54778059e01b18ba4cf6e1cd89e52&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/0e4802e2-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=166c5c51c22182aaf2843216b2a1734c3118ff3e9b0b3438a643950156b3c058&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/16c2fd50-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=a04fe3f6e69eff7e4e28209bd94899d2efb76a807cabec79c9b9b599e5d83628&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/1a4bf5a8-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=01dd15a7c6af19db8dbc1b05c0a224e1bc33b0f23cc9ecc00ec7f40a1e6a8c8d&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/1e64f7d4-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=f343b0bbd57a3e4d13aef11f8e2a43353c5eb7ce069de259b7eafc780df38920&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/27f874e2-9a67-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=3b47dba587500f0c9b53d5ebbc1cc2c0b91f49ae1c61e85c28a25cf515807654&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/878b7b78-9a6e-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=e46ea706ef8d5a16ae74a4a5625da8854e9540a1654feeb426b2cbfe46e2fd88&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/4443911a-9af6-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=01d1feaa46c82ec8c2be918645b7c89fe939ab7e6719d95f0263a15ed7357139&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/8d162866-9d69-11ee-97be-525400aa6c77.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=cba68abd923bcbefa171c6da1959d1d7c68e2b033e31be47e3c6b0e8ead19678&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/c449a045-9d69-11ee-a51d-525400aa6c77.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=32d1938270a6ccd701996d09679664e74e27a192f35d42561eac5c6db9217632&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/3e45a5f0-9d8a-11ee-a980-5d00e08b9823.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=5b8c5619dedea7dca1b0618b57f8b7215056d0cd0b7ae65fd52700035568af0e&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/9c93ec66-9eed-11ee-9afc-b3f8789cdf27.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=f341e0c00d9a76c00a2bfd402e885c5deb22f4c2150b356d3cd8cc9c2ce90dcb&X-Tos-SignedHeaders=host",
    #     "https://turbotest.tos-cn-beijing.ivolces.com/aafb0438-9eed-11ee-9afc-b3f8789cdf27.jpeg?X-Tos-Algorithm=TOS4-HMAC-SHA256&X-Tos-Credential=AKLTNmUxNTYyNjNhODA4NGE5N2E5ZTAwNWQzYzI1ODc5YzI%2F20231220%2Fcn-beijing%2Ftos%2Frequest&X-Tos-Expires=21600&X-Tos-Date=20231220T102241Z&X-Tos-Signature=27e5422d85138a921b973c1b720ecbc351e8c1345021fa636dc9bdb46bc4588f&X-Tos-SignedHeaders=host"]

    for url in url_lst:
        pil_img = get_oss_img(url)
        base64_img = img_to_base64(pil_img)

        s = time.time()
        timestamp = int(datetime.now().timestamp())
        js = {
            "account_id": 'user666',
            "data_id": str(uuid.uuid4()).replace("-", ""),
            "operate_time": timestamp,
            "biztype": 'aigc',
            "data": base64_img
        }
        # 调用实时接口
        req = {
            'AppId': 566067,
            'Service': "image_content_risk",
            'Parameters': json.dumps(js)
        }

        resp = riskDetector.image_content_risk_v2(params, req)
        print(f'consuming time = {time.time() - s}')
        print(resp)
