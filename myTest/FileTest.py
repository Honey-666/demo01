# @FileName：FileTest.py
# @Description：
# @Author：dyh
# @Time：2023/9/4 10:28
# @Website：www.xxx.com
# @Version：V1.0
import json
import re

import torch

# with open('C:\\Users\\bbw\\Desktop\\AI_qgn.txt', encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.replace('}|,', '}')
#         line = re.sub(r'\\(.)', r'\1', line)
#         js = json.loads(line)
#         print(json.dumps(line))
#         print(json.loads(json.dumps(line)))

control_type = 'x'
k = 'tile'
variant = None if (k in ['openpose'] and control_type == 'xl') else "fp16"
print(variant)