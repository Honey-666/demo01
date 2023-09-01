import requests
import json

headers = {'Cookie': 'm=2258:dGVzdDpxd2UxMjM%253D', 'Authorization': 'Basic dGVzdDpxd2UxMjM='}

json_str = '''
{
	"vhost": "/neptune.test",
	"name": "amq.default",
	"properties": {
		"delivery_mode": 1,
		"headers": {},
		"content_type": "application/json"
	},
	"routing_key": "direct.python.diffusers.general.queue",
	"delivery_mode": "1",
	"headers": {},
	"props": {
		"content_type": "application/json"
	},
	"payload_encoding": "string"
}
'''
js = json.loads(json_str)

data_str = '''
  {
  "controlnet_list": [
    {
      "conditioning_scale": 1.0,
      "control_guidance_end": 0.85,
      "control_guidance_start": 0.05,
      "controlnet_model": "canny",
      "img_url": "https://apin.bigurl.ink/refer-pin/236f02e5fc83441299687c236d8a811d.png?Expires=1693468783&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=zx40U4DlmRr6QxYN%2B4G86zDeVCA%3D&x-oss-process=image%2Fresize%2Cm_lfit%2Cw_768%2Ch_768%2Fformat%2Cjpg",
      "img_url_internal": "http://ai-pin.oss-cn-hangzhou-internal.aliyuncs.com/refer-pin/236f02e5fc83441299687c236d8a811d.png?Expires=1693468783&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=zx40U4DlmRr6QxYN%2B4G86zDeVCA%3D&x-oss-process=image%2Fresize%2Cm_lfit%2Cw_768%2Ch_768%2Fformat%2Cjpg",
      "preprocess": 1,
      "resolution": 512
    },
    {
      "conditioning_scale": 1.0,
      "control_guidance_end": 1.0,
      "control_guidance_start": 0.0,
      "controlnet_model": "depth",
      "img_url": "https://apin.bigurl.ink/refer-pin/cb4df404fd0d48228731a748803fc651.png?Expires=1693468783&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=sleW8zZmoMgPlmX774ohCcELeGE%3D&x-oss-process=image%2Fresize%2Cm_lfit%2Cw_768%2Ch_768%2Fformat%2Cjpg",
      "img_url_internal": "http://ai-pin.oss-cn-hangzhou-internal.aliyuncs.com/refer-pin/cb4df404fd0d48228731a748803fc651.png?Expires=1693468783&OSSAccessKeyId=LTAItRs09iNMHxzc&Signature=sleW8zZmoMgPlmX774ohCcELeGE%3D&x-oss-process=image%2Fresize%2Cm_lfit%2Cw_768%2Ch_768%2Fformat%2Cjpg",
      "preprocess": 1,
      "resolution": 512
    }
  ],
  "flag_audit": 0,
  "guidance_scale": 10.0,
  "height": 2048,
  "img2img2": {
    "height": 1536,
    "strength": 0.6,
    "width": 1536
  },
  "lora_file": "",
  "model_anime": 0,
  "model_version": "%s",
  "negative_prompt": "ng_deepnegative_v1_75t",
  "num_images_per_prompt": 1,
  "num_inference_steps": 30,
  "points_type": 3,
  "prompt": "red color",
  "sampling_method": "DPM2MK",
  "seed": -1,
  "single_point": 8,
  "start_time": 1693382383284,
  "suffix": "jpeg",
  "taskid": 35656,
  "text2img": {
    "height": 512,
    "width": 512
  },
  "tileable": false,
  "upscale": 2.0,
  "userid": 94332619,
  "uuid": "114787e11bc7864690a0c8c1956d928e77",
  "width": 2048
}
'''

for v in sorted('v1.5'.split(',')):
    for i in range(5):
        js['payload'] = data_str.replace("%s", v)
        resp = requests.post('http://47.98.234.186:15672/api/exchanges/%2Fneptune.test/amq.default/publish',
                             headers=headers, json=js)

        print(resp.status_code)
        print(resp.content.decode('utf-8'))
