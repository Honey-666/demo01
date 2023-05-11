import io
import json
import sys
import time
import uuid
from time import sleep

import cv2
import numpy
import requests
import torch
import inspect
from python_bigbigwork_util import DefaultNacos, LoggingUtil
from python_bigbigwork_util.OssService import OssService
from python_bigbigwork_util.RabbitReceiveMq import RabbitReceiveMq
from python_bigbigwork_util.RabbitSendMq import RabbitSendMq
from PIL import Image, ImageChops
from TranslateUtil import en_to_zh
from diffusers.controlNet.stable_diffusion_multi_controlnet import StableDiffusionMultiControlNetPipeline, ControlNetProcessor
from diffusers import (ControlNetModel)

my_logger = LoggingUtil.get_logging()


class ControlNet:
    def __init__(self):
        n = DefaultNacos.get_web_instance(env, 'img-controlNet-service', 5000, 'python', ephemeral=False)
        config = n.get_config('config-rabbitmq', 'python')
        self.r = RabbitReceiveMq(config, 'img-controlnet')
        self.s = RabbitSendMq(config, 'img-controlnet')

        oss_config = n.get_config('config-oss', 'python')
        self.o = OssService(n.get_config('config-oss', 'python'), connection_section='connection-model')
        self.bucket_url = oss_config.get('img-diffusers', 'bucket_url')
        self.oss_host = oss_config.get('img-diffusers', 'oss_host')
        self.bucket_name = oss_config.get('img-diffusers', 'bucket_name')

        self.pipe = StableDiffusionMultiControlNetPipeline.from_pretrained(
            "../models/stable-diffusion-v1-5", safety_checker=None, torch_dtype=torch.float16
        ).to("cuda")
        self.canny = ControlNetModel.from_pretrained("../models/sd-controlnet-canny", torch_dtype=torch.float16).to(
            "cuda")
        self.scribble = ControlNetModel.from_pretrained("../models/sd-controlnet-scribble",
                                                        torch_dtype=torch.float16).to("cuda")
        self.hed = ControlNetModel.from_pretrained("../models/sd-controlnet-hed", torch_dtype=torch.float16).to("cuda")
        self.depth = ControlNetModel.from_pretrained("../models/sd-controlnet-depth", torch_dtype=torch.float16).to(
            "cuda")
        self.mlsd = ControlNetModel.from_pretrained("../models/sd-controlnet-mlsd", torch_dtype=torch.float16).to(
            "cuda")
        self.normal = ControlNetModel.from_pretrained("../models/sd-controlnet-normal", torch_dtype=torch.float16).to(
            "cuda")
        self.openpose = ControlNetModel.from_pretrained("../models/sd-controlnet-openpose",
                                                        torch_dtype=torch.float16).to("cuda")
        self.seg = ControlNetModel.from_pretrained("../models/sd-controlnet-seg", torch_dtype=torch.float16).to("cuda")

        self.pipe.enable_xformers_memory_efficient_attention()

        self.keys = set(inspect.signature(StableDiffusionMultiControlNetPipeline.__call__).parameters.keys())

        # self.canny_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        #     "../models/stable-diffusion-v1-5", low_cpu_mem_usage=False, device_map=None, safety_checker=None,
        #     controlnet=ControlNetModel.from_pretrained("../models/sd-controlnet-canny")).to("cuda")
        #
        # self.scribble_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        #     "../models/stable-diffusion-v1-5", low_cpu_mem_usage=False, device_map=None, safety_checker=None,
        #     controlnet=ControlNetModel.from_pretrained("../models/sd-controlnet-scribble")).to("cuda")

        # self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        # self.scribble_pipe.scheduler = UniPCMultistepScheduler.from_config(self.scribble_pipe.scheduler.config)

    def run(self):
        try:
            self.r.run(self.callback)
        except Exception as e:
            my_logger.exception('run error:{}', e)
            self.close()
        sleep(100)

    def close(self):
        self.r.close()
        self.s.close()

    def upload(self, image: Image, key):
        s = time.time()
        img_byte_arr = io.BytesIO()
        # image.save expects a file-like as a argument
        image.save(img_byte_arr, format='png')
        # Turn the BytesIO object back into a bytes object
        img_byte_arr = img_byte_arr.getvalue()
        self.o.upload_object_to_oss(self.oss_host, self.bucket_name, key, img_byte_arr)
        oss_url = self.bucket_url + key
        spend = time.time() - s
        my_logger.info('oss_url=%s,time=%s', oss_url, spend)
        return oss_url

    # pli类型的图片转 cv2(其中numpy.array(img) 将一个pli转numpy)
    def pil_to_cv2(self, im):
        return cv2.cvtColor(numpy.array(im), cv2.COLOR_RGB2BGR)

    # def crop(self, img: Image, w, h):
    #     width, height = img.size
    #
    #     margin_left = (width - w)
    #     t = (width - margin_left) % 8
    #     if t != 0:
    #         margin_left += t
    #
    #     margin_top = (height - h)
    #     t = (height - margin_top) % 8
    #     if t != 0:
    #         margin_top += t
    #
    #     crop_img = img.crop((margin_left / 2, margin_top / 2, width - margin_left / 2, height - margin_top / 2))
    #     return crop_img
    #
    # def crop_handle(self, img: Image, target: tuple):
    #     width, height = img.size
    #     target_ratio = target[0] / target[1]
    #     source_ratio = width / height
    #
    #     if target_ratio == source_ratio:
    #         return img
    #
    #     if target_ratio == 1:
    #         return self.crop(img, min(width, height), min(width, height))
    #     else:
    #         if target_ratio > source_ratio:
    #             return self.crop(img, width, int(width / target_ratio))
    #         else:
    #             return self.crop(img, int(height * target_ratio), height)

    def callback(self, ch, method, properties, body):
        js = {'num_inference_steps': 30, 'guidance_scale': 7.5, 'prompt': '', 'negative_prompt': ''}
        js.update(json.loads(body.decode('utf-8')))
        my_logger.info(js)

        return_json = {"task_id": js['task_id']}
        try:
            s = time.time()
            oss_url = self.picture_additional_color(js)
            spend = time.time() - s
            return_json['result_url'] = oss_url
            my_logger.info(f'json={return_json}')
            return_json['spend'] = spend
            return_json["status"] = 200
        except Exception as e:
            my_logger.exception('error')
            return_json["error"] = str(e)

        dumps = json.dumps(return_json, ensure_ascii=False)
        my_logger.info(f'message={dumps}')
        self.s.send_message(dumps)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def picture_additional_color(self, js):
        control_version = js['control_version']
        img_url = js['oss_url']

        pose_image = Image.open(io.BytesIO(requests.get(img_url, timeout=(3, 20)).content)).convert('RGB')
        contro_model_arr = None
        if control_version == 1:
            # 如果是白底黑线要做反转，canny只能处理黑底白线的图片
            # 判断平均像素值是否大于128，如果是，则认为是白色，否则认为是黑色
            mean_pixel_value = cv2.mean(self.pil_to_cv2(pose_image))[0]
            if mean_pixel_value > 128:
                print('白色线框图，对颜色做反转~~~~~')
                pose_image = ImageChops.invert(pose_image)

            contro_model_arr = [ControlNetProcessor(self.canny, pose_image)]
        elif control_version == 2:
            contro_model_arr = [ControlNetProcessor(self.scribble, pose_image)]
        elif control_version == 3:
            contro_model_arr = [ControlNetProcessor(self.hed, pose_image)]
        elif control_version == 4:
            contro_model_arr = [ControlNetProcessor(self.depth, pose_image)]
        elif control_version == 5:
            contro_model_arr = [ControlNetProcessor(self.mlsd, pose_image)]
        elif control_version == 6:
            contro_model_arr = [ControlNetProcessor(self.normal, pose_image)]
        elif control_version == 7:
            contro_model_arr = [ControlNetProcessor(self.openpose, pose_image)]
        elif control_version == 8:
            contro_model_arr = [ControlNetProcessor(self.seg, pose_image)]
        elif control_version == 9:
            contro_model_arr = [ControlNetProcessor(self.seg, pose_image), ControlNetProcessor(self.canny, pose_image)]
        elif control_version == 10:
            contro_model_arr = [ControlNetProcessor(self.depth, pose_image),
                                ControlNetProcessor(self.scribble, pose_image)]

        self.pipe.safety_checker = lambda images, clip_input: (images, False)
        self.pipe.set_progress_bar_config(disable=None)

        prompt_cn = js['prompt']
        prompt_en = en_to_zh(prompt_cn)
        n_prompt_cn = js['negative_prompt']
        n_prompt_en = en_to_zh(n_prompt_cn)
        print('prompt_cn={0} ,prompt_en={1} ,n_prompt_cn={2} ,n_prompt_en={3}'.format(prompt_cn, prompt_en, n_prompt_cn,
                                                                                      n_prompt_en))
        js['prompt'] = prompt_en
        js['negative_prompt'] = n_prompt_en
        js['processors'] = contro_model_arr
        tmp = {k: v for k, v in js.items() if k in self.keys}
        s = time.time()
        image = self.pipe(**tmp).images[0]

        spend = time.time() - s
        print('generate img', spend)
        key = "{}/{}.png".format('controlnet', uuid.uuid1())
        return self.upload(image, key)


if __name__ == '__main__':
    rs = sys.argv
    env = 'dev' if len(rs) < 2 else rs[1]

    while True:
        ControlNet().run()
