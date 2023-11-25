# @FileName：ControlNetLiteTest.py
# @Description：
# @Author：dyh
# @Time：2023/10/19 16:35
# @Website：www.xxx.com
# @Version：V1.0
from PIL import Image
from library import sdxl_train_util

from sdxl_gen_img import main, setup_parser

parser = setup_parser()

args = parser.parse_args(args=[])
args.prompt = 'beautiful room'
img_path = 'C:\\work\\pythonProject\\aidazuo\\jupyter-script\\test-img\\mlsd-0000.png'
img = Image.open(img_path)
args.guide_image = [img]
# args.guide_image_path = [img_path]
args.ckpt = 'C:\\work\\pythonProject\\aidazuo\\models\\Stable-diffusion\\sd_xl_base_1.0'
args.control_net_lllite_models = [
    'C:\\work\\pythonProject\\aidazuo\models\\ControlNet\\bdsqlsz_controlllite_xl_mlsd_V2.safetensors']
args.vae = 'C:\\work\\pythonProject\\aidazuo\\models\\VAE\\sdxl-vae-fp16-fix'
args.tokenizer_cache_dir = 'C:\\work\\pythonProject\\aidazuo\\models'
args.steps = 10
args.sampler = 'dpmsolver++'
args.fp16 = True
args.H = 512
args.W = 512
images = main(args)

print(images)


