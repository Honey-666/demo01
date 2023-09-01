import time

from PIL import Image
from controlnet_aux import OpenposeDetector
# from controlnet_aux.open_pose import Body, Hand, Face

model_path = 'C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts'

# pose_body = Body('C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts\\body_pose_model.pth')
# pose_hand = Hand('C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts\\hand_pose_model.pth')
# pose_face = Face('C:\\work\\pythonProject\\aidazuo\\models\\ControlNet\\annotator\\ckpts\\facenet.pth')
#
# openpose = OpenposeDetector(pose_body, pose_hand, pose_face)
openpose = OpenposeDetector.from_pretrained(model_path).to('cuda')

img_path = '../../../img/control/cfe82f12f7814acca6c7ea9f61f722ad.jpg'
# img_path = 'C:\\Users\\bbw\\Desktop\\8eb316ba8f2642ada2c0a39206b21992.jpg'
img = Image.open(img_path)
for _ in range(5):
    s = time.time()
    im = openpose(img, 512, 512, include_hand=True)
    im.save('openpose_handle.png')
    print(time.time() - s)
