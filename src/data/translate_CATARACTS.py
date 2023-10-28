import os
import shutil

import torch
import tqdm
import cv2
import numpy as np
import torchvision.transforms as Tf

from src.agent.unit import UNIT
from src.utils.datasets import denormalize
from src.utils.train import read_config

root = '/local/scratch/Catarakt/videos/micro/'
target = '/local/scratch/CATARACTS-to-Cataract101/'

conf, _ = read_config('../../results/MotionUNIT_CATARACTS_Cataract101_192pix/2022_10_13-08_46_18/config.yml')
conf.device = 'cuda:0'

agent = UNIT(conf)
agent.gen_A.load_state_dict(
       torch.load('../../results/MotionUNIT_CATARACTS_Cataract101_192pix/'
                  '2022_10_13-08_46_18/checkpoints/gen_A_epoch199.pth', map_location='cpu'))
agent.gen_A.eval()
agent.gen_B.load_state_dict(
       torch.load('../../results/MotionUNIT_CATARACTS_Cataract101_192pix/'
                  '2022_10_13-08_46_18/checkpoints/gen_B_epoch199.pth', map_location='cpu'))
agent.gen_B.eval()

transforms = Tf.Compose([
    Tf.ToTensor(),
    Tf.Resize((192, 192)),
    Tf.Normalize(0.5, 0.5)
])

SAMPLE_FREQ = 2

assert os.path.isdir(root)

if os.path.exists(target) and os.path.isdir(target):
    shutil.rmtree(target)

os.makedirs(target, exist_ok=False)

for video_file in tqdm.tqdm(os.listdir(root)):

    if not video_file.endswith('.mp4'):
        continue

    os.makedirs(target + f'{video_file.replace(".mp4", "")}/')

    vidcap = cv2.VideoCapture(root + f'{video_file}')
    success, image = vidcap.read()
    count = 0
    while success:
        # Skip x frames
        if not count % SAMPLE_FREQ:
            # Translate frame
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = transforms(image).unsqueeze(0).to(conf.device)
            with torch.no_grad():
                h, n = agent.gen_A.encode(img_tensor)
                translated_img = agent.gen_B.decode(h + n)
            image = (denormalize(translated_img[0]) * 255).permute(1, 2, 0).cpu().numpy()
            image = np.float32(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Save frame as JPG file
            cv2.imwrite(target + f'{video_file.replace(".mp4", "")}/frame%d.jpg' % count, image)
        success, image = vidcap.read()
        count += 1
