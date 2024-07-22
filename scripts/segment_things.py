import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import argparse

import os
from PIL import Image
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from models.model import build_model
from dataset import transforms as T
from utils.misc import ImageList

image_orig = cv2.imread('/home/bibahaduri/dota_dataset/coco/test2017/P0086.1.0.jpg')
image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)

image = Image.open(os.path.join('/home/bibahaduri/dota_dataset/coco/test2017/P0086.1.0.jpg')).convert('RGB')  ## /home/bibahaduri/dota_dataset/coco/test2017/P0086.1.0.jpg  #/home/bibahaduri/segment-anything/assets/crop_1_0.jpeg
config_dict = {'pretrained_backbone_path': None,}

cfg = argparse.Namespace(**config_dict)

def create_transform():
    normalize = T.Compose(
        [T.Normalize2([123.675, 116.28, 103.53], [58.395, 57.12, 57.375], enc_img_size=1024)]
    )
    return T.Compose([T.ResizeLongestSide(target_length = 1024), normalize,])

model = build_model(num_classes=2, args=cfg)
preprocessor = create_transform()
target ={}
image, target = preprocessor(image, target)
target['orig_size'] = torch.tensor([512, 512])
target['boxes'] = torch.tensor([])
target['labels'] = torch.tensor([]).to(torch.int64)
targets = [target]

checkpoint = torch.load('/home/bibahaduri/detect_everything/output/checkpoint.pth', map_location="cpu")
missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model"], strict=False
        )
if len(missing_keys) > 0:
    print("Missing Keys: {}".format(missing_keys))

if len(unexpected_keys) > 0:
    print("Unexpected Keys: {}".format(unexpected_keys))
device = 'cuda'
model = model.to('cuda')
image = image.to('cuda')
model.eval()
images = image.unsqueeze(dim=0)
image_sizes = [img.shape[-2:] for img in images]
image_sizes_list = [(t['size'][0].item(), t['size'][0].item()) for t in targets] ##: List[Tuple[int, int]] = []

image_list = ImageList(images, image_sizes_list)
output = model(image_list, targets)

masks = output[1][0]['masks'].detach().cpu()
numpy_list = [np.round(masks[i, 0].numpy()).astype(np.int64) for i in range(masks.shape[0])]

masks = [{'segmentation': mask, 'area': mask.sum()} for mask in numpy_list]

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation'].astype(bool)

        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    ax.imshow(img)









def show_boxes(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in anns:
        # rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        box = ann['bbox']
        x0, y0 = box[0], box[1]
        w, h = box[2], box[3] #  - box[0], - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h,
                                fill=False, color='blue', linewidth=3))


plt.figure(figsize=(10,10))
plt.imshow(image_orig)
show_anns(masks)
# show_boxes(masks2)
plt.axis('off')
plt.savefig('./assets/200plotnibox8692.png')
plt.close()