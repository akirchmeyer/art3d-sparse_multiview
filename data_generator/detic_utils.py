# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
import torch
setup_logger()

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
path = '/home/akirchme/packages/Detic'
old_path = os.path.dirname(__file__)
os.chdir(f'{path}')
sys.path.append(f'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test

# Build the detector and download our pretrained weights
cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)

cfg.merge_from_file(f"configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
predictor = DefaultPredictor(cfg)


# Setup the model's vocabulary using build-in datasets

BUILDIN_CLASSIFIER = {
    'lvis': f'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': f'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': f'datasets/metadata/oid_clip_a+cname.npy',
    'coco': f'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

vocabulary = 'coco' # change to 'lvis', 'objects365', 'openimages', or 'coco'
print(f'Using "{vocabulary}" vocabulary ')
metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)
os.chdir(old_path)

def detic(image, cls, mode='cls'):
    preds = predictor(image)['instances']
    masks = preds.pred_masks
    classes = preds.pred_classes

    labels = metadata.thing_classes

    if mode == 'cls':
        if len(masks) == 0:
            return np.zeros(image.shape[:-1])
        output = torch.zeros_like(masks[0,:,:])
        for i in range(len(masks)):
            if labels[classes[i]] == cls:
                output = output | masks[i,:,:] 
        return output.cpu().numpy()
    elif mode == 'inst':
        return [masks[i, :, :].cpu().numpy() for i in range(len(masks)) if labels[classes[i]] == cls]
    else:
        raise "Unrecognized mode"