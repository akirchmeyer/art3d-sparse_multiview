import panopticapi
from panopticapi.utils import id2rgb, rgb2id
import torchvision.transforms as T
import requests
import io
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import seaborn as sns
import itertools

# These are the COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Detectron2 uses a different numbering scheme, we build a conversion table
coco2d2 = {}
count = 0
for i, c in enumerate(CLASSES):
  if c != "N/A":
    coco2d2[i] = count
    count+=1

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
model.eval();

# mean-std normalize the input image (batch-size: 1)
def detr(image, mode='fg'):
    img = transform(image).unsqueeze(0)
    out = model(img)
    # img = img.permute((0,2,3,1)).squeeze(0)
    # compute the scores, excluding the "no-object" class (the last one)
    # scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
    # threshold the confidence
    # keep = scores > 0.85

    # the post-processor expects as input the target size of the predictions (which we set here to the image size)
    result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]
    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    final_w, final_h = panoptic_seg.size
    panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
    # We retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb2id(panoptic_seg)
    panoptic_seg[:, :, :] = 0

    im = np.array(image.copy().resize((final_w, final_h)))#[:, :, ::-1]
    
    if mode == 'id':
        return im, panoptic_seg_id
    # Finally we color each mask individually
    if mode == 'fg':
        for id in range(1, panoptic_seg_id.max() + 1):
            panoptic_seg[panoptic_seg_id == id] = 255
    if mode == 'seg':
        # Finally we color each mask individually
        palette = itertools.cycle(sns.color_palette())
        for id in range(1, panoptic_seg_id.max() + 1):
            panoptic_seg[panoptic_seg_id == id] = np.asarray(next(palette)) * 255
    return im, panoptic_seg    