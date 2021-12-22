import mmcv
import os
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset, ImageNet
from mmcls.models import build_classifier
from mmcls.datasets.pipelines import Compose
from time import time
from typing import Union


class Arguments:
    config = "mmclassification/configs/swin_transformer/swin-tiny_16xb64_in1k.py"
    launcher = "none"
    checkpoint = "checkpoints/swin_tiny_patch4_window7_224-160bb0a5.pth"
    device = "cpu"
    tmp_dir = "tmp/"


args = Arguments()
cfg = mmcv.Config.fromfile(args.config)

if args.launcher == 'none':
    distributed = False
else:
    distributed = True
    init_dist(args.launcher, **cfg.dist_params)

model = build_classifier(cfg.model)
checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
pipeline = Compose(cfg.data.test.pipeline)

if args.device == 'cpu':
    model = model.cpu()
else:
    model = MMDataParallel(model, device_ids=[0])
model.CLASSES = ImageNet.CLASSES
model.eval()


def feature_generator(img_paths, batch_size=16):
    pending_tensors = []
    pending_paths = []
    for i, img_path in enumerate(mmcv.track_iter_progress(img_paths)):
        try:
            img = pipeline({
                'img_info': {'filename': img_path},
                'img_prefix': None
            })['img']
        except AttributeError:
            # Be aware of no-image files.
            continue
        pending_tensors.append(img)
        pending_paths.append(img_path)
        if (i + 1) % batch_size == 0 or i + 1 == len(img_paths):
            with torch.no_grad():
                features = model.extract_feat(
                    torch.stack(pending_tensors).to(args.device),
                    stage='neck')
            yield features, pending_paths
            pending_tensors = []
            pending_paths = []


def extract_feature(img: Union[np.ndarray, str]):
    if isinstance(img, np.ndarray):
        filename = os.path.join(args.tmp_dir, f"{time}.png")
        mmcv.imwrite(img, filename)
    else:
        filename = img
    img = pipeline({
        'img_info': {'filename': filename},
        'img_prefix': None
    })['img']
    with torch.no_grad():
        result = model.extract_feat(img[None, ...], stage='neck')
    return result[0].numpy().astype(np.float32)
