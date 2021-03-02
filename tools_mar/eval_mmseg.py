from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup

from segmentron.models.segbase import mmseg_evaluate
from mmseg.apis import init_segmentor, inference_segmentor
from tqdm import tqdm



class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode='testval', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)

        #####################
        # BATCH SIZE is always 1

        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        self.classes = val_dataset.classes

        ### Create network ###

        # Segmentron model
        # self.model = get_segmentation_model().to(self.device)

        # MMSeg model
        mmseg_config_file = "mmseg-configs/deeplabv3plus_r101-d8_512x512_80k_ade20k.py"
        mmseg_pretrained = "pretrained_weights/deeplabv3plus_r101-d8_512x512_80k_ade20k_20200615_014139-d5730af7.pth"
        self.model = init_segmentor(mmseg_config_file, mmseg_pretrained)

        self.model.to(self.device)
        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        logging.info("Using Val/Test img scale : {}".format(cfg.TEST.IMG_SCALE))
        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()
        pbar = tqdm(self.val_loader)
        for image, target, filename in pbar:
            image = image.to(self.device)
            target = target.to(self.device)

            assert image.shape[0] == 1, "Only batch-size 1 allowed when evaluating on test/val images"

            with torch.no_grad():
                output = mmseg_evaluate(model, image, target)

            self.metric.update(output, target)
            pixAcc, mIoU = self.metric.get()

            pbar.set_postfix_str("pixAcc: {:.3f}, mIoU: {:.3f}".format(pixAcc * 100, mIoU * 100))

        synchronize()
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(pixAcc * 100, mIoU * 100))


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.eval()