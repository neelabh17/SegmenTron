from __future__ import print_function
import io
from logging import log
# from new_tools import convcrf
from new_tools.convcrf import GaussCRF, get_default_conf

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import joblib

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.utils.score import SegmentationMetric
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.distributed import make_data_sampler, make_batch_data_sampler
import torch.utils.data as data

# from crfasrnn.crfrnn import CrfRnn
from PIL import Image
from tqdm import tqdm

from crf import DenseCRF

import os
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


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
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)

        self.dataset = val_dataset
        self.classes = val_dataset.classes
        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)

        # DEFINE data for noisy
        val_dataset_noisy = get_segmentation_dataset(cfg.DATASET.NOISY_NAME, split='val', mode='testval', transform=input_transform)
        val_sampler_noisy = make_data_sampler(val_dataset_noisy, False, args.distributed)
        val_batch_sampler_noisy = make_batch_data_sampler(val_sampler_noisy, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.val_loader_noisy = data.DataLoader(dataset=val_dataset_noisy,
                                          batch_sampler=val_batch_sampler_noisy,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)

        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        self.model.to(self.device)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    @torch.no_grad()
    def eval(self, val_loader, crf):
        self.metric.reset()
        self.model.eval()
        model = self.model
        
        logging.info("Start validation, Total sample: {:d}".format(len(val_loader)))
        import time
        time_start = time.time()

        for (image, target, filename) in tqdm(val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            # print(image.shape)
            
            output = model.evaluate(image)

            # using CRF --------------------
            filename = filename[0]
            # print(filename)
            raw_image = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.uint8)

            probmap = torch.softmax(output, dim=1)[0].cpu().numpy()
            # print(probmap.shape)
            output = crf(raw_image, probmap)

            # put numpy back on gpu since target is on gpu
            output = torch.from_numpy(output).cuda().unsqueeze(dim=0)
            # ---------------------------

            # print(output.shape)
            self.metric.update(output, target)
            # pixAcc, mIoU = self.metric.get()

        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                pixAcc * 100, mIoU * 100))

        return pixAcc * 100, mIoU * 100


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)

    crf = DenseCRF(
        iter_max=5,
        pos_w=3,
        pos_xy_std=1,
        bi_w=4,
        bi_xy_std=67,
        bi_rgb_std=3
    )

    # evaluator.eval(evaluator.val_loader, crf)
    evaluator.eval(evaluator.val_loader_noisy, crf)
