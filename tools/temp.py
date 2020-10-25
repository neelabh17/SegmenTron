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
import cv2
import numpy as np

from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup


from crf import DenseCRF


class Evaluator(object):
    def __init__(self, args):
        self.postprocessor= DenseCRF(iter_max=cfg.CRF.ITER_MAX,
                                        pos_xy_std=cfg.CRF.POS_XY_STD,
                                        pos_w=cfg.CRF.POS_W,
                                        bi_xy_std=cfg.CRF.BI_XY_STD,
                                        bi_rgb_std=cfg.CRF.BI_RGB_STD,
                                        bi_w=cfg.CRF.BI_W,
                                    )
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode='testval', transform=input_transform)
        # made shuffle true
        val_sampler = make_data_sampler(val_dataset, shuffle=True, distributed=args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        self.classes = val_dataset.classes
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
            cfg.MODEL.BN_EPS_FOR_ENCODER:
            logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
            self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
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

        temp = torch.nn.Parameter(torch.ones(1) * 1.5)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.DATASET.IGNORE_INDEX)
        optimizer = torch.optim.SGD([temp], lr=0.01)

        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()
        
        for epoch in range(10):

            logging.info("Epoch Started {}".format(epoch))
            loss_epoch = 0.0

            for i, (image, target, filename) in enumerate(self.val_loader):

                optimizer.zero_grad()
                
                image = image.to(self.device)
                # target = target.to(self.device)

                with torch.no_grad():
                    output = model.evaluate(image)

                output = output.cpu()
            
                # print(output.shape)
                # print(target.shape)

                loss = criterion(output/temp, target)
                loss_epoch += loss.item()
                loss.backward()
                optimizer.step()

                logging.info("Batch {} loss for Temp Scaling : {}".format(i, loss))
            
            logging.info("Epoch {} loss for Temp Scaling : {}".format(epoch, loss_epoch / (len(self.val_loader))))
            logging.info("Epoch {} Temp Scaling factor is : {}".format(temp))

        synchronize()
        print('Final scaled temp : {}'.format(temp))


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    # import pdb; pdb.set_trace()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.eval()