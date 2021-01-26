from __future__ import print_function
import io
from logging import log
from torch._C import device, dtype
from torch.utils.tensorboard import SummaryWriter
from calibration_library.metrics import CCELoss as perimageCCE


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

from PIL import Image
from tqdm import tqdm

import os
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

from calibration_library.cce_loss import CCELoss

class Evaluator(object):
    def __init__(self, args):

        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        self.lr = 0.1
        self.prefix = f"NoTrain_temp=1.5_bias=0"
        # self.prefix = f"loss=total_temp=1_random_bias_lr={self.lr}"
        self.writer = SummaryWriter(log_dir= f"cce_cityscapes_logs/{self.prefix}")
        # dataset and dataloader
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode='testval', transform=input_transform)
        # val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          shuffle=True,
                                          batch_size=cfg.TEST.BATCH_SIZE,
                                          drop_last=True,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)

        self.dataset = val_dataset
        self.classes = val_dataset.classes
        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)

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
    
    def eval(self):
        self.metric.reset()
        self.model.eval()
        model = self.model

        temp_weights = torch.eye(len(self.classes), device = "cuda")
        temp_weights/=1.5
        # temp_weights.requires_grad= True
        # temp_weights.requires_grad= True
        temp_bias = torch.zeros(len(self.classes), device = "cuda", requires_grad=False)
        # temp_weights = torch.rand(len(self.classes), len(self.classes), device="cuda", requires_grad=True)
        # temp_bias = torch.rand(len(self.classes), device="cuda", requires_grad=True)

        logging.info("Start training of temprature weights, Total sample: {:d}".format(len(self.val_loader)))

        cce_criterion = CCELoss(len(self.classes)).to(self.device)
        cross_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # optimizer = torch.optim.SGD([temp_weights, temp_bias], lr=self.lr)
        import time
        time_start = time.time()
        num_epochs = 1
        
        for epoch in range(num_epochs):
            eceEvaluator_perimage = perimageCCE(n_classes = len(self.classes))
            epoch_loss_cce_total = 0
            epoch_loss_cross_entropy_total = 0
            epoch_loss_total = 0
            for i, (images, targets, filenames) in enumerate(self.val_loader):
                # import pdb; pdb.set_trace()
                # optimizer.zero_grad()

                images = images.to(self.device)
                targets = targets.to(self.device)



                # print(image.shape)
                with torch.no_grad():
                    outputs = model.evaluate(images)

                outputs = outputs.permute(0, 2, 3, 1).contiguous()
                outputs = torch.matmul(outputs, temp_weights)
                outputs = outputs + temp_bias

                outputs = outputs.permute(0, 3, 1, 2).contiguous()
                
                loss_cce = cce_criterion.forward(outputs, targets)
                loss_cross_entropy = cross_criterion.forward(outputs, targets)

                total_loss = loss_cce + loss_cross_entropy

                epoch_loss_cce_total += loss_cce.item()
                epoch_loss_cross_entropy_total += loss_cross_entropy.item()
                epoch_loss_total += total_loss.item()

                # total_loss.backward()
                # optimizer.step()
                
                with torch.no_grad():
                    for output, target in zip(outputs,targets.detach()):
                        #older ece requires softmax and size output=[class,w,h] target=[w,h]
                        eceEvaluator_perimage.update(output.softmax(dim=0), target)
                # print(outputs.shape)
                # print(eceEvaluator_perimage.get_overall_CCELoss())
                print(f"batch :{i+1}/{len(self.val_loader)}" + "loss cce : {:.5f} | loss cls : {:.5f} | loss tot : {:.5f}".format(loss_cce, loss_cross_entropy, total_loss))
            
            print(temp_weights)
            print(temp_bias)
            epoch_loss_cce_total /= len(self.val_loader)
            epoch_loss_cross_entropy_total /= len(self.val_loader)
            epoch_loss_total /= len(self.val_loader)
            
            cce_table_image, dif_map= eceEvaluator_perimage.get_perc_table_img(self.classes)
            self.writer.add_image("CCE_table", cce_table_image, epoch, dataformats="HWC")
            self.writer.add_image("DifMap", dif_map, epoch, dataformats="HWC")
            
            self.writer.add_scalar(f"Cross EntropyLoss_LR", epoch_loss_cross_entropy_total, epoch)
            self.writer.add_scalar(f"CCELoss_LR", epoch_loss_cce_total, epoch)
            self.writer.add_scalar(f"Total Loss_LR", epoch_loss_total, epoch)
            self.writer.add_histogram("Weights", temp_weights, epoch)
            self.writer.add_histogram("Bias", temp_bias, epoch)
            # output = output/temp_weights
            # print(output.shape)
            # print(temp_weights, temp_bias)

            if epoch > 0 and epoch % 10 == 0:
                print("saving weights.")
                np.save("weights/foggy_cityscapes/wt_{}_{}.npy".format(epoch, self.prefix), temp_weights.cpu().detach().numpy())
                np.save("weights/foggy_cityscapes/b{}_{}.npy".format(epoch, self.prefix), temp_bias.cpu().detach().numpy())


            # print("epoch {} : loss {:.5f}".format(epoch, epoch_loss))
            # import pdb; pdb.set_trace()
        
        self.writer.close()


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
