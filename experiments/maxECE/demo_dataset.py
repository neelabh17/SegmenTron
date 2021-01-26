from __future__ import print_function
import io
from logging import log
from new_tools import convcrf
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

import os
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

from convcrf import *

import matplotlib.pyplot as plt
def giveComparisionImages(precrf_output,postcrf_output, raw_image, gt_label,classes,outname):
    '''
    precrf_output-> [1,21,h,w] cuda tensor
    postcrf_output-> [1,21,h,w] cuda tensor
    raw_image->[1,3,h,w] cuda tensor
    gt_label->[1,h,w] cuda tensor
    '''
    uncal_labels = np.unique(torch.argmax(precrf_output.squeeze(0),dim=0).cpu().numpy())
    cal_labels = np.unique(torch.argmax(postcrf_output.squeeze(0),dim=0).cpu().numpy())



    # Bringing the shapes to justice
    precrf_output=precrf_output.squeeze(0).cpu().numpy()
    postcrf_output=postcrf_output.squeeze(0).cpu().numpy()
    raw_image=raw_image.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
    gt_label=gt_label.squeeze(0).cpu().numpy()
    
    
    # import pdb; pdb.set_trace()

    
    # gt_label=get_gt_with_id(imageName)
    # if(np.sum((cal_labelmap!=uncal_labelmap).astype(np.float32))==0):
    if False:
        pass
    else:
        # Show result for each class
        cols = int(np.ceil((max(len(uncal_labels), len(cal_labels)) + 1)))+1
        rows = 4
        
        plt.figure(figsize=(20, 20))
        ax = plt.subplot(rows, cols, 1)
        ax.set_title("Input image")
        ax.imshow(raw_image[:, :, ::-1])
        ax.axis("off")
        ax = plt.subplot(rows, cols, cols + 1)
        # @ neelabh remove this
        loss=1.999999999999999
        ax.set_title("Accuracy dif = {:0.3f}".format(loss))
        ax.imshow(raw_image[:, :, ::-1])
        ax.axis("off")
        # ax = plt.subplot(rows, cols, 2 * cols + 1)
        # gradient = np.linspace(0, 1, 256)
        # gradient = np.vstack((gradient, gradient))
        # ax.imshow(gradient, cmap="nipy_spectral")
        # ax.set_title("Acc")
        # ax.imshow(raw_image[:, :, ::-1])
        # ax.axis("off")

        for i, label in enumerate(uncal_labels):
            ax = plt.subplot(rows, cols, i + 3)
            ax.set_title("Uncalibrated-" + classes[label])
            ax.imshow(precrf_output[label], cmap="nipy_spectral")

            ax.axis("off")

        for i, label in enumerate(cal_labels):
            ax = plt.subplot(rows, cols, cols + i + 3)
            ax.set_title("Calibrated-" + classes[label])
            ax.imshow(postcrf_output[label], cmap="nipy_spectral")
            ax.axis("off")

        for i, label in enumerate(cal_labels):
            ax = plt.subplot(rows, cols, 2 * cols + i + 3)
            
            min_dif=np.min(precrf_output[label] - postcrf_output[label])
            max_dif=np.max(precrf_output[label] - postcrf_output[label])

            dif_map=np.where((precrf_output[label] - postcrf_output[label])>0,(precrf_output[label] - postcrf_output[label]),0)
            
            ax.set_title(
                "decrease: "
                + classes[label]
                + " max={:0.3f}".format(
                    max_dif
                )
            )
            ax.imshow(
                dif_map
                / max_dif,
                cmap="nipy_spectral",
            )
            ax.axis("off")
        
        for i, label in enumerate(cal_labels):
            ax = plt.subplot(rows, cols, 3 * cols + i + 3)
            
            min_dif=np.min(precrf_output[label] - postcrf_output[label])
            max_dif=np.max(precrf_output[label] - postcrf_output[label])

            dif_map=np.where((precrf_output[label] - postcrf_output[label])<0,(precrf_output[label] - postcrf_output[label]),0)
            
            ax.set_title(
                "increase: "
                + classes[label]
                + " max={:0.3f}".format(
                    -min_dif
                )
            )
            ax.imshow(
                dif_map
                / min_dif,
                cmap="nipy_spectral",
            )
            ax.axis("off")
        
        # acc_cal_mask=(gt_label==cal_label).astype(np.float32)
        # acc_uncal_mask=(gt_label==uncal_label).astype(np.float32)
        # # for i, label in enumerate(cal_labels):
        # ax = plt.subplot(rows, cols, 2 * cols + 1)

        # maxi=np.max(acc_uncal_mask - acc_cal_mask)
        # un=np.unique(acc_uncal_mask - acc_cal_mask)
        # print(un)
        # dif_map=np.where((acc_uncal_mask - acc_cal_mask)<0,(acc_uncal_mask - acc_cal_mask),0)
        
        # ax.set_title(
        #     "increase ac: "
        #     + classes[label]
        #     + " max={:0.3f}".format(
        #         1
        #     )
        # )
        # ax.imshow(
        #     dif_map/(-1),
        #     cmap="nipy_spectral",
        # )

        # # for i, label in enumerate(cal_labels):
        # ax = plt.subplot(rows, cols, 3 * cols + 1)
        # dif_map=np.where((acc_uncal_mask - acc_cal_mask)>0,(acc_uncal_mask - acc_cal_mask),0)
        
        # ax.set_title(
        #     "decrease ac: "
        #     + classes[label]
        #     + " max={:0.3f}".format(
        #         1
        #     )
        # )
        # ax.imshow(
        #     dif_map/1,
        #     cmap="nipy_spectral",
        # )
            
        
        

        plt.tight_layout()
        plt.savefig(outname)


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


        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        temp=1.7
        usingCRF=True
        output_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, 'noCRF_foggy_conv9_full_dataset_comp_{}_{}_{}_{}_temp_{}_crf_{}'.format(
            cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE, cfg.DATASET.NAME, cfg.TIME_STAMP,temp,usingCRF))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        time_start = time.time()
        for (image, target, filename) in tqdm(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            # print(image.shape)
            with torch.no_grad():
                output = model.evaluate(image)
                no_cal_output=output.clone()
                
                forcrf_output=output/temp 

                # if use CRF
                filename = filename[0]
                raw_image = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.float32).transpose(2, 0, 1)
                raw_image = torch.from_numpy(raw_image).to(self.device)
                raw_image = raw_image.unsqueeze(dim=0)
                crf = GaussCRF(conf=get_default_conf(), shape=image.shape[2:], nclasses=len(self.classes), use_gpu=True)
                crf = crf.to(self.device)
                assert image.shape == raw_image.shape
                forcrf_output = crf.forward(forcrf_output, raw_image)
                forcrf_nocali_output = crf.forward(no_cal_output, raw_image)


                outname = os.path.splitext(os.path.split(filename)[-1])[0] + f'_temp_{temp}_crf_{usingCRF}.png'
                savename=os.path.join(output_dir, outname)
                plt=giveComparisionImages(output.softmax(dim=1), (no_cal_output/temp).softmax(dim=1), raw_image,target, self.classes,savename)
                

                # print(output.shape, forcrf_output.shape, target.shape)

                # import pdb; pdb.set_trace()
            

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
