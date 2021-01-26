from __future__ import print_function

from numpy.lib.function_base import vectorize
from calibration_library.metrics import CCELoss
import io
from logging import log
# from experiments.classCali_full_CRF import convcrf
# from experiments.classCali_full_CRF.convcrf import GaussCRF, get_default_conf

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
import matplotlib.pyplot as plt

from torchvision import transforms
from segmentron.utils.visualize import get_color_pallete
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.utils.score import SegmentationMetric
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.distributed import make_data_sampler, make_batch_data_sampler
import torch.utils.data as data

from crf import DenseCRF


# from crfasrnn.crfrnn import CrfRnn
from PIL import Image
from tqdm import tqdm

import os
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

# from convcrf import *


class Evaluator(object):
    def __init__(self, args):

        self.args = args
        self.device = torch.device(args.device)

        self.args = args
        self.device = torch.device(args.device)

        self.n_bins = 10
        self.ece_folder = "experiments/classCali_full_CRF/eceData"
        # self.postfix = "Conv13_PascalVOC_GPU"
        # self.postfix = "Min_Foggy_1_conv13_PascalVOC_GPU"
        # self.postfix = "Foggy_1_conv13_PascalVOC_GPU"
        self.postfix = "fggy_zurich_dec_21"
        self.temp = 1.5
        print("Current temp being used : {}".format(self.temp))
        self.showProbMaps = False
        # self.useCRF = False
        self.useCRF = True
        self.postprocessor= DenseCRF(iter_max=cfg.CRF.ITER_MAX,
                                        pos_xy_std=cfg.CRF.POS_XY_STD,
                                        pos_w=cfg.CRF.POS_W,
                                        bi_xy_std=cfg.CRF.BI_XY_STD,
                                        bi_rgb_std=cfg.CRF.BI_RGB_STD,
                                        bi_w=cfg.CRF.BI_W,
                                    )

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
        self.uncal_metric = SegmentationMetric(val_dataset.num_class, args.distributed)
        self.cal_metric = SegmentationMetric(val_dataset.num_class, args.distributed)

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
        eceEvaluator = CCELoss(n_classes = len(self.classes))
        time_start = time.time()
        for (image, target, filename) in tqdm(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            # print(image.shape)
            with torch.no_grad():
                output = model.evaluate(image)
                # output = torch.softmax(output, dim=1)
                # output =output*0 + 1/len(self.classes)

                output_uncal = output.clone()
                
                # Direchlet
                # output = torch.log(torch.softmax(output, dim=1))


                # Vector temps
                # vt=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
                # vt[0]= 1.54
                # vt[10] = 1.3 
                # vector_temp = torch.Tensor(vt).reshape(1,-1,1,1).cuda()
                # output/= vector_temp
                
                # print(vector_temp)

                # output_cal = output / vector_temp


                output_cal = output / self.temp

                # if use CRF
                if(self.useCRF):
                    filename = filename[0]
                    # print(filename)
                    raw_image_cal = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.uint8)
                    raw_image_uncal = raw_image_cal.copy()

                    cal_probs = torch.softmax(output_cal, dim=1).squeeze(dim=0).cpu().numpy()
                    uncal_probs = torch.softmax(output_uncal, dim=1).squeeze(dim=0).cpu().numpy()

                    # cal_probs = output_cal.squeeze(dim=0).cpu().numpy()
                    # uncal_probs = output_uncal.squeeze(dim=0).cpu().numpy()


                    output_cal_crf = self.postprocessor(raw_image_cal, cal_probs)
                    output_uncal_crf = self.postprocessor(raw_image_uncal, uncal_probs)

                    # print(output_cal_crf.shape)

                    output_cal_crf = torch.Tensor(output_cal_crf).unsqueeze(0)
                    output_uncal_crf = torch.Tensor(output_uncal_crf).unsqueeze(0)
                
                    comparisionFolder = "experiments/classCali_full_CRF/comparisionImages"
                    saveFolder = os.path.join(
                        comparisionFolder, self.postfix + f"_temp={self.temp}"
                    )
                    makedirs(saveFolder)
                    saveName = os.path.join(saveFolder, os.path.basename(filename))
                    
                    
                    image_pil = Image.open(filename).convert('RGB')
                    after_crf_uncal_mask = get_color_pallete(torch.argmax(output_uncal_crf, dim=1).squeeze(0).cpu().numpy(), cfg.DATASET.NAME)
                    after_crf_cal_mask = get_color_pallete(torch.argmax(output_cal_crf, dim=1).squeeze(0).cpu().numpy(), cfg.DATASET.NAME)
                    difference_map_mask = get_color_pallete((torch.argmax(output_uncal_crf, dim=1).squeeze(0).cpu().numpy() != torch.argmax(output_cal_crf, dim=1).squeeze(0).cpu().numpy())*19, cfg.DATASET.NAME)
                    gt_mask = get_color_pallete(target.squeeze(0).cpu().numpy(), cfg.DATASET.NAME)

                    # Concatenating horizontally [out_post_crf,out_pre_crf, rgb]
                    dst = Image.new('RGB', (5*gt_mask.width+12, gt_mask.height), color="white")
                    # dst = Image.new('RGB', (4*gt_mask.width+9, gt_mask.height), color="white")
                    dst.paste(after_crf_uncal_mask, (0, 0))
                    dst.paste(after_crf_cal_mask, (gt_mask.width+3, 0))
                    dst.paste(gt_mask, (2*gt_mask.width+6, 0))
                    dst.paste(image_pil, (3*gt_mask.width+9, 0))
                    dst.paste(difference_map_mask, (4*gt_mask.width+ 12, 0))
                    dst.save(saveName)
                # Required softmax inputs
                # eceEvaluator.update(output_cal.softmax(dim=1).squeeze(0), target.squeeze(0))

            # print(output.shape)
            if(self.useCRF):
                # self.metric.update(output_cal_crf, target)
                self.cal_metric.update(output_cal_crf.cuda(), target)
                
                self.uncal_metric.update(output_uncal_crf.cuda(), target)
                self.metric.update(output_cal, target)
            else:
                self.metric.update(output_cal, target)

        # eceEvaluator.get_perc_table(self.classes)
        # overallLoss = eceEvaluator.get_overall_CCELoss()
        # eceEvaluator.get_classVise_CCELoss(self.classes)

        # f= open("cityscapes_full_cali_testing.txt", "a")

        # f.write(f"Temp: {self.temp} \t CCE Loss: {overallLoss}\n")
        # f.close()
        if self.useCRF:
            print("Calibrated + CRF metrics :")
            pixAcc, mIoU, category_iou = self.cal_metric.get(return_category_iou=True)
            logging.info('pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                    pixAcc * 100, mIoU * 100))
            print("Uncalibrated + CRF metrics :") 
            pixAcc, mIoU, category_iou = self.uncal_metric.get(return_category_iou=True)
            logging.info('pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                    pixAcc * 100, mIoU * 100))
            print("Softmax Metrics : ")
            pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
            logging.info('pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                    pixAcc * 100, mIoU * 100))
        else:
            print("Softmax Metrics : ")
            pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
            logging.info('pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                    pixAcc * 100, mIoU * 100))

        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        
        # f= open("cityscapes_cali_Acc.txt", "a")

        # f.write("Temp: {} \t pixAcc: {:.3f}, mIoU: {:.3f} \n".format(self.temp,
        #         pixAcc * 100, mIoU * 100))
        # f.close()


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    # cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # temps=[0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55,
    #    0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95, 1.  , 1.05, 1.1 ,
    #    1.15, 1.2 , 1.25, 1.3 , 1.35, 1.4 , 1.45, 1.5 , 1.55, 1.6 , 1.65,
    #    1.7 , 1.75, 1.8 , 1.85, 1.9 , 1.95, 2.  , 2.05, 2.1 , 2.15, 2.2 ,
    #    2.25, 2.3 , 2.35, 2.4 , 2.45, 2.5 , 2.55, 2.6 , 2.65, 2.7 , 2.75,
    #    2.8 , 2.85, 2.9 , 2.95, 3.  , 3.05, 3.1 , 3.15, 3.2 , 3.25, 3.3 ,
    #    3.35, 3.4 , 3.45, 3.5 , 3.55, 3.6 , 3.65, 3.7 , 3.75, 3.8 , 3.85,
    #    3.9 , 3.95, 4.  , 4.05, 4.1 , 4.15, 4.2 , 4.25, 4.3 , 4.35, 4.4 ,
    #    4.45, 4.5 , 4.55, 4.6 , 4.65, 4.7 , 4.75, 4.8 , 4.85, 4.9 , 4.95,
    #    5.  ]

    default_setup(args)
    evaluator = Evaluator(args)
    # i=1

    # while(i<len(temps)):
    #     evaluator.temp= temps[i]
    evaluator.eval()    
        # i+=2
