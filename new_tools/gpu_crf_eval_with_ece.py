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
from calibration_library import metrics, visualization


# from crfasrnn.crfrnn import CrfRnn
from PIL import Image
from tqdm import tqdm

import os
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

from convcrf import *


class Evaluator(object):
    def __init__(self, args):

        self.args = args
        self.device = torch.device(args.device)

        self.n_bins=15
        self.ece_folder="eceData"
        self.postfix="Foggy_1_conv13_PascalVOC_GPU"
        self.temp=1.7
        # self.useCRF=False
        self.useCRF=True

        self.ece_criterion= metrics.IterativeECELoss()
        self.ece_criterion.make_bins(n_bins=self.n_bins)

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
    
    def eceOperations(self,bin_total, bin_total_correct, bin_conf_total):
        eceLoss=self.ece_criterion.get_interative_loss(bin_total, bin_total_correct, bin_conf_total)
        print('ECE with probabilties %f' % (eceLoss))
        
        saveDir=os.path.join(self.ece_folder,self.postfix)
        makedirs(saveDir)

        file=open(os.path.join(saveDir,"Results.txt"),"a")
        file.write(f"{self.postfix}_temp={self.temp}\t\t\t ECE Loss: {eceLoss}\n")


        plot_folder=os.path.join(saveDir,"plots")
        makedirs(plot_folder)
        

        rel_diagram = visualization.ReliabilityDiagramIterative()
        plt_test_2 = rel_diagram.plot(bin_total, bin_total_correct, bin_conf_total,title="Reliability Diagram")
        plt_test_2.savefig(os.path.join(plot_folder,f'rel_diagram_temp={self.temp}.png'),bbox_inches='tight')

    def eval(self):
        self.metric.reset()
        self.model.eval()
        model = self.model


        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()
        # if(not self.useCRF):
        bin_total=[]
        bin_total_correct=[] 
        bin_conf_total=[]
        for (image, target, filename) in tqdm(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            # print(image.shape)
            with torch.no_grad():
                output = model.evaluate(image)
                output /=self.temp 
                output_for_ece=output.clone()



                # if use CRF
                if(self.useCRF):
                    filename = filename[0]
                    raw_image = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.float32).transpose(2, 0, 1)
                    raw_image = torch.from_numpy(raw_image).to(self.device)
                    raw_image = raw_image.unsqueeze(dim=0)
                    crf = GaussCRF(conf=get_default_conf(), shape=image.shape[2:], nclasses=len(self.classes), use_gpu=True)
                    crf = crf.to(self.device)
                    assert image.shape == raw_image.shape
                    output = crf.forward(output, raw_image)
            
            # ECE Stuff
            conf = np.max(output_for_ece.softmax(dim=1).cpu().numpy(),axis=1)
            label=torch.argmax(output_for_ece,dim=1).cpu().numpy()
            # print(conf.shape,label.shape,target.shape)
            bin_total_current, bin_total_correct_current, bin_conf_total_current=self.ece_criterion.get_collective_bins(conf, label, target.cpu().numpy())
            # import pdb; pdb.set_trace()
            bin_total.append(bin_total_current)
            bin_total_correct.append(bin_total_correct_current) 
            bin_conf_total.append(bin_conf_total_current)

            # Accuracy Stuff
            self.metric.update(output, target)
            pixAcc, mIoU = self.metric.get()

        # ECE stuff
        # if(not self.useCRF):
        self.eceOperations(bin_total, bin_total_correct, bin_conf_total)

        # Accuracy stuff
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        # file=open("snow_1_conv9_VOC_final.txt","a")
        # file.write("Temp={} + crf\n".format(self.temp))
        # file.write('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
        #         pixAcc * 100, mIoU * 100))
        
        # file.write("\n\n")
        # file.close()

        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                pixAcc * 100, mIoU * 100))


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    
    # temperatures=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1.0, 1.05, 1.1, 1.15, 1.20, 1.25, 1.3, 1.35, 1.40, 1.45, 1.50,
    #  1.55, 1.6, 1.65, 1.70, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5]

    # for temperature in temperatures:
    #     evaluator.temp=temperature
    #     evaluator.eval()

    evaluator.eval()