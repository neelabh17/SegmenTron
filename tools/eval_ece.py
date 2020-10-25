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

def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

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
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
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

        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        import time
        time_start = time.time()

        tot_conf=torch.Tensor([]).reshape(-1,1)
        tot_obj=torch.Tensor([]).reshape(-1,1)
        tot_label_for_image=torch.Tensor([]).reshape(-1,1)

        for i, (image, target, filename) in enumerate(self.val_loader):
            
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output = model.evaluate(image)
            # import pdb; pdb.set_trace()

            doingCali=True
            usingCRF=True
            if(doingCali):
                # predetermined
                # temp=1.6127
                temp=2.8
            else:
                temp=1

            output=output/temp
            # to be removed for temp scaling
            if(not usingCRF):
                output_post=output
            else:
                output_post=[]

            output=F.softmax(output,dim=1)
            output_numpy=output.cpu().numpy()
            
            def get_raw_image(file_location):
                # load in bgr in H W C format
                raw_image = cv2.imread(file_location, cv2.IMREAD_COLOR).astype(np.float32)
                mean_bgr=np.array([103.53, 116.28, 123.675])
                # Do some subtraction
                raw_image-=mean_bgr
                # converted to C H W
                raw_image=raw_image.transpose(2,0,1)
                raw_image=raw_image.astype(np.uint8)
                raw_image=raw_image.transpose(1,2,0)
                return raw_image


            for j,image_file_loc in enumerate(filename):

                prob_to_use=output_numpy[j]
                if(usingCRF):
                    raw_image=get_raw_image(image_file_loc)
                    prob_post=self.postprocessor(raw_image,prob_to_use)
                    prob_to_use=prob_post
                    output_post.append(prob_post)

                # import pdb;pdb.set_trace()
                prob_to_use=torch.tensor(prob_to_use)
                # Neels implementation
                labels=torch.argmax(prob_to_use,dim=0)
                conf= torch.max(prob_to_use,dim=0)[0].cpu()
                obj= labels.cpu().float()
                label_for_image = target[j].view(-1,1).cpu().float()
                sel=(label_for_image>=0)

                tot_conf=torch.cat([tot_conf,conf.view(-1,1)[sel].view(-1,1)],dim=0)
                tot_obj=torch.cat([tot_obj,obj.view(-1,1)[sel].view(-1,1)],dim=0)
                tot_label_for_image=torch.cat([tot_label_for_image,label_for_image.view(-1,1)[sel].view(-1,1)],dim=0)
            
            if(usingCRF):
                output_post=np.array(output_post)
                output_post=torch.tensor(output_post)
                output_post=output_post.to(self.device)
                
            self.metric.update(output_post, target)
            # self.metric.update(output, target)
            pixAcc, mIoU = self.metric.get()
            logging.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))
            

        print(tot_conf.shape,tot_obj.shape,tot_label_for_image.shape)
        import pickle
        ece_folder="eceData"
        makedirs(ece_folder)

        # postfix="DLV2_UnCal"
        postfix="Foggy_Calibrated_DLV3Plus"
        saveDir=os.path.join(ece_folder,postfix)
        makedirs(saveDir)
        
        file=open(os.path.join(saveDir,"conf.pickle"),"wb")
        pickle.dump(tot_conf,file)
        file.close()
        file=open(os.path.join(saveDir,"obj.pickle"),"wb")
        pickle.dump(tot_obj,file)
        file.close()
        file=open(os.path.join(saveDir,"gt.pickle"),"wb")
        pickle.dump(tot_label_for_image,file)
        file.close()

        synchronize()
        pixAcc, mIoU, category_iou = self.metric.get(return_category_iou=True)
        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                pixAcc * 100, mIoU * 100))

        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(self.classes):
            table.append([cls_name, category_iou[i]])
        logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                           numalign='center', stralign='center')))


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
