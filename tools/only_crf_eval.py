from __future__ import print_function
import io

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
from segmentron.utils.score import batch_intersection_union, batch_pix_accuracy

import os
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

from crf import DenseCRF
from new_crf import do_crf

def process(i, dataset, postprocessor, num_classes, device, useCRF = True):
    image, gt_label, filename = dataset.__getitem__(i)

    # print(filename)

    logit = np.load('./npy_files_voc/' + os.path.basename(filename).strip('.jpg') + '.npy')

    _, H, W = image.shape
    logit = torch.FloatTensor(logit)
    
    logit = F.interpolate(logit.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=True)

    logit = logit / 2.77
    prob = F.softmax(logit, dim=1)[0].numpy()

    if useCRF:
        raw_image = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.uint8)
        prob = postprocessor(raw_image, prob)

    conf = np.max(prob, axis=0)
    label = np.argmax(prob, axis=0)

    conf = torch.tensor(conf).to(device).unsqueeze(dim=0)
    gt_label = gt_label.to(device).unsqueeze(dim=0)
    label = torch.tensor(label).to(device).unsqueeze(dim=0)

    # accuracy stuff
    area_inter, area_union = batch_intersection_union(label, gt_label, num_classes)

    area_inter = area_inter.cpu().numpy()
    area_union = area_union.cpu().numpy()

    correct, labeled = batch_pix_accuracy(label, gt_label)
    correct = correct.item()
    labeled = labeled.item()

    return area_inter, area_union, correct, labeled


class Evaluator(object):
    def __init__(self, args):
        self.postprocessor= DenseCRF(iter_max=cfg.CRF.ITER_MAX,
                                        pos_xy_std=cfg.CRF.POS_XY_STD,
                                        pos_w=cfg.CRF.POS_W,
                                        bi_xy_std=cfg.CRF.BI_XY_STD,
                                        bi_rgb_std=cfg.CRF.BI_RGB_STD,
                                        bi_w=cfg.CRF.BI_W,
                                    )
        
        # self.postprocessor = do_crf

        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        
        # dataset and dataloader
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode='testval', transform=input_transform)

        self.dataset = val_dataset
        self.classes = val_dataset.classes
        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def eval(self):
        self.metric.reset()

        logging.info("Start validation, Total sample: {:d}".format(len(self.dataset)))

        import time
        time_start = time.time()

        # CRF in multi-process
        results = joblib.Parallel(n_jobs=8, verbose=10)(
            [joblib.delayed(process)(i, self.dataset,self.postprocessor, len(self.classes), self.device) for i in range(len(self.dataset))]
        )

        area_inter, area_union, correct, labeled = zip(*results)

        # accuracy stuff
        total_correct = sum(correct)
        total_label = sum(labeled)
        
        area_inter = np.array(area_inter)
        area_union = np.array(area_union)

        total_inter = np.sum(area_inter, axis=0)
        total_union = np.sum(area_union, axis=0)

        pixAcc = 1.0 * total_correct / (2.2e-16 + total_label) 
        IoU = 1.0 * total_inter / (2.2e-16 + total_union)
        mIoU = np.mean(IoU)

        return pixAcc, mIoU


if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    pix, iou = evaluator.eval()

    print(pix, iou)
    print('Pixel Accuracy : ', pix*100)
    print('Mean IoU : ', iou*100)

    # import numpy as np

    # # cfg.CRF.BI_XY_STD=alpha
    # # cfg.CRF.BI_RGB_STD=beta
    # # cfg.CRF.BI_W=w

    # alphas=list(map(int,(np.linspace(1, 200, 41)).astype(np.int32)))
    # alphas.append(141)
    # alphas.append(121)
    # alphas.sort()

    # betas=list(map(int,(np.linspace(1, 50, 11)).astype(np.int32)))
    # betas.append(11)
    # betas.append(13)
    
    # #betas=[1,2,3,4,5,6,7,8,9,10]
    # betas.sort()

    # ws=[4,5,10]
    # total=[]
    # for w in ws:
    #     for beta in betas:
    #         for alpha in alphas:
    #             cfg.CRF.BI_XY_STD=alpha
    #             cfg.CRF.BI_RGB_STD=beta
    #             cfg.CRF.BI_W=w
    #             evaluator = Evaluator(args)
    #             pix, iou = evaluator.eval()
    #             print([w,alpha,beta],[pix,iou])
    #             print("-------------------------------\n")
    #             file=open("grid.txt","a")
    #             file.write(str([[w,alpha,beta],[pix,iou]]))
    #             file.write("\n")
    #             file.close()
    #             total.append([[w,alpha,beta],[pix,iou]])

    # import pickle
    # f=open("grid.pickle","wb")
    # pickle.dump(total,f)
    # f.close()


