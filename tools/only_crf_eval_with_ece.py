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

from calibration_library import metrics, visualization

from crfasrnn.crfrnn import CrfRnn
from PIL import Image

import os
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

from crf import DenseCRF
from new_crf import do_crf

def process(i, dataset, postprocessor,ece_criterion, temp, useCRF, num_classes, device):
    image, gt_label, filename = dataset.__getitem__(i)

    # print(filename)

    logit = np.load('./npy_files_voc/' + os.path.basename(filename).strip('.jpg') + '.npy')

    _, H, W = image.shape
    logit = torch.FloatTensor(logit)
    
    logit = F.interpolate(logit.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=True)

    logit = logit / temp
    prob = F.softmax(logit, dim=1)

    if useCRF:
        raw_image = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.float32).transpose(2, 0, 1)
        raw_image = torch.from_numpy(raw_image).unsqueeze(dim=0)
        # prob = postprocessor(raw_image, prob)
        prob = postprocessor(raw_image, logit)
        prob=prob.detach()
    
    
    conf = np.max(prob.numpy(), axis=1)
    # label = np.argmax(prob.numpy(), axis=0)

    prob = prob.to(device)
    label = torch.argmax(prob, dim=1)

    conf = torch.tensor(conf).to(device).unsqueeze(dim=0)
    gt_label = gt_label.to(device).unsqueeze(dim=0)

    #possible error
    # label = torch.tensor(label).to(device).unsqueeze(dim=0)
    
    # ECE stuff reshape @neelabh
    conf_for_ece = conf.squeeze(dim=0)
    gt_label_for_ece = gt_label
    label_for_ece = label
    print(prob.shape,conf_for_ece.shape,label_for_ece.shape,gt_label_for_ece.shape)
    bin_total, bin_total_correct, bin_conf_total=ece_criterion.get_collective_bins(conf_for_ece.cpu().numpy(), label_for_ece.cpu().numpy(), gt_label_for_ece.cpu().numpy())


    # accuracy stuff
    # print(label.shape,gt_label.shape,num_classes)
    area_inter, area_union = batch_intersection_union(label, gt_label, num_classes)

    area_inter = area_inter.cpu().numpy()
    area_union = area_union.cpu().numpy()

    correct, labeled = batch_pix_accuracy(label, gt_label)
    correct = correct.item()
    labeled = labeled.item()

    return area_inter, area_union, correct, labeled, bin_total, bin_total_correct, bin_conf_total


class Evaluator(object):
    def __init__(self, args):
        # self.postprocessor= DenseCRF(iter_max=cfg.CRF.ITER_MAX,
        #                                 pos_xy_std=cfg.CRF.POS_XY_STD,
        #                                 pos_w=cfg.CRF.POS_W,
        #                                 bi_xy_std=cfg.CRF.BI_XY_STD,
        #                                 bi_rgb_std=cfg.CRF.BI_RGB_STD,
        #                                 bi_w=cfg.CRF.BI_W,
        #                             )
        
        # self.postprocessor = do_crf

        self.args = args
        self.device = torch.device(args.device)

        self.n_bins=15
        self.ece_folder="eceData"
        self.postfix="Snow_VOC_1"
        self.temp=1.7
        self.useCRF=False
        # self.useCRF=True

        self.ece_criterion= metrics.IterativeECELoss()
        self.ece_criterion.make_bins(n_bins=self.n_bins)



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

        # self.postprocessor = CrfRnn(len(self.classes))
        # self.postprocessor.to(self.device)

    def eceOperations(self,bin_total, bin_total_correct, bin_conf_total):
        eceLoss=self.ece_criterion.get_interative_loss(bin_total, bin_total_correct, bin_conf_total)
        print('ECE with probabilties %f' % (eceLoss))
        
        saveDir=os.path.join(self.ece_folder,self.postfix)
        makedirs(saveDir)

        file=open(os.path.join(saveDir,"Results.txt"),"a")
        file.write(f"{self.postfix}_temp={self.temp}\t\t\t ECE Loss: {eceLoss}\n")


        plot_folder=os.path.join(saveDir,"plots")
        makedirs(plot_folder)
        # conf_hist = visualization.ConfidenceHistogram()
        # plt_test = conf_hist.plot(conf,obj,gt,title="Confidence Histogram")
        # plt_test.savefig(os.path.join(plot_folder,f'conf_histogram_bin={n_bins}_incBG={str(include_bg)}.png'),bbox_inches='tight')
        #plt_test.show()

        rel_diagram = visualization.ReliabilityDiagramIterative()
        plt_test_2 = rel_diagram.plot(bin_total, bin_total_correct, bin_conf_total,title="Reliability Diagram")
        plt_test_2.savefig(os.path.join(plot_folder,f'rel_diagram_temp={self.temp}.png'),bbox_inches='tight')
        #plt_test_2.show()


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
            [joblib.delayed(process)(i,self.dataset,self.postprocessor,self.ece_criterion,self.temp,self.useCRF, len(self.classes), self.device) for i in range(len(self.dataset))]
        )

        # ans = process(0, self.dataset,self.postprocessor, len(self.classes), self.device)

        area_inter, area_union, correct, labeled, bin_total, bin_total_correct, bin_conf_total = zip(*results)

        # ECE stuff
        if(not self.useCRF):
            self.eceOperations(bin_total, bin_total_correct, bin_conf_total)


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


