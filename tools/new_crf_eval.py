from __future__ import print_function

import os
import sys

from numpy.lib.npyio import save


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
import multiprocessing
import joblib

from tabulate import tabulate
from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.score import batch_intersection_union, batch_pix_accuracy

from calibration_library import metrics, visualization

import os
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

from crf import DenseCRF

def process(i, dataset, postprocessor,ece_criterion, temp, useCRF, num_classes, device):
    image, gt_label, filename = dataset.__getitem__(i)

    # print(filename)

    logit = np.load('./npy_files_foggy_dbf/' + os.path.basename(filename).strip('.png') + '.npy')

    _, H, W = image.shape
    logit = torch.FloatTensor(logit)
    
    logit = F.interpolate(logit.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=True)

    logit=logit / temp
    prob = F.softmax(logit, dim=1)[0].numpy()

    if(useCRF):
        raw_image = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.uint8)
        prob = postprocessor(raw_image, prob)
    conf = np.max(prob, axis=0)
    label = np.argmax(prob, axis=0)

    conf = torch.tensor(conf).to(device).unsqueeze(dim=0)
    gt_label = gt_label.to(device).unsqueeze(dim=0)
    label = torch.tensor(label).to(device).unsqueeze(dim=0)
    # ECE stuff reshape @neelabh

    # print(prob.shape,conf.shape,label.shape,gt_label.shape)
    bin_total, bin_total_correct, bin_conf_total=ece_criterion.get_collective_bins(conf.cpu().numpy(), label.cpu().numpy(), gt_label.cpu().numpy())

    # accuracy stuff
    area_inter, area_union = batch_intersection_union(label, gt_label, num_classes)

    area_inter = area_inter.cpu().numpy()
    area_union = area_union.cpu().numpy()

    correct, labeled = batch_pix_accuracy(label, gt_label)
    correct = correct.item()
    labeled = labeled.item()

    return area_inter, area_union, correct, labeled, bin_total, bin_total_correct, bin_conf_total



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

        self.n_bins=15
        self.ece_folder="eceData"
        self.postfix="Foggy_DBF_low_DLV3Plus"
        self.temp=2.3
        self.useCRF=False

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

        # made
        # val_sampler = make_data_sampler(val_dataset, shuffle=False, distributed=args.distributed)
        # val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)
        # self.val_loader = data.DataLoader(dataset=val_dataset,
        #                                   batch_sampler=val_batch_sampler,
        #                                   num_workers=cfg.DATASET.WORKERS,
        #                                   pin_memory=True)
        self.classes = val_dataset.classes
        # create network
        # self.model = get_segmentation_model().to(self.device)

        # if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
        #     cfg.MODEL.BN_EPS_FOR_ENCODER:
        #     logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
        #     self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        # if args.distributed:
        #     self.model = nn.parallel.DistributedDataParallel(self.model,
        #         device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        # self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)

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

        pixAcc = 1.0 * total_correct / (2.220446049250313e-16 + total_label)  # remove np.spacing(1)
        IoU = 1.0 * total_inter / (2.220446049250313e-16 + total_union)
        mIoU = np.mean(IoU)

        logging.info('Eval use time: {:.3f} second'.format(time.time() - time_start))
        logging.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
                pixAcc * 100, mIoU * 100))

        # headers = ['class id', 'class name', 'iou']
        # table = []
        # for i, cls_name in enumerate(self.classes):
        #     table.append([cls_name, category_iou[i]])
        # logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
        #                                                    numalign='center', stralign='center')))
        
        # pixAcc, mIoU = self.metric.get()
        # logging.info("All Samples: validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
        #     pixAcc * 100, mIoU * 100))


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
