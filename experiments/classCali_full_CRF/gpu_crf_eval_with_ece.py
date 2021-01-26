from __future__ import print_function
from ast import dump

from numpy.lib.npyio import save
from calibration_library.metrics import ECELoss, CCELoss
import io
from logging import log
from experiments.classCali_full_CRF import convcrf
from experiments.classCali_full_CRF.convcrf import GaussCRF, get_default_conf

import os
import sys
import pickle

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
from segmentron.utils.visualize import get_color_pallete
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

import matplotlib.pyplot as plt





class Evaluator(object):
    def __init__(self, args):

        self.args = args
        self.device = torch.device(args.device)

        self.n_bins = 15
        self.ece_folder = "experiments/classCali_full_CRF/eceData"
        # self.postfix = "Conv13_PascalVOC_GPU"
        # self.postfix = "Min_Foggy_1_conv13_PascalVOC_GPU"
        self.postfix = "MINFoggy_1_conv13_PascalVOC_GPU"
        self.temp = 1.7
        # self.useCRF=False
        self.useCRF = True

        self.ece_criterion = metrics.IterativeECELoss()
        self.ece_criterion.make_bins(n_bins=self.n_bins)

        # image transform
        input_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
            ]
        )

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(
            cfg.DATASET.NAME, split="val", mode="testval", transform=input_transform
        )
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(
            val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False
        )
        self.val_loader = data.DataLoader(
            dataset=val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=cfg.DATASET.WORKERS,
            pin_memory=True,
        )

        self.dataset = val_dataset
        self.classes = val_dataset.classes
        self.metric = SegmentationMetric(val_dataset.num_class, args.distributed)

        self.model = get_segmentation_model().to(self.device)

        if (
            hasattr(self.model, "encoder")
            and hasattr(self.model.encoder, "named_modules")
            and cfg.MODEL.BN_EPS_FOR_ENCODER
        ):
            logging.info(
                "set bn custom eps for bn in encoder: {}".format(
                    cfg.MODEL.BN_EPS_FOR_ENCODER
                )
            )
            self.set_batch_norm_attr(
                self.model.encoder.named_modules(), "eps", cfg.MODEL.BN_EPS_FOR_ENCODER
            )

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

        self.model.to(self.device)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    


    def giveComparisionImages_colormaps(self,pre_output, post_output, raw_image, gt_label, classes, outname):
        """
        pre_output-> [1,21,h,w] cuda tensor
        post_output-> [1,21,h,w] cuda tensor
        raw_image->[1,3,h,w] cuda tensor
        gt_label->[1,h,w] cuda tensor
        """
        metric = SegmentationMetric(nclass=21, distributed=False)
        metric.update(pre_output, gt_label)
        pre_pixAcc, pre_mIoU = metric.get()
        
        
        metric = SegmentationMetric(nclass=21, distributed=False)
        metric.update(post_output, gt_label)
        post_pixAcc, post_mIoU = metric.get()



        uncal_labels = np.unique(torch.argmax(pre_output.squeeze(0), dim=0).cpu().numpy())
        cal_labels = np.unique(torch.argmax(post_output.squeeze(0), dim=0).cpu().numpy())
        
        pre_label_map = torch.argmax(pre_output.squeeze(0), dim=0).cpu().numpy()
        post_label_map = torch.argmax(post_output.squeeze(0), dim=0).cpu().numpy()

        # Bringing the shapes to justice
        pre_output = pre_output.squeeze(0).cpu().numpy()
        post_output = post_output.squeeze(0).cpu().numpy()
        raw_image = raw_image.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        gt_label = gt_label.squeeze(0).cpu().numpy()


        if False:
            pass
        else:
            # Show result for each class
            cols = int(np.ceil((max(len(uncal_labels), len(cal_labels)) + 1))) + 1
            rows = 4

            plt.figure(figsize=(20, 20))
            # Plotting raw image
            ax = plt.subplot(rows, cols, 1)
            ax.set_title("Input image")
            ax.imshow(raw_image[:, :, ::-1])
            ax.axis("off")

            # Plottig GT
            ax = plt.subplot(rows, cols, cols + 1)
            ax.set_title("Difference MAP ")
            mask1 = get_color_pallete(pre_label_map, cfg.DATASET.NAME)
            mask2 = get_color_pallete(post_label_map, cfg.DATASET.NAME)
            # print(raw_image[:, :, ::-1].shape)
            ax.imshow(((pre_label_map!=post_label_map).astype(np.uint8)))
            ax.axis("off")

            # Plottig GT
            ax = plt.subplot(rows, cols, 2 * cols + 1)
            ax.set_title("ColorMap (uncal+crf) pixA={:.4f} mIoU={:.4f}".format(pre_pixAcc, pre_mIoU))

            mask = get_color_pallete(pre_label_map, cfg.DATASET.NAME)
            ax.imshow(np.array(mask))
            ax.axis("off")

            # Plottig GT
            ax = plt.subplot(rows, cols, 3 * cols + 1)
            # metric = SegmentationMetric(nclass=21, distributed=False)
            # metric.update(pre_output, gt_label)
            # pixAcc, mIoU = metric.get()
            ax.set_title("ColorMap (cal T = {} + CRF) pixA={:.4f} mIoU={:.4f}".format(self.temp,post_pixAcc, post_mIoU))
            mask = get_color_pallete(post_label_map, cfg.DATASET.NAME)
            ax.imshow(np.array(mask))
            ax.axis("off")

            for i, label in enumerate(uncal_labels):
                ax = plt.subplot(rows, cols, i + 3)
                ax.set_title("Uncalibrated-" + classes[label])
                ax.imshow(pre_output[label], cmap="nipy_spectral")

                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, cols + i + 3)
                ax.set_title("Calibrated-" + classes[label])
                ax.imshow(post_output[label], cmap="nipy_spectral")
                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 2 * cols + i + 3)

                min_dif = np.min(pre_output[label] - post_output[label])
                max_dif = np.max(pre_output[label] - post_output[label])

                dif_map = np.where(
                    (pre_output[label] - post_output[label]) > 0,
                    (pre_output[label] - post_output[label]),
                    0,
                )

                ax.set_title("decrease: " + classes[label] + " max={:0.3f}".format(max_dif))
                ax.imshow(
                    dif_map / max_dif,
                    cmap="nipy_spectral",
                )
                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 3 * cols + i + 3)

                min_dif = np.min(pre_output[label] - post_output[label])
                max_dif = np.max(pre_output[label] - post_output[label])

                dif_map = np.where(
                    (pre_output[label] - post_output[label]) < 0,
                    (pre_output[label] - post_output[label]),
                    0,
                )

                ax.set_title(
                    "increase: " + classes[label] + " max={:0.3f}".format(-min_dif)
                )
                ax.imshow(
                    dif_map / min_dif,
                    cmap="nipy_spectral",
                )
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(outname)


    def giveComparisionImages_after_crf(self,
        pre_output, post_output, raw_image, gt_label, classes, outname
    ):
        """
        pre_output-> [1,21,h,w] cuda tensor
        post_output-> [1,21,h,w] cuda tensor
        raw_image->[1,3,h,w] cuda tensor
        gt_label->[1,h,w] cuda tensor
        """
        uncal_labels = np.unique(torch.argmax(pre_output.squeeze(0), dim=0).cpu().numpy())
        cal_labels = np.unique(torch.argmax(post_output.squeeze(0), dim=0).cpu().numpy())

        # Bringing the shapes to justice
        pre_output = pre_output.squeeze(0).cpu().numpy()
        post_output = post_output.squeeze(0).cpu().numpy()
        raw_image = raw_image.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        gt_label = gt_label.squeeze(0).cpu().numpy()

        # import pdb; pdb.set_trace()

        # gt_label=get_gt_with_id(imageName)
        # if(np.sum((cal_labelmap!=uncal_labelmap).astype(np.float32))==0):
        if False:
            pass
        else:
            # Show result for each class
            cols = int(np.ceil((max(len(uncal_labels), len(cal_labels)) + 1))) + 1
            rows = 4

            plt.figure(figsize=(20, 20))
            ax = plt.subplot(rows, cols, 1)
            ax.set_title("Input image")
            ax.imshow(raw_image[:, :, ::-1])
            ax.axis("off")
            ax = plt.subplot(rows, cols, cols + 1)
            # @ neelabh remove this
            loss = 1.999999999999999
            ax.set_title("Difference Map")
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
                ax.set_title("Uncalibrated + crf-" + classes[label])
                ax.imshow(pre_output[label], cmap="nipy_spectral")

                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, cols + i + 3)
                ax.set_title("Calibrated (T={}) + CRF ".format(self.temp) + classes[label])
                ax.imshow(post_output[label], cmap="nipy_spectral")
                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 2 * cols + i + 3)

                min_dif = np.min(pre_output[label] - post_output[label])
                max_dif = np.max(pre_output[label] - post_output[label])

                dif_map = np.where(
                    (pre_output[label] - post_output[label]) > 0,
                    (pre_output[label] - post_output[label]),
                    0,
                )

                ax.set_title("decrease: " + classes[label] + " max={:0.3f}".format(max_dif))
                ax.imshow(
                    dif_map / max_dif,
                    cmap="nipy_spectral",
                )
                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 3 * cols + i + 3)

                min_dif = np.min(pre_output[label] - post_output[label])
                max_dif = np.max(pre_output[label] - post_output[label])

                dif_map = np.where(
                    (pre_output[label] - post_output[label]) < 0,
                    (pre_output[label] - post_output[label]),
                    0,
                )

                ax.set_title(
                    "increase: " + classes[label] + " max={:0.3f}".format(-min_dif)
                )
                ax.imshow(
                    dif_map / min_dif,
                    cmap="nipy_spectral",
                )
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(outname)


    def giveComparisionImages_before_crf(self,
        pre_output, post_output, raw_image, gt_label, classes, outname
    ):
        """
        pre_output-> [1,21,h,w] cuda tensor
        post_output-> [1,21,h,w] cuda tensor
        raw_image->[1,3,h,w] cuda tensor
        gt_label->[1,h,w] cuda tensor
        """
        uncal_labels = np.unique(torch.argmax(pre_output.squeeze(0), dim=0).cpu().numpy())
        cal_labels = np.unique(torch.argmax(post_output.squeeze(0), dim=0).cpu().numpy())

        # Bringing the shapes to justice
        pre_output = pre_output.squeeze(0).cpu().numpy()
        post_output = post_output.squeeze(0).cpu().numpy()
        raw_image = raw_image.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        gt_label = gt_label.squeeze(0).cpu().numpy()

        # import pdb; pdb.set_trace()

        # gt_label=get_gt_with_id(imageName)
        # if(np.sum((cal_labelmap!=uncal_labelmap).astype(np.float32))==0):
        if False:
            pass
        else:
            # Show result for each class
            cols = int(np.ceil((max(len(uncal_labels), len(cal_labels)) + 1))) + 1
            rows = 4

            plt.figure(figsize=(20, 20))
            ax = plt.subplot(rows, cols, 1)
            ax.set_title("Input image")
            ax.imshow(raw_image[:, :, ::-1])
            ax.axis("off")
            ax = plt.subplot(rows, cols, cols + 1)
            # @ neelabh remove this
            loss = 1.999999999999999
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
                ax.imshow(pre_output[label], cmap="nipy_spectral")

                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, cols + i + 3)
                ax.set_title("Calibrated (T = {}) ".format(self.temp) + classes[label])
                ax.imshow(post_output[label], cmap="nipy_spectral")
                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 2 * cols + i + 3)

                min_dif = np.min(pre_output[label] - post_output[label])
                max_dif = np.max(pre_output[label] - post_output[label])

                dif_map = np.where(
                    (pre_output[label] - post_output[label]) > 0,
                    (pre_output[label] - post_output[label]),
                    0,
                )

                ax.set_title("decrease: " + classes[label] + " max={:0.3f}".format(max_dif))
                ax.imshow(
                    dif_map / max_dif,
                    cmap="nipy_spectral",
                )
                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 3 * cols + i + 3)

                min_dif = np.min(pre_output[label] - post_output[label])
                max_dif = np.max(pre_output[label] - post_output[label])

                dif_map = np.where(
                    (pre_output[label] - post_output[label]) < 0,
                    (pre_output[label] - post_output[label]),
                    0,
                )

                ax.set_title(
                    "increase: " + classes[label] + " max={:0.3f}".format(-min_dif)
                )
                ax.imshow(
                    dif_map / min_dif,
                    cmap="nipy_spectral",
                )
                ax.axis("off")

            plt.tight_layout()
            plt.savefig(outname)


    def eceOperations(
        self, endNAme, bin_total, bin_total_correct, bin_conf_total, temp=None
    ):
        eceLoss = self.ece_criterion.get_interative_loss(
            bin_total, bin_total_correct, bin_conf_total
        )
        # print('ECE with probabilties %f' % (eceLoss))
        if temp == None:
            temp = self.temp

        saveDir = os.path.join(self.ece_folder, self.postfix + f"_temp={temp}")
        makedirs(saveDir)

        file = open(os.path.join(saveDir, "Results.txt"), "a")
        file.write(f"{endNAme.strip('.npy')}_temp={temp}\t\t\t ECE Loss: {eceLoss}\n")

        plot_folder = os.path.join(saveDir, "plots")
        makedirs(plot_folder)

        rel_diagram = visualization.ReliabilityDiagramIterative()
        plt_test_2 = rel_diagram.plot(
            bin_total, bin_total_correct, bin_conf_total, title="Reliability Diagram"
        )
        plt_test_2.savefig(
            os.path.join(plot_folder, f'{endNAme.strip(".npy")}_temp={temp}.png'),
            bbox_inches="tight",
        )
        plt_test_2.close()
        return eceLoss

    def give_ece_order(self, model):
        """
        Performs evaluation over the entire daatset
        Returns a array of [imageName, eceLoss] in sorted order (descending
        """

        eceLosses = []
        for (image, target, filename) in tqdm(self.val_loader):
            bin_total = []
            bin_total_correct = []
            bin_conf_total = []
            image = image.to(self.device)
            target = target.to(self.device)

            filename = filename[0]
            # print(filename)
            endName = os.path.basename(filename).replace(".jpg", ".npy")
            # print(endName)

            npy_target_directory = "datasets/VOC_targets"
            npy_file = os.path.join(npy_target_directory, endName)
            if os.path.isfile(npy_file):
                pass
            else:
                makedirs(npy_target_directory)
                np.save(npy_file, target.cpu().numpy())
                # print("Npy files not found | Going for onboard eval")

            # print(image.shape)
            with torch.no_grad():

                # Checking if npy preprocesssed exists or not

                # print(filename)
                # npy_output_directory = "npy_outputs/npy_VOC_outputs"
                npy_output_directory = "npy_outputs/npy_foggy1_VOC_outputs"
                npy_file = os.path.join(npy_output_directory, endName)

                # print (npy_file)
                if os.path.isfile(npy_file):
                    output = np.load(npy_file)
                    output = torch.Tensor(output).cuda()
                    # print("Reading Numpy Files")
                else:
                    # print("Npy files not found | Going for onboard eval")
                    makedirs(npy_output_directory)
                    output = model.evaluate(image)
                    np.save(npy_file, output.cpu().numpy())

                output_before_cali = output.clone()

                # ECE Stuff
                conf = np.max(output_before_cali.softmax(dim=1).cpu().numpy(), axis=1)
                label = torch.argmax(output_before_cali, dim=1).cpu().numpy()
                # print(conf.shape,label.shape,target.shape)
                (
                    bin_total_current,
                    bin_total_correct_current,
                    bin_conf_total_current,
                ) = self.ece_criterion.get_collective_bins(
                    conf, label, target.cpu().numpy()
                )
                # import pdb; pdb.set_trace()
                bin_total.append(bin_total_current)
                bin_total_correct.append(bin_total_correct_current)
                bin_conf_total.append(bin_conf_total_current)

                # ECE stuff
                # if(not self.useCRF):
                eceLosses.append(
                    [
                        endName,
                        filename,
                        self.eceOperations(
                            endName,
                            bin_total,
                            bin_total_correct,
                            bin_conf_total,
                            temp=1,
                        ),
                    ]
                )

        eceLosses.sort(key=lambda x: x[2], reverse=True)

        return eceLosses

    def eval(self):
        self.metric.reset()
        self.model.eval()
        model = self.model

        logging.info(
            "Start validation, Total sample: {:d}".format(len(self.val_loader))
        )
        import time

        time_start = time.time()
        # if(not self.useCRF):

        # first loop for finding ece errors
        if os.path.isfile("experiments/classCali_full_CRF/sorted_ecefoggy.pickle"):
            file = open("experiments/classCali_full_CRF/sorted_ecefoggy.pickle", "rb")
        # if os.path.isfile("experiments/classCali_full_CRF/sorted_ece.pickle"):
        #     file = open("experiments/classCali_full_CRF/sorted_ece.pickle", "rb")
            eceLosses = pickle.load(file)
            file.close()
        else:
            assert False
            eceLosses = self.give_ece_order(model)
            pickle.dump(eceLosses, open("experiments/classCali_full_CRF/sorted_ece.pickle", "wb"))

        print("ECE sorting completed....")

        top_k = 10

        assert top_k > 0
        eceLosses.reverse()
        # for i, (endName, imageLoc, eceLoss) in enumerate(tqdm(eceLosses[2:3])):
        for i, (endName, imageLoc, eceLoss) in enumerate(tqdm(eceLosses[:top_k])):
            # Loading outputs
            print(endName)
            # npy_output_directory = "npy_outputs/npy_VOC_outputs"
            npy_output_directory = "npy_outputs/npy_foggy1_VOC_outputs"
            npy_file = os.path.join(npy_output_directory, endName)
            output = np.load(npy_file)
            output = torch.Tensor(output).cuda()

            # loading targets
            npy_target_directory = "datasets/VOC_targets"
            npy_file = os.path.join(npy_target_directory, endName)
            target = np.load(npy_file)
            target = torch.Tensor(target).cuda()

            # print(image.shape)
            with torch.no_grad():

                output_uncal = output.clone()
                output_cal = output / self.temp
                
                # ECE Stuff
                bin_total = []
                bin_total_correct = []
                bin_conf_total = []
                conf = np.max(output_uncal.softmax(dim=1).cpu().numpy(), axis=1)
                label = torch.argmax(output_uncal, dim=1).cpu().numpy()
                # print(conf.shape,label.shape,target.shape)
                (
                    bin_total_current,
                    bin_total_correct_current,
                    bin_conf_total_current,
                ) = self.ece_criterion.get_collective_bins(
                    conf, label, target.cpu().numpy()
                )
                # import pdb; pdb.set_trace()
                bin_total.append(bin_total_current)
                bin_total_correct.append(bin_total_correct_current)
                bin_conf_total.append(bin_conf_total_current)

                # ECE stuff
                # if(not self.useCRF):
                self.eceOperations(
                            endName,
                            bin_total,
                            bin_total_correct,
                            bin_conf_total,
                            temp=1
                        )
                
                # ECE Stuff
                bin_total = []
                bin_total_correct = []
                bin_conf_total = []
                conf = np.max(output_cal.softmax(dim=1).cpu().numpy(), axis=1)
                label = torch.argmax(output_cal, dim=1).cpu().numpy()
                # print(conf.shape,label.shape,target.shape)
                (
                    bin_total_current,
                    bin_total_correct_current,
                    bin_conf_total_current,
                ) = self.ece_criterion.get_collective_bins(
                    conf, label, target.cpu().numpy()
                )
                # import pdb; pdb.set_trace()
                bin_total.append(bin_total_current)
                bin_total_correct.append(bin_total_correct_current)
                bin_conf_total.append(bin_conf_total_current)

                # ECE stuff
                # if(not self.useCRF):
                self.eceOperations(
                            endName,
                            bin_total,
                            bin_total_correct,
                            bin_conf_total,
                        )
                
                

                # REad raw image

                raw_image = (
                    cv2.imread(imageLoc, cv2.IMREAD_COLOR)
                    .astype(np.float32)
                    .transpose(2, 0, 1)
                )
                raw_image = torch.from_numpy(raw_image).to(self.device)
                raw_image = raw_image.unsqueeze(dim=0)

                # Setting up CRF
                crf = GaussCRF(
                    conf=get_default_conf(),
                    shape=output.shape[2:],
                    nclasses=len(self.classes),
                    use_gpu=True,
                )
                crf = crf.to(self.device)

                # Getting CRF outputs
                # print(output.shape, raw_image.shape)
                assert output.shape[2:] == raw_image.shape[2:]
                # import pdb; pdb.set_trace()
                # print(":here1:")
                output_cal_crf = crf.forward(output_cal, raw_image)
                # print(":here2:")
                output_uncal_crf = crf.forward(output_uncal, raw_image)

                # Comparision before CRF bw cali and uncali
                comparisionFolder = "experiments/classCali_full_CRF/comparisionImages"
                saveFolder = os.path.join(
                    comparisionFolder, "bcrf" + self.postfix + f"_temp={self.temp}"
                )
                makedirs(saveFolder)
                saveName = os.path.join(saveFolder, os.path.basename(imageLoc))
                self.giveComparisionImages_before_crf(
                    output_uncal.softmax(dim=1),
                    output_cal.softmax(dim=1),
                    raw_image,
                    target,
                    self.classes,
                    saveName,
                )

                # Comparision before CRF bw cali and uncali
                comparisionFolder = "experiments/classCali_full_CRF/comparisionImages"
                saveFolder = os.path.join(
                    comparisionFolder, "crf" + self.postfix + f"_temp={self.temp}"
                )
                makedirs(saveFolder)
                saveName = os.path.join(saveFolder, os.path.basename(imageLoc))
                self.giveComparisionImages_after_crf(
                    output_uncal_crf.softmax(dim=1),
                    output_cal_crf.softmax(dim=1),
                    raw_image,
                    target,
                    self.classes,
                    saveName,
                )

                # Comparision uncali vs  CRF after cali
                comparisionFolder = "experiments/classCali_full_CRF/comparisionImages"
                saveFolder = os.path.join(
                    comparisionFolder, "cmap_" + self.postfix + f"_temp={self.temp}"
                )
                makedirs(saveFolder)
                saveName = os.path.join(saveFolder, os.path.basename(imageLoc))
                self.giveComparisionImages_colormaps(
                    output_uncal_crf.softmax(dim=1),
                    output_cal_crf.softmax(dim=1),
                    raw_image,
                    target,
                    self.classes,
                    saveName,
                )
            # # Accuracy Stuff
            # self.metric.update(output, target)
            # pixAcc, mIoU = self.metric.get()


if __name__ == "__main__":
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = "test"
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