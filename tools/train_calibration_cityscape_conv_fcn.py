from __future__ import print_function
import io
from logging import log
from torch._C import device, dtype
from torch.utils.tensorboard import SummaryWriter
from calibration_library.metrics import CCELoss as perimageCCE
from calibration_library.info_entropy_loss import InfoEntropyLoss


import os
import sys
import matplotlib.pyplot as plt

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
from transformation_net.fcn import poolNet, FCNs


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
        self.lr = 7.5
        self.prefix = f"2_img_cce_only_lr={self.lr}"
        # self.prefix = f"overfit_with_bin_fraction_loss=no_bin_weights_ALPHA=0.5_lr={self.lr}"
        self.writer = SummaryWriter(log_dir= f"cce_cityscapes_conv_fcn_logs/{self.prefix}")
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

        self.poolnet = poolNet(len(self.classes)).to(self.device)
        self.fcn = FCNs(self.poolnet, len(self.classes)).to(self.device)
        

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
        torch.nn.init.xavier_uniform_(temp_weights, gain=1.0)
        temp_weights.requires_grad= True
        # temp_weights.requires_grad= True
        temp_bias = torch.zeros(len(self.classes), device = "cuda")
        # torch.nn.init.xavier_uniform_(temp_bias, gain=1.0)
        temp_bias.requires_grad= True
        # temp_weights = torch.rand(len(self.classes), len(self.classes), device="cuda", requires_grad=True)
        # temp_bias = torch.rand(len(self.classes), device="cuda", requires_grad=True)

        logging.info("Start training of temprature weights, Total sample: {:d}".format(len(self.val_loader)))

        cce_criterion = CCELoss(len(self.classes)).to(self.device)
        cross_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        info_entropy_criterion = InfoEntropyLoss()
        optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr)

        import time
        time_start = time.time()
        num_epochs = 200
        for epoch in range(num_epochs):
            eceEvaluator_perimage = perimageCCE(n_classes = len(self.classes))
            epoch_loss_cce_total = 0
            epoch_loss_cross_entropy_total = 0
            epoch_loss_total = 0
            epoch_loss_info_entropy = 0

            for i, (images, targets, filenames) in enumerate(self.val_loader):
                # import pdb; pdb.set_trace()
                optimizer.zero_grad()

                images = images.to(self.device)
                targets = targets.to(self.device)



                # print(image.shape)
                with torch.no_grad():
                    outputs = model.evaluate(images)
                

                # print(outputs.shape)
                outputs = self.fcn(outputs)
                # print(outputs.shape)

                # exit()

                
                # Image saving and stuff
                # save_imgs = torch.softmax(outputs, dim =1).squeeze(0)
                save_imgs = torch.softmax(outputs, dim =1)[0]
                for class_no, class_distri in enumerate(save_imgs):
                    plt.clf()
                    class_distri[0][0] = 0
                    class_distri[0][1] = 1

                    im = plt.imshow(class_distri.detach().cpu().numpy(),cmap="Greens")
                    plt.colorbar(im)
                    plt.savefig("temp.jpg")
                    plt.clf()
                    import cv2
                    img_dif = cv2.imread("temp.jpg")

                    self.writer.add_image(f"Class_{self.classes[class_no]}", img_dif, epoch, dataformats="HWC")
                
                loss_cce = cce_criterion.forward(outputs, targets)
                loss_cross_entropy = cross_criterion.forward(outputs, targets)
                loss_info_entropy = info_entropy_criterion.forward(outputs)


                alpha = 1
                # total_loss = loss_cce + alpha * loss_info_entropy
                total_loss = loss_cce

                epoch_loss_info_entropy += loss_info_entropy
                epoch_loss_cce_total += loss_cce.item()
                epoch_loss_cross_entropy_total += loss_cross_entropy.item()
                epoch_loss_total += total_loss.item()

                total_loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    for output, target in zip(outputs,targets.detach()):
                        #older ece requires softmax and size output=[class,w,h] target=[w,h]
                        eceEvaluator_perimage.update(output.softmax(dim=0), target)
                # print(outputs.shape)
                # print(eceEvaluator_perimage.get_overall_CCELoss())
                print(f"batch :{i+1}/{len(self.val_loader)}" + "loss cce : {:.5f} | loss cls : {:.5f} | loss tot : {:.5f}".format(loss_cce, loss_cross_entropy, total_loss))
            
            epoch_loss_cce_total /= len(self.val_loader)
            epoch_loss_cross_entropy_total /= len(self.val_loader)
            epoch_loss_total /= len(self.val_loader)
            epoch_loss_cross_entropy_total /= len(self.val_loader)
            
            count_table_image, _ = eceEvaluator_perimage.get_count_table_img(self.classes)
            cce_table_image, dif_map= eceEvaluator_perimage.get_perc_table_img(self.classes)
            self.writer.add_image("CCE_table", cce_table_image, epoch, dataformats="HWC")
            self.writer.add_image("Count table", count_table_image, epoch, dataformats="HWC")
            self.writer.add_image("DifMap", dif_map, epoch, dataformats="HWC")
            
            self.writer.add_scalar(f"Cross EntropyLoss_LR", epoch_loss_cross_entropy_total, epoch)
            self.writer.add_scalar(f"CCELoss_LR", epoch_loss_cce_total, epoch)
            self.writer.add_scalar(f"Info EntropyLoss_LR", epoch_loss_info_entropy, epoch)

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
