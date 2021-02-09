from __future__ import print_function
import io
from logging import log
from torch._C import device, dtype
from torch.utils.tensorboard import SummaryWriter
from calibration_library.metrics import CCELoss as perimageCCE
from calibration_library.info_entropy_loss import InfoEntropyLoss


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
import matplotlib.pyplot as plt

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

def analyse(outputs):
    print("Analysing output")
    print(outputs[0,0,100:200, 50:150])
    print(outputs[0,0,100:150, 250:300])
    print(outputs[0,1,100:200, 50:150])
    print(outputs[0,1,100:150, 250:300])
    print(outputs[0,2,100:200, 50:150])
    print(outputs[0,2,100:150, 250:300])
    
def accuracy(outputs):
    img = torch.zeros(300, 400)
    img[100:200, 50:150] = 1
    img[100:150, 250:300] = 2

    final_map = torch.argmax(outputs.cpu() , dim = 1).squeeze(0)

    assert final_map.shape == img.shape 

    print("Accuracy:\t",torch.sum(final_map == img))

    print("------------------------")
    # print("Accuracy:\t",torch.sum(final_map == img)/(final_map.shape[0]*final_map.shape[1]))
    # input()

class Evaluator(object):
    def __init__(self, args):

        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        self.lr = 2.5
        self.prefix = f"2_boxes_info_entropy_51_49_alpha=1_lr={self.lr}"
        # self.prefix = f"overfit__count_toy_experiment_3class_7_2_1_conf_loss=total_xavier_weights_xavier_bias_lr={self.lr}"
        self.writer = SummaryWriter(log_dir= f"cce_toy_entropy_logs/{self.prefix}")
        # self.writer = SummaryWriter(log_dir= f"cce_cityscapes_logs/{self.prefix}")
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

        # self.model = get_segmentation_model().to(self.device)

        # if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'named_modules') and \
        #     cfg.MODEL.BN_EPS_FOR_ENCODER:
        #     logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
        #     self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        # if args.distributed:
        #     self.model = nn.parallel.DistributedDataParallel(self.model,
        #         device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        # self.model.to(self.device)

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def eval(self):
        self.metric.reset()
        print(f"Length of classes: {len(self.classes)}")
        temp_weights = torch.eye(len(self.classes), device = "cuda")
        torch.nn.init.xavier_uniform_(temp_weights, gain=1.0)
        print(temp_weights)
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
        optimizer = torch.optim.SGD([temp_weights, temp_bias], lr=self.lr)
        import time
        time_start = time.time()
        num_epochs = 300
        for epoch in range(num_epochs):
            eceEvaluator_perimage = perimageCCE(n_classes = len(self.classes))
            epoch_loss_cce_total = 0
            epoch_loss_cross_entropy_total = 0
            epoch_loss_total = 0
            epoch_loss_info_entropy_total = 0
            for i, (images, targets, filenames) in enumerate(self.val_loader):
                # import pdb; pdb.set_trace()
                optimizer.zero_grad()

                images = images.to(self.device)
                targets = targets.to(self.device)

                # print(image.shape)
                with torch.no_grad():
                    # outputs = model.evaluate(images)

                    # outputs = torch.rand(1,3,300,400)
                    outputs = torch.ones(1,2,300,400)*(torch.Tensor([0.51,0.49]).reshape(1,-1,1,1))
                    # outputs = torch.ones(1,4,300,400)*(torch.Tensor([0.5,0.25,0.15, 0.1]).reshape(1,-1,1,1))
                    outputs = outputs.cuda()
                    outputs[0,0,:, :200] = 0.49
                    outputs[0,1,:, 200:] = 0.51

                    # outputs = torch.ones(1,3,300,400)*(torch.Tensor([0.7,0.2,0.1]).reshape(1,-1,1,1))
                    # # outputs = torch.ones(1,4,300,400)*(torch.Tensor([0.5,0.25,0.15, 0.1]).reshape(1,-1,1,1))
                    # outputs = outputs.cuda()
                    # outputs[0,0,100:200, 50:150] = 0.1
                    # outputs[0,0,100:150, 250:300] = 0.2
                    # outputs[0,1,100:200, 50:150] = 0.7
                    # outputs[0,1,100:150, 250:300] = 0.1
                    # outputs[0,2,100:200, 50:150] = 0.2
                    # outputs[0,2,100:150, 250:300] = 0.7


                    # Converting back to logits
                    outputs = torch.log(outputs)

                outputs = outputs.permute(0, 2, 3, 1).contiguous()
                outputs = torch.matmul(outputs, temp_weights)
                outputs = outputs + temp_bias


                outputs = outputs.permute(0, 3, 1, 2).contiguous()
                
                # Add image stuff
                save_imgs = torch.softmax(outputs, dim =1).squeeze(0)
                # analyse(outputs = save_imgs.unsqueeze(0))
                # accuracy(outputs = outputs)
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

                    self.writer.add_image(f"Class_{class_no}", img_dif, epoch, dataformats="HWC")

                loss_cce = cce_criterion.forward(outputs, targets)
                loss_cross_entropy = cross_criterion.forward(outputs, targets)
                loss_info_entropy = info_entropy_criterion.forward(outputs)

                alpha = 1
                total_loss = loss_cce + alpha * loss_info_entropy
                # total_loss = loss_cross_entropy

                epoch_loss_info_entropy_total += loss_info_entropy
                epoch_loss_cce_total += loss_cce.item()
                epoch_loss_cross_entropy_total += loss_cross_entropy.item()
                epoch_loss_total += total_loss.item()

                total_loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    for output, target in zip(outputs,targets.detach()):
                        # older ece requires softmax and size output=[class,w,h] target=[w,h]
                        eceEvaluator_perimage.update(output.softmax(dim=0), target)
                # print(outputs.shape)
                # print(eceEvaluator_perimage.get_overall_CCELoss())
                print(f"batch :{i+1}/{len(self.val_loader)}" + "loss cce : {:.5f} | loss cls : {:.5f} | loss tot : {:.5f}".format(loss_cce, loss_cross_entropy, total_loss))
            
            print(temp_weights)
            print(temp_bias)
            epoch_loss_cce_total /= len(self.val_loader)
            epoch_loss_cross_entropy_total /= len(self.val_loader)
            epoch_loss_total /= len(self.val_loader)
            
            count_table_image, _ = eceEvaluator_perimage.get_count_table_img(self.classes)
            cce_table_image, dif_map= eceEvaluator_perimage.get_perc_table_img(self.classes)
            self.writer.add_image("CCE_table", cce_table_image, epoch, dataformats="HWC")
            self.writer.add_image("Count table", count_table_image, epoch, dataformats="HWC")
            self.writer.add_image("DifMap", dif_map, epoch, dataformats="HWC")
            
            self.writer.add_scalar(f"Cross EntropyLoss_LR", epoch_loss_cross_entropy_total, epoch)
            self.writer.add_scalar(f"Info EntropyLoss_LR", epoch_loss_info_entropy_total, epoch)
            self.writer.add_scalar(f"CCELoss_LR", epoch_loss_cce_total, epoch)
            self.writer.add_scalar(f"Total Loss_LR", epoch_loss_total, epoch)
            self.writer.add_histogram("Weights", temp_weights, epoch)
            self.writer.add_histogram("Bias", temp_bias, epoch)
            # output = output/temp_weights
            # print(output.shape)
            # print(temp_weights, temp_bias)

            if epoch > 0 and epoch % 10 == 0:
                print("saving weights.")
                np.save("weights/toy/wt_{}_{}.npy".format(epoch, self.prefix), temp_weights.cpu().detach().numpy())
                np.save("weights/toy/b{}_{}.npy".format(epoch, self.prefix), temp_bias.cpu().detach().numpy())


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
