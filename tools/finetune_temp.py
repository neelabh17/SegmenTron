from pdb import set_trace
import time
import copy
import datetime
import os
import sys
import matplotlib.pyplot as plt

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.solver.loss import get_segmentation_loss
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import show_flops_params
from segmentron.config import cfg

from torch.utils.tensorboard import SummaryWriter
from calibration_library.metrics import CCELoss as perimageCCE
from calibration_library.cce_loss import CCELoss
from calibration_library.ece_loss import ECELoss


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # self.prefix = "randomtesting"
        # self.prefix = "cce_alpha={}_sigma=0.5".format(cfg.TRAIN.ALPHA)
        self.prefix = "temp fcn scaling testing full={}".format(cfg.TRAIN.ALPHA)
        self.writer = SummaryWriter(log_dir= f"CCE_train_both_eval/{self.prefix}")
        self.writer_noisy = SummaryWriter(log_dir= f"CCE_train_both_eval/{self.prefix}-foggy")

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}

        # OUR TRAIN AND VAL ARE ALWAYS NORMAL CITYSCAPES
        train_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='val', mode=cfg.DATASET.MODE, **data_kwargs)
        self.classes = val_dataset.classes
        self.iters_per_epoch = len(train_dataset) // (args.num_gpus * cfg.TRAIN.BATCH_SIZE)
        self.max_iters = cfg.TRAIN.EPOCHS * self.iters_per_epoch

        self.ece_evaluator = ECELoss(n_classes = len(self.classes))
        self.cce_evaluator = CCELoss(n_classes = len(self.classes))

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters, drop_last=True)

        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, cfg.TEST.BATCH_SIZE, drop_last=False)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            num_workers=cfg.DATASET.WORKERS,
                                            pin_memory=True)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)

        # DEFINE data for noisy
        val_dataset_noisy = get_segmentation_dataset(cfg.DATASET.NOISY_NAME, split='val', mode=cfg.DATASET.MODE, **data_kwargs)
        self.val_loader_noisy = data.DataLoader(dataset=val_dataset_noisy,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)

        # create network
        self.model = get_segmentation_model().to(self.device)

        # print (self.model)
        # import pdb; pdb,set_trace()
        for params in self.model.encoder.parameters():
            params.requires_grad = False

        
        # print params and flops
        if get_rank() == 0:
            try:
                show_flops_params(copy.deepcopy(self.model), args.device)
            except Exception as e:
                logging.warning('get flops and params error: {}'.format(e))

        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Not use SyncBatchNorm!')

        # create criterion
        self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
                                               aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
                                               ignore_index=cfg.DATASET.IGNORE_INDEX, n_classes=len(train_dataset.classes), alpha=cfg.TRAIN.ALPHA).to(self.device)

        # optimizer, for model just includes encoder, decoder(head and auxlayer).
        self.optimizer = get_optimizer(self.model)

        # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters,
                                          iters_per_epoch=self.iters_per_epoch)

        # resume checkpoint if needed
        self.start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            name, ext = os.path.splitext(args.resume)
            assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
            logging.info('Resuming training, loading {}...'.format(args.resume))
            resume_sate = torch.load(args.resume)
            self.model.load_state_dict(resume_sate['state_dict'])
            self.start_epoch = resume_sate['epoch']
            logging.info('resume train from epoch: {}'.format(self.start_epoch))
            if resume_sate['optimizer'] is not None and resume_sate['lr_scheduler'] is not None:
                logging.info('resume optimizer and lr scheduler from resume state..')
                self.optimizer.load_state_dict(resume_sate['optimizer'])
                self.lr_scheduler.load_state_dict(resume_sate['lr_scheduler'])

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class, args.distributed)
        self.best_pred = 0.0


    def train(self):
        self.save_to_disk = get_rank() == 0
        epochs, max_iters, iters_per_epoch = cfg.TRAIN.EPOCHS, self.max_iters, self.iters_per_epoch
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.iters_per_epoch

        start_time = time.time()
        logging.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0

        total_loss = 0
        for (images, targets, _) in self.train_loader:
            epoch = iteration // iters_per_epoch + 1
            iteration += 1

            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss_dict, loss_cal, loss_nll  = self.criterion(outputs, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            total_loss += losses_reduced.item()

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and self.save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], losses_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))

                self.writer.add_scalar("Loss", losses_reduced.item(), iteration )
                self.writer.add_scalar("CCE oe ECE part of Loss", loss_cal.item(), iteration )
                self.writer.add_scalar("NLL Part", loss_nll.item(), iteration )
                self.writer.add_scalar("LR", self.optimizer.param_groups[0]['lr'], iteration )

            if iteration % self.iters_per_epoch == 0 and self.save_to_disk:
                save_checkpoint(self.model, epoch, self.optimizer, self.lr_scheduler, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation(epoch, self.val_loader, self.writer)
                self.validation(epoch, self.val_loader_noisy, self.writer_noisy)
                self.model.train()

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logging.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self, val_loader, writer, temp = 1):
        self.metric.reset()
        self.ece_evaluator.reset()
        self.cce_evaluator.reset()
        eceEvaluator_perimage = perimageCCE(n_classes = len(self.classes))

        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()
        model.eval()

        for i, (image, target, filename) in enumerate(val_loader):
            image = image.to(self.device)
            target = target.to(self.device)


            with torch.no_grad():
                if cfg.DATASET.MODE == 'val' or cfg.TEST.CROP_SIZE is None:
                    output = model(image)[0]
                else:
                    size = image.size()[2:]
                    pad_height = cfg.TEST.CROP_SIZE[0] - size[0]
                    pad_width = cfg.TEST.CROP_SIZE[1] - size[1]
                    image = F.pad(image, (0, pad_height, 0, pad_width))
                    output = model(image)[0]
                    output = output[..., :size[0], :size[1]]

            # output is [1, 19, 1024, 2048] logits
            # target is [1, 1024, 2048]

            output /= temp
            self.metric.update(output, target)

            if (i==0):
                import cv2
                image_read = cv2.imread(filename[0])
                writer.add_image("Image[0] Read", image_read, temp*10, dataformats="HWC")

                save_imgs = torch.softmax(output, dim =1)[0]
                for class_no, class_distri in enumerate(save_imgs):
                    plt.clf()
                    class_distri[0][0] = 0
                    class_distri[0][1] = 1

                    im = plt.imshow(class_distri.detach().cpu().numpy(),cmap="Greens")
                    plt.colorbar(im)
                    plt.savefig("temp_files/temp.jpg")
                    plt.clf()
                    import cv2
                    img_dif = cv2.imread("temp_files/temp.jpg")

                    writer.add_image(f"Class_{self.classes[class_no]}", img_dif, temp*10, dataformats="HWC")
            
            with torch.no_grad():
                self.ece_evaluator.forward(output,target)
                self.cce_evaluator.forward(output,target)
                # for output, target in zip(output,target.detach()):
                #     #older ece requires softmax and size output=[class,w,h] target=[w,h]
                #     eceEvaluator_perimage.update(output.softmax(dim=0), target)                

            pixAcc, mIoU = self.metric.get()
            logging.info("[EVAL] Sample: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc * 100, mIoU * 100))
        
        pixAcc, mIoU = self.metric.get()
        logging.info("[EVAL END] temp: {:f}, pixAcc: {:.3f}, mIoU: {:.3f}".format(temp, pixAcc * 100, mIoU * 100))
        writer.add_scalar("Temp [EVAL END] pixAcc", pixAcc * 100, temp*10 )
        writer.add_scalar("Temp [EVAL END] mIoU", mIoU * 100, temp*10 )

        # count_table_image, _ = eceEvaluator_perimage.get_count_table_img(self.classes)
        # cce_table_image, dif_map = eceEvaluator_perimage.get_perc_table_img(self.classes)

        ece_count_table_image, _ = self.ece_evaluator.get_count_table_img(self.classes)
        ece_table_image, ece_dif_map = self.ece_evaluator.get_perc_table_img(self.classes)
        
        cce_count_table_image, _ = self.cce_evaluator.get_count_table_img(self.classes)
        cce_table_image, cce_dif_map = self.cce_evaluator.get_perc_table_img(self.classes)
        
        # writer.add_image("Temp CCE_table", cce_table_image, temp, dataformats="HWC")
        # writer.add_image("Temp CCE Count table", count_table_image, temp, dataformats="HWC")
        # writer.add_image("Temp CCE DifMap", dif_map, temp, dataformats="HWC")
        # writer.add_scalar("Temp CCE Score", eceEvaluator_perimage.get_overall_CCELoss(), temp)

        writer.add_image("Temp ece_table", ece_table_image, temp*10, dataformats="HWC")
        writer.add_image("Temp ece Count table", ece_count_table_image, temp*10, dataformats="HWC")
        writer.add_image("Temp ece DifMap", ece_dif_map, temp*10, dataformats="HWC")
        writer.add_scalar("Temp ece Score", self.ece_evaluator.get_overall_ECELoss(), temp*10)

        writer.add_image("Temp cce_table", cce_table_image, temp*10, dataformats="HWC")
        writer.add_image("Temp cce Count table", cce_count_table_image, temp*10, dataformats="HWC")
        writer.add_image("Temp cce DifMap", cce_dif_map, temp*10, dataformats="HWC")
        writer.add_scalar("Temp cce Score", self.cce_evaluator.get_overall_CCELoss(), temp*10)
        synchronize()
        # if self.best_pred < mIoU and self.save_to_disk:
        #     self.best_pred = mIoU
        #     logging.info('Epoch {} is the best model, best pixAcc: {:.3f}, mIoU: {:.3f}, save the model..'.format(temp*10, pixAcc * 100, mIoU * 100))
        #     save_checkpoint(model, temp*10, is_best=True)
    def temp_scaling_validation(self, temp = 1):
        self.validation( self.val_loader, self.writer, temp = temp)
        self.validation( self.val_loader_noisy, self.writer_noisy, temp = temp)



if __name__ == '__main__':
    args = parse_args()
    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)

    # create a trainer and start train
    trainer = Trainer(args)
    # trainer.train()
    # temps = [ 0.1, 0.2 ,0.3 ,0.4 ,0.5 ,0.6 ,0.7, 0.8 , 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,2.1, 2.2 ,2.3 ,2.4 ,2.5 ,2.6 ,2.7, 2.8 , 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
    temps = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,2.1, 2.2     ,2.3 ,2.4 ,2.5 ,2.6 ,2.7, 2.8 , 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
    for i, temp in enumerate(temps):
        if(i%2==1):
            trainer.temp_scaling_validation(temp = temp)
    # trainer.validation(0)

