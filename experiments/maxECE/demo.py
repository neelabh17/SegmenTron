import os
import sys
import torch
import cv2
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from torchvision import transforms
import torch.nn.functional as F

from PIL import Image
from segmentron.utils.visualize import get_color_pallete
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.config import cfg
from new_tools.convcrf import GaussCRF, get_default_conf
import matplotlib.pyplot as plt

classes=["__background__",
"aeroplane",
"bicycle",
"bird",
"boat",
"bottle",
"bus",
"car",
"cat",
"chair",
"cow",
"diningtable",
"dog",
"horse",
"motorbike",
"person",
"pottedplant",
"sheep",
"sofa",
"train",
"tvmonitor"]


def giveComparisionImages(precrf_output,postcrf_output, raw_image, gt_label):
    '''
    precrf_output-> [1,21,h,w] cuda tensor
    postcrf_output-> [1,21,h,w] cuda tensor
    raw_image->[3,h,w]
    gt_label->[h,w]
    '''
    uncal_labels = np.unique(torch.argmax(precrf_output.squeeze(0),dim=0).cpu().numpy())
    cal_labels = np.unique(torch.argmax(postcrf_output.squeeze(0),dim=0).cpu().numpy())
    
    
    # gt_label=get_gt_with_id(imageName)
    # if(np.sum((cal_labelmap!=uncal_labelmap).astype(np.float32))==0):
    if False:
        pass
    else:
        print("yay")
        # Show result for each class
        cols = int(np.ceil((max(len(uncal_labels), len(cal_labels)) + 1)))+1
        rows = 4
        
        

        plt.figure(figsize=(20, 20))
        ax = plt.subplot(rows, cols, 1)
        ax.set_title("Input image")
        ax.imshow(raw_image[:, :, ::-1])
        ax.axis("off")
        ax = plt.subplot(rows, cols, cols + 1)
        ax.set_title("Accuracy dif = {:0.3f}".format(loss))
        ax.imshow(raw_image[:, :, ::-1])
        ax.axis("off")
     

        for i, label in enumerate(uncal_labels):
            ax = plt.subplot(rows, cols, i + 3)
            ax.set_title("Uncalibrated-" + classes[label])
            ax.imshow(precrf_output[label], cmap="nipy_spectral")

            ax.axis("off")

        for i, label in enumerate(cal_labels):
            ax = plt.subplot(rows, cols, cols + i + 3)
            ax.set_title("Calibrated-" + classes[label])
            ax.imshow(postcrf_output[label], cmap="nipy_spectral")
            ax.axis("off")

        for i, label in enumerate(cal_labels):
            ax = plt.subplot(rows, cols, 2 * cols + i + 3)
            
            min_dif=np.min(precrf_output[label] - postcrf_output[label])
            max_dif=np.max(precrf_output[label] - postcrf_output[label])

            dif_map=np.where((precrf_output[label] - postcrf_output[label])>0,(precrf_output[label] - postcrf_output[label]),0)
            
            ax.set_title(
                "decrease: "
                + classes[label]
                + " max={:0.3f}".format(
                    max_dif
                )
            )
            ax.imshow(
                dif_map
                / max_dif,
                cmap="nipy_spectral",
            )
            ax.axis("off")
        
        for i, label in enumerate(cal_labels):
            ax = plt.subplot(rows, cols, 3 * cols + i + 3)
            
            min_dif=np.min(precrf_output[label] - postcrf_output[label])
            max_dif=np.max(precrf_output[label] - postcrf_output[label])

            dif_map=np.where((precrf_output[label] - postcrf_output[label])<0,(precrf_output[label] - postcrf_output[label]),0)
            
            ax.set_title(
                "increase: "
                + classes[label]
                + " max={:0.3f}".format(
                    -min_dif
                )
            )
            ax.imshow(
                dif_map
                / min_dif,
                cmap="nipy_spectral",
            )
            ax.axis("off")
        
        acc_cal_mask=(gt_label==cal_label).astype(np.float32)
        acc_uncal_mask=(gt_label==uncal_label).astype(np.float32)
        # for i, label in enumerate(cal_labels):
        ax = plt.subplot(rows, cols, 2 * cols + 1)

        maxi=np.max(acc_uncal_mask - acc_cal_mask)
        un=np.unique(acc_uncal_mask - acc_cal_mask)
        print(un)
        dif_map=np.where((acc_uncal_mask - acc_cal_mask)<0,(acc_uncal_mask - acc_cal_mask),0)
        
        ax.set_title(
            "increase ac: "
            + classes[label]
            + " max={:0.3f}".format(
                1
            )
        )
        ax.imshow(
            dif_map/(-1),
            cmap="nipy_spectral",
        )

        # for i, label in enumerate(cal_labels):
        ax = plt.subplot(rows, cols, 3 * cols + 1)
        dif_map=np.where((acc_uncal_mask - acc_cal_mask)>0,(acc_uncal_mask - acc_cal_mask),0)
        
        ax.set_title(
            "decrease ac: "
            + classes[label]
            + " max={:0.3f}".format(
                1
            )
        )
        ax.imshow(
            dif_map/1,
            cmap="nipy_spectral",
        )
            
        
        

        plt.tight_layout()
        return plt
        # save_file = os.path.join(saveDir, imageName+f"_{30-img_no}-30")
        # plt.savefig(save_file)



def demo():
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    # temp=1.8
    temp=1.6
    # usingCRF=False
    usingCRF=True
    # output folder
    output_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, 'snow_1_conv_9_{}_{}_{}_{}_temp_{}_crf_{}'.format(
        cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE, cfg.DATASET.NAME, cfg.TIME_STAMP,temp,usingCRF))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
    ])

    model = get_segmentation_model().to(args.device)
    model.eval()

    if os.path.isdir(args.input_img):
        img_paths = [os.path.join(args.input_img, x) for x in os.listdir(args.input_img)]
    else:
        img_paths = [args.input_img]
    for img_path in img_paths:
        image_pil = Image.open(img_path).convert('RGB')
        image = transform(image_pil).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output = model.evaluate(image).detach()
            # output shape is [1,21,w,h] connected to cuda

        # import pdb; pdb.set_trace()
        print(img_path)
        if(usingCRF):
            non_cali_crf_output = output.clone()
            output /=temp


            pre_crf_pred = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()
            pre_crf_mask = get_color_pallete(pre_crf_pred, cfg.DATASET.NAME)

            raw_image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32).transpose(2, 0, 1)
            raw_image = torch.from_numpy(raw_image).to(args.device)
            raw_image = raw_image.unsqueeze(dim=0)

            # output shape is [1,21,w,h]
            num_classes=output.shape[1]
            crf = GaussCRF(conf=get_default_conf(), shape=image.shape[2:], nclasses=num_classes, use_gpu=True)
            crf = crf.to(args.device)
            assert image.shape == raw_image.shape
            output = crf.forward(output, raw_image)
            # print(output.shape)

            # Saving the image
            pred = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()
            mask = get_color_pallete(pred, cfg.DATASET.NAME)
            outname = os.path.splitext(os.path.split(img_path)[-1])[0] + f'_temp_{temp}_crf_{usingCRF}.png'
            
            # Uncalibrated image withth crf
            non_cali_crf_output = crf.forward(non_cali_crf_output, raw_image)
            non_cali_crf_pred = torch.argmax(non_cali_crf_output, 1).squeeze(0).cpu().data.numpy()
            non_cali_crf_mask = get_color_pallete(non_cali_crf_pred, cfg.DATASET.NAME)

            # Concatenating horizontally [out_post_crf,out_pre_crf, rgb]
            dst = Image.new('RGB', (4*mask.width+9, mask.height), color="white")
            dst.paste(mask, (0, 0))
            dst.paste(non_cali_crf_mask, (mask.width+3, 0))
            dst.paste(pre_crf_mask, (2*mask.width+6, 0))
            dst.paste(image_pil, (3*mask.width+9, 0))
            dst.save(os.path.join(output_dir, outname))
            
        else:
            pred = torch.argmax(output, 1).squeeze(0).cpu().data.numpy()
            mask = get_color_pallete(pred, cfg.DATASET.NAME)
            
            # Concatenating horizontally [output, rgb]
            dst = Image.new('RGB', (mask.width + image_pil.width, mask.height))
            dst.paste(mask, (0, 0))
            dst.paste(image_pil, (mask.width, 0))
            outname = os.path.splitext(os.path.split(img_path)[-1])[0] + f'_temp_{temp}_crf_{usingCRF}.png'
            
            # mask.save(os.path.join(output_dir, outname))
            dst.save(os.path.join(output_dir, outname))


if __name__ == '__main__':
    demo()
