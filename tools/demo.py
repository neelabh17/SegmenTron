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
from crf import DenseCRF
from crfasrnn.crfrnn import CrfRnn



def demo():
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()
    default_setup(args)

    temp=1.8
    # temp=3
    usingCRF=False
    usingCRF=True
    # output folder
    output_dir = os.path.join(cfg.VISUAL.OUTPUT_DIR, 'vis_result_{}_{}_{}_{}_temp_{}_crf_{}'.format(
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
        image = Image.open(img_path).convert('RGB')
        images = transform(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            output = model(images)
        # import pdb;pdb.set_trace()
        # output=output
        _, H, W = images[0].shape
        logit = F.interpolate(output[0], size=(H, W), mode="bilinear", align_corners=True)

        print(img_path,logit.shape)
        logit/=temp
        # output_prob=F.softmax(logit/temp,dim=1)
        # output_prob=output_prob.cpu().numpy()
        if(usingCRF):
            # raw_image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)
            # mean_bgr=np.array([103.53, 116.28, 123.675])
            # # Do some subtraction
            # raw_image-=mean_bgr
            # # converted to C H W
            # raw_image=raw_image.transpose(2,0,1)
            # raw_image=raw_image.astype(np.uint8)
            # raw_image=raw_image.transpose(1,2,0)
            # raw_images.append(raw_image)
            # postprocessor= DenseCRF(iter_max=cfg.CRF.ITER_MAX,
            #                                 pos_xy_std=cfg.CRF.POS_XY_STD,
            #                                 pos_w=cfg.CRF.POS_W,
            #                                 bi_xy_std=cfg.CRF.BI_XY_STD,
            #                                 bi_rgb_std=cfg.CRF.BI_RGB_STD,
            #                                 bi_w=cfg.CRF.BI_W,
            #                             )
            postprocessor = CrfRnn(21)
            raw_image = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32).transpose(2, 0, 1)

            raw_image = torch.from_numpy(raw_image).unsqueeze(dim=0)
            
            prob_post=postprocessor(raw_image,logit.cpu().softmax(dim=1))
            print(prob_post.shape)
            pred = np.argmax(prob_post.squeeze(0).detach().numpy(), axis=0)
        else:
            pred = torch.argmax(logit, 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, cfg.DATASET.NAME)
        outname = os.path.splitext(os.path.split(img_path)[-1])[0] + f'_temp_{temp}_crf_{usingCRF}.png'
        mask.save(os.path.join(output_dir, outname))


if __name__ == '__main__':
    demo()
