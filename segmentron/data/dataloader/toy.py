"""Prepare Cityscapes dataset"""
import os
import torch
import numpy as np
import logging

from PIL import Image
from .seg_data_base import SegmentationDataset

class ToySegmentation(SegmentationDataset):
    
    """Cityscapes Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/cityscapes'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = CitySegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    

    NUM_CLASS = 3
    # NUM_CLASS = 4
    def __init__(self, root='datasets', split='train', mode=None, transform=None, **kwargs):
        pass
        

   
    def __getitem__(self, index):

        # Unequal boxes
        img = torch.zeros(300, 400)

        # 3 classes, 2 box
        img[:, :200] = 1

        # Equal boxes
        # img = torch.zeros(300, 400)
        # img[100:200, 50:150] = 1
        # img[100:200, 250:350] = 2



        # mask = torch.rand(300,400)

        # introducing a third class
        # img [ mask < (1/3) ] = 3


        return img, img.clone().long(), "nofile"
        

        
    
    
    def __len__(self):
        # return min(100, len(self.images))
        return 1
        # return 200
        # return len(self.images)

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        # return ("backGround","blue", "green", "phantom")
        # return ("backGround","blue", "green")
        return ("backGround","foreground")




if __name__ == '__main__':
    dataset = ToySegmentation()
    print(dataset)
