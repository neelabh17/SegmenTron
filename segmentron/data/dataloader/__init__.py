"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .mscoco import COCOSegmentation
from .cityscapes import CitySegmentation
from .ade import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .sbu_shadow import SBUSegmentation
from .toy import ToySegmentation
from .cityscapes_noisy import CitySegmentation_Noisy
from .pascal_context import VOCContextSegmentation
datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'coco': COCOSegmentation,
    'cityscape': CitySegmentation,
    'sbu': SBUSegmentation,
    "toy": ToySegmentation,
    "cityscape_noisy": CitySegmentation_Noisy,
    'pascal_context': VOCContextSegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
