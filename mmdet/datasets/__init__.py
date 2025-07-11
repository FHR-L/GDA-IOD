# Copyright (c) OpenMMLab. All rights reserved.
from .base_det_dataset import BaseDetDataset
from .cityscapes import CityscapesDataset
from .coco import CocoDataset, CocoDataset70ShuffleV1Base, CocoDataset70ShuffleV1Task2
from .coco_increase import CocoDataset40Order, CocoDataset40_20Order, CocoDataset40_20_20Order, CocoDataset70Order, \
    CocoDataset70_10Order
from .coco_panoptic import CocoPanopticDataset
from .crowdhuman import CrowdHumanDataset
from .dataset_wrappers import MultiImageMixDataset
from .deepfashion import DeepFashionDataset
from .lvis import LVISDataset, LVISV1Dataset, LVISV05Dataset
from .objects365 import Objects365V1Dataset, Objects365V2Dataset
from .openimages import OpenImagesChallengeDataset, OpenImagesDataset
from .samplers import (AspectRatioBatchSampler, ClassAwareSampler,
                       GroupMultiSourceSampler, MultiSourceSampler)
from .utils import get_loading_pipeline
from .voc import VOCDataset, VOCDataset10, VOCDataset10_10, VOCDataset10_5, VOCDataset10_5_5, VOCDataset15, VOCDataset5, \
    VOCDataset19, VOCDataset19_1, VOCDataset10_2, VOCDataset12_2, VOCDataset14_2, VOCDataset16_2, VOCDataset18_2, \
    VOCDataset16, VOCDataset18, VOCDataset17, VOCDataset11, VOCDataset12, VOCDataset13, VOCDataset14
from .wider_face import WIDERFaceDataset
from .xml_style import XMLDataset

__all__ = [
    'XMLDataset', 'CocoDataset', 'DeepFashionDataset', 'VOCDataset',
    'CityscapesDataset', 'LVISDataset', 'LVISV05Dataset', 'LVISV1Dataset',
    'WIDERFaceDataset', 'get_loading_pipeline', 'CocoPanopticDataset',
    'MultiImageMixDataset', 'OpenImagesDataset', 'OpenImagesChallengeDataset',
    'AspectRatioBatchSampler', 'ClassAwareSampler', 'MultiSourceSampler',
    'GroupMultiSourceSampler', 'BaseDetDataset', 'CrowdHumanDataset',
    'Objects365V1Dataset', 'Objects365V2Dataset',
    'CocoDataset70ShuffleV1Base', 'CocoDataset70ShuffleV1Task2',
    'CocoDataset40Order', 'CocoDataset40_20Order', 'CocoDataset40_20_20Order',
    'VOCDataset10', 'VOCDataset10_10', 'VOCDataset10_5', 'VOCDataset10_5_5',
    'VOCDataset5', 'VOCDataset15', 'CocoDataset70Order', 'CocoDataset70_10Order',
    'VOCDataset19',
    'VOCDataset19_1',
    'VOCDataset10_2',
    'VOCDataset12_2',
    'VOCDataset14_2',
    'VOCDataset16_2',
    'VOCDataset16',
    'VOCDataset17',
    'VOCDataset18',
    'VOCDataset18_2',
    'VOCDataset11',
    'VOCDataset12',
    'VOCDataset13',
    'VOCDataset14'
]
