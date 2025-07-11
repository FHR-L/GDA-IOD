# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_head import BBoxHead
from .bbox_head_increase import BBoxHeadIncrease
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .convfc_bbox_head_increase import Shared4Conv1FCBBoxHeadIncrease, ConvFCBBoxHeadIncrease, Shared2FCBBoxHeadIncrease
from .dii_head import DIIHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .multi_instance_bbox_head import MultiInstanceBBoxHead
from .sabl_head import SABLHead
from .scnet_bbox_head import SCNetBBoxHead

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead',
    'SCNetBBoxHead', 'MultiInstanceBBoxHead',
    'BBoxHeadIncrease', 'ConvFCBBoxHeadIncrease', 'Shared2FCBBoxHeadIncrease',
    'Shared4Conv1FCBBoxHeadIncrease'
]
