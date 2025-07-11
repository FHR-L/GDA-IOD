# Copyright (c) OpenMMLab. All rights reserved.
from .epoch_based_train_loop_voc_base import EpochBasedTrainLoopVOCBase
from .gpr import GPR
from .loops import TeacherStudentValLoop

__all__ = ['TeacherStudentValLoop',
           'GPR',
           'EpochBasedTrainLoopVOCBase',
]
