import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import quadprog
import torch
from mmengine.model import detect_anomalous_params
from mmengine.runner import BaseLoop
from mmengine.runner.utils import calc_dynamic_intervals
from torch import Tensor
from torch.overrides import handle_torch_function
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS


@LOOPS.register_module()
class EpochBasedTrainLoopVOCBase(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

        # self.runner.model._set_static_graph()

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.

        # outputs = self.runner.model.train_step(
        #     data_batch, optim_wrapper=self.runner.optim_wrapper)

        with self.runner.optim_wrapper.optim_context(self):
            data = self.runner.model.module.data_preprocessor(data_batch, training=True)
            losses = self.runner.model._run_forward(data, mode='loss')

        """
        self.runner.model 是MMDistributedDataParallel封装了的，MMDistributedDataParallel重写了
        train_step()方法，因此base_model中的train_step()方法不会生效。
        
        """
        loss_old_positive, loss_new_positive, loss_shared, log_vars = self.runner.model.module.parse_losses(losses)
        # -------------------------------------------------------
        # 替换：self.runner.optim_wrapper.update_params(parsed_loss)
        step_kwargs = {}
        zero_kwargs = {}
        loss_old_positive = self.runner.optim_wrapper.scale_loss(loss_old_positive)
        loss_new_positive = self.runner.optim_wrapper.scale_loss(loss_new_positive)
        loss_shared = self.runner.optim_wrapper.scale_loss(loss_shared)


        self.runner.optim_wrapper.backward(loss_new_positive + loss_shared + loss_old_positive)
        # self.runner.optim_wrapper.backward(loss_new_positive + loss_shared)

        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.runner.optim_wrapper.should_update():
            # self.runner.optim_wrapper.step(**step_kwargs)
            if self.runner.optim_wrapper.clip_grad_kwargs:
                self.runner.optim_wrapper._clip_grad()
            self.runner.optim_wrapper.optimizer.step(**step_kwargs)
            self.runner.optim_wrapper.zero_grad(**zero_kwargs)

        # -------------------------------------------------------
        if self.runner.model.detect_anomalous_params:
            detect_anomalous_params(loss_old_positive + loss_new_positive + loss_shared, model=self)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=log_vars)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]